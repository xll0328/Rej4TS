from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pywt
warnings.filterwarnings('ignore')

def haar_wavelet_transform(x):
    n = x.size(1)  
    coeffs = torch.zeros_like(x)
    
    for i in range(n // 2):
        avg = (x[:, 2*i] + x[:, 2*i+1]) / 2
        diff = (x[:, 2*i] - x[:, 2*i+1]) / 2
        coeffs[:, i] = avg
        coeffs[:, n // 2 + i] = diff
    
    return coeffs[:,0:n //2],coeffs[:,n //2 : n]

def inverse_haar_wavelet_transform(coeffs_avg, coeffs_diff):
    coeffs = torch.concatenate([coeffs_avg, coeffs_diff],dim=1)
    n = coeffs.size(1) 
    x = torch.zeros_like(coeffs)
    
    for i in range(n // 2):
        x[:, 2*i] = coeffs[:, i] + coeffs[:, n // 2 + i]
        x[:, 2*i+1] = coeffs[:, i] - coeffs[:, n // 2 + i]
    
    return x

def square_wave(t, frequency=1, amplitude=1):
    return amplitude * torch.sign(torch.sin(2 * np.pi * frequency * t))

def sawtooth_wave(t, frequency=1, amplitude=1):
    return amplitude * (2 * (t * frequency / (2 * np.pi) - torch.floor(0.5 + t * frequency / (2 * np.pi))))



def apply_wavelet_transform(batch_x):
    batch_size, seq_length, num_features = batch_x.shape
    approx_coeffs = []
    detail_coeffs = []

    for i in range(batch_size):
        approx_sample = []
        detail_sample = []
        for j in range(num_features):
            coeffs = pywt.dwt(batch_x[i, :, j], 'db1')  
            cA, cD = coeffs  
            approx_sample.append(cA)
            detail_sample.append(cD)
        approx_sample = np.stack(approx_sample, axis=1)  # [seq_length//2, num_features]
        detail_sample = np.stack(detail_sample, axis=1)
        approx_coeffs.append(approx_sample)
        detail_coeffs.append(detail_sample)

    approx_coeffs = np.stack(approx_coeffs, axis=0)  # [batch_size, seq_length//2, num_features]
    detail_coeffs = np.stack(detail_coeffs, axis=0)

    return approx_coeffs, detail_coeffs

class FeatureHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        # x: [batch_size, seq_length]
        return self.model(x)  # [batch_size, hidden_dim]

class VAEFeatureProcessor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(VAEFeatureProcessor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2 , hidden_dim),  # mu, log_var, recon_error
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.recon = nn.Linear(1, hidden_dim)

    def forward(self, recon_error, mu, log_var):
        # recon_error: [batch_size, 1]
        # mu, log_var: [batch_size, latent_dim]
        x = torch.cat([mu, log_var], dim=1)  # [batch_size, latent_dim * 2 + 1]
        recon_error_lin = self.recon(recon_error)
        return self.model(x) + recon_error_lin   # [batch_size, hidden_dim] self.model(x) + recon_error_lin 
    
class Rejector(nn.Module):
    def __init__(self, latent_dim, device, feature_dim):
        super(Rejector, self).__init__()
        self.num_features = feature_dim  
        self.seq_length = 48 
        self.hidden_dim = 256
        self.latent_dim = latent_dim
        self.device = device 
        self.approx_heads = nn.ModuleList([FeatureHead(self.seq_length, self.hidden_dim) for _ in range(self.num_features)])
        self.detail_heads = nn.ModuleList([FeatureHead(self.seq_length, self.hidden_dim) for _ in range(self.num_features)])

        self.vae_processor = VAEFeatureProcessor(self.latent_dim, self.hidden_dim * self.num_features * 2)

        total_features = self.hidden_dim * self.num_features * 2  
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, approx_coeffs, detail_coeffs, recon_error, mu, log_var, vae_weight = 2 ):
        batch_size = approx_coeffs.size(0)
        feature_outputs = []

        for i in range(self.num_features):
            approx_feature = approx_coeffs[:, :, i]  
            detail_feature = detail_coeffs[:, :, i]

            approx_output = self.approx_heads[i](approx_feature)  # [batch_size, hidden_dim]
            detail_output = self.detail_heads[i](detail_feature)

            feature_outputs.append(approx_output)
            feature_outputs.append(detail_output)

        vae_output = self.vae_processor(recon_error, mu, log_var)  # [batch_size, hidden_dim]
        vae_output = vae_output * vae_weight
        fusion_input = torch.cat(feature_outputs, dim=1)  + vae_output
        output = self.fusion_layer(fusion_input)  # [batch_size, 1]
        output = torch.sigmoid(output).squeeze()

        return output  # [batch_size]

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=192):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Linear(192, latent_dim * 2)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 192),
            nn.ReLU(),
            nn.Linear(192, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, log_var

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss(reduction='none')
        if self.args.loss == 'L1':
            criterion = nn.L1Loss(reduction='none')
        elif self.args.loss == 'MSE':
            criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {self.args.loss}")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                loss_mean = loss.mean().item()  # 聚合为标量
                total_loss.append(loss_mean)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test_with_rejection(self, test_loader, path,rejection_features_dim = 1072):
        data_iter = iter(test_loader)
        batch = next(data_iter)
        data = batch[0]
        dim_vae = data.shape[2]
        self.vae = VAE(input_dim=96 * dim_vae).to(self.device)
        vae_path = os.path.join(path, 'vae.pth')
        self.vae.load_state_dict(torch.load(vae_path))
        self.vae.eval()
        self.mu_train_mean = np.load(os.path.join(path, 'mu_train_mean.npy'))
        self.cov_train_inv = np.load(os.path.join(path, 'cov_train_inv.npy'))

        latent_dim = self.args.rejector_latent_dim
        self.rejector = Rejector(latent_dim=latent_dim, device=self.device,feature_dim = dim_vae).to(self.device)
        rejector_path = os.path.join(path, 'rejector.pth')
        self.rejector.load_state_dict(torch.load(rejector_path))
        self.rejector.eval()

        self.model.eval()
        test_loss = []
        all_predictions = []
        all_true = []
        reject_flags = []
        all_p_reject = []  
        features_list = []  
        all_recon_errors = [] 
        all_maha_distances = []  
        all_novelty_scores = []  
        lambda_recon = self.args.lambda_recon 
        lambda_maha = self.args.lambda_maha 
        batch_num = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_num = batch_num + 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = nn.MSELoss(reduction='none')(outputs, batch_y_target)  # [batch_size, pred_len, features]
                loss_per_sample = loss.mean(dim=(1, 2))  # [batch_size]
                test_loss.extend(loss_per_sample.cpu().numpy())

                all_predictions.append(outputs.cpu().numpy())
                all_true.append(batch_y_target.cpu().numpy())

                approx_coeffs_np, detail_coeffs_np = apply_wavelet_transform(batch_x.cpu().numpy())
                approx_coeffs = torch.tensor(approx_coeffs_np, dtype=torch.float32).to(self.device)
                detail_coeffs = torch.tensor(detail_coeffs_np, dtype=torch.float32).to(self.device)

                batch_size, seq_length, num_features = batch_x.size()
                reshaped_inputs = batch_x.view(batch_size, -1)  

                recon, mu, log_var = self.vae(reshaped_inputs.to(self.device))
                recon_error = nn.MSELoss(reduction='none')(recon, reshaped_inputs.to(self.device))
                recon_error = recon_error.mean(dim=1).cpu().numpy()  # [batch_size]
                recon_error_tensor = torch.from_numpy(recon_error).float().to(self.device)
                recon_error_tensor = recon_error_tensor.view(-1,1)
                all_recon_errors.extend(recon_error)  

                mu_np = mu.detach().cpu().numpy()  # [batch_size, latent_dim]
                delta = mu_np - self.mu_train_mean  # [batch_size, latent_dim]
                m_distance = np.sqrt(np.einsum('ij,jk,ik->i', delta, self.cov_train_inv, delta))  # [batch_size]
                all_maha_distances.extend(m_distance)

                novelty_score = lambda_recon * recon_error + lambda_maha * m_distance
                all_novelty_scores.extend(novelty_score)
                p_reject = self.rejector(approx_coeffs, detail_coeffs, recon_error_tensor, mu, log_var, vae_weight = self.args.vae_weight)
                p_reject = torch.reshape(p_reject, [batch_size])
                all_p_reject.extend(p_reject.cpu().numpy())
        threshold_reject_p = np.percentile(all_p_reject, self.args.reject_p)
        print(f"Rejection probability threshold set to: {threshold_reject_p:.4f}")

        threshold_novelty_score = np.percentile(all_novelty_scores, self.args.novelty_score)
        print(f"Reconstruction error threshold set to: {threshold_novelty_score:.4f}")
        # A threshold can also be set directly
        reject_flags = []
        for i in range(len(test_loss)):
            p_reject_i = all_p_reject[i]
            novelty_score_i = all_novelty_scores[i]
            if p_reject_i >= threshold_reject_p or novelty_score_i >= threshold_novelty_score:
                reject_flags.append(True)
            else:
                reject_flags.append(False)
        accepted_losses = []
        rejected_losses = []
        for i in range(len(test_loss)):
            if reject_flags[i]:
                rejected_losses.append(test_loss[i])
            else:
                accepted_losses.append(test_loss[i])

        if len(accepted_losses) > 0:
            test_loss_avg_accepted = np.average(accepted_losses)
        else:
            test_loss_avg_accepted = float('inf')  

        reject_rate = len(rejected_losses) / len(test_loss)

        c = self.args.c # 0.1

        test_loss_avg_all = np.average(test_loss)

        if reject_rate > 0:
            test_loss_avg_accepted = np.average(accepted_losses)
            overall_loss = c * reject_rate + test_loss_avg_accepted
        else:
            overall_loss = c
        
        print(f"Test Loss (All Samples): {test_loss_avg_all:.4f}")
        print(f"Test Loss (Accepted Samples): {test_loss_avg_accepted:.4f}")
        print(f"Number of Rejected Samples: {len(rejected_losses)} out of {len(test_loss)}")
        print(f"Reject Rate: {reject_rate:.4f}")
        print(f"Overall Expected Loss (Including Rejection Cost): {overall_loss:.4f}")


        return test_loss_avg_all, test_loss_avg_accepted, reject_flags

    def train_rejector(self, features, labels, path, feature_dim = 7): 
        latent_dim = 192 
        self.rejector = Rejector(latent_dim=latent_dim, device=self.device, feature_dim = feature_dim).to(self.device)
        optimizer = torch.optim.Adam(self.rejector.parameters(), lr=0.00001)
        criterion = nn.BCELoss()
        
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(features, labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        num_epochs = self.args.rejector_epochs
        for epoch in range(num_epochs):
            epoch_loss = 0
            itr = 0
            for batch_x, batch_y in train_loader:
                itr = itr + 1
                batch_x = batch_x.float().to(self.device)  # [batch_size, 96, 7]
                batch_y = batch_y.float().to(self.device)
                #if  batch_x. < 2:
                #      outputs = torch.reshape(outputs,[1])
                if itr ==67:
                    a = 0
                if batch_x.numel() == 672:
                    batch_x = batch_x.view(1,96,7)
                    batch_y = batch_y.view(1)
                if batch_x.numel() == 43680:
                    batch_x = batch_x.view(65,96,7)
                    batch_y = batch_y.view(65)
                if batch_x.numel() == 86016:
                    batch_x = batch_x.view(128,96,7)
                    batch_y = batch_y.view(128)
                else:
                    a = 0

                
                if batch_x.ndim == 3:
                    batch_size, seq_length, num_features = batch_x.shape
                else:
                    batch_x = batch_x.reshape(-1, 96,7) # ett 7 weather 21 exchange 8 
                #print(batch_x.size())
                approx_coeffs_np, detail_coeffs_np = apply_wavelet_transform(batch_x.cpu().numpy())
                approx_coeffs = torch.tensor(approx_coeffs_np, dtype=torch.float32).to(self.device)
            
                detail_coeffs = torch.tensor(detail_coeffs_np, dtype=torch.float32).to(self.device)
                reshaped_inputs = batch_x.view(batch_x.size(0), -1)  # [batch_size, 96 * 7]
                recon, mu, log_var = self.vae(reshaped_inputs)
                recon_error = nn.MSELoss(reduction='none')(recon, reshaped_inputs)
                recon_error = recon_error.mean(dim=1, keepdim=True)  # [batch_size, 1]

                optimizer.zero_grad()
                
                outputs = self.rejector(approx_coeffs, detail_coeffs, recon_error, mu, log_var, vae_weight = self.args.vae_weight)
                if batch_x.numel() == 672:
                    outputs = outputs.view(1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_y.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Rejector Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        rejector_path = os.path.join(path, 'rejector.pth')
        torch.save(self.rejector.state_dict(), rejector_path)
        print(f"Rejector model saved at {rejector_path}")
        self.rejector.eval()
        all_preds = []
        all_labels = []

    def test_vali(self, vali_data, vali_loader, criterion):
        def format_and_print_losses(sample_losses, values_per_row=16, decimal_places=4, max_rows=8):
            if isinstance(sample_losses, torch.Tensor):
                sample_losses = sample_losses.tolist()

            num_samples = len(sample_losses)
            rows_to_print = min(max_rows, (num_samples + values_per_row - 1) // values_per_row)

            print(f"Sample Losses (showing {min(num_samples, rows_to_print * values_per_row)} out of {num_samples} samples):")
            
            for row in range(rows_to_print):
                start_idx = row * values_per_row
                end_idx = min(start_idx + values_per_row, num_samples)
                row_values = sample_losses[start_idx:end_idx]
                
                formatted_row = " ".join([f"{value:.{decimal_places}f}".rjust(decimal_places + 3) for value in row_values])
                print(formatted_row)
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                sample_losses = []
                for i in range(pred.shape[0]):
                    sample_loss = criterion(pred[i], true[i])
                    sample_losses.append(sample_loss.item())
                format_and_print_losses(sample_losses)
                loss = sum(sample_losses) / pred.shape[0]

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train_vae(self, ideal_samples, path):
        if ideal_samples.size(0) == 0:
            print("No ideal samples to train VAE.")
            return
        
        ideal_samples = ideal_samples.to(self.device)
        self.args.vae_epochs # = 400
        self.vae.train()
        for vae_epoch in range(self.args.vae_epochs):
            self.vae_optimizer.zero_grad()
            recon, mu, log_var = self.vae(ideal_samples)
            recon_loss = self.vae_criterion(recon, ideal_samples)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = recon_loss + kl_loss
            vae_loss.backward()
            self.vae_optimizer.step()
            
            if (vae_epoch + 1) % 50 == 0 or vae_epoch == 0:
                print(f"VAE Epoch [{vae_epoch+1}/{self.args.vae_epochs}], Loss: {vae_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
        
        print("VAE training completed.")
        vae_path = os.path.join(path, 'vae.pth')
        torch.save(self.vae.state_dict(), vae_path)
        print(f"VAE model saved at {vae_path}")
        with torch.no_grad():
            recon, mu, log_var = self.vae(ideal_samples)
            mu_np = mu.detach().cpu().numpy()  # [num_samples, latent_dim]

        mu_train_mean = np.mean(mu_np, axis=0)  # [latent_dim]
        cov_train = np.cov(mu_np, rowvar=False)  # [latent_dim, latent_dim]
        epsilon = 1e-6
        cov_train += epsilon * np.eye(cov_train.shape[0])
        cov_train_inv = np.linalg.inv(cov_train)

        self.mu_train_mean = mu_train_mean
        self.cov_train_inv = cov_train_inv
        np.save(os.path.join(path, 'mu_train_mean.npy'), mu_train_mean)
        np.save(os.path.join(path, 'cov_train_inv.npy'), cov_train_inv)
        print("mu_train_mean and cov_train_inv saved.")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        self.ideal_features = []
        self.amb_features = []
        ideal_threshold = None  
        data_iter = iter(test_loader)
        batch = next(data_iter)
        data = batch[0]
        dim_vae = data.shape[2]
        self.vae = VAE(input_dim=96 * dim_vae).to(self.device) 
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=0.0001)
        self.vae_criterion = nn.MSELoss(reduction='mean')  
        rejection_features = []
        rejection_labels = []
        rejection_features_rej = []
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_rej = nn.MSELoss(reduction='none')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []       
            epoch_losses = []      
            epoch_losses_for_rej = []      
            epoch_inputs = []     
            epoch_targets = []    
            rejection_features = []
            rejection_labels = []
            rejection_features_rej = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    loss_rej = criterion_rej(outputs, batch_y) 
                    loss_per_sample = loss_rej.mean(dim=(1, 2)) 
                    epoch_losses.extend(loss_per_sample.detach().cpu().numpy())  
                    epoch_losses_for_rej.extend(loss_rej.detach().cpu().numpy())
                    # train_loss.append(loss.item())

                epoch_inputs.extend(batch_x.detach().cpu())
                epoch_targets.extend(batch_y.detach().cpu())


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.mean().backward()
                    model_optim.step()

            ideal_threshold = np.percentile(epoch_losses, 100) 
            if epoch > self.args.vae_train_start:
                epoch_losses_np = np.array(epoch_losses)
                epoch_losses_np_for_rej = np.array(epoch_losses_for_rej)
                epoch_inputs_tensor = torch.stack(epoch_inputs, dim=0)  
                epoch_targets_tensor = torch.stack(epoch_targets, dim=0) 


                threshold_loss = np.percentile(epoch_losses_np, self.args.labels)
                threshold_loss_for_rej = np.percentile(epoch_losses_np_for_rej, self.args.labels) 
                print(f"Initial prediction error threshold set to: {threshold_loss_for_rej}")
                labels = (epoch_losses_np > threshold_loss).astype(int) 
                
                batch_size, seq_length, num_features = epoch_inputs_tensor.size()
                reshaped_inputs = epoch_inputs_tensor.view(batch_size, -1)  # [num_samples, seq_length * num_features]
                self.train_vae(reshaped_inputs, path)
                recon, mu, log_var = self.vae(reshaped_inputs.to(self.device))
                recon_error = nn.MSELoss(reduction='none')(recon, reshaped_inputs.to(self.device))
                recon_error = recon_error.mean(dim=1).detach().cpu().numpy() 

                input_features = reshaped_inputs.cpu().numpy()  # [num_samples, input_dim]

                mu_np = mu.detach().cpu().numpy()        # [num_samples, latent_dim]
                log_var_np = log_var.detach().cpu().numpy()  # [num_samples, latent_dim]
                features = np.concatenate([input_features], axis=1)
                rejection_features.append(features)
                rejection_labels.append(labels)
            epoch_losses = [] 
            epoch_inputs = []
            epoch_targets = [] 
            self.ideal_features = []

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        rejection_features = np.concatenate(rejection_features, axis=0)  
        rejection_labels = np.concatenate(rejection_labels, axis=0)     
        rejection_features_tensor = torch.tensor(rejection_features, dtype=torch.float32).to(self.device)
        rejection_features_dim = rejection_features_tensor.size(1) 
        self.train_rejector(rejection_features, rejection_labels, path, feature_dim = dim_vae)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # print("Starting testing with rejection strategy...")
        test_loss_all, test_loss_accepted, reject_flags = self.test_with_rejection(test_loader, path, rejection_features_dim)
        print(test_loss_accepted)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        return self.model   

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return
    
   