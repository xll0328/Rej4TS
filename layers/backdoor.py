import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pywt


def haar_wavelet_transform(x):
    # 计算Haar小波变换
    n = x.size(1)  # 时间步长
    coeffs = torch.zeros_like(x)
    
    for i in range(n // 2):
        avg = (x[:, 2*i] + x[:, 2*i+1]) / 2
        diff = (x[:, 2*i] - x[:, 2*i+1]) / 2
        coeffs[:, i] = avg
        coeffs[:, n // 2 + i] = diff
    
    return coeffs[:,0:n //2],coeffs[:,n //2 : n]

def inverse_haar_wavelet_transform(coeffs_avg, coeffs_diff):
    coeffs = torch.concatenate([coeffs_avg, coeffs_diff],dim=1)
    n = coeffs.size(1)  # 小波系数数量
    x = torch.zeros_like(coeffs)
    
    for i in range(n // 2):
        x[:, 2*i] = coeffs[:, i] + coeffs[:, n // 2 + i]
        x[:, 2*i+1] = coeffs[:, i] - coeffs[:, n // 2 + i]
    
    return x

# 假设 batch_x 是你的输入数据
batch_x = torch.rand(32, 96, 7)
batch_x_attack = batch_x.clone()


# 对每个特征进行小波变换和攻击特征添加
for i in range(batch_x_attack.shape[2]):
    coeffs_avg, coeffs_diff = haar_wavelet_transform(batch_x_attack[:, :, i])

    # 添加攻击特征
    # 确保形状匹配，使用扩展. torch.linspace 0 到 2 * np.pi 之间生成 96 个等间隔的点。  torch.sin每个值应用正弦函数
    attack_feature = torch.sin(torch.linspace(0, 2 * np.pi, 48)).unsqueeze(0).expand(32, -1)  # 形状为[32, 96]
    coeffs_diff+= attack_feature 

    # 逆小波变换
    batch_x_attack[:, :, i] = inverse_haar_wavelet_transform(coeffs_avg, coeffs_diff)

# 现在 batch_x 已经过小波变换和攻击特征添加

feature_index = 0
batch_index = 0
original_image = torch.reshape(batch_x[batch_index, :, feature_index] ,[1, 96])
original_image = original_image.numpy()

transformed_image = torch.reshape(batch_x_attack[batch_index, :, feature_index] ,[1, 96])
transformed_image = transformed_image.numpy()
# 可视hua
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(transformed_image, cmap='gray')
axes[1].set_title('Transformed Image')
plt.show()