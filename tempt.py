# xxxxxx
# # test_meta_feature_extraction.py

# import torch
# import numpy as np
# import pandas as pd
# import numpy as np
# import pandas as pd
# from scipy.stats import skew, kurtosis, entropy
# from scipy.signal import periodogram
# from statsmodels.tsa.stattools import acf, adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.ar_model import AutoReg


# def extract_meta_feature(data):
#     """
#     Extracts meta-features from a given time series data.

#     Parameters:
#     - data: np.ndarray, shape (n_samples, n_features), time series data

#     Returns:
#     - features: dict, contains the extracted meta-features
#     """
#     features = {}

#     # basic statistics
#     # axis=0 表示沿着时间轴计算（即对每个变量单独计算）
#     # .mean() 再在所有变量上取平均 → 得到一个综合性的标量特征
#     features["mean"] = np.mean(data, axis=0).mean()
#     features["std"] = np.std(data, axis=0).mean()
#     features["min"] = np.min(data, axis=0).mean()
#     features["max"] = np.max(data, axis=0).mean()
#     features["skewness"] = np.nanmean(skew(data, axis=0))
#     features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))

#     # time series decomposition
#     acfs = [acf(data[:, i], nlags=10, fft=True) for i in range(data.shape[1])]
#     features["autocorrelation_mean"] = np.nanmean(
#         [acf_val[1] for acf_val in acfs]
#     )  # first lag
#     adf_results = [adfuller(data[:, i]) for i in range(data.shape[1])]
#     features["stationarity"] = np.mean([result[1] < 0.05 for result in adf_results])

#     # rate_of_change = np.diff(data, axis=0) / data[:-1]
#     # Deal with 0 division
#     safe_data = np.where(data[:-1] == 0, np.nan, data[:-1])
#     rate_of_change = np.diff(data, axis=0) / safe_data
#     features["rate_of_change_mean"] = np.nanmean(rate_of_change)
#     features["rate_of_change_std"] = np.nanstd(rate_of_change)

#     # Landmarker features
#     autoreg_coefs, residual_stds = [], []
#     for i in range(data.shape[1]):
#         model = AutoReg(data[:, i], lags=1).fit()
#         autoreg_coefs.append(model.params[1])
#         residual_stds.append(np.std(model.resid))
#     features["autoreg_coef_mean"] = np.mean(autoreg_coefs)
#     features["residual_std_mean"] = np.mean(residual_stds)

#     # frequency domain features
#     freq_means, freq_peaks, spectral_entropies = [], [], []
#     spectral_variations, spectral_skewnesses, spectral_kurtoses = [], [], []

#     for i in range(data.shape[1]):
#         freqs, psd = periodogram(data[:, i])
#         freq_means.append(np.mean(psd))
#         freq_peaks.append(freqs[np.argmax(psd)])
#         spectral_entropies.append(entropy(psd))
#         if i > 0:
#             prev_psd = periodogram(data[:, i - 1])[1]
#             spectral_variations.append(np.sqrt(np.sum((psd - prev_psd) ** 2)))
#         else:
#             spectral_variations.append(0)  # 第一个变量无法计算变化
#         spectral_skewnesses.append(skew(psd))
#         spectral_kurtoses.append(kurtosis(psd))

#     features["frequency_mean"] = np.mean(freq_means)
#     features["frequency_peak"] = np.mean(freq_peaks)
#     features["spectral_entropy"] = np.nanmean(spectral_entropies)
#     features["spectral_variation"] = np.nanmean(spectral_variations)
#     features["spectral_skewness"] = np.nanmean(spectral_skewnesses)
#     features["spectral_kurtosis"] = np.nanmean(spectral_kurtoses)

#     cov_matrix = np.cov(data, rowvar=False)
#     features["covariance_mean"] = np.mean(cov_matrix)
#     features["covariance_max"] = np.max(cov_matrix)
#     features["covariance_min"] = np.min(cov_matrix)
#     features["covariance_std"] = np.std(cov_matrix)

#     return features


# # Step 2: 批量提取并转换为张量的函数
# def batch_extract_meta_features(batch_x):

#     print(batch_x.shape)
#     try:
#         batch_x = batch_x.numpy()  # 如果是 tensor 就转成 numpy
#     except AttributeError:
#         pass  # 已经是 numpy，无需处理

#     batch_meta_features = []
#     for i in range(len(batch_x)):
#         meta_features = extract_meta_feature(batch_x[i])
#         batch_meta_features.append(meta_features)

#     # 转为 PyTorch 张量 (float32 类型)
#     meta_tensor = torch.tensor(batch_meta_features, dtype=torch.float32)
#     # 转为 DataFrame
#     batch_meta_features = pd.DataFrame(batch_meta_features)

#     return batch_meta_features, meta_tensor


# # Step 3: 测试用例
# def test_batch_extract_meta_features():
#     print("🚀 开始测试 batch_extract_meta_features 函数...\n")

#     # 创建模拟输入：(batch_size=4, sequence_length=100) 的时间序列数据
#     np.random.seed(42)
#     fake_data = np.random.randn(4, 10, 100)  # 4 个样本，每个 100 个点

#     # 包装成 PyTorch Tensor（模拟 DataLoader 输出）
#     batch_x = torch.tensor(fake_data, dtype=torch.float32)

#     print(f"输入数据形状: {batch_x.shape} (batch_size, sequence_length)")

#     # 调用函数提取元特征并转为张量
#     batch_meta_features, meta_features_tensor = batch_extract_meta_features(batch_x)

#     print(f"输出张量形状: {meta_features_tensor.shape} (应为 [4, 7] 因为有 7 个元特征)")
#     print("输出张量内容:")
#     print(meta_features_tensor)

#     # 额外：打印原始 DataFrame 查看结构

#     print(batch_meta_features.round(4))

#     print("\n✅ 测试通过！")


# # Step 4: 运行测试
# if __name__ == "__main__":
#     test_batch_extract_meta_features()

# xxxxxxx 测试meta_feature_v计算元特征的方法

# import torch
# import torch.nn as nn

# import time
# import torch
# from layers.meta_feature_v import (
#     batch_extract_meta_features_gpu,
#     extract_meta_features_per_variable_gpu,
#     batch_extract_meta_features,
# )


# def test_meta_batch():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     B, T, N = 32, 512, 200
#     x = torch.randn(B, T, N, device=device)
#     seq_len, pred_len = T, 720
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     m_batch = batch_extract_meta_features_gpu(x, seq_len, pred_len)
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     t1 = time.perf_counter()
#     #     feats = []
#     #     if device.type == "cuda":
#     #         torch.cuda.synchronize()
#     #     t2 = time.perf_counter()
#     #     for b in range(B):
#     #         feats.append(extract_meta_features_per_variable_gpu(x[b], seq_len, pred_len))
#     #     m_stack = torch.stack(feats, dim=0)
#     #     if device.type == "cuda":
#     #         torch.cuda.synchronize()
#     #     t3 = time.perf_counter()
#     #     eq = torch.allclose(m_batch, m_stack, atol=1e-6, rtol=1e-5)
#     #     diff = (m_batch - m_stack).abs().max().item()
#     print("shape_batch", m_batch.shape)
#     # print("shape_stack", m_stack.shape)
#     # print("equal", eq)
#     # print("max_abs_diff", diff)
#     print("time_batch_s", t1 - t0)
#     # print("time_stack_s", t3 - t2)
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     t2 = time.perf_counter()
#     meta_cpu = batch_extract_meta_features(x, seq_len, pred_len)
#     if device.type == "cuda":
#         torch.cuda.synchronize()
#     t3 = time.perf_counter()
#     print("shape_batch_cpu", meta_cpu.shape)
#     print("cpu_time_batch_s", t3 - t2)


# if __name__ == "__main__":
#     test_meta_batch()


# def fuse_ts2img_select_best(ts2img_tensor_list, weights):
#     """
#     对每个样本，选择权重最大的那种 TS2Img 表示进行融合（样本级自适应选择）。

#     Args:
#         ts2img_tensor_list (list of torch.Tensor): 每个元素形状为 (B, C, H, W)
#         weights (torch.Tensor): 形状为 (B, d_ts2img)，表示每个样本在每种 TS2Img 方法上的权重

#     Returns:
#         fused_tensor (torch.Tensor): 形状为 (B, C, H, W)，融合后的结果
#     """
#     assert len(ts2img_tensor_list) > 0, "ts2img_tensor_list 不能为空"
#     B, C, H, W = ts2img_tensor_list[0].shape
#     d_ts2img = len(ts2img_tensor_list)

#     # 检查 shapes 是否一致
#     for i, tensor in enumerate(ts2img_tensor_list):
#         assert tensor.shape == (
#             B,
#             C,
#             H,
#             W,
#         ), f"第 {i} 个 tensor 形状不匹配: {tensor.shape}"

#     assert weights.shape == (
#         B,
#         d_ts2img,
#     ), f"权重形状应为 (B, d_ts2img)，但得到 {weights.shape}"

#     # 获取每个样本权重最大的方法索引: shape (B,)
#     best_indices = torch.argmax(weights, dim=-1)  # (B,)

#     # 构建输出: 对每个样本 i，取 ts2img_tensor_list[best_indices[i]][i]
#     fused_tensors = []
#     A = weights.shape[0]
#     print(A, B)
#     for i in range(B):
#         chosen_idx = best_indices[i].item()  # 转为 Python int
#         selected_tensor = ts2img_tensor_list[chosen_idx][i]  # (C, H, W)
#         fused_tensors.append(selected_tensor)

#     # 堆叠回 batch 维度
#     fused_tensor = torch.stack(fused_tensors)  # (B, C, H, W)
#     return fused_tensor


# # ========================================
# # 🔧 测试代码
# # ========================================


# def test_fuse_ts2img_select_best():
#     print("开始测试 fuse_ts2img_select_best 函数...\n")

#     # 设置随机种子以保证可复现
#     torch.manual_seed(42)

#     # 模拟参数
#     B = 4  # batch size
#     C = 3  # channel
#     H = 32  # height
#     W = 32  # width
#     d_ts2img = 3  # 三种 TS2Img 方法：比如 GAF, MTF, RP

#     print(f"构造数据：B={B}, C={C}, H={H}, W={W}, d_ts2img={d_ts2img}")

#     # 构造 ts2img_tensor_list: 3 个 (B, C, H, W) 的张量
#     ts2img_tensor_list = []
#     method_names = ["GAF", "MTF", "RP"]
#     for i in range(d_ts2img):
#         # 模拟不同方法生成的图像表示（随机初始化）
#         tensor = torch.randn(B, C, H, W) + i * 0.5  # 稍微偏移以便区分
#         ts2img_tensor_list.append(tensor)
#         print(f"{method_names[i]} tensor shape: {tensor.shape}")

#     # 构造 weights: (B, d_ts2img)，表示每个样本对三种方法的偏好
#     weights = torch.softmax(torch.randn(B, d_ts2img), dim=-1)
#     print(f"\nWeights (softmax 后): \n{weights}\n")

#     # 打印每个样本选择的是哪个方法
#     best_indices = torch.argmax(weights, dim=-1)
#     print("每个样本选择的方法索引:", best_indices.tolist())
#     print("对应方法:", [method_names[i] for i in best_indices.tolist()])

#     # 执行融合
#     fused_tensor = fuse_ts2img_select_best(ts2img_tensor_list, weights)

#     print(f"\n融合后输出形状: {fused_tensor.shape}")
#     assert fused_tensor.shape == (B, C, H, W), "输出形状错误！"

#     # 验证某个样本是否正确选择
#     for i in range(B):
#         chosen_idx = best_indices[i].item()
#         expected = ts2img_tensor_list[chosen_idx][i]
#         actual = fused_tensor[i]
#         assert torch.allclose(expected, actual), f"样本 {i} 选择错误！"
#     print("\n✅ 所有测试通过！融合逻辑正确。")

#     return fused_tensor


# # 运行测试
# if __name__ == "__main__":
#     result = test_fuse_ts2img_select_best()

# xxxxxxx
# import torch
# import torch.nn as nn


# # 判断一个形状为 [C, H, W] 的 Tensor 是否为灰度图像
# def is_grayscale_tensor(tensor, tol=1e-6):

#     if tensor.shape[0] == 1:
#         return True
#     elif tensor.shape[0] == 3:
#         # 拆分三个通道
#         r, g, b = tensor[0], tensor[1], tensor[2]

#         # 检查 R 和 G 的差异，R 和 B 的差异
#         diff_rg = torch.abs(r - g)
#         diff_rb = torch.abs(r - b)

#         # 如果所有差异都小于容忍度，则认为是灰度图
#         return (diff_rg < tol).all() and (diff_rb < tol).all()
#     else:
#         raise ValueError("Warning: Unexpected number of channels")


# def fuse_ts2img_top3_grayscale_stack(ts2img_tensor_list, weights):
#     """
#     对每个样本选择权重最高的 Top-3 TS2Img 表示，
#     若为灰度图则取单通道，最后在通道维度堆叠成新的 (B, 3, H, W) 表示。

#     Args:
#         ts2img_tensor_list (list of torch.Tensor): 每个形状为 (B, 3, H, W)
#         weights (torch.Tensor): 形状为 (B, d_ts2img)

#     Returns:
#         fused_tensor (torch.Tensor): (B, 3, H, W)，由 Top-3 的单通道图像堆叠而成
#     """
#     assert (
#         len(ts2img_tensor_list) >= 3
#     ), "ts2img_tensor_list 至少要有 3 个表示才能选 Top-3"
#     B, C, H, W = ts2img_tensor_list[0].shape
#     assert C == 3, "每个 TS2Img 表示应为 3 通道"
#     d_ts2img = len(ts2img_tensor_list)
#     assert weights.shape == (
#         B,
#         d_ts2img,
#     ), f"权重形状应为 (B, {d_ts2img})，但得到 {weights.shape}"

#     # 获取 Top-3 索引 (B, 3)
#     top3_indices = torch.topk(weights, k=3, dim=-1).indices  # (B, 3)

#     fused_batch = []
#     for i in range(B):
#         # 当前样本选择的三种方法索引
#         idx0, idx1, idx2 = top3_indices[i].tolist()

#         channels = []
#         for method_idx in [idx0, idx1, idx2]:
#             img_3ch = ts2img_tensor_list[method_idx][i]  # (3, H, W)

#             # 判断是否为灰度图
#             if is_grayscale_tensor(img_3ch):
#                 # 取第一个通道作为灰度值
#                 gray_channel = img_3ch[0:1]  # (1, H, W)
#             else:
#                 # 如果不是灰度图，也取第一个通道（或可改为平均）
#                 gray_channel = img_3ch[0:1]  # (1, H, W)

#             channels.append(gray_channel)

#         # 将三个 (1, H, W) 通道堆叠成 (3, H, W)
#         fused_img = torch.cat(channels, dim=0)  # (3, H, W)
#         fused_batch.append(fused_img)

#     # 堆叠成 batch
#     fused_tensor = torch.stack(fused_batch)  # (B, 3, H, W)
#     return fused_tensor


# # ========================================
# # 🔧 测试代码
# # ========================================


# def test_fuse_top3_grayscale_stack():
#     print("开始测试 Top-3 灰度图堆叠融合策略...\n")
#     torch.manual_seed(42)

#     # 参数
#     B = 2
#     H, W = 16, 16
#     d_ts2img = 5  # 有5种 TS2Img 方法

#     print(f"构造数据：B={B}, H={H}, W={W}, d_ts2img={d_ts2img}")

#     # 构造 ts2img_tensor_list
#     ts2img_tensor_list = []
#     method_names = [f"Method_{i}" for i in range(d_ts2img)]

#     for i in range(d_ts2img):
#         if i % 2 == 0:
#             # 偶数方法：构造灰度图（三通道相同）
#             gray_value = torch.randn(1, H, W)
#             tensor = torch.cat([gray_value] * 3, dim=0)  # (3, H, W)
#             print(f"{method_names[i]}: 灰度图")
#         else:
#             # 奇数方法：构造彩色图（三通道不同）
#             tensor = torch.randn(3, H, W)
#             print(f"{method_names[i]}: 非灰度图")

#         # 扩展为 batch
#         batch_tensor = tensor.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 3, H, W)
#         ts2img_tensor_list.append(batch_tensor)

#     # 构造权重 (B, d_ts2img)
#     weights = torch.softmax(torch.randn(B, d_ts2img), dim=-1)
#     print(f"\nWeights:\n{weights}\n")

#     # 获取每个样本的 Top-3
#     top3_indices = torch.topk(weights, 3, dim=-1).indices
#     print("每个样本选择的 Top-3 方法索引:")
#     for i in range(B):
#         selected = [method_names[idx] for idx in top3_indices[i].tolist()]
#         print(f"  样本 {i}: {selected}")

#     # 执行融合
#     fused_tensor = fuse_ts2img_top3_grayscale_stack(ts2img_tensor_list, weights)
#     print(f"\n融合后形状: {fused_tensor.shape}")
#     assert fused_tensor.shape == (B, 3, H, W), "输出形状错误！"

#     # 验证：检查是否真的从 Top-3 中提取并堆叠
#     for i in range(B):
#         idx0, idx1, idx2 = top3_indices[i].tolist()
#         ch0 = ts2img_tensor_list[idx0][i][0]  # 取第一个通道
#         ch1 = ts2img_tensor_list[idx1][i][0]
#         ch2 = ts2img_tensor_list[idx2][i][0]

#         expected = torch.stack([ch0, ch1, ch2], dim=0)  # (3, H, W)
#         actual = fused_tensor[i]

#         assert torch.allclose(expected, actual), f"样本 {i} 融合结果不匹配！"
#     print("\n✅ 所有测试通过！Top-3 灰度图堆叠融合逻辑正确。")

#     return fused_tensor


# # 运行测试
# if __name__ == "__main__":
#     result = test_fuse_top3_grayscale_stack()

# xxxxxxxxxxxxxxxxxxx

# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image
# import torch
# from torchvision import transforms

# # 生成测试数据（模拟小波变换结果）
# np.random.seed(42)  # 固定随机种子，确保可复现
# wavelet_data = np.random.rand(20, 100)  # 20个尺度，100个时间点
# single_series = np.arange(100)  # 模拟时间序列
# scales = np.arange(20)  # 模拟尺度
# W, H = 224, 224  # 目标图像尺寸


# def generate_tensor_and_save(color_map, save_path):
#     """根据指定的colormap生成图像张量并保存图像到本地"""
#     fig, ax = plt.subplots(figsize=(5, 3))  # 固定画布大小

#     # 绘制小波图
#     im = ax.imshow(
#         wavelet_data,
#         origin="upper",
#         aspect="auto",
#         extent=[0, len(single_series), 0, len(scales)],
#         cmap=color_map,
#     )
#     plt.axis("off")  # 关闭坐标轴，避免干扰

#     # 保存到内存缓冲区
#     with BytesIO() as buf:
#         plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
#         buf.seek(0)

#         # 转换为灰度图
#         with Image.open(buf) as img:
#             img_gray = img.convert("L")  # 转为单通道灰度图
#             img_resized = img_gray.resize((W, H), Image.Resampling.LANCZOS)

#             # 保存图像到本地
#             img_resized.save(save_path)

#             # 转换为张量
#             transform = transforms.ToTensor()
#             tensor = transform(img_resized)

#     plt.close(fig)  # 释放内存
#     return tensor


# # 生成并保存两种图像
# tensor_rainbow = generate_tensor_and_save("rainbow", "rainbow_to_gray.png")
# tensor_gray = generate_tensor_and_save("gray", "gray_to_gray.png")

# # 打印张量信息
# print(f"彩虹色映射转灰度张量形状: {tensor_rainbow.shape}")
# print(f"灰度映射转灰度张量形状: {tensor_gray.shape}")
# print(f"两个张量是否完全相同: {torch.allclose(tensor_rainbow, tensor_gray, atol=1e-6)}")

# # 计算张量差异（统计最大值）
# diff = torch.abs(tensor_rainbow - tensor_gray)
# print(f"张量元素最大差异: {diff.max().item():.6f}")


# xxxxxxxxxxxxxxxxxxxxxxxxx
# import torch
# import torch.nn.functional as F


# def adaptive_fusion(ts2img_tensor_list, ts2img_weights):
#     """
#     自适应融合多个时序图像化表示。

#     Args:
#         ts2img_tensor_list: list of tensors, each shape (B, C, H, W)
#         ts2img_weights: tensor of shape (B, d_ts2img), 权重越大越重要

#     Returns:
#         fused_tensor: (B, 3, H, W), 每个样本由top3加权表示拼接而成
#     """
#     B = ts2img_weights.size(0)
#     d_ts2img = len(ts2img_tensor_list)

#     assert d_ts2img == ts2img_weights.size(1), "权重维度应与表示数量一致"

#     # Step 1: 处理每个表示：若 C != 1，则在通道维度取平均 -> 变成 (B, 1, H, W)
#     processed_tensors = []
#     for tensor in ts2img_tensor_list:
#         assert tensor.ndim == 4, f"期望4D张量 (B,C,H,W)，得到 {tensor.shape}"
#         if tensor.size(1) != 1:
#             # 在通道维度取平均，并保持维度
#             print("yasuo")
#             squeezed = tensor.mean(dim=1, keepdim=True)  # (B, 1, H, W)
#         else:
#             squeezed = tensor  # 已经是 (B, 1, H, W)
#         processed_tensors.append(squeezed)

#     # Step 2: 获取每个样本 top-3 的 index (基于权重)
#     _, topk_indices = torch.topk(ts2img_weights, k=3, dim=1)  # (B, 3)

#     # Step 3: 构造输出张量 (B, 3, H, W)
#     device = processed_tensors[0].device
#     dtype = processed_tensors[0].dtype
#     H, W = processed_tensors[0].shape[2], processed_tensors[0].shape[3]

#     fused_tensor = torch.zeros(B, 3, H, W, device=device, dtype=dtype)

#     for b in range(B):
#         for i in range(3):
#             modality_idx = topk_indices[b, i].item()
#             # 取出对应模态的 (B, 1, H, W) 中第 b 个样本
#             tempt = processed_tensors[modality_idx][b : b + 1, :, :, :]
#             # print(f"shape{tempt.shape}")
#             fused_tensor[b, i : i + 1, :, :] = tempt
#     # print(f"shape2{fused_tensor.shape}")
#     return fused_tensor


# def test_adaptive_fusion():
#     print("开始测试 adaptive_fusion 函数...")

#     # 设置随机种子以便复现
#     torch.manual_seed(42)

#     B = 2
#     H, W = 2, 2
#     d_ts2img = 5

#     # 构造输入：不同 C 值的张量列表
#     ts2img_tensor_list = [
#         torch.randn(B, 1, H, W),  # C=1，无需处理
#         torch.randn(B, 1, H, W),  # C=4，需平均
#         torch.randn(B, 1, H, W),  # C=1
#         torch.randn(B, 1, H, W),  # C=3，需平均
#         torch.randn(B, 1, H, W),  # C=1
#     ]

#     print(ts2img_tensor_list)
#     # 权重：(B, d_ts2img)
#     ts2img_weights = torch.randn(B, d_ts2img)
#     print(f"权重矩阵:\n{ts2img_weights}\n")

#     # 执行融合
#     fused_output = adaptive_fusion(ts2img_tensor_list, ts2img_weights)

#     print(fused_output)
#     # 检查输出形状
#     assert fused_output.shape == (B, 3, H, W), f"输出形状错误: {fused_output.shape}"
#     print(f"✅ 输出形状正确: {fused_output.shape}")

#     # 验证每个样本确实是来自 top-3 权重对应的模态
#     _, topk_indices = torch.topk(ts2img_weights, k=3, dim=1)

#     print("\n逐样本验证...")
#     for b in range(B):
#         print(f"\n样本 {b}:")
#         print(f"  Top-3 模态索引: {topk_indices[b].tolist()}")

#         for i in range(3):
#             mod_idx = topk_indices[b, i].item()
#             expected_slice = ts2img_tensor_list[mod_idx][b]
#             if expected_slice.size(0) != 1:  # 如果原始 C > 1，应已平均
#                 expected_slice = expected_slice.mean(dim=0, keepdim=True)  # (1, H, W)

#             actual_slice = fused_output[b, i]  # (H, W)

#             # 检查是否相等
#             diff = (actual_slice - expected_slice.squeeze()).abs().max()
#             assert diff < 1e-6, f"样本{b}, 位置{i}: 不匹配, 最大误差={diff}"
#             print(f"    位置 {i}: 来自模态 {mod_idx}, 匹配 ✓")

#     print("\n🎉 所有测试通过！")


# # 运行测试
# if __name__ == "__main__":
#     test_adaptive_fusion()

# xxxxxxxxxxxxxxxxxxxx
# import torch


# def weighted_sum_fusion(ts2img_tensor_list, ts2img_weights):
#     """
#     对多个时序图像化表示进行加权加和融合。

#     Args:
#         ts2img_tensor_list (list of torch.Tensor): 每个张量形状为 (B, C, H, W)
#         ts2img_weights (torch.Tensor): 形状为 (B, d_ts2img)

#     Returns:
#         fused_tensor (torch.Tensor): 形状为 (B, C, H, W)，加权融合结果
#     """
#     if len(ts2img_tensor_list) == 0:
#         raise ValueError("ts2img_tensor_list 不能为空")

#     B, d_ts2img = ts2img_weights.shape
#     assert (
#         len(ts2img_tensor_list) == d_ts2img
#     ), f"表示数量({len(ts2img_tensor_list)})应与权重第二维({d_ts2img})一致"

#     # 检查所有张量形状是否一致
#     ref_shape = ts2img_tensor_list[0].shape
#     C, H, W = ref_shape[1], ref_shape[2], ref_shape[3]
#     for i, tensor in enumerate(ts2img_tensor_list):
#         if tensor.shape != ref_shape:
#             raise ValueError(
#                 f"张量 {i} 形状 {tensor.shape} 与参考形状 {ref_shape} 不一致"
#             )

#     # 将列表堆叠成 (B, d_ts2img, C, H, W)
#     stacked = torch.stack(ts2img_tensor_list, dim=1)  # → (B, d_ts2img, C, H, W)

#     print(f"stacked.shape{stacked.shape}")
#     # 扩展权重到 (B, d_ts2img, 1, 1, 1)，以便广播
#     weights_expanded = (
#         ts2img_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#     )  # → (B, d_ts2img, 1, 1, 1)

#     print(f"weights_expanded.shape{weights_expanded.shape}")
#     # 加权求和: (B, d_ts2img, C, H, W) * (B, d_ts2img, 1, 1, 1) → (B, C, H, W)
#     fused_tensor = torch.sum(stacked * weights_expanded, dim=1)

#     print(f"fused_tensor.shape{fused_tensor.shape}")
#     return fused_tensor


# def test_weighted_sum_fusion():
#     print("开始测试 weighted_sum_fusion 函数...\n")
#     torch.manual_seed(42)

#     # 设置参数
#     B = 2
#     C = 3
#     H, W = 2, 2
#     d_ts2img = 4

#     # 构造输入：4 种不同的时序图像化表示
#     ts2img_tensor_list = [
#         torch.randn(B, C, H, W) * 1.0 + 0.0,
#         torch.randn(B, C, H, W) * 1.0 + 1.0,
#         torch.randn(B, C, H, W) * 1.0 + 2.0,
#         torch.randn(B, C, H, W) * 1.0 + 3.0,
#     ]

#     # 构造权重：(B, d_ts2img)，每行和不一定为1（支持任意权重）
#     ts2img_weights = torch.softmax(
#         torch.randn(B, d_ts2img), dim=1
#     )  # 使用 softmax 归一化
#     print(f"权重矩阵 (softmax 归一化):\n{ts2img_weights}\n")

#     # 执行融合
#     fused_output = weighted_sum_fusion(ts2img_tensor_list, ts2img_weights)

#     # 检查输出形状
#     assert fused_output.shape == (B, C, H, W), f"输出形状错误: {fused_output.shape}"
#     print(f"✅ 输出形状正确: {fused_output.shape}")

#     # 逐样本验证
#     print("\n逐样本验证...")
#     for b in range(B):
#         print(f"\n--- 样本 {b} ---")
#         expected = torch.zeros(C, H, W)  # 手动计算加权和
#         for i in range(d_ts2img):
#             weight = ts2img_weights[b, i].item()
#             rep = ts2img_tensor_list[i][b]  # (C, H, W)
#             expected += weight * rep
#             print(f"  模态 {i}: 权重={weight:.3f}")

#         actual = fused_output[b]
#         diff = (actual - expected).abs().max()
#         assert diff < 1e-6, f"样本 {b} 验证失败，最大误差={diff}"
#         print(f"✅ 样本 {b} 验证通过，最大误差: {diff:.2e}")

#     print("\n🎉 所有测试通过！加权加和融合功能正常。")


# # 运行测试
# if __name__ == "__main__":
#     test_weighted_sum_fusion()

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import numpy as np
# from utils.dtw_metric import dtw, accelerated_dtw
# from utils.augmentation import run_augmentation, run_augmentation_single

# warnings.filterwarnings("ignore")


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion):
#         total_loss = []
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
#                 vali_loader
#             ):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
#                 dec_inp = (
#                     torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
#                     .float()
#                     .to(self.device)
#                 )
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(
#                             batch_x, batch_x_mark, dec_inp, batch_y_mark
#                         )
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == "MS" else 0
#                 outputs = outputs[:, -self.args.pred_len :, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

#                 pred = outputs.detach().cpu()
#                 true = batch_y.detach().cpu()

#                 loss = criterion(pred, true)

#                 total_loss.append(loss)
#         total_loss = np.average(total_loss)
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag="train")
#         vali_data, vali_loader = self._get_data(flag="val")
#         test_data, test_loader = self._get_data(flag="test")

#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
#                 train_loader
#             ):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
#                 dec_inp = (
#                     torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
#                     .float()
#                     .to(self.device)
#                 )

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(
#                             batch_x, batch_x_mark, dec_inp, batch_y_mark
#                         )

#                         f_dim = -1 if self.args.features == "MS" else 0
#                         outputs = outputs[:, -self.args.pred_len :, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
#                             self.device
#                         )
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                     f_dim = -1 if self.args.features == "MS" else 0
#                     outputs = outputs[:, -self.args.pred_len :, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
#                     loss = criterion(outputs, batch_y)
#                     train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print(
#                         "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
#                             i + 1, epoch + 1, loss.item()
#                         )
#                     )
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * (
#                         (self.args.train_epochs - epoch) * train_steps - i
#                     )
#                     print(
#                         "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
#                             speed, left_time
#                         )
#                     )
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     loss.backward()
#                     model_optim.step()

#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)
#             vali_loss = self.vali(vali_data, vali_loader, criterion)
#             test_loss = self.vali(test_data, test_loader, criterion)

#             print(
#                 "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#                     epoch + 1, train_steps, train_loss, vali_loss, test_loss
#                 )
#             )
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             adjust_learning_rate(model_optim, epoch + 1, self.args)

#         best_model_path = path + "/" + "checkpoint.pth"
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag="test")
#         if test:
#             print("loading model")
#             self.model.load_state_dict(
#                 torch.load(
#                     os.path.join("./checkpoints/" + setting, "checkpoint.pth"),
#                     map_location=self.device,
#                 )
#             )

#         preds = []
#         trues = []
#         folder_path = "./test_results/" + setting + "/"
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
#                 test_loader
#             ):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
#                 dec_inp = (
#                     torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
#                     .float()
#                     .to(self.device)
#                 )
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(
#                             batch_x, batch_x_mark, dec_inp, batch_y_mark
#                         )
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                 f_dim = -1 if self.args.features == "MS" else 0
#                 outputs = outputs[:, -self.args.pred_len :, :]
#                 batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = outputs.shape
#                     outputs = test_data.inverse_transform(
#                         outputs.reshape(shape[0] * shape[1], -1)
#                     ).reshape(shape)
#                     batch_y = test_data.inverse_transform(
#                         batch_y.reshape(shape[0] * shape[1], -1)
#                     ).reshape(shape)

#                 outputs = outputs[:, :, f_dim:]
#                 batch_y = batch_y[:, :, f_dim:]

#                 pred = outputs
#                 true = batch_y

#                 preds.append(pred)
#                 trues.append(true)
#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     if test_data.scale and self.args.inverse:
#                         shape = input.shape
#                         input = test_data.inverse_transform(
#                             input.reshape(shape[0] * shape[1], -1)
#                         ).reshape(shape)
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

#         preds = np.concatenate(preds, axis=0)
#         trues = np.concatenate(trues, axis=0)
#         print("test shape:", preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print("test shape:", preds.shape, trues.shape)

#         # result save
#         folder_path = "./results/" + setting + "/"
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         # dtw calculation
#         if self.args.use_dtw:
#             dtw_list = []
#             manhattan_distance = lambda x, y: np.abs(x - y)
#             for i in range(preds.shape[0]):
#                 x = preds[i].reshape(-1, 1)
#                 y = trues[i].reshape(-1, 1)
#                 if i % 100 == 0:
#                     print("calculating dtw iter:", i)
#                 d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
#                 dtw_list.append(d)
#             dtw = np.array(dtw_list).mean()
#         else:
#             dtw = "not calculated"

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw))
#         f = open("result_long_term_forecast.txt", "a")
#         f.write(setting + "  \n")
#         f.write("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw))
#         f.write("\n")
#         f.write("\n")
#         f.close()

#         np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + "pred.npy", preds)
#         np.save(folder_path + "true.npy", trues)

#         return
# xxxxxxxxxxxxxxxxxxxxx

import torch
import torch.nn as nn

# ===================== 配置你的路径 =====================
PATH_A = "0202/TimeApart/Checkpoint_TimeApart/heat_ETTh1_seasonal_512_192_TimeApart_ETTh1_ETTh1_seasonal_sl512_pl192_dp0.3_0/checkpoint.pth"  # 第一个模型
PATH_B = "0317/TimeApart_old/Checkpoint_TimeApart/cwt_OT_trend_512_96_TimeApart_ETTh1_OT_trend_sl512_pl96_dp0.3_0/checkpoint.pth"  # 第二个模型
SAVE_PATH = "model_checkpoint_diff.txt"
# ======================================================

def load_model(path):
    """加载 checkpoint，返回模型 state_dict"""
    checkpoint = torch.load(path, map_location="cpu")
    # 兼容常见保存格式：直接存model / 存dict / 存state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    return state_dict

def get_model_layers(state_dict):
    """提取层名称、形状、参数数量"""
    layers = []
    for name, param in state_dict.items():
        shape = list(param.shape)
        numel = param.numel()  # 参数总数
        layers.append(f"{name:<60} | shape={str(shape):<25} | params={numel}")
    return sorted(layers)  # 排序后方便对比

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

def find_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "params"]:
            v = checkpoint.get(k)
            if isinstance(v, dict) and sum(torch.is_tensor(x) for x in v.values()) > 0:
                return v, k
        if sum(torch.is_tensor(v) for v in checkpoint.values()) > 0:
            return checkpoint, "<root>"
    raise ValueError("No state_dict found")

def summarize_top_level(checkpoint):
    out = {}
    if not isinstance(checkpoint, dict):
        out["<root_type>"] = type(checkpoint).__name__
        return out
    for k, v in checkpoint.items():
        if torch.is_tensor(v):
            out[k] = f"tensor{tuple(v.shape)}"
        elif isinstance(v, dict):
            out[k] = f"dict(len={len(v)})"
        elif isinstance(v, (list, tuple)):
            out[k] = f"{type(v).__name__}(len={len(v)})"
        else:
            out[k] = repr(v) if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__
    return out

def compare_state_dicts(sd_a, sd_b, topk=50, max_diff_items=2000000):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    shape_mismatch = []
    diffs = []

    for k in common:
        ta = sd_a[k]
        tb = sd_b[k]
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            continue
        if ta.shape != tb.shape:
            shape_mismatch.append((k, tuple(ta.shape), tuple(tb.shape)))
            continue

        numel = ta.numel()
        if numel > max_diff_items:
            idx = torch.randint(0, numel, (max_diff_items,), device=ta.device)
            da = ta.reshape(-1).index_select(0, idx).float()
            db = tb.reshape(-1).index_select(0, idx).float()
            d = (da - db).abs()
            mx = d.max().item()
            mn = d.mean().item()
            sampled = True
        else:
            d = (ta.float() - tb.float()).abs()
            mx = d.max().item()
            mn = d.mean().item()
            sampled = False

        diffs.append((k, mx, mn, numel, sampled))

    diffs.sort(key=lambda x: x[1], reverse=True)
    shape_mismatch.sort(key=lambda x: x[0])

    return {
        "only_a": only_a,
        "only_b": only_b,
        "shape_mismatch": shape_mismatch,
        "diffs": diffs[:topk],
    }

ckpt_a = load_checkpoint(PATH_A)
ckpt_b = load_checkpoint(PATH_B)
sd_a, sd_src_a = find_state_dict(ckpt_a)
sd_b, sd_src_b = find_state_dict(ckpt_b)

layers_a = get_model_layers(sd_a)
layers_b = get_model_layers(sd_b)
cmp = compare_state_dicts(sd_a, sd_b)

# 保存到文件
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write("TOP-LEVEL (A)\n")
    f.write("=" * 100 + "\n")
    for k, v in summarize_top_level(ckpt_a).items():
        f.write(f"{k:<30} {v}\n")
    f.write("\n")

    f.write("=" * 100 + "\n")
    f.write("TOP-LEVEL (B)\n")
    f.write("=" * 100 + "\n")
    for k, v in summarize_top_level(ckpt_b).items():
        f.write(f"{k:<30} {v}\n")
    f.write("\n")

    f.write("=" * 100 + "\n")
    f.write(f"STATE_DICT SOURCE: A={sd_src_a}, B={sd_src_b}\n")
    f.write("=" * 100 + "\n\n")

    f.write("=" * 100 + "\n")
    f.write("KEYS ONLY IN A\n")
    f.write("=" * 100 + "\n")
    f.write("\n".join(cmp["only_a"]) + "\n\n")

    f.write("=" * 100 + "\n")
    f.write("KEYS ONLY IN B\n")
    f.write("=" * 100 + "\n")
    f.write("\n".join(cmp["only_b"]) + "\n\n")

    f.write("=" * 100 + "\n")
    f.write("SHAPE MISMATCHES\n")
    f.write("=" * 100 + "\n")
    if cmp["shape_mismatch"]:
        for k, sa, sb in cmp["shape_mismatch"]:
            f.write(f"{k:<60} {sa} -> {sb}\n")
    else:
        f.write("(none)\n")
    f.write("\n")

    f.write("=" * 100 + "\n")
    f.write("TOP PARAMETER DIFFS (max_abs, mean_abs)\n")
    f.write("=" * 100 + "\n")
    for k, mx, mn, n, sampled in cmp["diffs"]:
        f.write(f"{k:<60} max={mx:.6g} mean={mn:.6g} n={n} sampled={sampled}\n")
    f.write("\n")

    f.write("=" * 100 + "\n")
    f.write(f"MODEL A (state_dict) LAYERS: {PATH_A}\n")
    f.write("=" * 100 + "\n")
    f.write("\n".join(layers_a))

    f.write("\n\n" + "=" * 100 + "\n")
    f.write(f"MODEL B (state_dict) LAYERS: {PATH_B}\n")
    f.write("=" * 100 + "\n")
    f.write("\n".join(layers_b))

print(f"✅ checkpoint 对比报告已导出到: {SAVE_PATH}")
