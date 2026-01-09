import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image
from contextlib import contextmanager
import psutil
import time
import threading

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage_v
from layers.TimeSeries_To_Image import time_series_to_simple_image_v
from layers.models_mae import *
from transformers.models.vilt import *

# 尝试导入GPU监控库（需单独安装）
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except pynvml.NVMLError:
    GPU_AVAILABLE = False


@contextmanager
def monitor_performance(interval=0.1):
    """
    上下文管理器：监控代码块的CPU占用率、GPU占用率和运行时间
    :param interval: 监控采样间隔（秒）
    """
    # 初始化监控变量
    start_time = time.time()
    cpu_percent_list = []
    gpu_percent_list = []
    monitoring = True

    # CPU监控线程
    def cpu_monitor():
        while monitoring:
            # 获取当前进程的CPU占用率（百分比）
            cpu_percent = psutil.Process().cpu_percent(interval=interval)
            cpu_percent_list.append(cpu_percent)
            time.sleep(interval)

    # GPU监控线程（如果可用）
    def gpu_monitor():
        if not GPU_AVAILABLE:
            return
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 获取第0块GPU
        while monitoring:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent_list.append(util.gpu)  # GPU使用率（百分比）
            time.sleep(interval)

    # 启动监控线程
    cpu_thread = threading.Thread(target=cpu_monitor, daemon=True)
    gpu_thread = threading.Thread(target=gpu_monitor, daemon=True)
    cpu_thread.start()
    gpu_thread.start()

    try:
        yield  # 执行被监控的代码块
    finally:
        # 停止监控
        monitoring = False
        cpu_thread.join()
        gpu_thread.join()

        # 计算统计结果
        elapsed_time = time.time() - start_time
        avg_cpu = (
            sum(cpu_percent_list) / len(cpu_percent_list) if cpu_percent_list else 0
        )
        max_cpu = max(cpu_percent_list) if cpu_percent_list else 0

        avg_gpu = 0
        max_gpu = 0
        if GPU_AVAILABLE and gpu_percent_list:
            avg_gpu = sum(gpu_percent_list) / len(gpu_percent_list)
            max_gpu = max(gpu_percent_list)

        # 打印结果
        print("\n" + "=" * 50)
        print(f"代码块执行时间: {elapsed_time:.4f} 秒")
        print(f"CPU 平均占用率: {avg_cpu:.2f}% | 最大占用率: {max_cpu:.2f}%")
        if GPU_AVAILABLE:
            print(f"GPU 平均占用率: {avg_gpu:.2f}% | 最大占用率: {max_gpu:.2f}%")
        else:
            print("GPU监控不可用（请安装pynvml或检查GPU驱动）")
        print("=" * 50 + "\n")


class Model(nn.Module):
    """
    Time-VLM model with image and text modalities for enhanced time series forecasting.
    """

    # 初始化模型
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        self.device = torch.device("cuda:{}".format(self.config.gpu))

        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    # 初始化各个模块
    def _init_modules(self, config):
        self.learnable_image_module = LearnableTimeSeriesToImage_v(
            input_dim=3,
            hidden_dim=48,
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size,
            periodicity=config.periodicity,
            grid_size=10,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size, config.pred_len * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.Dropout(config.dropout),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        # 1. 输入归一化
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Convert time series data to images and generate text prompts
        # 2. 图像增强学习器：将时间序列数据转换为图像
        images = self.vision_augmented_learner(
            x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity
        )
        print("[DEBUG] Images shape:", images.shape)

        # CHANGE: 消融文本分支
        # Process inputs with the VLM
        # 3. 利用VLM处理图像和文本输入
        with monitor_performance(interval=0.2):
            vision_embeddings = self.vlm_manager.process_images_inputs(B * D, images)
        print("[DEBUG] Vision embeddings -1 shape:", vision_embeddings.shape)

        # TEST:测试另一种编码方式处理结果是否相同
        # Main prediction branch
        # 4. 前向预测
        images = einops.rearrange(images, "(b d) c h w -> d b c h w", b=B, d=D)
        with monitor_performance(interval=0.2):
            encoded_features_list = []
            for d in range(D):
                batch_images = images[d, :, :, :, :]  # 或者 images[d]
                encoded_batch = self.vlm_manager.process_images_inputs(B, batch_images)
                encoded_features_list.append(encoded_batch)
            encoded_features_stacked = torch.stack(encoded_features_list, dim=0)
            final_encoded_features = einops.rearrange(
                encoded_features_stacked, "d b h -> (b d) h"
            )
        print("[DEBUG] Vision embeddings -2 shape:", final_encoded_features.shape)

        print(f"JUDGE:{torch.equal(vision_embeddings, final_encoded_features)}")
        print(
            f"JUDGE:{torch.allclose(vision_embeddings, final_encoded_features, atol=1e-6)}"
        )
        exit(0)

        predictions = self.prediction_head(vision_embeddings)
        print("[DEBUG] Predictions shape before rearrange:", predictions.shape)
        predictions = einops.rearrange(
            predictions, "(b d) l -> b l d", b=B, d=D
        )  # [B, pred_len, D]

        # Denormalize output
        # 5. 输出反归一化
        y = self._denormalize_output(predictions, means, stdev)
        return y

    # 对输入的时间序列数据进行归一化处理
    def _normalize_input(self, x):
        # 计算均值
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        # 计算标准差
        # unbiased=False：使用无偏估计，计算样本方差
        # 1e-5：防止除零错误-接近零的方差会导致标准差接近零，在反向传播时可能引发梯度爆炸
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # 使用配置中的norm_const进一步缩放标准差
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    # 对输出进行反归一化处理
    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    # 【视觉增强学习器】
    # CHANGE：通道独立设计，取消原始设计中的多变量平均池化计算，保留更多细节，输出images [B*D, output_channels, H, W]
    # 将时间序列数据转换为3通道的图像张量
    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """

        # 1. 时序图像化
        # self.config.learnable_image为True，则使用可学习的图像生成模块self.learnable_image_module
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        # 否则使用简单的转换函数time_series_to_simple_image
        else:
            images = time_series_to_simple_image_v(
                x_enc, image_size, context_len, periodicity
            )

        # Normalize images to [0, 255] as uint8
        # 2. 标准化处理
        images = self._normalize_images(images)

        # Optionally save images
        # 3. 选择性保存
        if self.config.save_images:
            self.save_images(images)

        return images

    @staticmethod
    # 图像归一化函数
    # 输入的图像张量归一化到 [0, 255] 范围并转换为 uint8 数据类型
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.

        Args:
        - images (Tensor): Input images with shape [B, C, H, W]

        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        # 1. 计算每张图像的最小值和最大值（跨所有通道和像素
        # .reshape(images.size(0), -1)：[B, C, H, W]重塑为[B, C*H*W]，方便对每个样本的所有像素值计算统计量
        # .min(dim=1, keepdim=True)[0]：在第1维上计算最小值，保持维度，注意min() 返回一个元组 (values, indices)，values和indices维度都是[B, 1]
        # [0]：提取的是values张量
        # .view(-1, 1, 1, 1)：将 [B, 1] 形状的张量重塑为 [B, 1, 1, 1]，为后续的广播操作做准备
        min_vals = (
            images.reshape(images.size(0), -1)
            .min(dim=1, keepdim=True)[0]
            .view(-1, 1, 1, 1)
        )
        max_vals = (
            images.reshape(images.size(0), -1)
            .max(dim=1, keepdim=True)[0]
            .view(-1, 1, 1, 1)
        )

        # Avoid division by zero by adding a small epsilon
        # 2. 计算缩放因子，同时防止除零错误
        #  .clamp(min=epsilon)：将张量中的每个元素限制在 [epsilon, +∞) 范围内
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)

        # Normalize to [0, 1]
        # 3. 归一化图像到 [0, 1] 范围
        images = (images - min_vals) / scale

        # Scale to [0, 255] and clamp to ensure valid range
        # 转换到 [0, 255] 范围并转为整数
        # .clamp(0, 255)：确保所有值在有效范围内（防止浮点数运算误差导致越界）
        images = (images * 255).clamp(0, 255).to(torch.uint8)

        return images

    @torch.no_grad()
    # 图像保存
    def save_images(self, images):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved with shape [B, C, H, W]
        """
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            # Move to CPU and convert to numpy
            img_tensor = img_tensor.cpu().numpy()

            # Check channel count and handle accordingly
            if img_tensor.shape[0] == 3:
                # RGB image: Convert from [C, H, W] to [H, W, C]
                img_tensor = np.transpose(img_tensor, (1, 2, 0))
                mode = "RGB"
            elif img_tensor.shape[0] == 1:
                # Grayscale image: Convert from [C, H, W] to [H, W]
                img_tensor = np.squeeze(img_tensor, 0)
                mode = "L"
            else:
                print(
                    f"Warning: Unexpected number of channels {img_tensor.shape[0]} for image {i}. Skipping..."
                )
                continue

            # Ensure data type is uint8
            if img_tensor.dtype != np.uint8:
                img_tensor = img_tensor.astype(np.uint8)

            # Create PIL image and save
            try:
                img = Image.fromarray(img_tensor, mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue
