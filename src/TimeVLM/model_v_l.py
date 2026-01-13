import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage_v
from layers.TimeSeries_To_Image import time_series_to_simple_image_v
from layers.models_mae import *
from transformers.models.vilt import *

# INTRO: TimeVLM的视觉分支，提取图像表征后，通过【简单线性层预测】，但是不同之处在于【通道独立性】
# 原始处理中，视觉分支进行时序图像化转换时，对于多变量进行平均池化处理，以获取全局特征，作为时间序列特征的补充
# 但是预测前维度扩增是通过拷贝完成的，如果仅将视觉表征输入参数共享的预测头中，最终不同变量的预测结果是相同的，显然不符合逻辑
# 因此视觉分支的单独实验中，存在两种选择：
#    1 - 保留平均池化的视觉特征处理，但是为不同变量分配独立的预测头
#    2 - 在时序图像化转换时，保留不同变量的通道信息，预测头仍然参数共享
# 个人认为第二种选择更合理一些

# CHANGE:2025/12/17更新，完成堆叠输入VLM的计算优化，可以加快运算时间


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

        # self.prediction_head = nn.Sequential(
        #     nn.Linear(self.vlm_manager.hidden_size, config.pred_len * 2),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.pred_len * 2, config.pred_len),
        #     nn.Dropout(config.dropout),
        # )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.vlm_manager.hidden_size),
            nn.Linear(self.vlm_manager.hidden_size, self.vlm_manager.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.vlm_manager.hidden_size, config.pred_len),
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
        )  # [B, D, C, H, W]
        # print("[DEBUG] Images shape:", images.shape)

        # CHANGE: 消融文本分支，同时通过堆叠完成计算优化
        # Process inputs with the VLM
        # 3. 利用VLM处理图像输入
        # vision_embeddings = self.vlm_manager.process_images_inputs(B * D, images)
        feats = []
        for d in range(D):
            img_d = images[:, d]
            f = self.vlm_manager.process_images_inputs(B, img_d)
            feats.append(f)
        vision_embeddings = torch.stack(feats, dim=1).reshape(
            B * D, -1
        )  # [B*D, hidden_size]
        # print("[DEBUG] Vision embeddings shape:", vision_embeddings.shape)

        # Main prediction branch
        # 4. 前向预测
        predictions = self.prediction_head(vision_embeddings)
        # print("[DEBUG] Predictions shape before rearrange:", predictions.shape)
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
    # CHANGE：通道独立设计，取消原始设计中的多变量平均池化计算，保留更多细节，输出images [B, D, output_channels, H, W]
    # 将时间序列数据转换为3通道的图像张量
    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """

        # 1. 时序图像化 [B, D, output_channels, H, W]
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

        return images

    @staticmethod
    # 图像归一化函数
    # 输入的图像张量归一化
    # CHANGE: 处理images形状为  [B, D, C, H, W]
    def _normalize_images(images):

        B, D, C, H, W = images.shape
        images = images.reshape(B * D, C, H, W)

        # Compute min and max per image across all channels and spatial dimensions
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
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)

        # Normalize to [0, 1]
        images = (images - min_vals) / scale

        # 原始TimeVLM还有一个额外的归一化步骤，将图像转换为uint8类型，但这是错误的，将导致梯度断裂
        # Scale to [0, 255] and clamp to ensure valid range
        # images = (images * 255).clamp(0, 255).to(torch.uint8)

        images = images.reshape(B, D, C, H, W)

        return images
