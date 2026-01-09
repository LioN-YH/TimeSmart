import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image
import math

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *

# 使用【GPU优化计算】的元特征提取和时序图像化模块
from layers.meta_feature_v import batch_extract_meta_features_gpu
from layers.VE_v import *

# INTRO: TimeSmart的视觉分支，采用通道独立策略


# Router：路由模块，单层线性层，使用 Kaiming 均匀初始化
class Router(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Router, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))

    def forward(self, meta_features):
        weights = torch.softmax(self.fc(meta_features), dim=-1)
        return weights


class Model(nn.Module):

    # 初始化模型
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        self.device = torch.device("cuda:{}".format(self.config.gpu))
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

        # CHANGE：监控元特征和时序图像化类型权重
        self.meta_records = []
        self.ts2img_records = []

    # 初始化各个模块
    def _init_modules(self, config):

        # 路由模块
        # input_dim：元特征维度 / output_dim：时序图像化类型数量
        self.router = Router(input_dim=config.d_meta, output_dim=config.d_ts2img)

        # 时序图像化模块
        self.mt2v_encoder = MT2VEncoderFusion(config)

        # 预测头
        # CHANGE: 与TimeVLM_v采用的预测头设计不同
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

    # CHANGE: 主要函数
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        # 1. 输入归一化
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Compute meta-features
        # 2. 元特征提取
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len
        meta_tensor = batch_extract_meta_features_gpu(
            x_enc, seq_len, pred_len
        )  # [B, D, d_meta]
        # print("meta_tensor:", meta_tensor.shape)

        # Compute weights for time-series imaging types
        # 3. 计算时序图像化类型权重
        ts2img_weights = self.router(meta_tensor)  # [B, D, d_ts2img]
        # print("ts2img_weights:", ts2img_weights.shape)

        # CHANGE：仅在训练时记录元特征和时序图像化类型权重
        if not self.training:
            self.meta_records.append(meta_tensor.detach().cpu())
            self.ts2img_records.append(ts2img_weights.detach().cpu())

        # Convert time series data to images based on the fusion strategy and weights
        # 4. 根据融合策略和权重，获取时序图像化结果
        fusion_strategy = self.config.ts2img_fusion_strategy
        save_images = self.config.save_images
        images = self.mt2v_encoder(
            x_enc, ts2img_weights, fusion_strategy, save_images
        )  # [B, D, C, H, W]
        images = self._normalize_images(images)  # [B, D, 3, H, W]
        # print("images:", images.shape)

        # Process images with the VLM
        # 5. 输入VLM获取图像编码
        feats = []
        for d in range(D):
            img_d = images[:, d]
            f = self.vlm_manager.process_images_inputs(B, img_d)
            feats.append(f)
        vision_embeddings = torch.stack(feats, dim=1).reshape(
            B * D, -1
        )  # [B*D, hidden_size]
        # print("vision_embeddings:", vision_embeddings.shape)

        # Make predictions
        # 6. 预测
        predictions = self.prediction_head(vision_embeddings)
        predictions = einops.rearrange(
            predictions, "(b d) l -> b l d", b=B, d=D
        )  # [B, pred_len, D]
        # print("predictions:", predictions.shape)

        # Denormalize output
        # 7. 输出反归一化
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

    @staticmethod
    # 图像归一化函数
    # CHANGE：匹配VLM的图像输入要求，包括RGB通道以及[0, 255]范围
    def _normalize_images(images):
        # 1. 值域转换与类型转换
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        B, D, C, H, W = images.shape
        # 2. 确保 RGB 通道 (若 C=1)
        if C == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        return images
