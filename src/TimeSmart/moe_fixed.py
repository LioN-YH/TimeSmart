import os
import sys
import json
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
from layers.VE_v import *

import time

# Model for validating a single fixed imaging method (e.g., seg)
# Skips Router and Meta Feature Extraction
class Model(nn.Module):

    # 初始化模型
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        self.device = torch.device("cuda:{}".format(self.config.gpu))
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

        # Monitor (Optional, keep empty lists for compatibility if needed)
        self.meta_records = []
        self.ts2img_records = []

    # 初始化各个模块
    def _init_modules(self, config):
        # NO ROUTER

        # 时序图像化模块
        self.mt2v_encoder = MT2VEncoderFusionOpt(config)

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.vlm_manager.hidden_size),
            nn.Linear(self.vlm_manager.hidden_size, self.vlm_manager.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.vlm_manager.hidden_size, config.pred_len),
        )

    # 主要函数
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        # 1. 输入归一化
        x_enc, means, stdev = self._normalize_input(x_enc)

        # 2. Skip Meta Features & Router
        
        # 3. Direct Imaging (seg)
        # Call underlying encoder directly for "seg"
        # x_enc: [B, L, D] -> get_ts2img_tensor -> [B, D, H, W] (assuming compress_vars=False)
        save_images = self.config.save_images
        
        # Hardcode method to "seg"
        method_name = "seg"
        
        # Note: get_ts2img_tensor returns [B, D, H, W]
        images_raw = self.mt2v_encoder.encoder.get_ts2img_tensor(x_enc, method_name, save_images)
        
        # Reshape to [B, D, C, H, W]
        # C is usually 1 from get_ts2img_tensor unless 3-channel
        images = images_raw.unsqueeze(2) # [B, D, 1, H, W]
        

        # 4. Normalize Images
        images = self._normalize_images(images)  # [B, D, 3, H, W]

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

        # Make predictions
        # 6. 预测
        predictions = self.prediction_head(vision_embeddings)
        predictions = einops.rearrange(
            predictions, "(b d) l -> b l d", b=B, d=D
        )  # [B, pred_len, D]

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
        stdev = stdev / self.config.norm_const
        x = x / stdev
        return x, means, stdev

    # 对输出进行反归一化处理
    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    @staticmethod
    # 图像归一化函数
    def _normalize_images(images):
        B, D, C, H, W = images.shape
        # 2. 确保 RGB 通道 (若 C=1)
        if C == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        return images
