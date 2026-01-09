import time
import psutil
import subprocess
import re
import threading
from contextlib import contextmanager

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
import math
from PIL import Image

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *
from layers.meta_feature import batch_extract_meta_features
from layers.VE import *

from torch.profiler import record_function

# INTRO: 检测模型运行过程中各个模块的耗时以及CPU/GPU占用情况
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


# ModelFusor：融合模块，单层线性层，使用 Kaiming 均匀初始化
class ModelFusor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelFusor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))

    def forward(self, meta_features):
        weights = torch.softmax(self.fc(meta_features), dim=-1)
        return weights


# PatchMemoryBank: 用于存储和检索时间序列的块特征
class PatchMemoryBank:
    # 初始化记忆库
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize the patch memory bank.

        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device("cpu")
        # 形状为 [max_size, feature_dim] 的张量，用来存储所有 patch 的特征向量
        self.patches = torch.zeros(
            (max_size, feature_dim), device=self.device
        )  # [100, d_model]
        # 当前插入位置的指针，初始化为0
        self.ptr = 0

    # 更新记忆库
    def update(self, new_patches):
        """
        Update the patch memory bank with new patches using circular buffer strategy.

        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        # new_patches的形状为[n, Np, d_model]
        n = new_patches.size(0)
        # 论文3.1Retrieval-Augmented Memory提到的：在时间维度上进行平均池化
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]

        if self.ptr + n > self.max_size:
            # 如果当前指针+新增数量超过最大容量，则从当前位置填到末尾，然后回到开头继续填充
            # Wrap around if the memory bank is full
            remaining_space = self.max_size - self.ptr
            self.patches[self.ptr :] = new_patches_flat[:remaining_space]
            remaining_patches = n - remaining_space
            if remaining_patches >= self.max_size:
                self.patches[:] = new_patches_flat[-self.max_size :]
                self.ptr = 0
            else:
                self.patches[:remaining_patches] = new_patches_flat[remaining_space:]
                self.ptr = remaining_patches
        else:
            # 正常填充
            self.patches[self.ptr : self.ptr + n] = new_patches_flat
            self.ptr += n

    # 检索记忆库
    def retrieve(self, query_patches, top_k=5):
        """
        Retrieve the top-k most similar patches from the memory bank.

        Args:
            query_patches (Tensor): Query patches for retrieval.
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            retrieved_patches (Tensor): Retrieved patches from the memory bank.
            indices (Tensor): Indices of the retrieved patches.
        """
        query_flat = query_patches.mean(dim=1)  # [224, d_model]
        memory_flat = self.patches  # [100, d_model]

        similarity = torch.matmul(query_flat, memory_flat.T)  # [224, 100]
        # _ 表示最大值本身（即相似度值）
        _, indices = similarity.topk(top_k, dim=-1)

        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices


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
        self.use_mem_gate = config.use_mem_gate

        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=config.patch_memory_size,  # e.g., 100 patches
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            device=self.device,
        )

        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    # 初始化各个模块
    def _init_modules(self, config):

        # Patch Embedding 层
        self.patch_embedding = PatchEmbedding(
            config.d_model,
            config.patch_len,
            config.stride,
            config.padding,
            config.dropout,
        )

        self.head_nf = config.d_model * int(
            (config.seq_len - config.patch_len) / config.stride + 2
        )
        # Flatten 层
        self.flatten = nn.Flatten(start_dim=-2)  # 展平最后两个维度

        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len), nn.Dropout(config.dropout)
        )

        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model), nn.Dropout(config.dropout)
        )

        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(
                self.vlm_manager.hidden_size * 2, config.d_model
            ),  # Combine vision and text
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        # Gating mechanism
        # 门控机制
        # 记忆融合门控
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1),
            )

        # 预测融合门控
        # Prediction fusion gate
        self.gate = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Linear(config.pred_len, 2),
            nn.Softmax(dim=-1),
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Memory-related modules
        # 与记忆库相关的模块
        # ① 局部记忆 MLP：有两个线性层组成，瓶颈结构（扩张 - 收缩）
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )

        # ② 全局记忆注意力：多头自注意力
        # embed_dim：输入嵌入的维度
        # batch_first=True：输入形状为[batch_size, n, d_model]
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        self.layer_norm = nn.LayerNorm(config.d_model)

        # CHANGE
        # 融合模块
        self.fusor = ModelFusor(input_dim=config.d_meta, output_dim=config.d_ts2img)

        # 时序图像化模块
        self.mt2v_encoder = MT2VEncoder(config)
        self.ts2img_methods = ["seg", "gaf", "rp", "stft", "wavelet", "mel", "mtf"]

    # 计算局部记忆：对应论文3.1的Local Memory
    def _compute_local_memory(self, patches):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank

        # 检索记忆库中的Topk个相似块
        retrieved_patches, _ = self.patch_memory_bank.retrieve(
            patches, top_k=self.config.top_k
        )

        # Process retrieved patches with local MLP
        # 使用局部MLP处理检索到的块
        local_memory = self.local_memory_mlp(retrieved_patches)

        # Average over retrieved patches
        # 在patch维度上进行平均池化
        local_memory = local_memory.mean(dim=1, keepdim=True)

        # Residual connection with original patches
        # 残差连接：将局部记忆与原始块特征相加
        local_memory = local_memory + patches

        return local_memory

    # 计算全局记忆：对应论文3.1的Global Memory
    def _compute_global_memory(self, patches):
        """Compute global memory by aggregating information across all patches"""
        # Self-attention to capture global dependencies
        # 多头自注意力机制来捕捉全局依赖关系
        attn_output, _ = self.memory_attention(
            query=patches, key=patches, value=patches
        )

        # Update patch memory bank with current patches
        # 更新记忆库：将当前块特征添加到记忆库中
        self.patch_memory_bank.update(patches.detach())

        if self.use_mem_gate:
            # Return full attention output for advanced gating
            # 使用门控融合机制：返回完整的注意力输出
            return attn_output
        else:
            # Return global context for simple gating (original behavior)
            # 不使用门控融合机制：在patch维度上进行平均池化
            return attn_output.mean(dim=1, keepdim=True)

    # CHANGE：元特征提取器
    def meta_feature_extractor(self, x_enc):
        # x_enc [B, L, D]
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len
        # 提取元特征
        meta_tensor = batch_extract_meta_features(x_enc, seq_len, pred_len)
        return meta_tensor

    # CHANGE：根据指定的融合策略，自适应的选择或融合时序图像化表示
    def _fuse_ts2img_representations(self, ts2img_tensor_list, ts2img_weights):

        B = ts2img_weights.shape[0]
        fused_tensor = None
        # 1. 选择最优
        if self.config.ts2img_fusion_strategy == "select_best":
            # 获取每个样本权重最大的方法索引: shape (B,)
            best_indices = torch.argmax(ts2img_weights, dim=-1)
            # 构建输出: 对每个样本 i，取 ts2img_tensor_list[best_indices[i]][i]
            fused_tensors = []
            for c in range(B):
                chosen_idx = best_indices[c].item()  # 转为 Python int
                # print(
                #     f"[DEBUG]Best method:{self.ts2img_methods[best_indices[c].item()]}"
                # )
                selected_tensor = ts2img_tensor_list[chosen_idx][c]  # (C, H, W)
                fused_tensors.append(selected_tensor)
            # 堆叠回 batch 维度
            fused_tensor = torch.stack(fused_tensors)  # (B, C, H, W)

        # 2. 选择Top3堆叠
        elif self.config.ts2img_fusion_strategy == "top3_stack":
            # 预处理每个时序图像化表示
            # 若 C != 1，则在通道维度取平均 -> (B, 1, H, W)
            processed_tensors = []
            for tensor in ts2img_tensor_list:
                if tensor.size(1) != 1:
                    squeezed = tensor.mean(dim=1, keepdim=True)  # (B, 1, H, W)
                else:
                    squeezed = tensor  # 已经是 (B, 1, H, W)
                processed_tensors.append(squeezed)
            # 获取每个样本Top3权重的方法索引
            _, topk_indices = torch.topk(ts2img_weights, k=3, dim=1)  # (B, 3)
            # 构建输出：对于每个样本 i，堆叠Top3的时序图像化表示
            fused_tensor = torch.zeros(
                B, 3, self.config.image_size, self.config.image_size, device=self.device
            )
            for i in range(B):
                for c in range(3):
                    method_idx = topk_indices[i, c].item()
                    fused_tensor[i, c : c + 1, :, :] = processed_tensors[method_idx][
                        i : i + 1, :, :, :
                    ]
        # 3. 加权融合
        elif self.config.ts2img_fusion_strategy == "weighted_sum":
            # 堆叠ts2img_tensor_list
            stacked = torch.stack(ts2img_tensor_list, dim=1)  #  (B, d_ts2img, C, H, W)
            # 扩展权重到 (B, d_ts2img, 1, 1, 1)，以便广播
            weights_expanded = (
                ts2img_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # (B, d_ts2img, 1, 1, 1)
            # 加权求和: (B, d_ts2img, C, H, W) * (B, d_ts2img, 1, 1, 1) → (B, C, H, W)
            fused_tensor = torch.sum(stacked * weights_expanded, dim=1)
        else:
            raise ValueError(
                f"Unknown ts2img_fusion_strategy: {self.config.ts2img_fusion_strategy}"
            )
        return fused_tensor

    # 【前向预测主函数】
    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings):
        B, L, n_vars = x_enc.shape

        # 1. Process temporal features
        # 处理时序特征：采用patch_embedding
        patches, _ = self.patch_embedding(
            x_enc.transpose(1, 2)
        )  # [B * n_vars, n_patches, d_model]

        # 2. Compute local and global memory
        # 计算局部和全局记忆
        local_memory = self._compute_local_memory(
            patches
        )  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(
            patches
        )  # [B * n_vars, n_patches, d_model] or [B * n_vars, 1, d_model]

        # 3. Combine local and global memory
        # 融合局部和全局记忆
        if self.use_mem_gate:
            # Advanced memory fusion with gating
            # 采用门控融合：local_memory [B * n_vars, n_patches, d_model] // global_memory [B * n_vars, n_patches, d_model]
            # 最后一个维度上连接局部记忆和全局记忆，通过门控组件计算门控权重，加权融合
            combined_features = torch.cat(
                [local_memory, global_memory], dim=-1
            )  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(
                combined_features
            )  # [B * n_vars, n_patches, 2]

            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory
                + gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # Simple addition (original behavior)
            # 不采用门控融合：local_memory [B * n_vars, n_patches, d_model] // global_memory [B * n_vars, 1, d_model]
            # 直接简单相加
            memory_features = (
                local_memory + global_memory
            )  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        # 处理时序特征和记忆特征
        memory_features = self.flatten(
            memory_features
        )  # 展平最后两个维度 [B * n_vars, head_nf]
        temporal_features = self.temporal_head(
            memory_features
        )  # 维度转换 [B, n_vars, d_model]
        memory_features = self.memory_head(
            memory_features
        )  # 维度转换 [B * n_vars, pred_len]
        temporal_features = einops.rearrange(
            temporal_features, "(b n) d -> b n d", b=B, n=n_vars
        )  # [B, n_vars, d_model]
        memory_features = einops.rearrange(
            memory_features, "(b n) d -> b n d", b=B, n=n_vars
        )  # [B, n_vars, pred_len]

        # 5. Process multimodal features
        # 处理多模态特征：连接 - 维度转换 - 扩增 - 归一化
        multimodal_features = torch.cat(
            [vision_embeddings, text_embeddings], dim=-1
        )  # [B, hidden_size * 2]
        multimodal_features = self.multimodal_enhancement(
            multimodal_features
        )  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(
            -1, n_vars, -1
        )  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(
            multimodal_features
        )  # [B, n_vars, d_model]

        # 6. Cross-modal attention enhancement
        # 采用交叉模态注意力增强：本质是计算时序特征与多模态特征之间的相似度
        # 归一化特征向量
        temporal_features = temporal_features / torch.norm(
            temporal_features, dim=-1, keepdim=True
        )
        multimodal_features = multimodal_features / torch.norm(
            multimodal_features, dim=-1, keepdim=True
        )
        multimodal_features, _ = self.cross_attention(
            query=temporal_features, key=multimodal_features, value=multimodal_features
        )  # [B, n_vars, d_model]

        # 7. Normalize cross attention output
        # 归一化交叉注意力输出，同时进行维度转换
        multimodal_features = self.layer_norm(
            multimodal_features
        )  # [B, n_vars, d_model]
        multimodal_features = self.multimodal_head(
            multimodal_features
        )  # [B, n_vars, pred_len]

        # 8. Compute gating weights
        # 计算门控权重
        combined_features = torch.cat(
            [memory_features, multimodal_features], dim=-1
        )  # [B, n_vars, pred_len * 2]
        gate_weights = self.gate(combined_features)  # [B, n_vars, 2]

        # 9. Weighted fusion
        # 加权融合：将记忆特征和多模态特征进行加权融合
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features
            + gate_weights[:, :, 1:2] * multimodal_features
        )  # [B, n_vars, pred_len]

        # 10. Final fusion
        # 最终融合：将记忆特征和融合特征进行线性变换
        predictions = (
            self.fusion_layer(torch.cat([memory_features, fused_features], dim=-1))
            + memory_features
        )  # [B, n_vars, pred_len]

        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        # 1. 输入归一化
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Convert time series data to images and generate text prompts
        # 2. 图像增强学习器：将时间序列数据转换为图像
        #    文本增强学习器：为语言模型生成文本提示
        images = self.vision_augmented_learner(x_enc)
        prompts = self.text_augmented_learner(
            x_enc, self.config.content, self.config.pred_len, self.config.seq_len
        )

        # Process inputs with the VLM
        # 3. 利用VLM处理图像和文本输入
        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(
            B, images, prompts
        )

        # Main prediction branch
        # 4. 前向预测
        predictions = self.forward_prediction(x_enc, vision_embeddings, text_embeddings)

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

    # 【文本增强学习器】
    # 根据时间序列数据为语言模型生成文本提示
    # 时间序列中的每个变量都有自己的提示
    def text_augmented_learner(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, T, n_vars = (
            x_enc.shape
        )  # Get batch size, sequence length, and number of variables

        # Initialize a list to store prompts for each batch
        prompts = []

        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            # 1. 对于batch中的每个样本，计算其统计信息
            min_value = torch.min(
                x_enc[b]
            ).item()  # Overall minimum value for the batch
            max_value = torch.max(
                x_enc[b]
            ).item()  # Overall maximum value for the batch
            median_value = torch.median(
                x_enc[b]
            ).item()  # Overall median value for the batch
            # 通过差分累加计算趋势
            # diff(dim=0) 对时间维度（第 0 维）计算相邻元素的差值
            trend = x_enc[b].diff(dim=0).sum().item()  # Overall trend for the batch

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"

            # 2. 提示模板
            prompt_parts = [
                "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}.",
            ]
            # 3. 将列表 prompt_parts 中的所有字符串元素，用空格 " " 作为分隔符，连接成一个完整的字符串
            prompt = " ".join(prompt_parts)

            # 4. 截断超长文本，确保生成的提示不超过语言模型的最大输入长度限制
            prompt = (
                prompt[: self.vlm_manager.max_input_text_length]
                if len(prompt) > self.vlm_manager.max_input_text_length
                else prompt
            )
            prompts.append(prompt)

        return prompts

    # CHANGE:【视觉增强学习器】
    def vision_augmented_learner(self, x_enc):
        """
        Convert time series data into image tensors.
        """

        # 1. 元特征提取
        with monitor_performance(interval=0.2):
            meta_tensor = self.meta_feature_extractor(x_enc=x_enc).to(self.device)
        print("meta_feature_extractor")
        # 缓存meta_tensor到模型实例，供钩子捕获（仅增加这一行）
        self._meta_tensor_cache = meta_tensor

        # 2. 计算时序图像化类型权重
        with monitor_performance(interval=0.2):
            ts2img_weights = self.fusor(meta_tensor)  # [B, d_ts2img]
        print("compute_ts2img_weights")

        # 3. 获取候选的时序图像化表示
        with monitor_performance(interval=0.2):
            ts2img_tensor_list = self.mt2v_encoder(
                x_enc, save_images=self.config.save_images
            )  # List of tensors, each [B, C, H, W]
        print("transform_ts2img_tensor")

        # 4. 加权融合时序图像化表示
        with monitor_performance(interval=0.2):
            images = self._fuse_ts2img_representations(
                ts2img_tensor_list, ts2img_weights
            )
        print("fuse_ts2img_representations")

        # 选择性保存
        if self.config.save_images:
            self.save_images(images)

        print(f"[DEBUG]GET-IMAGE {images.shape}")
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
