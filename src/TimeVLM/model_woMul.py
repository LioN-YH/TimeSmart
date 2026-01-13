import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *

""" 
PatchMemoryBank: 用于存储和检索时间序列的块特征
"""


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

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len, config.pred_len * 2),
            nn.GELU(),
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

        self.layer_norm = nn.LayerNorm(config.d_model)

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

    # 【前向预测主函数】
    def forward_prediction(self, x_enc):
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
        memory_features = self.memory_head(
            memory_features
        )  # 维度转换 [B * n_vars, pred_len]
        memory_features = einops.rearrange(
            memory_features, "(b n) d -> b n d", b=B, n=n_vars
        )  # [B, n_vars, pred_len]

        # print("memory_features.shape:", memory_features.shape)
        # 5. Final fusion
        # 最终融合：将记忆特征进行线性变换
        predictions = (
            self.fusion_layer(memory_features) + memory_features
        )  # [B, n_vars, pred_len]

        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Normalize input
        # 1. 输入归一化
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Main prediction branch
        # 2. 前向预测
        predictions = self.forward_prediction(x_enc)

        # Denormalize output
        # 3. 输出反归一化
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
