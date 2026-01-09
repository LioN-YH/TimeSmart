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
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image
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
        self.cross_attention_Mul2TS = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        self.cross_attention_EnhanceTS = nn.MultiheadAttention(
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

        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3,
            hidden_dim=48,
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size,
            periodicity=config.periodicity,
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
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

        # TODO:待修改-先对齐再检索
        # 双向检索
        aligned_features, _ = self.cross_attention_Mul2TS(
            query=multimodal_features, key=temporal_features, value=temporal_features
        )

        aligned_features = self.layer_norm(aligned_features)  # [B, n_vars, d_model]
        aligned_features = aligned_features / torch.norm(
            aligned_features, dim=-1, keepdim=True
        )
        enhanced_features, _ = self.cross_attention_EnhanceTS(
            query=temporal_features, key=aligned_features, value=aligned_features
        )  # [B, n_vars, d_model]

        # 7. Normalize cross attention output
        # 归一化交叉注意力输出，同时进行维度转换
        enhanced_features = self.layer_norm(enhanced_features)  # [B, n_vars, d_model]
        enhanced_features = self.multimodal_head(
            enhanced_features
        )  # [B, n_vars, pred_len]

        # 8. Compute gating weights
        # 计算门控权重
        combined_features = torch.cat(
            [memory_features, enhanced_features], dim=-1
        )  # [B, n_vars, pred_len * 2]
        gate_weights = self.gate(combined_features)  # [B, n_vars, 2]

        # 9. Weighted fusion
        # 加权融合：将记忆特征和多模态特征进行加权融合
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features
            + gate_weights[:, :, 1:2] * enhanced_features
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
        images = self.vision_augmented_learner(
            x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity
        )
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

    # 【视觉增强学习器】
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
            images = time_series_to_simple_image(
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
