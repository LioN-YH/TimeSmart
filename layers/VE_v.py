import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .VE import MT2VEncoder, ts2img_methods

# INTRO:优化后的适用于通道独立性策略的时序图像转化模块，支持seg, gaf, rp, stft, wavelet, mel, mtf等多种方法，已完成【GPU版本的计算优化】
# 迁移融合过程于本模块中，减少一些冗余计算


class MT2VEncoderFusion(nn.Module):
    def __init__(self, config):
        super(MT2VEncoderFusion, self).__init__()
        cfg = copy.deepcopy(config)
        cfg.compress_vars = False
        self.encoder = MT2VEncoder(cfg)
        self.image_size = cfg.image_size
        self.three_channel_image = cfg.three_channel_image

    # 计算单样本单变量时序图像化结果 (C,H,W)
    def ts2img_single(self, series, method, force_one_channel=False):
        # 扩增series维度，使其适配批样本多变量时序图像化处理函数get_ts2img_tensor，完成代码复用
        if series.dim() == 1:
            series = series.unsqueeze(0).unsqueeze(-1)  # (L,) → (1, L, 1)
        elif series.dim() == 2:
            series = series.unsqueeze(-1)  # (B, L) → (B, L, 1)
        # (B,D,H,W) or (B,1,H,W)
        out = self.encoder.get_ts2img_tensor(series, method, save_images=False)
        out = out.unsqueeze(2)  # (B,D,1,H,W)
        out = out[:, 0]  # (B,1,H,W)
        img = out[0]  # (1,H,W)
        if self.three_channel_image and not force_one_channel:
            img = img.repeat(3, 1, 1)  # (3,H,W)
        return img

    # 预计算批多变量时序数据多种时序图像化结果 (B,D,7,C,H,W)
    def compute_all_method_images(self, x, save_images=False):
        B, L, D = x.shape
        imgs = []
        for m in ts2img_methods:
            t = self.encoder.get_ts2img_tensor(x, m, save_images)
            imgs.append(t.unsqueeze(2))  # (B,D,1,H,W)
        stack = torch.stack(imgs, dim=2)  # (B,D,7,1,H,W)
        if self.three_channel_image:
            stack = stack.repeat(1, 1, 1, 3, 1, 1)
        return stack

    # 根据融合策略和权重信息进行时序图像化
    def fuse(self, x, ts2img_weights, fusion_strategy, save_images=False):
        B, L, D = x.shape
        # 1. 选择最优 (使用条件计算+Straight-Through Estimator保持梯度传导)
        if fusion_strategy == "select_best":
            # idx: (B, D, 1)
            idx = torch.argmax(ts2img_weights, dim=-1, keepdim=True)

            # 准备输出张量
            C = 3 if self.three_channel_image else 1
            out = torch.zeros(
                B,
                D,
                C,
                self.image_size,
                self.image_size,
                device=x.device,
                dtype=x.dtype,
            )

            # 条件计算：仅计算被选中的方法
            unique_methods = torch.unique(idx)

            # 将x reshape为(B*D, L, 1)以便处理
            x_flat = x.reshape(B * D, L, 1)
            idx_flat = idx.reshape(B * D)
            out_flat = out.view(B * D, C, self.image_size, self.image_size)

            for m_idx in unique_methods:
                m_idx_int = int(m_idx.item())
                method_name = ts2img_methods[m_idx_int]

                # 找到选择该方法的样本
                mask_flat = idx_flat == m_idx_int
                if not mask_flat.any():
                    continue

                # 提取子集并计算
                x_sub = x_flat[mask_flat]  # (N_sub, L, 1)
                # out_sub: (N_sub, 1, H, W)
                out_sub = self.encoder.get_ts2img_tensor(
                    x_sub, method_name, save_images=False
                )

                # 调整维度
                if self.three_channel_image:
                    # (N_sub, 1, H, W) -> (N_sub, 3, H, W)
                    if out_sub.shape[1] == 1:
                        out_sub = out_sub.repeat(1, 3, 1, 1)
                    # 如果本来就是3通道则不用repeat

                # 填回结果
                out_flat[mask_flat] = out_sub

            # 恢复形状
            out = out_flat.view(B, D, C, self.image_size, self.image_size)

            # Straight-Through Estimator: 仅对选中的权重应用梯度
            # w_selected: (B, D, 1)
            w_selected = torch.gather(ts2img_weights, -1, idx)
            # w_selected: (B, D, 1, 1, 1)
            w_selected = w_selected.unsqueeze(-1).unsqueeze(-1)

            # out = out * (1 - w.detach() + w)
            out = out * (1.0 - w_selected.detach() + w_selected)

            return out

        # 2. 选择Top3堆叠 (使用条件计算+Straight-Through Estimator保持梯度传导)
        elif fusion_strategy == "top3_stack":
            # 获取Top3索引
            _, topk_indices = torch.topk(ts2img_weights, k=3, dim=-1)  # (B, D, 3)

            out_channels = []
            x_flat = x.reshape(B * D, L, 1)

            for k in range(3):
                # 第k个channel对应第k大的方法
                idx_k = topk_indices[:, :, k : k + 1]  # (B, D, 1)
                idx_flat = idx_k.reshape(B * D)

                # 准备该层的输出 (B*D, 1, H, W)
                # 强制单通道
                layer_out = torch.zeros(
                    B * D,
                    1,
                    self.image_size,
                    self.image_size,
                    device=x.device,
                    dtype=x.dtype,
                )

                unique_methods = torch.unique(idx_k)
                for m_idx in unique_methods:
                    m_idx_int = int(m_idx.item())
                    method_name = ts2img_methods[m_idx_int]

                    mask_flat = idx_flat == m_idx_int
                    if not mask_flat.any():
                        continue

                    x_sub = x_flat[mask_flat]
                    out_sub = self.encoder.get_ts2img_tensor(
                        x_sub, method_name, save_images=False
                    )
                    # out_sub: (N_sub, 1, H, W) or (N_sub, 3, H, W)

                    # Ensure single channel
                    if out_sub.shape[1] > 1:
                        out_sub = out_sub[:, 0:1, :, :]

                    layer_out[mask_flat] = out_sub

                # Reshape back
                layer_out = layer_out.view(B, D, 1, self.image_size, self.image_size)

                # Apply STE gradient
                w_selected = torch.gather(ts2img_weights, -1, idx_k)
                w_selected = w_selected.unsqueeze(-1).unsqueeze(-1)

                layer_out = layer_out * (1.0 - w_selected.detach() + w_selected)
                out_channels.append(layer_out)

            out = torch.cat(out_channels, dim=2)  # (B, D, 3, H, W)
            return out

        # 3. 加权融合
        # 计算每个样本每个变量每个方法的时序图像化结果，根据权重进行加权求和
        elif fusion_strategy == "weighted_sum":
            imgs = self.compute_all_method_images(x, save_images)
            weights = ts2img_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = (imgs * weights).sum(dim=2)
            return out  # [B, D, C, H, W]
        else:
            raise ValueError("Unknown fusion_strategy")

    def forward(self, x, ts2img_weights, fusion_strategy, save_images=False):
        return self.fuse(x, ts2img_weights, fusion_strategy, save_images)


# CHANGE:替换使用GPU版本的wavelet_transform
class MT2VEncoderFusionOpt(nn.Module):
    def __init__(self, config):
        super(MT2VEncoderFusionOpt, self).__init__()
        cfg = copy.deepcopy(config)
        cfg.compress_vars = False
        self.encoder = MT2VEncoder(cfg)
        self.image_size = cfg.image_size
        self.three_channel_image = cfg.three_channel_image

    # 计算单样本单变量时序图像化结果 (C,H,W)
    def ts2img_single(self, series, method, force_one_channel=False):
        # 扩增series维度，使其适配批样本多变量时序图像化处理函数get_ts2img_tensor_opt，完成代码复用
        if series.dim() == 1:
            series = series.unsqueeze(0).unsqueeze(-1)  # (L,) → (1, L, 1)
        elif series.dim() == 2:
            series = series.unsqueeze(-1)  # (B, L) → (B, L, 1)
        # (B,D,H,W) or (B,1,H,W)
        out = self.encoder.get_ts2img_tensor_opt(series, method, save_images=False)
        out = out.unsqueeze(2)  # (B,D,1,H,W)
        out = out[:, 0]  # (B,1,H,W)
        img = out[0]  # (1,H,W)
        if self.three_channel_image and not force_one_channel:
            img = img.repeat(3, 1, 1)  # (3,H,W)
        return img

    # 预计算批多变量时序数据多种时序图像化结果 (B,D,7,C,H,W)
    def compute_all_method_images(self, x, save_images=False):
        B, L, D = x.shape
        imgs = []
        for m in ts2img_methods:
            t = self.encoder.get_ts2img_tensor_opt(x, m, save_images)
            imgs.append(t.unsqueeze(2))  # (B,D,1,H,W)
        stack = torch.stack(imgs, dim=2)  # (B,D,7,1,H,W)
        if self.three_channel_image:
            stack = stack.repeat(1, 1, 1, 3, 1, 1)
        return stack

    # 根据融合策略和权重信息进行时序图像化
    def fuse(self, x, ts2img_weights, fusion_strategy, save_images=False):
        B, L, D = x.shape
        # 1. 选择最优 (使用条件计算+Straight-Through Estimator保持梯度传导)
        if fusion_strategy == "select_best":
            # idx: (B, D, 1)
            idx = torch.argmax(ts2img_weights, dim=-1, keepdim=True)

            # 准备输出张量
            C = 3 if self.three_channel_image else 1
            out = torch.zeros(
                B,
                D,
                C,
                self.image_size,
                self.image_size,
                device=x.device,
                dtype=x.dtype,
            )

            # 条件计算：仅计算被选中的方法
            unique_methods = torch.unique(idx)

            # 将x reshape为(B*D, L, 1)以便处理
            x_flat = x.reshape(B * D, L, 1)
            idx_flat = idx.reshape(B * D)
            out_flat = out.view(B * D, C, self.image_size, self.image_size)

            for m_idx in unique_methods:
                m_idx_int = int(m_idx.item())
                method_name = ts2img_methods[m_idx_int]

                # 找到选择该方法的样本
                mask_flat = idx_flat == m_idx_int
                if not mask_flat.any():
                    continue

                # 提取子集并计算
                x_sub = x_flat[mask_flat]  # (N_sub, L, 1)
                # out_sub: (N_sub, 1, H, W)
                out_sub = self.encoder.get_ts2img_tensor_opt(
                    x_sub, method_name, save_images=False
                )

                # 调整维度
                if self.three_channel_image:
                    # (N_sub, 1, H, W) -> (N_sub, 3, H, W)
                    if out_sub.shape[1] == 1:
                        out_sub = out_sub.repeat(1, 3, 1, 1)
                    # 如果本来就是3通道则不用repeat

                # 填回结果
                out_flat[mask_flat] = out_sub

            # 恢复形状
            out = out_flat.view(B, D, C, self.image_size, self.image_size)

            # Straight-Through Estimator: 仅对选中的权重应用梯度
            # w_selected: (B, D, 1)
            w_selected = torch.gather(ts2img_weights, -1, idx)
            # w_selected: (B, D, 1, 1, 1)
            w_selected = w_selected.unsqueeze(-1).unsqueeze(-1)

            # out = out * (1 - w.detach() + w)
            out = out * (1.0 - w_selected.detach() + w_selected)

            return out

        # 2. 选择Top3堆叠 (使用条件计算+Straight-Through Estimator保持梯度传导)
        elif fusion_strategy == "top3_stack":
            # 获取Top3索引
            _, topk_indices = torch.topk(ts2img_weights, k=3, dim=-1)  # (B, D, 3)

            out_channels = []
            x_flat = x.reshape(B * D, L, 1)

            for k in range(3):
                # 第k个channel对应第k大的方法
                idx_k = topk_indices[:, :, k : k + 1]  # (B, D, 1)
                idx_flat = idx_k.reshape(B * D)

                # 准备该层的输出 (B*D, 1, H, W)
                # 强制单通道
                layer_out = torch.zeros(
                    B * D,
                    1,
                    self.image_size,
                    self.image_size,
                    device=x.device,
                    dtype=x.dtype,
                )

                unique_methods = torch.unique(idx_k)
                for m_idx in unique_methods:
                    m_idx_int = int(m_idx.item())
                    method_name = ts2img_methods[m_idx_int]

                    mask_flat = idx_flat == m_idx_int
                    if not mask_flat.any():
                        continue

                    x_sub = x_flat[mask_flat]
                    out_sub = self.encoder.get_ts2img_tensor_opt(
                        x_sub, method_name, save_images=False
                    )
                    # out_sub: (N_sub, 1, H, W) or (N_sub, 3, H, W)

                    # Ensure single channel
                    if out_sub.shape[1] > 1:
                        out_sub = out_sub[:, 0:1, :, :]

                    layer_out[mask_flat] = out_sub

                # Reshape back
                layer_out = layer_out.view(B, D, 1, self.image_size, self.image_size)

                # Apply STE gradient
                w_selected = torch.gather(ts2img_weights, -1, idx_k)
                w_selected = w_selected.unsqueeze(-1).unsqueeze(-1)

                layer_out = layer_out * (1.0 - w_selected.detach() + w_selected)
                out_channels.append(layer_out)

            out = torch.cat(out_channels, dim=2)  # (B, D, 3, H, W)
            return out

        # 3. 加权融合
        # 计算每个样本每个变量每个方法的时序图像化结果，根据权重进行加权求和
        elif fusion_strategy == "weighted_sum":
            imgs = self.compute_all_method_images(x, save_images)
            weights = ts2img_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = (imgs * weights).sum(dim=2)
            return out  # [B, D, C, H, W]
        else:
            raise ValueError("Unknown fusion_strategy")

    def forward(self, x, ts2img_weights, fusion_strategy, save_images=False):
        return self.fuse(x, ts2img_weights, fusion_strategy, save_images)
