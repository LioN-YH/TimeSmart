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
        # 1. 选择最优 (使用Straight-Through Estimator保持梯度传导)
        if fusion_strategy == "select_best":
            # 计算所有方法的图像 (B, D, 7, C, H, W)
            imgs = self.compute_all_method_images(x, save_images)

            # 生成硬选择掩码，但保留梯度
            # idx: (B, D, 1)
            idx = torch.argmax(ts2img_weights, dim=-1, keepdim=True)
            mask_hard = torch.zeros_like(ts2img_weights).scatter_(-1, idx, 1.0)

            # Straight-Through Estimator: forward=mask_hard, backward=ts2img_weights
            mask = mask_hard - ts2img_weights.detach() + ts2img_weights

            # (B, D, 7, 1, 1, 1)
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # (B, D, C, H, W)
            out = (imgs * mask).sum(dim=2)
            return out

        # 2. 选择Top3堆叠 (使用Straight-Through Estimator保持梯度传导)
        elif fusion_strategy == "top3_stack":
            imgs = self.compute_all_method_images(x, save_images)
            # 确保使用单通道进行堆叠
            if imgs.shape[3] == 3:
                imgs_1c = imgs[:, :, :, 0:1, :, :]  # (B, D, 7, 1, H, W)
            else:
                imgs_1c = imgs

            # 获取Top3索引
            _, topk_indices = torch.topk(ts2img_weights, k=3, dim=-1)

            out_channels = []
            for k in range(3):
                # 第k个channel对应第k大的方法
                idx = topk_indices[:, :, k : k + 1]
                mask_hard = torch.zeros_like(ts2img_weights).scatter_(-1, idx, 1.0)
                mask = mask_hard - ts2img_weights.detach() + ts2img_weights

                mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # (B, D, 1, H, W)
                c_img = (imgs_1c * mask).sum(dim=2)
                out_channels.append(c_img)

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
        # 1. 选择最优 (使用Straight-Through Estimator保持梯度传导)
        if fusion_strategy == "select_best":
            # 计算所有方法的图像 (B, D, 7, C, H, W)
            imgs = self.compute_all_method_images(x, save_images)

            # 生成硬选择掩码，但保留梯度
            # idx: (B, D, 1)
            idx = torch.argmax(ts2img_weights, dim=-1, keepdim=True)
            mask_hard = torch.zeros_like(ts2img_weights).scatter_(-1, idx, 1.0)

            # Straight-Through Estimator: forward=mask_hard, backward=ts2img_weights
            mask = mask_hard - ts2img_weights.detach() + ts2img_weights

            # (B, D, 7, 1, 1, 1)
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # (B, D, C, H, W)
            out = (imgs * mask).sum(dim=2)
            return out

        # 2. 选择Top3堆叠 (使用Straight-Through Estimator保持梯度传导)
        elif fusion_strategy == "top3_stack":
            imgs = self.compute_all_method_images(x, save_images)
            # 确保使用单通道进行堆叠
            if imgs.shape[3] == 3:
                imgs_1c = imgs[:, :, :, 0:1, :, :]  # (B, D, 7, 1, H, W)
            else:
                imgs_1c = imgs

            # 获取Top3索引
            _, topk_indices = torch.topk(ts2img_weights, k=3, dim=-1)

            out_channels = []
            for k in range(3):
                # 第k个channel对应第k大的方法
                idx = topk_indices[:, :, k : k + 1]
                mask_hard = torch.zeros_like(ts2img_weights).scatter_(-1, idx, 1.0)
                mask = mask_hard - ts2img_weights.detach() + ts2img_weights

                mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # (B, D, 1, H, W)
                c_img = (imgs_1c * mask).sum(dim=2)
                out_channels.append(c_img)

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
