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
        # 1. 选择最优
        # 获取每个样本每个变量权重最大的方法索引，对应进行图像化转换
        if fusion_strategy == "select_best":
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
            for b in range(B):
                for d in range(D):
                    w = ts2img_weights[b, d]
                    idx = torch.argmax(w).item()
                    method = ts2img_methods[idx]
                    img = self.ts2img_single(
                        x[b, :, d],
                        method,
                        force_one_channel=not self.three_channel_image,
                    )
                    out[b, d] = img
            return out  # [B, D, C, H, W]

        # 2. 选择Top3堆叠
        # 获取每个样本每个变量Top3方法，转换后堆叠为RGB三通道图像
        elif fusion_strategy == "top3_stack":
            out = torch.zeros(
                B,
                D,
                3,
                self.image_size,
                self.image_size,
                device=x.device,
                dtype=x.dtype,
            )
            for b in range(B):
                for d in range(D):
                    w = ts2img_weights[b, d]
                    topk = torch.topk(w, k=3, dim=-1).indices
                    for c in range(3):
                        method = ts2img_methods[int(topk[c].item())]
                        img = self.ts2img_single(
                            x[b, :, d], method, force_one_channel=True
                        )
                        out[b, d, c] = img[0]
            return out  # [B, D, 3, H, W]
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
        # 1. 选择最优
        # 获取每个样本每个变量权重最大的方法索引，对应进行图像化转换
        if fusion_strategy == "select_best":
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
            for b in range(B):
                for d in range(D):
                    w = ts2img_weights[b, d]
                    idx = torch.argmax(w).item()
                    method = ts2img_methods[idx]
                    img = self.ts2img_single(
                        x[b, :, d],
                        method,
                        force_one_channel=not self.three_channel_image,
                    )
                    out[b, d] = img
            return out  # [B, D, C, H, W]

        # 2. 选择Top3堆叠
        # 获取每个样本每个变量Top3方法，转换后堆叠为RGB三通道图像
        elif fusion_strategy == "top3_stack":
            out = torch.zeros(
                B,
                D,
                3,
                self.image_size,
                self.image_size,
                device=x.device,
                dtype=x.dtype,
            )
            for b in range(B):
                for d in range(D):
                    w = ts2img_weights[b, d]
                    topk = torch.topk(w, k=3, dim=-1).indices
                    for c in range(3):
                        method = ts2img_methods[int(topk[c].item())]
                        img = self.ts2img_single(
                            x[b, :, d], method, force_one_channel=True
                        )
                        out[b, d, c] = img[0]
            return out  # [B, D, 3, H, W]
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
