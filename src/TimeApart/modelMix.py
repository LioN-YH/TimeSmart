import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import ConvNextModel, ConvNextConfig
from layers.VE import MT2VEncoder
from .norm import Normalize
from .adapter import Adapter


class MethodInputProjector(nn.Module):
    def __init__(self, method):
        super(MethodInputProjector, self).__init__()
        self.method = method
        self.freq_methods = ["stft", "wavelet", "cwt", "mel", "st"]
        self.visual_methods = ["plot", "smooth"]
        self.heat_methods = ["heat"]

        if method in self.freq_methods:
            self.conv_h = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))
            self.conv_v = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
            self.conv_t = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        elif method in self.heat_methods:
            self.conv_s = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
            self.conv_m = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3))
            self.conv_l = nn.Conv2d(1, 1, kernel_size=(1, 11), padding=(0, 5))
        elif method in self.visual_methods:
            self.proj = nn.Conv2d(1, 3, kernel_size=7, padding=3)
        else:
            self.proj = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        self.act = nn.GELU()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if self.method in self.freq_methods:
            h = self.conv_h(x)
            v = self.conv_v(x)
            t = self.conv_t(x)
            out = torch.cat([h, v, t], dim=1)
        elif self.method in self.heat_methods:
            s = self.conv_s(x)
            m = self.conv_m(x)
            l = self.conv_l(x)
            out = torch.cat([s, m, l], dim=1)
        else:
            out = self.proj(x)

        return self.act(out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_features = configs.enc_in

        self.router_periodicity = getattr(configs, "periodicity", 24)
        self.router_temperature = getattr(configs, "router_temperature", 5.0)
        self.router_bias = getattr(configs, "router_bias", 0.0)
        self.router_detach_score = getattr(configs, "router_detach_score", False)
        self.router_aux_weight = getattr(configs, "router_aux_weight", 0.0)
        self.router_balance_target = getattr(configs, "router_balance_target", 0.5)
        self.router_entropy_weight = getattr(configs, "router_entropy_weight", 0.0)
        self.router_eps = 1e-6
        self.trend_ma_window = getattr(configs, "trend_ma_window", 0)
        self.trend_low_freq_weight = getattr(configs, "trend_low_freq_weight", 0.65)
        self.trend_direction_weight = getattr(configs, "trend_direction_weight", 0.15)
        self.trend_linearity_weight = getattr(configs, "trend_linearity_weight", 0.20)

        self.revin = Normalize(self.num_features, affine=False)

        if not hasattr(configs, "image_size"):
            configs.image_size = 224
        if not hasattr(configs, "interpolation"):
            configs.interpolation = "bilinear"
        if not hasattr(configs, "three_channel_image"):
            configs.three_channel_image = False
        if not hasattr(configs, "periodicity"):
            configs.periodicity = 24

        configs.compress_vars = False
        self.img_encoder = MT2VEncoder(configs)

        self.seg_method = "seg"
        self.smooth_method = "smooth"
        self.seg_input_projector = MethodInputProjector(self.seg_method)
        self.smooth_input_projector = MethodInputProjector(self.smooth_method)

        print("Loading ConvNeXt backbone...")
        try:
            self.backbone = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
        except Exception as e:
            print(f"Warning: Could not load pretrained ConvNeXt: {e}. Using random initialization.")
            config = ConvNextConfig(image_size=configs.image_size)
            self.backbone = ConvNextModel(config)

        if hasattr(configs, "finetune_vlm") and not configs.finetune_vlm:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.neck_channels = 256
        self.neck = nn.Sequential(
            nn.Conv2d(
                self.backbone.config.hidden_sizes[-1],
                self.neck_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, self.neck_channels),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy_h = configs.image_size // 32
            dummy_input = torch.zeros(
                1, self.backbone.config.hidden_sizes[-1], dummy_h, dummy_h
            )
            dummy_output = self.neck(dummy_input)
            self.backbone_dim = dummy_output.numel()

        print(f"Feature dimension after Neck: {self.backbone_dim}")

        self.seg_adapter = Adapter(self.backbone_dim, self.backbone_dim // 4, self.backbone_dim)
        self.smooth_adapter = Adapter(self.backbone_dim, self.backbone_dim // 4, self.backbone_dim)

        head_hidden_dim = 512
        self.seg_head = nn.Sequential(
            nn.Linear(self.backbone_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(head_hidden_dim, self.pred_len),
        )
        self.smooth_head = nn.Sequential(
            nn.Linear(self.backbone_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(head_hidden_dim, self.pred_len),
        )

        self.last_router_info = {}
        self.last_router_aux_loss = None
        self.last_trend_components = {}

    def _compute_periodic_score(self, x):
        B, L, D = x.shape
        lag = min(self.router_periodicity, max(1, L // 2))
        if lag >= L:
            return x.new_zeros(B)

        x1 = x[:, lag:, :]
        x2 = x[:, :-lag, :]
        x1 = x1 - x1.mean(dim=1, keepdim=True)
        x2 = x2 - x2.mean(dim=1, keepdim=True)

        numerator = (x1 * x2).sum(dim=1)
        denominator = torch.sqrt(
            (x1.pow(2).sum(dim=1) + self.router_eps)
            * (x2.pow(2).sum(dim=1) + self.router_eps)
        )
        corr = numerator / (denominator + self.router_eps)
        return corr.abs().mean(dim=-1)

    def _moving_average_trend(self, x, ma_window):
        xt = x.permute(0, 2, 1)
        pad = ma_window // 2
        xt_pad = F.pad(xt, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xt_pad, kernel_size=ma_window, stride=1)
        return trend.permute(0, 2, 1)

    def _compute_trend_score(self, x):
        B, L, D = x.shape

        if self.trend_ma_window is None or self.trend_ma_window <= 0:
            ma_window = min(
                max(9, self.router_periodicity * 2 + 1),
                L if L % 2 == 1 else L - 1,
            )
        else:
            ma_window = min(
                self.trend_ma_window,
                L if L % 2 == 1 else L - 1,
            )

        if ma_window < 3:
            ma_window = 3
        if ma_window % 2 == 0:
            ma_window += 1
        if ma_window > L:
            ma_window = L if L % 2 == 1 else max(3, L - 1)

        trend = self._moving_average_trend(x, ma_window=ma_window)

        x_centered = x - x.mean(dim=1, keepdim=True)
        trend_centered = trend - trend.mean(dim=1, keepdim=True)

        total_var = x_centered.pow(2).mean(dim=1)
        trend_var = trend_centered.pow(2).mean(dim=1)
        low_freq_ratio = (trend_var / (total_var + self.router_eps)).clamp(0.0, 1.0)

        dtrend = trend[:, 1:, :] - trend[:, :-1, :]
        direction_consistency = (
            dtrend.mean(dim=1).abs() / (dtrend.abs().mean(dim=1) + self.router_eps)
        ).clamp(0.0, 1.0)

        t = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, L, 1)
        tc = t - t.mean(dim=1, keepdim=True)

        numerator = (trend_centered * tc).sum(dim=1)
        denominator = torch.sqrt(
            (trend_centered.pow(2).sum(dim=1) + self.router_eps)
            * (tc.pow(2).sum(dim=1) + self.router_eps)
        )
        linearity = (numerator / (denominator + self.router_eps)).abs().clamp(0.0, 1.0)

        weight_sum = (
            self.trend_low_freq_weight
            + self.trend_direction_weight
            + self.trend_linearity_weight
        )
        trend_score_per_var = (
            self.trend_low_freq_weight * low_freq_ratio
            + self.trend_direction_weight * direction_consistency
            + self.trend_linearity_weight * linearity
        ) / weight_sum

        self.last_trend_components = {
            "low_freq_ratio_mean": low_freq_ratio.mean().item(),
            "direction_consistency_mean": direction_consistency.mean().item(),
            "linearity_mean": linearity.mean().item(),
        }

        return trend_score_per_var.mean(dim=-1)

    def _compute_router_weights(self, x):
        periodic_score = self._compute_periodic_score(x)
        trend_score = self._compute_trend_score(x)

        if self.router_detach_score:
            periodic_for_gate = periodic_score.detach()
            trend_for_gate = trend_score.detach()
        else:
            periodic_for_gate = periodic_score
            trend_for_gate = trend_score

        score_gap = periodic_for_gate - trend_for_gate - self.router_bias
        w_seg = torch.sigmoid(self.router_temperature * score_gap)
        w_smooth = 1.0 - w_seg

        prefer_seg = (w_seg >= 0.5).float()
        prefer_smooth = 1.0 - prefer_seg

        aux_terms = []
        if self.router_aux_weight > 0:
            seg_ratio = w_seg.mean()
            balance_loss = (seg_ratio - self.router_balance_target).pow(2)
            aux_terms.append(self.router_aux_weight * balance_loss)

        if self.router_entropy_weight > 0:
            entropy = -(
                w_seg * torch.log(w_seg + self.router_eps)
                + w_smooth * torch.log(w_smooth + self.router_eps)
            ).mean()
            aux_terms.append(-self.router_entropy_weight * entropy)

        aux_loss = sum(aux_terms) if len(aux_terms) > 0 else None
        router_info = {
            "periodic_score_mean": periodic_score.mean().item(),
            "trend_score_mean": trend_score.mean().item(),
            "seg_weight_mean": w_seg.mean().item(),
            "smooth_weight_mean": w_smooth.mean().item(),
            "seg_prefer_ratio": prefer_seg.mean().item(),
            "smooth_prefer_ratio": prefer_smooth.mean().item(),
        }
        if hasattr(self, "last_trend_components"):
            router_info.update(self.last_trend_components)
        return w_seg, w_smooth, aux_loss, router_info

    def _encode_images(self, images, input_projector):
        B, D, _, _, _ = images.shape
        feats = []
        out = None

        for d in range(D):
            img_d = images[:, d]
            img_d_proj = input_projector(img_d)
            out = self.backbone(img_d_proj).last_hidden_state
            feats.append(out)

        feat = torch.stack(feats, dim=1).reshape(B * D, -1, out.size(2), out.size(3))
        feat = self.neck(feat)
        feat = feat.reshape(feat.size(0), -1)
        return feat

    def _forward_branch(self, x, method, input_projector, adapter, head):
        B, _, D = x.shape
        images = self.img_encoder.get_ts2img_tensor(x, method)
        images = images.reshape(B, D, 1, self.configs.image_size, self.configs.image_size)
        feat = self._encode_images(images, input_projector)
        feat = feat + adapter(feat)
        pred = head(feat)
        pred = pred.reshape(B, D, self.pred_len).permute(0, 2, 1)
        return pred

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B, _, D = x.shape
        x = self.revin(x, "norm")

        w_seg, w_smooth, aux_loss, router_info = self._compute_router_weights(x)

        pred_seg = self._forward_branch(
            x, self.seg_method, self.seg_input_projector, self.seg_adapter, self.seg_head
        )
        pred_smooth = self._forward_branch(
            x, self.smooth_method, self.smooth_input_projector, self.smooth_adapter, self.smooth_head
        )

        pred = w_seg.view(B, 1, 1) * pred_seg + w_smooth.view(B, 1, 1) * pred_smooth
        pred = self.revin(pred, "denorm")

        self.last_router_info = router_info
        self.last_router_aux_loss = aux_loss
        return pred
