import torch
import torch.nn as nn
import sys
import os

# Add project root to path to allow importing from layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import ViTModel, ViTConfig
from layers.VE import MT2VEncoder

# Local imports
from .norm import Normalize
from .adapter import Adapter


class MethodInputProjector(nn.Module):
    def __init__(self, method):
        super(MethodInputProjector, self).__init__()
        self.method = method

        # Classification of methods
        self.freq_methods = ["stft", "wavelet", "cwt", "mel", "st"]
        self.visual_methods = ["plot", "smooth"]
        self.heat_methods = ["heat"]
        # Others: ["gaf", "rp", "mtf", "seg", "hilbert"]

        if method in self.freq_methods:
            # Frequency-domain images: emphasize time/freq anisotropy
            self.conv_h = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))
            self.conv_v = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
            self.conv_t = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        elif method in self.heat_methods:
            # Heatmap-like representation: emphasize temporal horizontal structures
            self.conv_s = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
            self.conv_m = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3))
            self.conv_l = nn.Conv2d(1, 1, kernel_size=(1, 11), padding=(0, 5))

        elif method in self.visual_methods:
            self.proj = nn.Conv2d(1, 3, kernel_size=7, padding=3)

        else:
            self.proj = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, 1, H, W] or [B, H, W]
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


class TokenFusionNeck(nn.Module):
    """
    Fuse CLS token and mean pooled patch tokens into a compact feature.
    Input:
        tokens: [B, 1 + N, C]
    Output:
        feat: [B, neck_dim]
    """

    def __init__(self, hidden_size, neck_dim, dropout=0.1):
        super(TokenFusionNeck, self).__init__()
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, neck_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, tokens):
        cls_feat = tokens[:, 0]  # [B, C]
        patch_feat = tokens[:, 1:].mean(dim=1)  # [B, C]
        fused = torch.cat([cls_feat, patch_feat], dim=-1)  # [B, 2C]
        fused = self.norm(fused)
        feat = self.proj(fused)
        return feat


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_features = configs.enc_in
        self.method = getattr(configs, "ts2img_method", "stft")

        # 1. RevIN
        self.revin = Normalize(self.num_features, affine=False)

        # 2. Time series -> image encoder
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

        # 3. Method-specific input projector
        self.input_projector = MethodInputProjector(self.method)

        # 4. ViT backbone
        print("Loading ViT backbone...")
        vit_name = getattr(
            configs, "vit_model_name", "google/vit-base-patch16-224-in21k"
        )
        vit_hidden_size = getattr(configs, "vit_hidden_size", 768)
        vit_patch_size = getattr(configs, "vit_patch_size", 16)

        try:
            self.backbone = ViTModel.from_pretrained(vit_name)
            print(f"Loaded pretrained ViT from: {vit_name}")
        except Exception as e:
            print(
                f"Warning: Could not load pretrained ViT: {e}. Using random initialization."
            )
            vit_config = ViTConfig(
                image_size=configs.image_size,
                patch_size=vit_patch_size,
                num_channels=3,
                hidden_size=vit_hidden_size,
                intermediate_size=vit_hidden_size * 4,
                num_hidden_layers=12,
                num_attention_heads=12,
            )
            self.backbone = ViTModel(vit_config)

        # Optional freezing
        if hasattr(configs, "finetune_vlm") and not configs.finetune_vlm:
            print("Freezing ViT backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size

        # 5. Token fusion neck
        self.neck_dim = getattr(configs, "neck_dim", 512)
        self.neck = TokenFusionNeck(
            hidden_size=self.hidden_size,
            neck_dim=self.neck_dim,
            dropout=getattr(configs, "dropout", 0.1),
        )

        # 6. Adapter
        adapter_hidden = max(32, self.neck_dim // 4)
        self.adapter = Adapter(
            self.neck_dim,
            adapter_hidden,
            self.neck_dim,
            dropout=getattr(configs, "dropout", 0.1),
        )

        # 7. Prediction head
        head_hidden_dim = getattr(configs, "head_hidden_dim", 512)
        self.head = nn.Sequential(
            nn.Linear(self.neck_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(getattr(configs, "dropout", 0.1)),
            nn.Linear(head_hidden_dim, self.pred_len),
        )

    def encode_one_variable_image(self, img_d):
        """
        img_d: [B, 1, H, W]
        return: [B, neck_dim]
        """
        img_d_proj = self.input_projector(img_d)  # [B, 3, H, W]

        # ViT output:
        # last_hidden_state: [B, 1 + N, C]
        out = self.backbone(pixel_values=img_d_proj).last_hidden_state

        feat = self.neck(out)  # [B, neck_dim]
        feat = feat + self.adapter(feat)  # residual adapter
        return feat

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x: [B, L, D]
        B, L, D = x.shape

        # 1. RevIN normalize
        x = self.revin(x, "norm")

        # 2. Time series -> image
        images = self.img_encoder.get_ts2img_tensor(x, self.method)
        # [B, D, H, W] -> [B, D, 1, H, W]
        images = images.reshape(
            B, D, 1, self.configs.image_size, self.configs.image_size
        )

        # 3. Encode each variable image
        feats = []
        for d in range(D):
            img_d = images[:, d]  # [B, 1, H, W]
            feat_d = self.encode_one_variable_image(img_d)  # [B, neck_dim]
            feats.append(feat_d)

        # 4. Stack per-variable features
        # [B, D, neck_dim]
        feat = torch.stack(feats, dim=1)

        # 5. Prediction head
        # reshape -> [B*D, neck_dim]
        feat = feat.reshape(B * D, self.neck_dim)
        pred = self.head(feat)  # [B*D, pred_len]

        # 6. Reshape back to [B, pred_len, D]
        pred = pred.reshape(B, D, self.pred_len)
        pred = pred.permute(0, 2, 1)  # [B, pred_len, D]

        # 7. RevIN denormalize
        pred = self.revin(pred, "denorm")

        return pred
