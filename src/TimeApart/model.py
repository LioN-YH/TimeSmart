import torch
import torch.nn as nn
import sys
import os

# Add project root to path to allow importing from layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import ConvNextModel, ConvNextConfig
from layers.VE import MT2VEncoder

# Local imports
from .norm import Normalize
from .adapter import Adapter


class MethodInputProjector(nn.Module):
    def __init__(self, method):
        super(MethodInputProjector, self).__init__()
        self.method = method

        self.freq_mag_methods = ["stft", "wavelet", "mel"]
        self.freq_phase_methods = ["cwt", "st"]
        self.relation_methods = ["gaf", "rp", "mtf"]
        self.structured_methods = ["seg", "hilbert"]
        self.plot_methods = ["plot"]
        self.heat_methods = ["heat"]
        self.smooth_methods = ["smooth"]

        self.act = nn.GELU()

        # shared residual shortcut
        self.shortcut = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        if method in self.freq_mag_methods:
            self.branch1 = nn.Conv2d(
                1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False
            )
            self.branch2 = nn.Conv2d(
                1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False
            )
            self.branch3 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        elif method in self.freq_phase_methods:
            self.branch1 = nn.Conv2d(
                1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False
            )
            self.branch2 = nn.Conv2d(
                1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False
            )
            self.branch3 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)

        elif method in self.relation_methods:
            self.branch1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            self.branch2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
            self.branch3 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False),
                nn.GELU(),
                nn.Conv2d(1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False),
            )

        elif method in self.structured_methods:
            self.branch1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            self.branch2 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
            self.branch3 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

        elif method in self.plot_methods:
            self.branch1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
            self.branch2 = nn.Conv2d(
                1, 1, kernel_size=(1, 11), padding=(0, 5), bias=False
            )
            self.branch3 = nn.Conv2d(
                1, 1, kernel_size=(11, 1), padding=(5, 0), bias=False
            )

        elif method in self.heat_methods:
            self.branch1 = nn.Conv2d(
                1, 1, kernel_size=(1, 3), padding=(0, 1), bias=False
            )
            self.branch2 = nn.Conv2d(
                1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False
            )
            self.branch3 = nn.Conv2d(
                1, 1, kernel_size=(1, 11), padding=(0, 5), bias=False
            )

        elif method in self.smooth_methods:
            self.branch1 = nn.Conv2d(
                1, 1, kernel_size=(1, 3), padding=(0, 1), bias=False
            )
            self.branch2 = nn.Conv2d(
                1, 1, kernel_size=(1, 9), padding=(0, 4), bias=False
            )
            self.branch3 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        else:
            self.proj = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)

        self.fuse = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, bias=False), nn.GELU())

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if hasattr(self, "proj"):
            out = self.proj(x)
            out = out + self.shortcut(x)
            return self.act(out)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        out = torch.cat([b1, b2, b3], dim=1)  # [B, 3, H, W]
        out = self.fuse(out) + self.shortcut(x)
        return self.act(out)


class InputAlignStem(nn.Module):
    """
    Align ts2img pseudo-images to a distribution that is easier for pretrained ConvNeXt to consume.
    Lightweight, trainable, and stable for small batch sizes.
    """

    def __init__(self, in_channels=3, hidden_channels=16, out_channels=3):
        super(InputAlignStem, self).__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(
                1, in_channels
            ),  # channel-wise normalization, batch-size friendly
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GELU(),
            nn.GroupNorm(4 if hidden_channels >= 4 else 1, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        y = self.net(x)
        return x + y


class StructuredPoolingHead(nn.Module):
    """
    Replace naive flatten with structured pooling.

    Input:
        x: [B, C, H, W]

    Output:
        pooled feature: [B, out_dim]
    """

    def __init__(self, in_channels, in_h, in_w, proj_dim=1024, dropout=0.1):
        super(StructuredPoolingHead, self).__init__()

        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w

        # Global summaries
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Structured summaries
        # row_pool: preserve height structure, average over width
        # shape -> [B, C, H]
        # col_pool: preserve width structure, average over height
        # shape -> [B, C, W]

        total_dim = (
            in_channels  # GAP
            + in_channels  # GMP
            + in_channels * in_h  # Row pooling
            + in_channels * in_w  # Col pooling
        )

        self.proj = nn.Sequential(
            nn.Linear(total_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        self.out_dim = proj_dim

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        gap = self.gap(x).flatten(1)  # [B, C]
        gmp = self.gmp(x).flatten(1)  # [B, C]
        row_pool = x.mean(dim=3).reshape(b, c * h)  # [B, C*H]
        col_pool = x.mean(dim=2).reshape(b, c * w)  # [B, C*W]

        feat = torch.cat([gap, gmp, row_pool, col_pool], dim=1)
        feat = self.proj(feat)
        return feat


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_features = configs.enc_in  # Number of variates/channels
        self.method = getattr(
            configs, "ts2img_method", "stft"
        )  # Default to stft if not specified

        # 1. RevIN (Reversible Instance Normalization)
        self.revin = Normalize(self.num_features, affine=False)

        # 2. Image Transformer (Time Series -> Image)
        # Reusing existing VE implementation

        if not hasattr(configs, "image_size"):
            configs.image_size = 224
        if not hasattr(configs, "interpolation"):
            configs.interpolation = "bilinear"

        configs.compress_vars = False

        if not hasattr(configs, "three_channel_image"):
            configs.three_channel_image = False  # We handle channel expansion manually
        if not hasattr(configs, "periodicity"):
            configs.periodicity = 24  # Default periodicity

        self.img_encoder = MT2VEncoder(configs)

        # 3. Method-Specific Input Projector
        # Replaces simple channel repetition with learnable, method-aware projection
        self.input_projector = MethodInputProjector(self.method)

        # 3.5 Input Alignment Stem
        # Learnable distribution alignment before ConvNeXt
        self.input_align = InputAlignStem(
            in_channels=3,
            hidden_channels=16,
            out_channels=3,
        )

        # 4. Backbone (Shared Pretrained ConvNeXt)
        print("Loading ConvNeXt backbone...")
        try:
            # Try loading pretrained weights
            # Using 'facebook/convnext-tiny-224' as a standard efficient backbone
            self.backbone = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
        except Exception as e:
            print(
                f"Warning: Could not load pretrained ConvNeXt: {e}. Using random initialization."
            )
            config = ConvNextConfig(image_size=configs.image_size)
            self.backbone = ConvNextModel(config)

        if hasattr(configs, "finetune_vlm") and not configs.finetune_vlm:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get hidden size (ConvNeXt Tiny: 768)
        # We are using last_hidden_state, so we need to account for spatial dimensions
        # ConvNeXt downsamples by a factor of 32
        # Original: 7x7 spatial map for 224x224 input

        # 4. Neck (Dimensionality Reduction)
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

        # Calculate neck output shape dynamically using a real dummy image
        with torch.no_grad():
            ref_param = next(self.backbone.parameters())
            dummy_img = torch.zeros(
                1,
                3,
                configs.image_size,
                configs.image_size,
                device=ref_param.device,
                dtype=ref_param.dtype,
            )

            # Real backbone forward
            dummy_feat = self.backbone(dummy_img).last_hidden_state  # [1, C, H', W']
            _, backbone_c, backbone_h, backbone_w = dummy_feat.shape
            print(
                f"Backbone output shape: C={backbone_c}, H={backbone_h}, W={backbone_w}"
            )

            # Neck forward
            dummy_output = self.neck(dummy_feat)  # [1, Cn, Hn, Wn]
            _, neck_c, neck_h, neck_w = dummy_output.shape

        print(f"Neck output shape: C={neck_c}, H={neck_h}, W={neck_w}")

        # Structured pooling head
        self.pooling_head = StructuredPoolingHead(
            in_channels=neck_c,
            in_h=neck_h,
            in_w=neck_w,
            proj_dim=1024,
            dropout=configs.dropout,
        )

        self.backbone_dim = self.pooling_head.out_dim
        print(f"Feature dimension after structured pooling: {self.backbone_dim}")

        # Adapter
        self.adapter = Adapter(
            self.backbone_dim,
            self.backbone_dim // 4,
            self.backbone_dim,
        )

        # Prediction head
        head_hidden_dim = 512
        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(head_hidden_dim, self.pred_len),
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x: [B, L, D]
        B, L, D = x.shape

        # 1. RevIN Normalize (Input)
        x = self.revin(x, "norm")

        # 2. Transform to Images using selected method
        images = self.img_encoder.get_ts2img_tensor(x, self.method)
        # Prepare for ConvNeXt: [B, D, 1, H, W]
        images = images.reshape(
            B, D, 1, self.configs.image_size, self.configs.image_size
        )
        # Note: We use input_projector to expand channels instead of simple repeat

        # 3. Backbone Encoding (Iterate over D to save GPU memory)
        feats = []
        # If the backbone is frozen (requires_grad=False), its parameters will not be updated.
        # However, because we do NOT wrap the backbone forward with torch.no_grad(),
        # gradients can still flow through the backbone computation graph and update
        # upstream trainable modules such as input_projector.

        for d in range(D):
            img_d = images[:, d]  # [B, 1, H, W]

            # 1) method-aware projection
            img_d_proj = self.input_projector(img_d)  # [B, 3, H, W]

            # 2) learnable alignment before pretrained ConvNeXt
            img_d_proj = self.input_align(img_d_proj)  # [B, 3, H, W]

            # 3) ConvNeXt encoding
            out = self.backbone(img_d_proj).last_hidden_state  # [B, C, H', W']

            feats.append(out)

        # Stack features: [B, D, C, H', W']
        # Reshape to [B*D, C, H', W'] for batch processing through Neck
        feat = torch.stack(feats, dim=1).reshape(B * D, -1, out.size(2), out.size(3))

        # 4. Neck
        feat = self.neck(feat)  # [B*D, C_neck, H_neck, W_neck]
        feat = self.pooling_head(feat)  # [B*D, backbone_dim]

        # 5. Adapter
        feat = feat + self.adapter(feat)
        # 6. Prediction Head
        pred = self.head(feat)  # [B*D, pred_len]
        # Reshape back to [B, D, pred_len]
        pred = pred.reshape(B, D, self.pred_len)
        # Permute to [B, pred_len, D]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, D]

        # 7. RevIN Denormalize (Output)
        pred = self.revin(pred, "denorm")

        return pred
