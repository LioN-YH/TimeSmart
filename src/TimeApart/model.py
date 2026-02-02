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

        # 3. Backbone (Shared Pretrained ConvNeXt)
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
        # Reduce 768x7x7 (37632) to something manageable using a Conv layer
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

        # Calculate backbone_dim dynamically after Neck
        with torch.no_grad():
            # Dummy input representing last_hidden_state: [1, 768, 7, 7]
            dummy_h = configs.image_size // 32
            dummy_input = torch.zeros(
                1, self.backbone.config.hidden_sizes[-1], dummy_h, dummy_h
            )
            dummy_output = self.neck(dummy_input)
            self.backbone_dim = (
                dummy_output.numel()
            )  # Flattened size: e.g. 256 * 4 * 4 = 4096

        print(f"Feature dimension after Neck: {self.backbone_dim}")

        # 5. Adapter
        # Specific adapter for the chosen branch/method
        self.adapter = Adapter(
            self.backbone_dim, self.backbone_dim // 4, self.backbone_dim
        )

        # 5. Prediction Head (MLP)
        # Maps backbone features to prediction length
        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(512, self.pred_len),
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x: [B, L, D]
        B, L, D = x.shape

        # 1. RevIN Normalize (Input)
        x = self.revin(x, "norm")

        # 2. Transform to Images using selected method
        images = self.img_encoder.get_ts2img_tensor(x, self.method)
        # Prepare for ConvNeXt: [B, D, C, H, W]
        images = images.reshape(
            B, D, 1, self.configs.image_size, self.configs.image_size
        )
        images = images.repeat(1, 1, 3, 1, 1)  # Expand to 3 channels: [B, D, 3, H, W]

        # 3. Backbone Encoding (Iterate over D to save GPU memory)
        feats = []
        is_frozen = (
            hasattr(self.configs, "finetune_vlm") and not self.configs.finetune_vlm
        )
        for d in range(D):
            img_d = images[:, d]  # [B, 3, H, W]
            if is_frozen:
                with torch.no_grad():
                    out = self.backbone(img_d).last_hidden_state
            else:
                out = self.backbone(img_d).last_hidden_state  # [B, C, H', W']

            feats.append(out)

        # Stack features: [B, D, C, H', W']
        # Reshape to [B*D, C, H', W'] for batch processing through Neck
        feat = torch.stack(feats, dim=1).reshape(B * D, -1, out.size(2), out.size(3))

        # 4. Neck
        feat = self.neck(feat)  # [B*D, C_neck, H_neck, W_neck]
        feat = feat.reshape(feat.size(0), -1)  # Flatten: [B*D, backbone_dim]

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
