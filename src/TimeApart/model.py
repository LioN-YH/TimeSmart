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
        
        # Classification of methods
        self.freq_methods = ["stft", "wavelet", "cwt", "mel", "st"]
        self.visual_methods = ["plot", "smooth"]
        self.heat_methods = ["heat"]
        # Others: ["gaf", "rp", "mtf"] (Correlation) and ["seg", "hilbert"] (Structured) -> Default handling

        if method in self.freq_methods:
            # Frequency Domain: Anisotropic filtering to capture time/freq axes
            # Channel 0: Horizontal (Time continuity)
            self.conv_h = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))
            # Channel 1: Vertical (Frequency/Scale spread)
            self.conv_v = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
            # Channel 2: Local Texture
            self.conv_t = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        elif method in self.heat_methods:
            # Heatmap is vertically redundant (time series expanded to image)
            # We use multi-scale horizontal convolutions to capture temporal patterns
            # Channel 0: Short-term details (Kernel 1x3)
            self.conv_s = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
            # Channel 1: Medium-term trends (Kernel 1x7)
            self.conv_m = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3))
            # Channel 2: Long-term trends (Kernel 1x11)
            self.conv_l = nn.Conv2d(1, 1, kernel_size=(1, 11), padding=(0, 5))
        elif method in self.visual_methods:
            # Visual/Line plots: Larger kernel to capture sparse line features
            self.proj = nn.Conv2d(1, 3, kernel_size=7, padding=3)
            
        else:
            # Correlation/Structured: Standard local feature extraction
            # Using 3x3 kernel to map 1 channel to 3 channels
            self.proj = nn.Conv2d(1, 3, kernel_size=3, padding=1)
            
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, 1, H, W] or [B, H, W] (if squeezed)
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
        # Increased hidden dimension from 512 to 2048 to prevent information bottleneck for long horizons
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
        is_frozen = (
            hasattr(self.configs, "finetune_vlm") and not self.configs.finetune_vlm
        )
        
        # Warning: If backbone is frozen, input_projector cannot be trained!
        # We assume that if input_projector is used, finetune_vlm should be True
        # or at least the user is aware.
        
        for d in range(D):
            img_d = images[:, d]  # [B, 1, H, W]
            
            # Apply Method-Specific Projection
            # This maps 1 channel to 3 channels using method-specific kernels
            img_d_proj = self.input_projector(img_d) # [B, 3, H, W]
            
            # Even if backbone is frozen (finetune_vlm=False), we avoid torch.no_grad() here.
            # This ensures gradients can flow back through the backbone to update the 
            # input_projector parameters, while the backbone weights remain static 
            # (due to requires_grad=False set in __init__).
            out = self.backbone(img_d_proj).last_hidden_state  # [B, C, H', W']

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
