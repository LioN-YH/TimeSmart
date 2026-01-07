from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import numpy as np
import einops


class TimeSeriesVisualizer:
    """Visualization tools for time series to image conversion"""

    @staticmethod
    def plot_feature_maps(features, title="Feature Maps"):
        """Plot feature maps from intermediate layers"""
        if not isinstance(features, torch.Tensor):
            return

        # Convert to numpy and normalize
        features = features.detach().cpu().numpy()
        features = (features - features.min()) / (
            features.max() - features.min() + 1e-8
        )

        # Plot first batch item
        num_channels = min(4, features.shape[1])
        if num_channels == 1:
            plt.figure(figsize=(5, 5))
            plt.imshow(features[0, 0], cmap="viridis")
            plt.axis("on")
            plt.title("Channel 1")
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(features[0, i], cmap="viridis")
                ax.axis("off")
                ax.set_title(f"Channel {i+1}")
        plt.suptitle(title)
        plt.tight_layout()

        # Save figure
        import os

        os.makedirs("ts-images/ts-visualizer", exist_ok=True)
        filename = f"ts-images/ts-visualizer/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {filename}")
        plt.close()

    @staticmethod
    def plot_attention(attention_map, title="Attention Map"):
        """Plot attention weights"""
        if not isinstance(attention_map, torch.Tensor):
            return

        attention_map = attention_map.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map[0, 0], cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.show()


# 代码实现部分和论文介绍略有差别，但是大体是一致的
class LearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""

    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1
        )

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape

        # Generate periodicity encoding (sin/cos)
        time_steps = (
            torch.arange(L, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(B, 1)
            .to(x_enc.device)
        )
        periodicity_encoding = torch.cat(
            [
                torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
                torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            ],
            dim=-1,
        )
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(
            1, 1, D, 1
        )  # [B, L, D, 2]

        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        # 零填充操作
        # 如果变换后的频谱长度（L//2+1）小于目标长度L，则在右侧填充零
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(
                B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype
            )
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodicity
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat(
            [x_enc, x_fft_mag, periodicity_encoding], dim=-1
        )  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D Convolution processing
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))

        # CHANGE: 【2026/1/6】删除多余的插值处理，重写的processor将进行统一的resize
        # # Resize to target image size
        # x_enc = F.interpolate(
        #     x_enc,
        #     size=(self.image_size, self.image_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        return x_enc  # [B, output_channels, H, W]


# CHANGE: 通道独立性版本，最终获得[B, D, output_channels, H, W]的表示
class LearnableTimeSeriesToImage_v(nn.Module):
    """Learnable module to convert time series data into image tensors"""

    def __init__(
        self, input_dim, hidden_dim, output_channels, image_size, periodicity, grid_size
    ):
        super(LearnableTimeSeriesToImage_v, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity
        self.grid_size = grid_size

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1
        )

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x_enc):
        """Convert input time series to image tensor [B*D, output_channels, H, W]"""
        # 流程大致如下：
        # 1. 周期编码 [B, L, D, 2]
        # 2. FFT编码 [B, L, D, 1]
        # 3. 原始输入 [B, L, D, 1]
        # 4. 堆叠上述三种编码 [B, L, D, 4]
        # 5. 1D卷积处理 [B*D, hidden_dim, L]
        # 6. 自适应池化 [B*D, hidden_dim, G^2]
        # 7. 重塑 [B*D, hidden_dim, G, G]
        # 8. 2D卷积处理 [B*D, output_channels, G, G]
        # 9. 插值调整 [B*D, output_channels, H, W]

        B, L, D = x_enc.shape

        # Generate periodicity encoding (sin/cos)
        time_steps = (
            torch.arange(L, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(B, 1)
            .to(x_enc.device)
        )
        periodicity_encoding = torch.cat(
            [
                torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
                torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            ],
            dim=-1,
        )
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(
            1, 1, D, 1
        )  # [B, L, D, 2]

        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        # 零填充操作
        # 如果变换后的频谱长度（L//2+1）小于目标长度L，则在右侧填充零
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(
                B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype
            )
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodicity
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat(
            [x_enc, x_fft_mag, periodicity_encoding], dim=-1
        )  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = F.tanh(self.conv1d(x_enc))  # [B*D, hidden_dim, L]

        # Adaptive pooling to grid size
        grid_area = self.grid_size * self.grid_size
        x_enc = F.adaptive_avg_pool1d(
            x_enc, output_size=grid_area
        )  # [B*D, hidden_dim, grid_area]

        # Reshape to square feature map
        x_enc = x_enc.reshape(
            B * D, self.hidden_dim, self.grid_size, self.grid_size
        )  # [B*D, hidden_dim, grid_size, grid_size]

        # 2D Convolution processing
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(
            self.conv2d_2(x_enc)
        )  # [B*D, output_channels, grid_size, grid_size]

        # CHANGE: 【2026/1/6】删除多余的插值处理，重写的processor将进行统一的resize
        # # Resize to target image size
        # x_enc = F.interpolate(
        #     x_enc,
        #     size=(self.image_size, self.image_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        # Reshape to recover D dimension
        x_enc = einops.rearrange(x_enc, "(b d) c h w -> b d c h w", b=B, d=D)
        return x_enc  # [B, D, output_channels, H, W]
