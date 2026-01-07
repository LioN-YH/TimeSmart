import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


# 简单的时序图像化处理
def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.

    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.

    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    # 调整填充以使 context_len 成为 periodicity 的倍数
    pad_left = 0
    if context_len % periodicity != 0:
        # 计算需要填充的左侧长度 pad_left
        # 使得 (context_len + pad_left) 可以被 periodicity 整除
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")

    # Pad the time series
    # 在序列左侧进行复制边缘值的填充
    x_pad = F.pad(x_enc, (pad_left, 0), mode="replicate")

    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=periodicity)

    # CHANGE: 【2026/1/6】删除多余的插值处理，重写的processor将进行统一的resize
    # # Resize the time series data
    # x_2d = F.interpolate(
    #     x_2d, size=(image_size, image_size), mode="bilinear", align_corners=False
    # )

    # Convert to 3-channel image
    # 转化为三通道图像
    images = einops.repeat(x_2d, "b 1 h w -> b c h w", c=3)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    # 重塑回 [B, nvars, 3, H, W] 并在 nvars 上求平均
    images = einops.rearrange(
        images, "(b n) c h w -> b n c h w", b=B, n=nvars
    )  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]

    return images


# CHANGE: 通道独立性版本，最终获得[B, nvars, 3, H, W]的表示
def time_series_to_simple_image_v(x_enc, image_size, context_len, periodicity):

    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    # 调整填充以使 context_len 成为 periodicity 的倍数
    pad_left = 0
    if context_len % periodicity != 0:
        # 计算需要填充的左侧长度 pad_left
        # 使得 (context_len + pad_left) 可以被 periodicity 整除
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")

    # Pad the time series
    # 在序列左侧进行复制边缘值的填充
    x_pad = F.pad(x_enc, (pad_left, 0), mode="replicate")

    # Reshape to [B, nvars, f, p]
    x_2d = einops.rearrange(x_pad, "b n (p f) -> b n f p", f=periodicity)

    # CHANGE: 【2026/1/6】删除多余的插值处理，重写的processor将进行统一的resize
    # # Resize the time series data
    # x_2d = F.interpolate(
    #     x_2d, size=(image_size, image_size), mode="bilinear", align_corners=False
    # )

    # Convert to 3-channel image
    # 转化为三通道图像
    images = einops.repeat(x_2d, "b n h w -> b n c h w", c=3)  # [B, nvars, 3, H, W]

    return images


# 采用傅里叶变换和小波变换的时序图像化处理
def time_series_to_image_with_fft_and_wavelet(
    x_enc, image_size, context_len, periodicity
):
    """
    Convert time series data into 3-channel image tensors using FFT and Wavelet transforms.

    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.

    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """

    def _apply_fourier_transform(x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        # 对输入张量x_2d沿最后一个维度（dim=-1，即时间维度）执行离散傅里叶变换（DFT）
        # 复数张量 x_fft，形状与输入相同 [N, L]，但每个元素是复数（实部和虚部），表示不同频率成分的振幅和相位。
        x_fft = torch.fft.fft(x_2d, dim=-1)

        # 提取幅度谱 torch.abs
        # 对于复数 a + bi，其模为 \(\sqrt{a^2 + b^2}\)，表示该频率成分的振幅大小。
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        # 创建一个正向离散小波变换（DWT）处理器
        # J=1：小波分解的级数（此处为 1 级分解，将原始信号分解为低频 cA1 和高频 cD1）
        # wave='haar'：使用 Haar 小波（最简单的正交小波基）
        dwt = DWTForward(J=1, wave="haar")

        # cA: Low-frequency components, cD: High-frequency components
        # 执行小波变换，返回低频分量 cA 和高频分量 cD
        # cA：低频分量，形状为 [B*nvars, 1, f', p']
        # cD：高频分量，是一个包含 3 个张量的元组，每个张量形状为 [B*nvars, 1, f', p']，分别对应水平、垂直和对角方向的高频信息
        cA, cD = dwt(x_2d)  # [B * nvars, 1, f, p]

        # cD[0]：提取高频分量元组中的第一个元素（包含三个方向的高频信息）
        # squeeze(1)：移除维度为 1 的通道维度
        cD_reshaped = cD[0].squeeze(1)  # [B * nvars, 3, f, p]

        # Concatenate low-frequency and high-frequency components
        # 合并低频和高频分量
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [B * nvars, 4, f, p]

        # Average across the channel dimension to reduce to 1 channel
        # 通道降维，通过平均化将通道数减少到 1
        wavelet_result = wavelet_result.mean(
            dim=1, keepdim=True
        )  # [B * nvars, 1, f, p]
        return wavelet_result

    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode="replicate")

    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=periodicity)

    # Resize the time series data
    # 对一个二维数据张量进行插值（上采样或下采样）操作，使其尺寸变为指定的图像大小
    # x_resized_2d：(B * nvars, 1, image_size, image_size)
    x_resized_2d = F.interpolate(
        x_2d, size=(image_size, image_size), mode="bilinear", align_corners=False
    )

    # Apply Fourier transform or wavelet transform
    # 应用傅里叶变换和小波变换
    # [B * nvars, 1, f, p]
    x_fft = _apply_fourier_transform(x_2d)
    # [B * nvars, 1, f, p]
    x_wavelet = _apply_wavelet_transform(x_2d)

    # Resize the Fourier or wavelet transformed data as image input using interpolation
    # 插值处理
    x_resized_fft = F.interpolate(
        x_fft, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    x_resized_wavelet = F.interpolate(
        x_wavelet, size=(image_size, image_size), mode="bilinear", align_corners=False
    )

    # Concatenate along the channel dimension to form a 3-channel image
    # 连接三个向量，构成三通道图像
    images = torch.concat(
        [x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1
    )  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    # 重塑回 [B, nvars, 3, H, W] 并在 nvars 上求平均
    images = einops.rearrange(
        images, "(b n) c h w -> b n c h w", b=B, n=nvars
    )  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]

    return images
