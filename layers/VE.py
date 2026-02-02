import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import inspect
import time
from PIL import Image
from torchvision.transforms import Resize
import pywt
import matplotlib.cm as cm
import os

# INTRO:时序图像转化模块，支持seg, gaf, rp, stft, wavelet, mel, mtf等多种方法

# ts2img_methods = ["seg", "gaf", "rp", "stft", "wavelet", "mel", "mtf"]

ts2img_methods = [
    "wavelet",
    "cwt",
    "mel",
    "mtf",
    "seg",
    "gaf",
    "rp",
    "stft",
    "st",
    "hilbert",
    "plot",
    "heat",
    "smooth",
]


class MT2VEncoder(nn.Module):
    def __init__(self, config):
        super(MT2VEncoder, self).__init__()

        # CHANGE：添加部分参数的加载
        # universal
        self.image_size = config.image_size
        self.interpolation = config.interpolation
        self.compress_vars = config.compress_vars
        self.three_channel_image = config.three_channel_image

        # seg
        self.periodicity = config.periodicity

        # gaf
        self.gaf_method = getattr(config, "gaf_method", "summation")

        # rp
        self.rp_threshold = getattr(config, "rp_threshold", "point")
        self.rp_percentage = getattr(config, "rp_percentage", 10)

        # stft
        self.stft_window_size = getattr(config, "stft_window_size", 128)
        self.stft_hop_length = getattr(config, "stft_hop_length", 32)
        self.use_log_scale = getattr(config, "use_log_scale", True)

        # wavelet
        self.wavelet_type = getattr(config, "wavelet_type", "morl")

        # mel
        self.use_mel = getattr(config, "use_mel", False)
        self.num_filters = getattr(config, "num_filters", 32)

        # mtf
        self.mtf_downsample_threshold = getattr(config, "mtf_downsample_threshold", 256)
        self.use_fast_mode = getattr(config, "use_fast_mode", False)

        # hilbert
        self.hilbert_curve_cache = {}

        # plot
        self.plot_line_thickness = getattr(config, "plot_line_thickness", 6)

        # add
        self.method_times = {}

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[self.interpolation]

        self.input_resize = self.safe_resize(
            (self.image_size, self.image_size), interpolation=interpolation
        )

    def normalize_per_series(self, x, eps=1e-8):
        # x: B, L, D -> normalize along L
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def standardize_per_series(self, x, eps=1e-8):
        # x: B, L, D or B*D, L -> normalize along L (dim 1)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        return (x - x_mean) / (x_std + eps)

    def normalize_per_image(self, x, eps=1e-8):
        # x: B, D, H, W -> normalize along H, W
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def normalize_minmax(self, x, eps=1e-8):
        if x.numel() == 0:
            return torch.zeros_like(x)

        x_min = x.min()
        x_max = x.max()

        if x_max - x_min < eps:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    def segmentation(self, x):
        B, L, D = x.shape

        # CHANGE: Use Autocorrelation to find the dominant period
        # periods, _ = self.FFT_for_Period(x, k=1)
        # period = int(periods[0])
        period = self.Autocorrelation_for_Period(x)

        # Safety check for period
        if period < 2:
            period = 2

        x = einops.rearrange(x, "b s d -> b d s")
        pad_left = 0
        if L % period != 0:
            pad_left = period - L % period
        x_pad = F.pad(x, (pad_left, 0), mode="replicate")

        x_2d = einops.rearrange(
            x_pad,
            "b d (p f) -> b d f p",
            p=x_pad.size(-1) // period,
            f=period,
        )

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            x_combined = torch.mean(x_2d, dim=1, keepdim=True)
        else:
            x_combined = x_2d

        x_resize = F.interpolate(
            x_combined,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Vectorized Normalization per image
        x_norm = self.normalize_per_image(x_resize)
        x_combined = x_norm

        grid_size = self.image_size // 8
        grid_mask = torch.ones_like(x_combined)
        grid_mask[:, :, ::grid_size, :] = 0.95
        grid_mask[:, :, :, ::grid_size] = 0.95
        x_combined = x_combined * grid_mask

        return x_combined

    def gramian_angular_field(self, x):
        B, L, D = x.shape

        # Normalize per series (instance normalization)
        x_norm = self.normalize_per_series(x) * 2 - 1
        theta = torch.arccos(x_norm.clamp(-1 + 1e-6, 1 - 1e-6))

        # Vectorized GAF
        angle_i = theta.unsqueeze(2)  # (B, L, 1, D)
        angle_j = theta.unsqueeze(1)  # (B, 1, L, D)

        if self.gaf_method == "summation":
            gaf_matrix = torch.cos(angle_i + angle_j)
        else:
            gaf_matrix = torch.cos(angle_i - angle_j)

        # gaf_matrix: (B, L, L, D) -> (B, D, L, L)
        gaf = gaf_matrix.permute(0, 3, 1, 2)

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            gaf = gaf.mean(dim=1, keepdim=True)

        gaf = F.interpolate(
            gaf,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize result to [0, 1] per image (After interpolation to preserve contrast)
        gaf = self.normalize_per_image(gaf)

        return gaf

    # CHANGE: Univariate Recurrence Plot
    def recurrence_plot_u(self, x):
        B, L, D = x.shape

        # Vectorized RP
        # x: (B, L, D) -> (B, L, 1, D) and (B, 1, L, D)
        s_i = x.unsqueeze(2)
        s_j = x.unsqueeze(1)
        distances = torch.abs(s_i - s_j)  # (B, L, L, D)

        # Move D to dim 1 -> (B, D, L, L)
        distances = distances.permute(0, 3, 1, 2)

        if self.rp_threshold == "point":
            # quantile over last two dims
            flat_dist = distances.reshape(B, D, -1)
            threshold = torch.quantile(
                flat_dist, self.rp_percentage / 100.0, dim=2, keepdim=True
            )
            threshold = threshold.unsqueeze(3)  # (B, D, 1, 1)

            # Use Sigmoid for soft thresholding to enable gradient flow
            rp = torch.sigmoid(10.0 * (threshold - distances))
        elif self.rp_threshold == "distance":
            threshold = self.rp_percentage / 100.0
            rp = torch.sigmoid(10.0 * (threshold - distances))
        else:  # 'fan' or Gaussian
            flat_dist = distances.reshape(B, D, -1)
            sigma = torch.std(flat_dist, dim=2, keepdim=True).unsqueeze(3)
            rp = torch.exp(-(distances**2) / (2 * sigma**2 + 1e-8))

        # Now interpolate each channel independently
        rp_resized = F.interpolate(
            rp,  # (B, D, L, L)
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # (B, D, H, W)

        return rp_resized

    # CHANGE：Multivariate Recurrence Plot
    def recurrence_plot_m(self, x):
        B, L, D = x.shape

        # Normalize first to ensure variables contribute equally
        x_norm = self.normalize_per_series(x)

        x_i = x_norm.unsqueeze(2)  # (B, L, 1, D)
        x_j = x_norm.unsqueeze(1)  # (B, 1, L, D)

        # Euclidean distance in phase space (D dimensions)
        distances = torch.norm(x_i - x_j, dim=3)  # (B, L, L)

        # distances is (B, L, L). We want output (B, 1, L, L)
        distances = distances.unsqueeze(1)  # (B, 1, L, L)

        if self.rp_threshold == "point":
            flat_dist = distances.reshape(B, 1, -1)
            threshold = torch.quantile(
                flat_dist, self.rp_percentage / 100.0, dim=2, keepdim=True
            ).unsqueeze(3)
            rp = torch.sigmoid(10.0 * (threshold - distances))
        elif self.rp_threshold == "distance":
            threshold = self.rp_percentage / 100.0
            rp = torch.sigmoid(10.0 * (threshold - distances))
        else:
            flat_dist = distances.reshape(B, 1, -1)
            sigma = torch.std(flat_dist, dim=2, keepdim=True).unsqueeze(3)
            rp = torch.exp(-(distances**2) / (2 * sigma**2 + 1e-8))

        rp = F.interpolate(
            rp,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return rp

    # CHANGE：Recurrence Plot
    def recurrence_plot(self, x):
        if self.compress_vars:
            return self.recurrence_plot_m(x)
        else:
            return self.recurrence_plot_u(x)

    def stft_spectrogram(self, x):
        B, L, D = x.shape

        n_fft = min(self.stft_window_size, L)
        hop_length = max(1, self.stft_hop_length)
        win_length = n_fft
        window = torch.hann_window(win_length, device=x.device)

        # Vectorized STFT
        # Reshape to (B*D, L)
        x_flat = x.permute(0, 2, 1).reshape(B * D, L)

        # Z-score normalization per series
        x_norm = self.standardize_per_series(x_flat)

        stft_result = torch.stft(
            x_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            pad_mode="reflect",
        )
        magnitude = torch.abs(stft_result)  # (B*D, F, T)

        if self.use_log_scale:
            magnitude = torch.log1p(magnitude * 10)

        # Reshape back: (B, D, F, T)
        magnitude = magnitude.reshape(B, D, magnitude.size(1), magnitude.size(2))

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            spectrograms = magnitude.mean(dim=1, keepdim=True)
        else:
            spectrograms = magnitude

        spectrograms = F.interpolate(
            spectrograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize per image (After interpolation)
        spectrograms = self.normalize_per_image(spectrograms)

        spectrograms = torch.flip(spectrograms, [2])

        return spectrograms

    def wavelet_transform(self, x):
        B, L, D = x.shape

        wavelet_type = self.wavelet_type

        scales = torch.logspace(0, np.log10(L / 2), 32, base=10, device=x.device)

        scalograms = torch.zeros(B, D, int(len(scales)), L, device=x.device)

        for b in range(B):
            for d in range(D):
                ts = self.normalize_minmax(x[b, :, d])

                ts_np = ts.cpu().numpy()

                try:
                    coeffs, _ = pywt.cwt(ts_np, scales.cpu().numpy(), wavelet_type)

                    coeff_tensor = torch.tensor(coeffs, device=x.device)

                    if self.use_log_scale:
                        coeff_tensor = torch.log1p(torch.abs(coeff_tensor))
                    else:
                        coeff_tensor = torch.abs(coeff_tensor)

                    scalograms[b, d] = self.normalize_minmax(coeff_tensor)

                except ImportError:
                    print(
                        "PyWavelets library not available, using simplified wavelet transform implementation"
                    )

                    for i, scale in enumerate(scales):
                        scale_val = scale.item()
                        wavelet = torch.zeros(L, device=x.device)

                        t = torch.arange(L, device=x.device)
                        center = L / 2

                        sigma = scale_val
                        wavelet = torch.sin(2 * np.pi * t / scale_val) * torch.exp(
                            -((t - center) ** 2) / (2 * sigma**2)
                        )

                        wavelet = wavelet / torch.norm(wavelet)

                        for j in range(L):
                            indices = torch.arange(L, device=x.device)
                            valid_idx = (indices >= 0) & (indices < L)

                            if j + wavelet.shape[0] <= L:
                                scalograms[b, d, i, j] = torch.sum(
                                    ts[j : j + wavelet.shape[0]] * wavelet[valid_idx]
                                )
                            else:
                                overlap = L - j
                                scalograms[b, d, i, j] = torch.sum(
                                    ts[j:] * wavelet[:overlap]
                                )

                scalogram = scalograms[b, d]

                if scalogram.max() - scalogram.min() < 1e-6:
                    scalogram = scalogram + 0.1 * torch.rand_like(scalogram)

                scalogram = 0.2 + 0.8 * self.normalize_minmax(scalogram)

                scalograms[b, d] = scalogram

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            scalograms = scalograms.mean(dim=1, keepdim=True)

        scalograms = F.interpolate(
            scalograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        scalograms = self.normalize_per_image(scalograms)

        return scalograms

    # CHANGE：GPU优化版本
    def wavelet_transform_gpu(self, x):
        B, L, D = x.shape
        S = 32  # Number of scales (frequencies)

        # 1. Generate Scales (Log-spaced)
        # Scales correspond to 1/frequency. We range from small scale (high freq) to large scale (low freq).
        end_exp = torch.log10(torch.tensor(L / 2.0, device=x.device))
        scales = torch.logspace(0, float(end_exp.item()), S, base=10, device=x.device)

        # 2. Construct Morlet Wavelet Bank in Frequency Domain
        # t: Time points [0, 1, ..., L-1]
        t = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        center = L / 2.0  # Center the wavelet in the time window

        # Morlet Wavelet formula: psi(t) = sin(2*pi*t/s) * exp(-(t-center)^2 / (2*s^2))
        # scales.unsqueeze(1) -> (S, 1) to broadcast over time
        sin_term = torch.sin(2 * torch.pi * (t / scales.unsqueeze(1)))
        gauss = torch.exp(-((t - center) ** 2) / (2 * (scales.unsqueeze(1) ** 2)))
        wavelets = sin_term * gauss  # (S, L)

        # Normalize wavelets to have unit energy
        wavelets = wavelets / (torch.linalg.norm(wavelets, dim=1, keepdim=True) + 1e-8)

        # 3. Frequency Domain Convolution (Theorem: Conv(x, w) <=> IFFT(FFT(x) * FFT(w)))
        # Reshape x to (B*D, L) for batch processing
        x_bd = x.permute(0, 2, 1).reshape(B * D, L)

        # FFT of input signal
        Xf = torch.fft.rfft(x_bd, dim=-1)  # (B*D, L//2 + 1)
        # FFT of wavelet bank
        Wf = torch.fft.rfft(wavelets, dim=-1)  # (S, L//2 + 1)

        # Broadcasting multiply: (B*D, 1, F) * (1, S, F) -> (B*D, S, F)
        Yf = Xf.unsqueeze(1) * Wf.unsqueeze(0)

        # Inverse FFT to get CWT coefficients
        coeff = torch.fft.irfft(Yf, n=L, dim=-1)  # (B*D, S, L)

        # Reshape back to (B, D, S, L)
        coeff = coeff.reshape(B, D, S, L)

        # 4. Post-processing (Magnitude & Log-scale)
        if self.use_log_scale:
            coeff = torch.log1p(torch.abs(coeff))
        else:
            coeff = torch.abs(coeff)

        # 5. Normalization
        # Normalize per image (S, L) dimensions
        # mn = coeff.amin(dim=(-2, -1), keepdim=True)
        # mx = coeff.amax(dim=(-2, -1), keepdim=True)
        # coeff = (coeff - mn) / (mx - mn + 1e-8)

        # Contrast adjustment and clamping
        # coeff = 0.2 + 0.8 * coeff

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            coeff = coeff.mean(dim=1, keepdim=True)

        # 6. Interpolate to target image size
        coeff = F.interpolate(
            coeff,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # coeff = torch.clamp(coeff, 0.05, 1.0)

        # Normalize per image after interpolation
        coeff = self.normalize_per_image(coeff)

        return coeff

    def cwt_spectrogram_real(self, x):
        """
        Continuous Wavelet Transform (CWT) using Real Wavelet (Mexican Hat / Ricker).
        Unlike standard Spectrograms or Wavelet Magnitude, this method preserves
        PHASE information (sign of the signal) by avoiding abs() and mapping
        the real-valued coefficients directly to [0, 1].

        Output mapping:
        - Positive coefficients (Peaks) -> > 0.5 (Brighter)
        - Zero coefficients (Silence)   -> 0.5 (Gray)
        - Negative coefficients (Troughs)-> < 0.5 (Darker)
        """
        B, L, D = x.shape
        S = 32  # Number of scales

        # 1. Generate Scales
        # Ricker wavelet is well defined for scales.
        end_exp = torch.log10(torch.tensor(L / 2.0, device=x.device))
        scales = torch.logspace(0, float(end_exp.item()), S, base=10, device=x.device)

        # 2. Construct Ricker (Mexican Hat) Wavelet in Frequency Domain
        # Ricker time domain: A * (1 - t^2/a^2) * exp(-t^2/2a^2)
        # Ricker freq domain: A * (w^2 * a^2) * exp(-w^2 * a^2 / 2)
        # It's computationally more stable to build in Time Domain and FFT

        t = torch.arange(L, device=x.device).unsqueeze(0) - L / 2.0  # Center at 0
        # t: (1, L), scales: (S) -> (S, L)
        t_scaled = t / scales.unsqueeze(1)

        # Ricker Wavelet Formula
        # psi(t) = (1 - t^2) * exp(-t^2/2)
        ricker = (1 - t_scaled**2) * torch.exp(-0.5 * t_scaled**2)

        # Energy Normalization
        ricker = ricker / (torch.linalg.norm(ricker, dim=1, keepdim=True) + 1e-8)

        # 3. Frequency Domain Convolution
        x_bd = x.permute(0, 2, 1).reshape(B * D, L)

        Xf = torch.fft.rfft(x_bd, dim=-1)  # (B*D, L//2+1)
        Wf = torch.fft.rfft(ricker, dim=-1)  # (S, L//2+1)

        Yf = Xf.unsqueeze(1) * Wf.unsqueeze(0)  # (B*D, S, L//2+1)

        # Inverse FFT -> Real values (Phase preserved)
        coeff = torch.fft.irfft(Yf, n=L, dim=-1)  # (B*D, S, L)
        coeff = coeff.reshape(B, D, S, L)

        # 4. Phase-Preserving Normalization
        # Instead of abs(), we map the signed range to [0, 1]
        # We want 0 to be mapped to 0.5 to preserve "neutrality"

        # Global scaling strategy to preserve relative amplitude across frequencies
        # Find max absolute value per image to determine the range [-max, +max]
        max_val = torch.abs(coeff).amax(dim=(-2, -1), keepdim=True)

        # Map [-max, +max] -> [0, 1]
        # x_norm = (x / max_val + 1) / 2
        # -max -> (-1 + 1)/2 = 0
        # 0    -> (0 + 1)/2 = 0.5
        # +max -> (1 + 1)/2 = 1
        coeff = (coeff / (max_val + 1e-8) + 1) / 2.0

        # 5. Pooling & Interpolate
        if self.compress_vars:
            coeff = coeff.mean(dim=1, keepdim=True)

        coeff = F.interpolate(
            coeff,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Note: We do NOT use normalize_per_image again here,
        # because we intentionally set the center (0) to 0.5.
        # Standard Min-Max would shift the zero-point if the signal is not symmetric.
        # But we clip to ensure safety.
        coeff = torch.clamp(coeff, 0.0, 1.0)

        return coeff

    def mel_filterbank(self, x):
        B, L, D = x.shape

        n_fft = min(self.stft_window_size, L)
        hop_length = max(1, self.stft_hop_length)
        win_length = n_fft
        window = torch.hann_window(win_length, device=x.device)
        n_freq_bins = n_fft // 2 + 1

        # 1. Vectorized STFT
        x_flat = x.permute(0, 2, 1).reshape(B * D, L)

        # Z-score normalization per series (Standard for audio-like processing)
        x_norm = self.standardize_per_series(x_flat)

        stft_result = torch.stft(
            x_norm,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        # Power Spectrogram: |STFT|^2
        power_spec = torch.abs(stft_result) ** 2  # (B*D, n_freq, n_time)
        n_time_bins = power_spec.shape[-1]

        # 2. Create Mel Filter Bank
        sample_rate = 1.0  # Normalized sample rate
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate, device=x.device)

        if self.use_mel:
            mel_min = 2595 * torch.log10(1 + freqs[0] / 700)
            mel_max = 2595 * torch.log10(1 + freqs[-1] / 700)
            mel_points = torch.linspace(
                mel_min, mel_max, self.num_filters + 2, device=x.device
            )
            freq_points = 700 * (10 ** (mel_points / 2595) - 1)
            filter_bank = torch.zeros(self.num_filters, n_freq_bins, device=x.device)
            for i in range(self.num_filters):
                f_left = freq_points[i]
                f_center = freq_points[i + 1]
                f_right = freq_points[i + 2]
                left_mask = (freqs >= f_left) & (freqs <= f_center)
                right_mask = (freqs >= f_center) & (freqs <= f_right)
                left_slope = (freqs[left_mask] - f_left) / (f_center - f_left + 1e-8)
                right_slope = (f_right - freqs[right_mask]) / (
                    f_right - f_center + 1e-8
                )
                filter_bank[i, left_mask] = left_slope
                filter_bank[i, right_mask] = right_slope
        else:
            # Gaussian filter bank
            bandwidth = freqs[-1] / self.num_filters
            filter_bank = torch.zeros(self.num_filters, n_freq_bins, device=x.device)
            for i in range(self.num_filters):
                center = (i + 0.5) * bandwidth
                filter_bank[i] = torch.exp(
                    -0.5 * ((freqs - center) / (bandwidth / 2 + 1e-8)) ** 2
                )

        # 3. Apply Mel Filter Bank
        # filter_bank: (n_mels, n_freq)
        # power_spec: (B*D, n_freq, n_time)
        # result: (B*D, n_mels, n_time)
        mel_spectrograms = torch.matmul(filter_bank, power_spec)

        # 4. Log Scale (dB)
        mel_spectrograms = 10 * torch.log10(mel_spectrograms + 1e-6)

        # 5. Reshape back and Normalize
        mel_spectrograms = mel_spectrograms.reshape(B, D, self.num_filters, n_time_bins)

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            mel_spectrograms = mel_spectrograms.mean(dim=1, keepdim=True)

        mel_spectrograms = F.interpolate(
            mel_spectrograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize per image after interpolation
        mel_spectrograms = self.normalize_per_image(mel_spectrograms)

        return mel_spectrograms

    def s_transform(self, x):
        """
        Stockwell Transform (S-Transform) implementation on GPU.
        S-Transform provides frequency-dependent resolution (multi-resolution),
        similar to Wavelet Transform but directly linked to Fourier spectrum (preserves phase).

        S(tau, f) = IFFT( X(v) * W(v-f) )
        where W(v) is a Gaussian window with width proportional to f.
        """
        B, L, D = x.shape

        # 1. Prepare Data & FFT
        x_flat = x.permute(0, 2, 1).reshape(B * D, L)
        # Standardize to ensure consistent spectral magnitude
        x_norm = self.standardize_per_series(x_flat)

        # Compute Full FFT
        Xf = torch.fft.fft(x_norm, dim=-1)  # (B*D, L)

        # 2. Define Frequencies
        # FFT frequencies indices: [0, 1, ..., L/2, -L/2, ..., -1]
        # We construct a vector 'k' representing the frequency indices
        if L % 2 == 0:
            k = torch.cat(
                [
                    torch.arange(L // 2, device=x.device),
                    torch.arange(-L // 2, 0, device=x.device),
                ]
            ).float()
        else:
            k = torch.cat(
                [
                    torch.arange((L - 1) // 2 + 1, device=x.device),
                    torch.arange(-(L - 1) // 2, 0, device=x.device),
                ]
            ).float()

        # Target frequencies 'f' to analyze
        # We analyze positive frequencies from 1 to L//2
        # (ignoring DC component and negative frequencies for the "image")
        max_f = L // 2
        f = torch.arange(1, max_f + 1, device=x.device).float().unsqueeze(1)  # (F, 1)

        # 3. Construct Gaussian Windows Matrix
        # Window function: G(k, f) = exp( -2 * pi^2 * (k - f)^2 / f^2 )
        # k: (1, L)
        # f: (F, 1)
        # Broadcasting -> (F, L)

        # Note: We need to handle the circular frequency distance for correct windowing?
        # In standard S-transform, it's linear frequency.
        # But since we use DFT, the window should ideally wrap around.
        # However, for meaningful f (>=1) and large L, the Gaussian is narrow enough
        # that wrapping is negligible, except maybe for very highest frequencies.
        # We'll use direct difference.

        exponent = -2 * (torch.pi**2) * (k.unsqueeze(0) - f) ** 2 / (f**2)
        mask = torch.exp(exponent)  # (F, L)

        # 4. Apply Windows and IFFT
        # Xf: (B*D, L) -> (B*D, 1, L)
        # mask: (F, L) -> (1, F, L)
        Y = Xf.unsqueeze(1) * mask.unsqueeze(0)  # (B*D, F, L)

        # IFFT along the last dimension to get time domain complex S-series
        S_complex = torch.fft.ifft(Y, dim=-1)  # (B*D, F, L)

        # 5. Magnitude
        # We use magnitude for visualization
        S_mag = torch.abs(S_complex)

        # 6. Reshape and Post-process
        # Reshape back to (B, D, F, L)
        S_mag = S_mag.reshape(B, D, S_mag.shape[1], L)

        # Compress Variables if needed
        if self.compress_vars:
            S_mag = S_mag.mean(dim=1, keepdim=True)

        # Interpolate to target image size
        # Input (B, D, F, L), Output (B, D, H, W)
        S_img = F.interpolate(
            S_mag,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Flip Y axis (to have low freq at bottom)
        S_img = torch.flip(S_img, [2])

        # Normalize per image
        S_img = self.normalize_per_image(S_img)

        return S_img

    def markov_transition_field(self, x, n_bins=8):
        B, L, D = x.shape

        # 1. Normalization
        x_norm = self.normalize_per_series(x)

        # 2. Downsampling
        downsample_factor = 1
        if L > self.mtf_downsample_threshold and self.use_fast_mode:
            downsample_factor = L // self.mtf_downsample_threshold + 1
            x_norm = x_norm[:, ::downsample_factor, :]
            effective_L = x_norm.size(1)
            print(
                f"Downsampling time series from {L} to {effective_L} for MTF calculation"
            )
        else:
            effective_L = L

        # 3. Binning (Soft)
        bins = torch.linspace(0, 1, n_bins + 1, device=x.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # (n_bins,)

        # x_norm: (B, L_eff, D)
        # bin_centers: (n_bins,)
        # dists: (B, L_eff, D, n_bins)
        dists = torch.abs(x_norm.unsqueeze(-1) - bin_centers)
        soft_digitized = F.softmax(-10.0 * dists, dim=-1)  # (B, L_eff, D, n_bins)

        # 4. Transitions
        # prob_t: (B, L_eff-1, D, n_bins)
        prob_t = soft_digitized[:, :-1, :, :]
        prob_t_plus_1 = soft_digitized[:, 1:, :, :]

        # transitions: (B, D, n_bins, n_bins)
        # Sum over time (dim 1)
        transitions = torch.einsum("btki,btkj->bkij", prob_t, prob_t_plus_1)

        # Normalize transitions
        row_sums = transitions.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1
        transitions = transitions / row_sums

        # 5. MTF Matrix
        # soft_digitized: (B, L_eff, D, n_bins) -> permute to (B, D, L_eff, n_bins)
        Q = soft_digitized.permute(0, 2, 1, 3)

        # P_projected = Q @ transitions
        # (B, D, L_eff, n_bins) @ (B, D, n_bins, n_bins) -> (B, D, L_eff, n_bins)
        P_projected = torch.matmul(Q, transitions)

        # mtf_small = P_projected @ Q.T
        # (B, D, L_eff, n_bins) @ (B, D, n_bins, L_eff) -> (B, D, L_eff, L_eff)
        mtf_small = torch.matmul(P_projected, Q.transpose(-1, -2))

        mtf = mtf_small

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            mtf = mtf.mean(dim=1, keepdim=True)

        mtf = F.interpolate(
            mtf,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return mtf

    # 通过自相关识别周期
    def Autocorrelation_for_Period(self, x):
        B, L, D = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * D, L)
        x_flat = x_flat - x_flat.mean(dim=1, keepdim=True)

        # FFT padding
        n_fft = 2 * 2 ** int(np.ceil(np.log2(L)))
        f = torch.fft.rfft(x_flat, n=n_fft, dim=1)
        acf = torch.fft.irfft(f * torch.conj(f), n=n_fft, dim=1)[:, :L]

        # Average ACF across all samples
        mean_acf = acf.mean(dim=0)  # (L,)

        # Mask lag 0 and very small lags (to avoid trivial peaks)
        mean_acf[0:4] = -float("inf")

        # Find peak
        period = torch.argmax(mean_acf).item()
        return period

    def hilbert_curve_d2xy(self, n, d):
        """
        Convert 1D index d to 2D Hilbert curve coordinates (x, y).
        n: order of the curve (image size will be 2^n x 2^n)
        d: 1D index
        """
        x, y = 0, 0
        s = 1
        while s < (1 << n):
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)

            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x

            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    def get_hilbert_indices(self, side_length, device):
        """
        Generate and cache Hilbert curve indices.
        """
        cache_key = side_length
        if cache_key in self.hilbert_curve_cache:
            return self.hilbert_curve_cache[cache_key].to(device)

        # Calculate order n
        # side_length must be power of 2 for standard Hilbert curve
        # If not, we find the next power of 2 that covers the image,
        # but here we assume user wants the exact side_length.
        # However, Hilbert curve is strictly defined for 2^n.
        # We will use the largest 2^n <= side_length or force side_length to be 2^n.
        # For simplicity and correctness, we assume side_length is a power of 2,
        # or we pad/crop. Let's enforce 2^n internally for the curve generation.

        n = int(np.ceil(np.log2(side_length)))
        # We need to cover the whole image.

        # Actually, let's just generate for the exact target size if it is a power of 2.
        # If not, we might need a more complex space filling curve or just resize later.
        # For this implementation, we generate for 2^n >= side_length, then crop/resize.

        curve_side = 2**n
        num_points = curve_side * curve_side

        indices = torch.zeros((num_points, 2), dtype=torch.long)

        # This loop is slow in Python, but only runs once per size
        # Optimization: Could be pre-computed or JIT compiled if critical
        for d in range(num_points):
            x, y = self.hilbert_curve_d2xy(n, d)
            indices[d, 0] = x
            indices[d, 1] = y

        self.hilbert_curve_cache[cache_key] = indices
        return indices.to(device)

    def hilbert_curve_mapping(self, x):
        B, L, D = x.shape

        # 1. Determine target size based on image_size
        # Ideally image_size should be a power of 2.
        target_side = self.image_size
        n = int(np.ceil(np.log2(target_side)))
        curve_side = 2**n
        target_len = curve_side * curve_side

        # 2. Resize time series to match the curve length
        # x: (B, L, D) -> (B, D, L)
        x_perm = x.permute(0, 2, 1)

        if L != target_len:
            x_resized = F.interpolate(
                x_perm, size=target_len, mode="linear", align_corners=True
            )  # (B, D, target_len)
        else:
            x_resized = x_perm

        # 3. Get mapping indices
        indices = self.get_hilbert_indices(curve_side, x.device)  # (N, 2)

        # 4. Fill the image
        # Create canvas: (B, D, H, W)
        canvas = torch.zeros(B, D, curve_side, curve_side, device=x.device)

        # x_indices and y_indices
        x_idx = indices[:, 0]
        y_idx = indices[:, 1]

        # Scatter values
        # x_resized: (B, D, N)
        # We assign x_resized[..., i] to canvas[..., y_idx[i], x_idx[i]]

        canvas[:, :, y_idx, x_idx] = x_resized

        # 5. Resize to exact image_size if needed (if image_size was not power of 2)
        if curve_side != self.image_size:
            canvas = F.interpolate(
                canvas,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # 6. Normalize per image
        canvas = self.normalize_per_image(canvas)

        # 7. Compress vars if needed
        if self.compress_vars:
            canvas = canvas.mean(dim=1, keepdim=True)

        return canvas

    def plot_mapping(self, x):
        """
        Maps time series to a 2D image where x-axis is time and y-axis is value.
        Uses soft rendering (Gaussian) for differentiability.
        """
        B, L, D = x.shape
        H, W = self.image_size, self.image_size

        # 1. Normalize x to [0, H-1] range
        # Use min-max normalization per series to fit in the image height
        x_norm = self.normalize_per_series(x)  # (B, L, D) in [0, 1]
        x_scaled = x_norm * (H - 1)

        # 2. Resize to W (Time axis)
        # Permute to (B, D, L) for interpolate
        x_perm = x_scaled.permute(0, 2, 1)
        # Linear interpolation to match image width
        x_resized = F.interpolate(
            x_perm, size=W, mode="linear", align_corners=True
        )  # (B, D, W)

        # 3. Create Grid
        # y_grid: (1, 1, H, 1) -> Represents coordinate of each pixel row
        y_grid = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)

        # x_resized: (B, D, 1, W) -> Represents the target coordinate for each column
        y_centers = x_resized.unsqueeze(2)

        # 4. Gaussian Soft Rendering
        # Sigma controls the "thickness".
        sigma = max(0.5, self.plot_line_thickness / 2.0)

        # Expand for broadcasting: (B, D, H, W)
        # Distance squared from the center line
        y_centers_expanded = y_centers.expand(-1, -1, H, -1)

        dist_sq = (y_grid - y_centers_expanded) ** 2

        # Intensity = exp(-dist / 2sigma^2)
        # This creates a soft band around the target value
        intensity = torch.exp(-dist_sq / (2 * sigma**2))

        # Result is already (B, D, H, W)
        return intensity

    def heat_mapping(self, x):
        """
        Maps time series to a 2D heatmap where x-axis is time and y-axis is value.
        The value at time t determines the intensity of the entire column t.
        """
        B, L, D = x.shape
        H, W = self.image_size, self.image_size

        # 1. Normalize x to [0, 1] range for intensity
        x_norm = self.normalize_per_series(x)  # (B, L, D) in [0, 1]

        # 2. Resize to W (Time axis)
        # Permute to (B, D, L) for interpolate
        x_perm = x_norm.permute(0, 2, 1)
        # Linear interpolation to match image width
        x_resized = F.interpolate(
            x_perm, size=W, mode="linear", align_corners=True
        )  # (B, D, W)

        # 3. Broadcast to H (Y axis)
        # Expand to (B, D, H, W)
        image = x_resized.unsqueeze(2).expand(-1, -1, H, -1)

        return image

    def smooth_mapping(self, x):
        """
        Maps time series to a 2D image where x-axis is time and y-axis is smoothing granularity.
        Bottom (y=H-1) is window size 1 (raw). Top (y=0) is max window size.
        """
        B, L, D = x.shape
        H, W = self.image_size, self.image_size

        # 1. Normalize x to [0, 1] range
        x_norm = self.normalize_per_series(x)  # (B, L, D)

        # 2. Resize to W (Time axis)
        x_perm = x_norm.permute(0, 2, 1)  # (B, D, L)
        x_resized = F.interpolate(
            x_perm, size=W, mode="linear", align_corners=True
        )  # (B, D, W)

        # 3. Construct Convolution Kernels for Moving Average
        # Max window size is roughly W/4. Ensure it's odd for symmetric padding.
        # Ensure at least size 3 to have some smoothing effect at the top.
        target_max = max(3, W // 4)
        K_max = target_max if target_max % 2 == 1 else target_max - 1

        # Initialize weights: (H, 1, K_max)
        weights = torch.zeros(H, 1, K_max, device=x.device, dtype=x.dtype)

        # Generate window sizes for each row h
        # h=0 (Top) -> K_max
        # h=H-1 (Bottom) -> 1
        for h in range(H):
            # Linearly interpolate window size
            # h goes 0 -> H-1.
            # We want k to go K_max -> 1.
            # k = 1 + (K_max - 1) * (H - 1 - h) / (H - 1)

            progress = (H - 1 - h) / (H - 1)  # 1.0 at h=0, 0.0 at h=H-1
            k_float = 1 + (K_max - 1) * progress
            k = int(round(k_float))

            # Ensure k is odd
            if k % 2 == 0:
                k += 1
            # Clamp k to be at most K_max
            k = min(k, K_max)

            # Create average kernel
            start = (K_max - k) // 2
            weights[h, 0, start : start + k] = 1.0 / k

        # 4. Apply Convolution
        # If D > 1, we need to repeat weights for grouped convolution
        # Input: (B, D, W)
        # Weights: (D*H, 1, K_max)
        # Groups: D
        if D > 1:
            weights = weights.repeat(D, 1, 1)

        # Padding 'same': pad = K_max // 2
        padding = K_max // 2

        # Output: (B, D*H, W)
        output = F.conv1d(x_resized, weights, padding=padding, groups=D)

        # 5. Reshape to (B, D, H, W)
        output = output.view(B, D, H, W)

        return output

    @torch.no_grad()
    def save_images(self, images, method, batch_idx):
        save_dir = "image_visualization"
        os.makedirs(save_dir, exist_ok=True)

        for i, img_tensor in enumerate(images):
            if img_tensor.shape[0] == 1:
                gray_img = img_tensor[0].cpu().numpy()
                colored_img = cm.viridis(gray_img)
                colored_img = colored_img[:, :, :3]
                colored_img = (colored_img * 255).astype(np.uint8)
                img = Image.fromarray(colored_img)
            elif img_tensor.shape[0] == 3:
                rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
                rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(rgb_img)
            else:
                other_img = img_tensor.mean(dim=0).cpu().numpy()
                other_img = (other_img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(other_img, mode="L")
            img.save(os.path.join(save_dir, f"image_{method}_{batch_idx}_{i}.png"))

    def forward(self, x):
        # x: (B, L, D)
        # return: list of (B, C, H, W) tensors for each method in ts2img_methods

        output_list = []
        for method in ts2img_methods:
            img = self.get_ts2img_tensor(x, method)

            # Post-processing: channel expansion
            if self.three_channel_image:
                if img.shape[1] == 1:
                    img = img.repeat(1, 3, 1, 1)
                elif img.shape[1] != 3:
                    # Simple projection if needed, or just repeat/mean
                    pass

            output_list.append(img)

        return output_list

    def get_ts2img_tensor(self, x, method):
        output = None
        B, L, D = x.shape
        if method == "seg":
            output = self.segmentation(x)
        elif method == "gaf":
            output = self.gramian_angular_field(x)
        elif method == "rp":
            output = self.recurrence_plot(x)
        elif method == "stft":
            output = self.stft_spectrogram(x)
        elif method == "wavelet":
            output = self.wavelet_transform_gpu(x)
        elif method == "cwt":
            output = self.cwt_spectrogram_real(x)
        elif method == "mel":
            output = self.mel_filterbank(x)
        elif method == "mtf":
            output = self.markov_transition_field(x)
        elif method == "st":
            output = self.s_transform(x)
        elif method == "hilbert":
            output = self.hilbert_curve_mapping(x)
        elif method == "plot":
            output = self.plot_mapping(x)
        elif method == "heat":
            output = self.heat_mapping(x)
        elif method == "smooth":
            output = self.smooth_mapping(x)
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'seg', 'gaf', 'rp', 'stft', 'wavelet', 'mel', 'mtf', 'cwt', 'hilbert', 'plot', 'heat', 'smooth'"
            )
        return output

    @staticmethod
    def safe_resize(size, interpolation):
        signature = inspect.signature(Resize)
        params = signature.parameters
        if "antialias" in params:
            return Resize(size, interpolation, antialias=False)
        else:
            return Resize(size, interpolation)
