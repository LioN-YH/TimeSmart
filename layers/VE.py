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
        # Increased default thickness to ensure visibility after super-sampling
        self.plot_line_thickness = getattr(config, "plot_line_thickness", 10)

        # add
        self.method_times = {}

        # ===== runtime caches for differentiable ts2img =====
        # These caches only store constants derived from shape / device / dtype / config.
        # They do NOT depend on input x, so they do not affect gradient flow to x.
        self._mel_filter_cache = {}
        self._s_transform_cache = {}
        self._wavelet_cache = {}
        self._cwt_cache = {}
        self._smooth_kernel_cache = {}
        self._plot_grid_cache = {}

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

    def _cache_key(self, *items):
        return tuple(items)

    def _get_plot_y_grid(self, H, device, dtype):
        key = self._cache_key("plot_y_grid", H, str(device), str(dtype))
        if key not in self._plot_grid_cache:
            self._plot_grid_cache[key] = torch.arange(
                H, device=device, dtype=dtype
            ).view(1, 1, H, 1)
        return self._plot_grid_cache[key]

    def _get_smooth_weights(self, H, W, D, device, dtype):
        """Build and cache grouped 1D moving-average kernels for smooth_mapping."""
        target_max = max(3, W // 4)
        K_max = target_max if target_max % 2 == 1 else target_max - 1

        key = self._cache_key("smooth", H, W, D, str(device), str(dtype))
        if key in self._smooth_kernel_cache:
            return self._smooth_kernel_cache[key], K_max

        h_idx = torch.arange(H, device=device, dtype=torch.float32)
        if H == 1:
            progress = torch.ones_like(h_idx)
        else:
            progress = (H - 1 - h_idx) / (H - 1)

        k_float = 1 + (K_max - 1) * progress
        k = torch.round(k_float).to(torch.long)
        k = k + (k % 2 == 0).to(torch.long)
        k = torch.clamp(k, max=K_max)

        pos = torch.arange(K_max, device=device).view(1, K_max)
        start = ((K_max - k) // 2).view(H, 1)
        end = (start.squeeze(1) + k).view(H, 1)

        mask = ((pos >= start) & (pos < end)).to(dtype)
        weights = mask / k.to(dtype).view(H, 1)
        weights = weights.view(H, 1, K_max)

        if D > 1:
            weights = weights.repeat(D, 1, 1)

        self._smooth_kernel_cache[key] = weights
        return weights, K_max

    def _get_mel_filter_bank(self, n_fft, device, dtype):
        key = self._cache_key(
            "mel",
            n_fft,
            self.num_filters,
            bool(self.use_mel),
            str(device),
            str(dtype),
        )
        if key in self._mel_filter_cache:
            return self._mel_filter_cache[key]

        n_freq_bins = n_fft // 2 + 1
        sample_rate = 1.0
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate, device=device).to(dtype)

        if self.use_mel:
            mel_min = 2595 * torch.log10(1 + freqs[0] / 700)
            mel_max = 2595 * torch.log10(1 + freqs[-1] / 700)
            mel_points = torch.linspace(
                mel_min, mel_max, self.num_filters + 2, device=device, dtype=dtype
            )
            freq_points = 700 * (10 ** (mel_points / 2595) - 1)

            f_left = freq_points[:-2].view(-1, 1)
            f_center = freq_points[1:-1].view(-1, 1)
            f_right = freq_points[2:].view(-1, 1)
            freqs_row = freqs.view(1, -1)

            left = (freqs_row - f_left) / (f_center - f_left + 1e-8)
            right = (f_right - freqs_row) / (f_right - f_center + 1e-8)

            left = torch.clamp(left, min=0.0)
            right = torch.clamp(right, min=0.0)
            filter_bank = torch.minimum(left, right)
        else:
            bandwidth = freqs[-1] / self.num_filters
            centers = (
                (torch.arange(self.num_filters, device=device, dtype=dtype) + 0.5)
                * bandwidth
            ).view(-1, 1)
            filter_bank = torch.exp(
                -0.5 * ((freqs.view(1, -1) - centers) / (bandwidth / 2 + 1e-8)) ** 2
            )

        self._mel_filter_cache[key] = filter_bank
        return filter_bank

    def _get_s_transform_mask(self, L, device, dtype):
        key = self._cache_key("st_mask", L, str(device), str(dtype))
        if key in self._s_transform_cache:
            return self._s_transform_cache[key]

        if L % 2 == 0:
            k = torch.cat(
                [
                    torch.arange(L // 2, device=device),
                    torch.arange(-L // 2, 0, device=device),
                ]
            ).to(dtype)
        else:
            k = torch.cat(
                [
                    torch.arange((L - 1) // 2 + 1, device=device),
                    torch.arange(-(L - 1) // 2, 0, device=device),
                ]
            ).to(dtype)

        max_f = L // 2
        f = torch.arange(1, max_f + 1, device=device, dtype=dtype).unsqueeze(1)
        exponent = -2 * (torch.pi**2) * (k.unsqueeze(0) - f) ** 2 / (f**2)
        mask = torch.exp(exponent)

        self._s_transform_cache[key] = mask
        return mask

    def _get_wavelet_bank(self, L, S, device, dtype):
        key = self._cache_key(
            "morlet_bank", L, S, bool(self.use_log_scale), str(device), str(dtype)
        )
        if key in self._wavelet_cache:
            return self._wavelet_cache[key]

        end_exp = torch.log10(torch.tensor(L / 2.0, device=device, dtype=dtype))
        scales = torch.logspace(
            0, float(end_exp.item()), S, base=10, device=device, dtype=dtype
        )

        t = torch.arange(L, device=device, dtype=dtype).unsqueeze(0)
        center = torch.tensor(L / 2.0, device=device, dtype=dtype)
        sin_term = torch.sin(2 * torch.pi * (t / scales.unsqueeze(1)))
        gauss = torch.exp(-((t - center) ** 2) / (2 * (scales.unsqueeze(1) ** 2)))
        wavelets = sin_term * gauss
        wavelets = wavelets / (torch.linalg.norm(wavelets, dim=1, keepdim=True) + 1e-8)

        Wf = torch.fft.rfft(wavelets, dim=-1)
        self._wavelet_cache[key] = Wf
        return Wf

    def _get_cwt_bank(self, L, S, device, dtype):
        key = self._cache_key("ricker_bank", L, S, str(device), str(dtype))
        if key in self._cwt_cache:
            return self._cwt_cache[key]

        end_exp = torch.log10(torch.tensor(L / 2.0, device=device, dtype=dtype))
        scales = torch.logspace(
            0, float(end_exp.item()), S, base=10, device=device, dtype=dtype
        )

        t = torch.arange(L, device=device, dtype=dtype).unsqueeze(0) - (L / 2.0)
        t_scaled = t / scales.unsqueeze(1)
        ricker = (1 - t_scaled**2) * torch.exp(-0.5 * t_scaled**2)
        ricker = ricker / (torch.linalg.norm(ricker, dim=1, keepdim=True) + 1e-8)

        Wf = torch.fft.rfft(ricker, dim=-1)
        self._cwt_cache[key] = Wf
        return Wf

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
            n = flat_dist.size(-1)
            max_samples = 1000000
            if n > max_samples:
                idx = torch.randint(0, n, (max_samples,), device=flat_dist.device)
                sample = flat_dist.index_select(2, idx)
                threshold = torch.quantile(
                    sample, self.rp_percentage / 100.0, dim=2, keepdim=True
                )
            else:
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
            n = flat_dist.size(-1)
            max_samples = 1000000
            if n > max_samples:
                idx = torch.randint(0, n, (max_samples,), device=flat_dist.device)
                sample = flat_dist.index_select(2, idx)
                threshold = torch.quantile(
                    sample, self.rp_percentage / 100.0, dim=2, keepdim=True
                ).unsqueeze(3)
            else:
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
        S = 32

        x_bd = x.permute(0, 2, 1).reshape(B * D, L)
        Xf = torch.fft.rfft(x_bd, dim=-1)
        Wf = self._get_wavelet_bank(L, S, x.device, x.dtype)

        Yf = Xf.unsqueeze(1) * Wf.unsqueeze(0)
        coeff = torch.fft.irfft(Yf, n=L, dim=-1)
        coeff = coeff.reshape(B, D, S, L)

        if self.use_log_scale:
            coeff = torch.log1p(torch.abs(coeff))
        else:
            coeff = torch.abs(coeff)

        if self.compress_vars:
            coeff = coeff.mean(dim=1, keepdim=True)

        coeff = F.interpolate(
            coeff,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        coeff = self.normalize_per_image(coeff)

        return coeff

    def cwt_spectrogram_real(self, x):
        """
        Continuous Wavelet Transform (CWT) using Real Wavelet (Mexican Hat / Ricker).
        Output mapping preserves coefficient sign by mapping 0 to 0.5.
        """
        B, L, D = x.shape
        S = 32

        x_bd = x.permute(0, 2, 1).reshape(B * D, L)
        Xf = torch.fft.rfft(x_bd, dim=-1)
        Wf = self._get_cwt_bank(L, S, x.device, x.dtype)

        Yf = Xf.unsqueeze(1) * Wf.unsqueeze(0)
        coeff = torch.fft.irfft(Yf, n=L, dim=-1)
        coeff = coeff.reshape(B, D, S, L)

        max_val = torch.abs(coeff).amax(dim=(-2, -1), keepdim=True)
        coeff = (coeff / (max_val + 1e-8) + 1) / 2.0

        if self.compress_vars:
            coeff = coeff.mean(dim=1, keepdim=True)

        coeff = F.interpolate(
            coeff,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        coeff = torch.clamp(coeff, 0.0, 1.0)

        return coeff

    def mel_filterbank(self, x):
        B, L, D = x.shape

        n_fft = min(self.stft_window_size, L)
        hop_length = max(1, self.stft_hop_length)
        win_length = n_fft
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

        x_flat = x.permute(0, 2, 1).reshape(B * D, L)
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
        power_spec = torch.abs(stft_result) ** 2
        n_time_bins = power_spec.shape[-1]

        filter_bank = self._get_mel_filter_bank(
            n_fft=n_fft, device=x.device, dtype=power_spec.dtype
        )

        mel_spectrograms = torch.matmul(filter_bank, power_spec)
        mel_spectrograms = 10 * torch.log10(mel_spectrograms + 1e-6)
        mel_spectrograms = mel_spectrograms.reshape(B, D, self.num_filters, n_time_bins)

        if self.compress_vars:
            mel_spectrograms = mel_spectrograms.mean(dim=1, keepdim=True)

        mel_spectrograms = F.interpolate(
            mel_spectrograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mel_spectrograms = self.normalize_per_image(mel_spectrograms)

        return mel_spectrograms

    def s_transform(self, x):
        """
        Cached-mask S-transform.
        Fully differentiable w.r.t. x.
        """
        B, L, D = x.shape

        x_flat = x.permute(0, 2, 1).reshape(B * D, L)
        x_norm = self.standardize_per_series(x_flat)

        Xf = torch.fft.fft(x_norm, dim=-1)
        mask = self._get_s_transform_mask(L, x.device, Xf.real.dtype)

        Y = Xf.unsqueeze(1) * mask.unsqueeze(0)
        S_complex = torch.fft.ifft(Y, dim=-1)
        S_mag = torch.abs(S_complex)

        S_mag = S_mag.reshape(B, D, S_mag.shape[1], L)

        if self.compress_vars:
            S_mag = S_mag.mean(dim=1, keepdim=True)

        S_img = F.interpolate(
            S_mag,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        S_img = torch.flip(S_img, [2])
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
        Differentiable soft line rendering with configurable supersampling.
        """
        B, L, D = x.shape
        H, W = self.image_size, self.image_size

        scale_factor = 2 if H >= 224 else 4
        W_scaled = W * scale_factor

        x_norm = self.normalize_per_series(x)
        x_scaled = x_norm * (H - 1)

        x_resized = F.interpolate(
            x_scaled.permute(0, 2, 1),
            size=W_scaled,
            mode="linear",
            align_corners=True,
        )

        y_grid = self._get_plot_y_grid(H, x.device, x.dtype)
        y_centers = x_resized.unsqueeze(2)

        sigma = max(0.5, self.plot_line_thickness / 2.0)
        dist_sq = (y_grid - y_centers) ** 2
        intensity = torch.exp(-dist_sq / (2 * sigma**2))

        if scale_factor > 1:
            intensity = F.avg_pool2d(
                intensity,
                kernel_size=(1, scale_factor),
                stride=(1, scale_factor),
            )

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
        Fully differentiable w.r.t. x.
        """
        B, L, D = x.shape
        H, W = self.image_size, self.image_size

        x_norm = self.normalize_per_series(x)
        x_resized = F.interpolate(
            x_norm.permute(0, 2, 1), size=W, mode="linear", align_corners=True
        )

        weights, K_max = self._get_smooth_weights(
            H=H, W=W, D=D, device=x.device, dtype=x.dtype
        )
        padding = K_max // 2

        output = F.conv1d(x_resized, weights, padding=padding, groups=D)
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
