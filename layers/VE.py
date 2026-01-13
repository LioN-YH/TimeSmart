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
# TODO：后续可以优化的方向在于，将融合策略合并至本模块中，可以减少一些冗余计算（例如对于Select-best，实际上只有一种图像化有效）

# ts2img_methods = ["seg", "gaf", "rp", "stft", "wavelet", "mel", "mtf"]

ts2img_methods = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]


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
        x = einops.rearrange(x, "b s d -> b d s")
        pad_left = 0
        if L % self.periodicity != 0:
            pad_left = self.periodicity - L % self.periodicity
        x_pad = F.pad(x, (pad_left, 0), mode="replicate")

        x_2d = einops.rearrange(
            x_pad,
            "b d (p f) -> b d f p",
            p=x_pad.size(-1) // self.periodicity,
            f=self.periodicity,
        )

        x_resize = F.interpolate(
            x_2d,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # x_channels = torch.zeros(B, D, 1, self.image_size, self.image_size, device=x.device)

        # for i in range(D):
        #     channel = x_resize[:, i:i+1]
        #     channel = self.normalize_minmax(channel)
        #     x_channels[:, i] = channel

        # x_combined = torch.mean(x_channels, dim=1)

        # grid_size = self.image_size // 8
        # grid_mask = torch.ones_like(x_combined)
        # grid_mask[:, :, ::grid_size] = 0.95
        # grid_mask[:, :, :, ::grid_size] = 0.95
        # x_combined = x_combined * grid_mask

        # CHANGE：Change the initialization dimensions of x_norm to ensure that the final output tensor has dim() = 4
        x_norm = torch.zeros(B, D, self.image_size, self.image_size, device=x.device)
        for i in range(D):
            channel = x_resize[:, i : i + 1]
            channel = self.normalize_minmax(channel)
            x_norm[:, i] = channel[:, 0]

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            x_combined = torch.mean(x_norm, dim=1, keepdim=True)
        else:
            x_combined = x_norm

        grid_size = self.image_size // 8
        grid_mask = torch.ones_like(x_combined)
        grid_mask[:, :, ::grid_size, :] = 0.95
        grid_mask[:, :, :, ::grid_size] = 0.95
        x_combined = x_combined * grid_mask

        return x_combined

    def gramian_angular_field(self, x):
        B, L, D = x.shape

        x_norm = self.normalize_minmax(x) * 2 - 1
        theta = torch.arccos(x_norm.clamp(-1 + 1e-6, 1 - 1e-6))
        gaf = torch.zeros(B, D, L, L, device=x.device)

        for b in range(B):
            for d in range(D):
                angle_i = theta[b, :, d].unsqueeze(1)
                angle_j = theta[b, :, d].unsqueeze(0)

                if self.gaf_method == "summation":
                    gaf_matrix = torch.cos(angle_i + angle_j)
                else:
                    gaf_matrix = torch.sin(angle_i - angle_j)

                gaf[b, d] = self.normalize_minmax(gaf_matrix)

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            gaf = gaf.mean(dim=1, keepdim=True)

        gaf = F.interpolate(
            gaf,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return gaf

    # CHANGE: Univariate Recurrence Plot
    def recurrence_plot_u(self, x):
        B, L, D = x.shape

        # Pre-allocate output: (B, D, L, L)
        rp = torch.zeros(B, D, L, L, device=x.device)

        for b in range(B):
            for d in range(D):
                series = x[b, :, d]  # (L,)
                # Reshape to (L, 1) for broadcasting
                s_i = series.unsqueeze(1)  # (L, 1)
                s_j = series.unsqueeze(0)  # (1, L)
                distances = torch.abs(s_i - s_j)  # (L, L), 1D distance

                if self.rp_threshold == "point":
                    threshold = torch.quantile(distances, self.rp_percentage / 100.0)
                    binary_matrix = (distances <= threshold).float()
                elif self.rp_threshold == "distance":
                    threshold = self.rp_percentage / 100.0
                    binary_matrix = (distances <= threshold).float()
                else:  # 'fan' or Gaussian
                    sigma = torch.std(distances)
                    binary_matrix = torch.exp(-(distances**2) / (2 * sigma**2))

                rp[b, d] = binary_matrix

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
        rp = torch.zeros(B, 1, L, L, device=x.device)

        for b in range(B):
            x_b = x[b]
            x_i = x_b.unsqueeze(1)
            x_j = x_b.unsqueeze(0)

            distances = torch.norm(x_i - x_j, dim=2)

            if self.rp_threshold == "point":
                threshold = torch.quantile(distances, self.rp_percentage / 100.0)
                binary_matrix = (distances <= threshold).float()
                rp[b, 0] = binary_matrix
            elif self.rp_threshold == "distance":
                threshold = self.rp_percentage / 100.0
                binary_matrix = (distances <= threshold).float()
                rp[b, 0] = binary_matrix
            else:
                sigma = torch.std(distances)
                rp[b, 0] = torch.exp(-(distances**2) / (2 * sigma**2))

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

        # CHANGE: No longer precompute n_time_bins
        # n_freq_bins = n_fft // 2 + 1
        # n_time_bins = max(1, (L - win_length) // hop_length + 1)
        # spectrograms = torch.zeros(B, D, n_freq_bins, n_time_bins, device=x.device)

        spectrogram_list = []
        for b in range(B):
            channel_specs = []
            for d in range(D):
                ts = x[b, :, d]
                ts = (ts - ts.mean()) / (ts.std() + 1e-10)
                stft_result = torch.stft(
                    ts,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    return_complex=True,
                    pad_mode="reflect",  # 自动 padding
                )
                magnitude = torch.abs(stft_result)  # shape: [n_freq, n_time_actual]
                if self.use_log_scale:
                    magnitude = torch.log1p(magnitude * 10)

                # spectrograms[b, d] = self.normalize_minmax(magnitude)
                magnitude = self.normalize_minmax(magnitude)
                channel_specs.append(magnitude)
            # Stack channels: [D, F, T]
            channel_specs = torch.stack(channel_specs, dim=0)
            spectrogram_list.append(channel_specs)

        # Final shape: [B, D, F, T_actual]
        spectrograms = torch.stack(spectrogram_list, dim=0)

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            spectrograms = spectrograms.mean(dim=1, keepdim=True)

        spectrograms = F.interpolate(
            spectrograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

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

        scalograms = torch.clamp(scalograms, 0.05, 1.0)

        return scalograms

    # CHANGE：GPU优化版本
    def wavelet_transform_gpu(self, x):
        B, L, D = x.shape
        S = 32
        end_exp = torch.log10(torch.tensor(L / 2.0, device=x.device))
        scales = torch.logspace(0, float(end_exp.item()), S, base=10, device=x.device)
        t = torch.arange(L, device=x.device).unsqueeze(0)
        center = L / 2.0
        sin_term = torch.sin(2 * torch.pi * (t / scales.unsqueeze(1)))
        gauss = torch.exp(-((t - center) ** 2) / (2 * (scales.unsqueeze(1) ** 2)))
        wavelets = sin_term * gauss
        wavelets = wavelets / (torch.linalg.norm(wavelets, dim=1, keepdim=True) + 1e-8)
        x_bd = x.permute(0, 2, 1).reshape(B * D, L)
        Xf = torch.fft.rfft(x_bd, dim=-1)
        Wf = torch.fft.rfft(wavelets, dim=-1)
        Yf = Xf.unsqueeze(1) * Wf.unsqueeze(0)
        coeff = torch.fft.irfft(Yf, n=L, dim=-1)
        coeff = coeff.reshape(B, D, S, L)
        if self.use_log_scale:
            coeff = torch.log1p(torch.abs(coeff))
        else:
            coeff = torch.abs(coeff)
        mn = coeff.amin(dim=(-2, -1), keepdim=True)
        mx = coeff.amax(dim=(-2, -1), keepdim=True)
        coeff = (coeff - mn) / (mx - mn + 1e-8)
        coeff = 0.2 + 0.8 * coeff
        if self.compress_vars:
            coeff = coeff.mean(dim=1, keepdim=True)
        coeff = F.interpolate(
            coeff,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        coeff = torch.clamp(coeff, 0.05, 1.0)
        return coeff

    def mel_filterbank(self, x):
        B, L, D = x.shape

        n_fft = min(self.stft_window_size, L)
        hop_length = max(1, self.stft_hop_length)
        win_length = n_fft
        window = torch.hann_window(win_length, device=x.device)
        n_freq_bins = n_fft // 2 + 1
        n_time_bins = max(1, (L - win_length) // hop_length + 1)
        spectrograms = torch.zeros(B, D, n_freq_bins, n_time_bins, device=x.device)

        for b in range(B):
            for d in range(D):
                ts = self.normalize_minmax(x[b, :, d])
                stft_result = torch.stft(
                    ts,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    return_complex=True,
                    pad_mode="constant",
                )
                power_spec = torch.abs(stft_result) ** 2
                if power_spec.shape != spectrograms[b, d].shape:
                    power_spec = (
                        F.interpolate(
                            power_spec.unsqueeze(0).unsqueeze(0),
                            size=spectrograms[b, d].shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                spectrograms[b, d] = power_spec
        sample_rate = 1.0
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
                left_slope = (freqs[left_mask] - f_left) / (f_center - f_left)
                right_slope = (f_right - freqs[right_mask]) / (f_right - f_center)
                filter_bank[i, left_mask] = left_slope
                filter_bank[i, right_mask] = right_slope
        else:
            bandwidth = freqs[-1] / self.num_filters
            filter_bank = torch.zeros(self.num_filters, n_freq_bins, device=x.device)
            for i in range(self.num_filters):
                center = (i + 0.5) * bandwidth
                filter_bank[i] = torch.exp(
                    -0.5 * ((freqs - center) / (bandwidth / 2)) ** 2
                )

        mel_spectrograms = torch.zeros(
            B, D, self.num_filters, n_time_bins, device=x.device
        )
        for b in range(B):
            for d in range(D):
                try:
                    mel_spectrograms[b, d] = torch.matmul(
                        filter_bank, spectrograms[b, d]
                    )
                except RuntimeError as e:
                    print(
                        f"Dimension mismatch: {filter_bank.shape}, {spectrograms[b, d].shape}"
                    )
                    if filter_bank.shape[1] != spectrograms[b, d].shape[0]:
                        resized_filter = F.interpolate(
                            filter_bank.unsqueeze(0),
                            size=(spectrograms[b, d].shape[0]),
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0)
                        mel_spectrograms[b, d] = torch.matmul(
                            resized_filter, spectrograms[b, d]
                        )
        mel_spectrograms = 10 * torch.log10(mel_spectrograms + 1e-6)
        mel_spectrograms = self.normalize_minmax(mel_spectrograms)

        # CHANGE：Multivariate Average Pooling
        if self.compress_vars:
            mel_spectrograms = mel_spectrograms.mean(dim=1, keepdim=True)

        mel_spectrograms = F.interpolate(
            mel_spectrograms,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return mel_spectrograms

    def markov_transition_field(self, x, n_bins=8):
        B, L, D = x.shape
        mtf = torch.zeros(B, D, L, L, device=x.device)
        downsample_factor = 1
        if L > self.mtf_downsample_threshold and self.use_fast_mode:
            downsample_factor = L // self.mtf_downsample_threshold + 1
            effective_L = L // downsample_factor
            print(
                f"Downsampling time series from {L} to {effective_L} for MTF calculation"
            )
        else:
            effective_L = L

        for b in range(B):
            for d in range(D):
                ts = self.normalize_minmax(x[b, :, d])

                if downsample_factor > 1:
                    ts = ts[::downsample_factor]
                    curr_L = len(ts)
                else:
                    curr_L = L

                bins = torch.linspace(0, 1, n_bins + 1, device=x.device)
                digitized = torch.zeros(curr_L, dtype=torch.long, device=x.device)

                for i in range(n_bins):
                    if i < n_bins - 1:
                        mask = (ts >= bins[i]) & (ts < bins[i + 1])
                    else:
                        mask = ts >= bins[i]
                    digitized[mask] = i

                transitions = torch.zeros(n_bins, n_bins, device=x.device)

                for i in range(curr_L - 1):
                    transitions[digitized[i], digitized[i + 1]] += 1

                row_sums = transitions.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1
                transitions = transitions / row_sums

                i_indices = digitized.unsqueeze(1).expand(curr_L, curr_L)
                j_indices = digitized.unsqueeze(0).expand(curr_L, curr_L)

                mtf_small = transitions[i_indices, j_indices]

                if downsample_factor > 1:
                    mtf_resized = (
                        F.interpolate(
                            mtf_small.unsqueeze(0).unsqueeze(0),
                            size=(L, L),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    mtf[b, d] = mtf_resized
                else:
                    mtf[b, d] = mtf_small
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

    # 通过FFT识别周期
    def FFT_for_Period(self, x, k=2):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list

        # 周期和对应的频率振幅
        return period, abs(xf).mean(-1)[:, top_list]

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

    # def forward(self, x, method="seg", save_images=False):
    #     B, L, D = x.shape
    #     start_time = time.time()
    #     try:
    #         if method == "seg":
    #             output = self.segmentation(x)
    #         elif method == "gaf":
    #             output = self.gramian_angular_field(x)
    #         elif method == "rp":
    #             output = self.recurrence_plot(x)
    #         elif method == "stft":
    #             output = self.stft_spectrogram(x)
    #         elif method == "wavelet":
    #             output = self.wavelet_transform(x)
    #         elif method == "mel":
    #             output = self.mel_filterbank(x)
    #         elif method == "mtf":
    #             output = self.markov_transition_field(x)
    #         else:
    #             raise ValueError(
    #                 f"Unknown method: {method}. Choose from 'seg', 'gaf', 'rp', 'stft', 'wavelet', 'mel', 'mtf'"
    #             )

    #         if (
    #             output.dim() != 4
    #             or output.size(1) != 1
    #             or output.size(2) != self.image_size
    #             or output.size(3) != self.image_size
    #         ):
    #             if output.dim() == 2:
    #                 output = output.unsqueeze(1).unsqueeze(1)
    #             elif output.dim() == 3:
    #                 if output.size(1) == output.size(2):
    #                     output = output.unsqueeze(1)
    #                 else:
    #                     output = output.unsqueeze(2)
    #             output = F.interpolate(
    #                 output,
    #                 size=(self.image_size, self.image_size),
    #                 mode="bilinear",
    #                 align_corners=False,
    #             )
    #             if output.size(1) != 1:
    #                 output = output.mean(dim=1, keepdim=True)

    #         exec_time = time.time() - start_time
    #         if method not in self.method_times:
    #             self.method_times[method] = []
    #         self.method_times[method].append(exec_time)
    #         output = self.norm(output)
    #         if save_images:
    #             self.save_images(output, method, B)
    #         return output

    #     except Exception as e:
    #         print(f"Error in {method} method: {e}. Falling back to segmentation.")
    #         output = self.segmentation(x)
    #         output = self.norm(output)
    #         return output

    # CHANGE: 重构forward方法
    # !!! 注意，目前这个方法倾向于批次样本的多变量时序数据使用
    def forward(self, x, save_images=False):
        ts2img_tensor_list = []
        for method in ts2img_methods:
            ts2img_tensor = self.get_ts2img_tensor(x, method, save_images)
            # print(f"[DEBUG]{method}-tensor shape: {ts2img_tensor.shape}")
            if self.three_channel_image:
                ts2img_tensor = ts2img_tensor.repeat(1, 3, 1, 1)
            ts2img_tensor_list.append(ts2img_tensor)
        return ts2img_tensor_list

    def get_ts2img_tensor(self, x, method, save_images=False):
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
            output = self.wavelet_transform(x)
        elif method == "mel":
            output = self.mel_filterbank(x)
        elif method == "mtf":
            output = self.markov_transition_field(x)
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'seg', 'gaf', 'rp', 'stft', 'wavelet', 'mel', 'mtf'"
            )
        if save_images:
            self.save_images(output, method, B)
        # compress_vars=True → B,1,H,W
        # compress_vars=False → B,D,H,W
        return output

    # CHANGE: 新增GPU加速的wavelet变换
    def get_ts2img_tensor_opt(self, x, method, save_images=False):
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
        elif method == "mel":
            output = self.mel_filterbank(x)
        elif method == "mtf":
            output = self.markov_transition_field(x)
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'seg', 'gaf', 'rp', 'stft', 'wavelet', 'mel', 'mtf'"
            )
        if save_images:
            self.save_images(output, method, B)
        # compress_vars=True → B,1,H,W
        # compress_vars=False → B,D,H,W
        return output

    @staticmethod
    def safe_resize(size, interpolation):
        signature = inspect.signature(Resize)
        params = signature.parameters
        if "antialias" in params:
            return Resize(size, interpolation, antialias=False)
        else:
            return Resize(size, interpolation)
