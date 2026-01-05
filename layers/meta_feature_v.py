import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.ar_model import AutoReg
from numpy.linalg import LinAlgError
import torch

# INTRO: 针对多变量时间序列通道独立预测的设计，提取每个变量的元特征，同时完成了【GPU版本的计算优化】


# CPU版本
def batch_extract_meta_features(batch_x, seq_len, pred_len):
    """
    Extract meta-features for each variable in each sample.

    Input:
        batch_x: (B, T, N) or (B, N, T) — we assume (B, T, N): B samples, T time steps, N variables
        seq_len: int
        pred_len: int

    Output:
        meta_tensor: (B, N, 20) — PyTorch tensor of meta-features per variable
    """
    if isinstance(batch_x, torch.Tensor):
        batch_x = batch_x.detach().cpu().numpy()  # Ensure on CPU and numpy

    batch_meta_features = []
    for sample in batch_x:
        # Assume sample shape is (T, N)
        meta_per_var = extract_meta_features_per_variable(
            sample, seq_len, pred_len
        )  # (N, 20)
        batch_meta_features.append(meta_per_var)

    meta_tensor = torch.tensor(
        np.stack(batch_meta_features), dtype=torch.float32
    )  # (B, N, 20)
    return meta_tensor


def extract_meta_features_per_variable(data, seq_len, pred_len):
    """
    Extract 20 meta-features for each variable in a multivariate time series.

    Parameters:
        data: np.ndarray of shape (T, N) — T time steps, N variables

    Returns:
        features_per_var: np.ndarray of shape (N, 20)
    """
    T, N = data.shape
    # Pre-allocate output array
    features_per_var = np.full((N, 20), np.nan, dtype=np.float32)

    # Time index for trend slope
    t = np.arange(T, dtype=np.float32)

    for i in range(N):
        x = data[:, i]

        # Basic stats
        mean_val = np.mean(x)
        std_val = np.std(x)
        min_val = np.min(x)
        max_val = np.max(x)
        skew_val = skew(x)
        kurt_val = kurtosis(x)

        # Autocorrelation at lag 1
        try:
            acf_vals = acf(x, nlags=1, fft=True)
            acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0.0
        except Exception:
            acf_lag1 = 0.0

        # Stationarity via ADF test
        if np.allclose(x, x[0], atol=1e-8):
            pvalue = 0.0  # constant → stationary
        else:
            try:
                _, pvalue, _, _, _, _ = adfuller(x, autolag="AIC")
            except (ValueError, LinAlgError, TypeError):
                pvalue = 0.0
        stationary = float(pvalue < 0.05)

        # Rate of change
        safe_x = np.where(x[:-1] == 0, np.nan, x[:-1])
        roc = np.diff(x) / safe_x
        roc_mean = np.nanmean(roc)
        roc_std = np.nanstd(roc)

        # AutoRegressive(1) model
        try:
            ar_model = AutoReg(x, lags=1).fit()
            ar_coef = ar_model.params[1] if len(ar_model.params) > 1 else 0.0
            resid_std = np.std(ar_model.resid)
        except Exception:
            ar_coef = 0.0
            resid_std = 0.0

        # Frequency domain
        freqs, psd = periodogram(x)
        freq_mean = np.mean(psd)
        freq_peak = freqs[np.argmax(psd)] if psd.size > 0 else 0.0
        psd_nonzero = psd + 1e-12  # avoid log(0) in entropy
        spec_entropy = entropy(psd_nonzero)
        spec_skew = skew(psd)
        spec_kurt = kurtosis(psd)

        # Trend slope (linear regression of x ~ t)
        # Solve least squares: x = a * t + b → slope = a
        A = np.vstack([t, np.ones_like(t)]).T
        try:
            slope, _ = np.linalg.lstsq(A, x, rcond=None)[0]
        except np.linalg.LinAlgError:
            slope = 0.0

        # Assemble features (20 in total)
        features = [
            mean_val,
            std_val,
            min_val,
            max_val,
            skew_val,
            kurt_val,
            acf_lag1,
            stationary,
            roc_mean,
            roc_std,
            ar_coef,
            resid_std,
            freq_mean,
            freq_peak,
            spec_entropy,
            spec_skew,
            spec_kurt,
            slope,  # trend_slope
            seq_len,
            pred_len,
        ]

        features_per_var[i, :] = features

    return features_per_var


# --------------------------------------------------------
# CHANGE：GPU 版本
def _skew_kurt(x, dim):
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, unbiased=False, keepdim=True)
    std = torch.sqrt(var + 1e-12)
    z = (x - mean) / (std + 1e-12)
    skew = (z.pow(3)).mean(dim=dim)
    kurt = (z.pow(4)).mean(dim=dim) - 3.0
    return skew, kurt


def _nanstd(x, dim, keepdim=False):
    m = torch.nanmean(x, dim=dim, keepdim=True)
    mask = ~torch.isnan(x)
    zero = torch.zeros_like(x)
    d = torch.where(mask, x - m, zero)
    count = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    v = (d.pow(2).sum(dim=dim, keepdim=True) / count).squeeze(
        dim if not keepdim else -1
    )
    return torch.sqrt(v + 1e-12)


def extract_meta_features_per_variable_gpu(data, seq_len, pred_len):
    T, N = data.shape
    x = data
    min_val = x.min(dim=0).values
    max_val = x.max(dim=0).values
    skew_val, kurt_val = _skew_kurt(x, dim=0)
    x0 = x[:-1, :]
    x1 = x[1:, :]
    x0_mean = x0.mean(dim=0)
    x1_mean = x1.mean(dim=0)
    cov = ((x0 - x0_mean) * (x1 - x1_mean)).mean(dim=0)
    var0 = (x0 - x0_mean).pow(2).mean(dim=0)
    var1 = (x1 - x1_mean).pow(2).mean(dim=0)
    acf_lag1 = cov / torch.sqrt(var0 * var1 + 1e-12)
    diff_std = x1.std(dim=0, unbiased=False) - x0.std(dim=0, unbiased=False)
    stationary = (
        (acf_lag1.abs() < 0.8)
        & ((x1 - x0).std(dim=0, unbiased=False) < x.std(dim=0, unbiased=False))
    ).float()
    safe_denom = torch.where(x0 == 0, torch.full_like(x0, float("nan")), x0)
    roc = (x1 - x0) / safe_denom
    roc_mean = torch.nanmean(roc, dim=0)
    roc_std = _nanstd(roc, dim=0)
    x0_mean = x0.mean(dim=0)
    y_mean = x1.mean(dim=0)
    cov_xy = ((x0 - x0_mean) * (x1 - y_mean)).mean(dim=0)
    var_x = (x0 - x0_mean).pow(2).mean(dim=0)
    ar_coef = cov_xy / (var_x + 1e-12)
    intercept = y_mean - ar_coef * x0_mean
    resid = x1 - (ar_coef.unsqueeze(0) * x0 + intercept.unsqueeze(0))
    resid_std = resid.std(dim=0, unbiased=False)
    Xf = torch.fft.rfft(x, dim=0)
    psd = (Xf.abs() ** 2) / T
    psd_sum = psd.sum(dim=0)
    p = psd / (psd_sum + 1e-12)
    spec_entropy = -(p * torch.log(p + 1e-12)).sum(dim=0)
    freq_bins = torch.fft.rfftfreq(T, d=1.0).to(x.device)
    peak_idx = psd.argmax(dim=0)
    freq_peak = freq_bins.index_select(0, peak_idx)
    spec_skew, spec_kurt = _skew_kurt(psd, dim=0)
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    t_mean = t.mean()
    cov_tx = ((t - t_mean).unsqueeze(1) * (x - x.mean(dim=0))).mean(dim=0)
    var_t = ((t - t_mean) ** 2).mean()
    slope = cov_tx / (var_t + 1e-12)
    features = torch.stack(
        [
            min_val,
            max_val,
            skew_val,
            kurt_val,
            acf_lag1,
            stationary,
            roc_mean,
            roc_std,
            ar_coef,
            resid_std,
            freq_peak,
            spec_entropy,
            spec_skew,
            spec_kurt,
            slope,
        ],
        dim=1,
    )
    # 处理NaN值
    features = torch.nan_to_num(features, nan=0.0)
    return features.to(dtype=torch.float32)


def batch_extract_meta_features_gpu(batch_x, seq_len, pred_len):
    x = batch_x
    if torch.isnan(x).any():
        print("[meta_feature_v] NaN detected in input, cleaning ...")
        x = torch.nan_to_num(x)
    B, T, N = x.shape

    out = []
    for b in range(B):
        feats = extract_meta_features_per_variable_gpu(x[b], seq_len, pred_len)
        out.append(feats)
    return torch.stack(out, dim=0)  # (B,N,15)


# CHANGE: 新增标准化元特征计算版本
def batch_extract_meta_features_gpu_Norm(batch_x, meta_mean, meta_std):
    x = batch_x
    # 输入检查
    if torch.isnan(x).any():
        print("[meta_feature_v] NaN detected in input, cleaning ...")
        x = torch.nan_to_num(x)
    B, T, N = x.shape

    out = []
    for b in range(B):
        feats = extract_meta_features_per_variable_gpu(x[b], None, None)
        out.append(feats)
    meta_features = torch.stack(out, dim=0)  # (B,N,15)

    if meta_mean is not None and meta_std is not None:
        # Avoid division by zero
        safe_std = torch.where(meta_std == 0, torch.ones_like(meta_std), meta_std)
        meta_features = (meta_features - meta_mean) / safe_std
    return meta_features
