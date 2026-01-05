import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
import torch
from numpy.linalg import LinAlgError


# 提取批次数据的元特征
def batch_extract_meta_features(batch_x, seq_len, pred_len):
    try:
        batch_x = batch_x.cpu().numpy()
    except AttributeError:
        pass

    batch_meta_features = []
    # print("The shape of the data is " + str(batch_x[0].shape))
    # 遍历每个样本调用 extract_meta_feature() 提取其元特征
    for i in range(len(batch_x)):
        _, meta_features = extract_meta_feature(batch_x[i], seq_len, pred_len)
        batch_meta_features.append(meta_features)

    # 转为PyTorch张量
    meta_tensor = torch.tensor(batch_meta_features, dtype=torch.float32)

    return meta_tensor


# 这部分的计算逻辑，其实也和TimeVLM差不多，跨变量取平均
def extract_meta_feature(data, seq_len, pred_len):
    """
    Extracts meta-features from a given time series data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features), time series data

    Returns:
    - features: dict, contains the extracted meta-features
    """
    features = {}

    # basic statistics
    # axis=0 表示沿着时间轴计算（即对每个变量单独计算）
    # .mean() 再在所有变量上取平均 → 得到一个综合性的标量特征
    features["mean"] = np.mean(data, axis=0).mean()
    features["std"] = np.std(data, axis=0).mean()
    features["min"] = np.min(data, axis=0).mean()
    features["max"] = np.max(data, axis=0).mean()
    features["skewness"] = np.nanmean(skew(data, axis=0))
    features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))

    # time series decomposition
    acfs = [acf(data[:, i], nlags=10, fft=True) for i in range(data.shape[1])]
    features["autocorrelation_mean"] = np.nanmean(
        [acf_val[1] for acf_val in acfs]
    )  # first lag

    # CHANGE: adfuller() 无法对常数序列进行检验，当传入的某列时间序列所有值都一样时，将会报错
    # adf_results = [adfuller(data[:, i]) for i in range(data.shape[1])]
    # features["stationarity"] = np.mean([result[1] < 0.05 for result in adf_results])

    adf_pvalues = []
    for i in range(data.shape[1]):
        series = data[:, i]

        # 检查是否为常数序列（允许浮点误差）
        if np.allclose(series, series[0], atol=1e-8):
            # 常数序列：视为完全平稳 → p-value = 0.0（< 0.05，算作平稳）
            pvalue = 0.0
            # print("p-value = 0.0")
        else:
            try:
                _, pvalue, _, _, _, _ = adfuller(series, autolag="AIC")
            except (ValueError, LinAlgError):
                # 极端情况（如全零但有微小扰动仍导致数值问题），也视为平稳
                pvalue = 0.0

        adf_pvalues.append(pvalue)

    # 原逻辑：计算平稳变量的比例
    features["stationarity"] = np.mean([p < 0.05 for p in adf_pvalues])

    # rate_of_change = np.diff(data, axis=0) / data[:-1]
    # Deal with 0 division
    safe_data = np.where(data[:-1] == 0, np.nan, data[:-1])
    rate_of_change = np.diff(data, axis=0) / safe_data
    features["rate_of_change_mean"] = np.nanmean(rate_of_change)
    features["rate_of_change_std"] = np.nanstd(rate_of_change)

    # Landmarker features
    autoreg_coefs, residual_stds = [], []
    for i in range(data.shape[1]):
        model = AutoReg(data[:, i], lags=1).fit()
        autoreg_coefs.append(model.params[1])
        residual_stds.append(np.std(model.resid))
    features["autoreg_coef_mean"] = np.mean(autoreg_coefs)
    features["residual_std_mean"] = np.mean(residual_stds)

    # frequency domain features
    freq_means, freq_peaks, spectral_entropies = [], [], []
    spectral_variations, spectral_skewnesses, spectral_kurtoses = [], [], []

    for i in range(data.shape[1]):
        freqs, psd = periodogram(data[:, i])
        freq_means.append(np.mean(psd))
        freq_peaks.append(freqs[np.argmax(psd)])
        spectral_entropies.append(entropy(psd))
        if i > 0:
            prev_psd = periodogram(data[:, i - 1])[1]
            spectral_variations.append(np.sqrt(np.sum((psd - prev_psd) ** 2)))
        else:
            spectral_variations.append(0)  # 第一个变量无法计算变化
        spectral_skewnesses.append(skew(psd))
        spectral_kurtoses.append(kurtosis(psd))

    features["frequency_mean"] = np.mean(freq_means)
    features["frequency_peak"] = np.mean(freq_peaks)
    features["spectral_entropy"] = np.nanmean(spectral_entropies)
    features["spectral_variation"] = np.nanmean(spectral_variations)
    features["spectral_skewness"] = np.nanmean(spectral_skewnesses)
    features["spectral_kurtosis"] = np.nanmean(spectral_kurtoses)

    cov_matrix = np.cov(data, rowvar=False)
    features["covariance_mean"] = np.mean(cov_matrix)
    features["covariance_max"] = np.max(cov_matrix)
    features["covariance_min"] = np.min(cov_matrix)
    features["covariance_std"] = np.std(cov_matrix)

    # CHANGE：回望窗口&预测长度也算是重要的元特征
    features["seq_len"] = seq_len
    features["pred_len"] = pred_len

    key_order = [
        "mean",
        "std",
        "min",
        "max",
        "skewness",
        "kurtosis",
        "autocorrelation_mean",
        "stationarity",
        "rate_of_change_mean",
        "rate_of_change_std",
        "autoreg_coef_mean",
        "residual_std_mean",
        "frequency_mean",
        "frequency_peak",
        "spectral_entropy",
        "spectral_variation",
        "spectral_skewness",
        "spectral_kurtosis",
        "covariance_mean",
        "covariance_max",
        "covariance_min",
        "covariance_std",
        "seq_len",
        "pred_len",
    ]

    # Extract feature values in the defined order
    feature_values = [features[key] for key in key_order]

    return features, feature_values
