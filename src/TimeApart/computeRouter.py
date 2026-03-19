
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

sys.path.append("/mnt/data")
from norm import Normalize


def build_minimal_args():
    return SimpleNamespace(augmentation_ratio=0)


class Dataset_ETT_hour_Fallback(Dataset):
    """
    Fallback that mirrors Dataset_ETT_hour logic when direct import fails.
    """
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ("M", "MS"):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features={self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def get_dataset_class():
    try:
        from data_provider.data_loader import Dataset_ETT_hour
        return Dataset_ETT_hour, "imported_from_data_loader"
    except Exception as e:
        return Dataset_ETT_hour_Fallback, f"fallback_exact_logic:{type(e).__name__}:{e}"


def compute_periodic_score(x, periodicity=24, eps=1e-6):
    """
    Same periodic score as the original model:
    normalized lag-correlation averaged over variables.
    x: (B, L, D)
    """
    B, L, D = x.shape
    lag = min(periodicity, max(1, L // 2))
    x1 = x[:, lag:, :]
    x2 = x[:, :-lag, :]

    x1 = x1 - x1.mean(dim=1, keepdim=True)
    x2 = x2 - x2.mean(dim=1, keepdim=True)

    numerator = (x1 * x2).sum(dim=1)
    denominator = torch.sqrt(
        (x1.pow(2).sum(dim=1) + eps) * (x2.pow(2).sum(dim=1) + eps)
    )
    corr = numerator / (denominator + eps)
    return corr.abs().mean(dim=-1)


def _moving_average_trend(x, ma_window):
    """
    x: (B, L, D)
    return trend: (B, L, D)
    """
    xt = x.permute(0, 2, 1)  # (B, D, L)
    pad = ma_window // 2
    xt_pad = F.pad(xt, (pad, pad), mode="replicate")
    trend = F.avg_pool1d(xt_pad, kernel_size=ma_window, stride=1)
    return trend.permute(0, 2, 1)


def compute_trend_score(
    x,
    eps=1e-6,
    periodicity=24,
    trend_ma_window=None,
    trend_low_freq_weight=0.50,
    trend_direction_weight=0.30,
    trend_linearity_weight=0.20,
    return_components=False,
):
    """
    A broader trend score designed to reward:
    1) low-frequency dominance
    2) consistent direction after smoothing
    3) linearity of the smoothed trend

    This is intentionally more trend-friendly than the original
    pure linear-correlation score.
    """
    B, L, D = x.shape

    if trend_ma_window is None or trend_ma_window <= 0:
        ma_window = min(max(9, periodicity * 2 + 1), L if L % 2 == 1 else L - 1)
    else:
        ma_window = min(trend_ma_window, L if L % 2 == 1 else L - 1)

    if ma_window < 3:
        ma_window = 3
    if ma_window % 2 == 0:
        ma_window += 1
    if ma_window > L:
        ma_window = L if L % 2 == 1 else max(3, L - 1)

    trend = _moving_average_trend(x, ma_window=ma_window)

    x_centered = x - x.mean(dim=1, keepdim=True)
    trend_centered = trend - trend.mean(dim=1, keepdim=True)

    total_var = x_centered.pow(2).mean(dim=1)      # (B, D)
    trend_var = trend_centered.pow(2).mean(dim=1)  # (B, D)
    low_freq_ratio = (trend_var / (total_var + eps)).clamp(0.0, 1.0)

    dtrend = trend[:, 1:, :] - trend[:, :-1, :]
    direction_consistency = (
        dtrend.mean(dim=1).abs() / (dtrend.abs().mean(dim=1) + eps)
    ).clamp(0.0, 1.0)

    t = torch.linspace(-1.0, 1.0, steps=L, device=x.device, dtype=x.dtype).view(1, L, 1)
    tc = t - t.mean(dim=1, keepdim=True)
    numerator = (trend_centered * tc).sum(dim=1)
    denominator = torch.sqrt(
        (trend_centered.pow(2).sum(dim=1) + eps) * (tc.pow(2).sum(dim=1) + eps)
    )
    linearity = (numerator / (denominator + eps)).abs().clamp(0.0, 1.0)

    weight_sum = trend_low_freq_weight + trend_direction_weight + trend_linearity_weight
    if weight_sum <= 0:
        raise ValueError("Trend score weights must sum to a positive value.")

    trend_score_per_var = (
        trend_low_freq_weight * low_freq_ratio +
        trend_direction_weight * direction_consistency +
        trend_linearity_weight * linearity
    ) / weight_sum

    trend_score = trend_score_per_var.mean(dim=-1)

    if return_components:
        return trend_score, {
            "ma_window": ma_window,
            "low_freq_ratio": low_freq_ratio.mean(dim=-1),
            "direction_consistency": direction_consistency.mean(dim=-1),
            "linearity": linearity.mean(dim=-1),
        }
    return trend_score


def summarize_array(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def infer_num_features(csv_path, features, target):
    df = pd.read_csv(csv_path, nrows=5)
    if features in ("M", "MS"):
        return df.shape[1] - 1
    if features == "S":
        return 1
    raise ValueError(features)


def run_split(args, split, dataset_cls):
    ds = dataset_cls(
        args=build_minimal_args(),
        root_path=str(Path(args.csv_path).parent),
        data_path=Path(args.csv_path).name,
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=0,
        freq="h",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    revin = Normalize(
        num_features=args.normalize_num_features or infer_num_features(args.csv_path, args.features, args.target),
        eps=args.normalize_eps,
        affine=args.normalize_affine,
        subtract_last=args.normalize_subtract_last,
        non_norm=args.normalize_non_norm,
    )
    revin.eval()

    periodic_scores = []
    trend_scores = []
    seg_weights = []
    smooth_weights = []
    prefer_seg = []
    prefer_smooth = []
    low_freq_ratio_vals = []
    direction_consistency_vals = []
    linearity_vals = []

    with torch.no_grad():
        for batch in loader:
            seq_x = batch[0].float()
            x = revin(seq_x, "norm")

            periodic_score = compute_periodic_score(
                x, periodicity=args.periodicity, eps=args.router_eps
            )
            trend_score, trend_components = compute_trend_score(
                x,
                eps=args.router_eps,
                periodicity=args.periodicity,
                trend_ma_window=args.trend_ma_window,
                trend_low_freq_weight=args.trend_low_freq_weight,
                trend_direction_weight=args.trend_direction_weight,
                trend_linearity_weight=args.trend_linearity_weight,
                return_components=True,
            )

            score_gap = periodic_score - trend_score - args.router_bias
            w_seg = torch.sigmoid(args.router_temperature * score_gap)
            w_smooth = 1.0 - w_seg

            # Use actual routing weights to define preference
            pref_seg = (w_seg >= 0.5).float()
            pref_smooth = 1.0 - pref_seg

            periodic_scores.append(periodic_score.cpu().numpy())
            trend_scores.append(trend_score.cpu().numpy())
            seg_weights.append(w_seg.cpu().numpy())
            smooth_weights.append(w_smooth.cpu().numpy())
            prefer_seg.append(pref_seg.cpu().numpy())
            prefer_smooth.append(pref_smooth.cpu().numpy())

            low_freq_ratio_vals.append(trend_components["low_freq_ratio"].cpu().numpy())
            direction_consistency_vals.append(trend_components["direction_consistency"].cpu().numpy())
            linearity_vals.append(trend_components["linearity"].cpu().numpy())

    periodic_scores = np.concatenate(periodic_scores) if periodic_scores else np.array([])
    trend_scores = np.concatenate(trend_scores) if trend_scores else np.array([])
    seg_weights = np.concatenate(seg_weights) if seg_weights else np.array([])
    smooth_weights = np.concatenate(smooth_weights) if smooth_weights else np.array([])
    prefer_seg = np.concatenate(prefer_seg) if prefer_seg else np.array([])
    prefer_smooth = np.concatenate(prefer_smooth) if prefer_smooth else np.array([])
    low_freq_ratio_vals = np.concatenate(low_freq_ratio_vals) if low_freq_ratio_vals else np.array([])
    direction_consistency_vals = np.concatenate(direction_consistency_vals) if direction_consistency_vals else np.array([])
    linearity_vals = np.concatenate(linearity_vals) if linearity_vals else np.array([])

    score_gap = periodic_scores - trend_scores - args.router_bias

    return {
        "num_windows": int(len(ds)),
        "periodic_score": summarize_array(periodic_scores),
        "trend_score": summarize_array(trend_scores),
        "score_gap_after_bias": summarize_array(score_gap),
        "seg_weight": summarize_array(seg_weights),
        "smooth_weight": summarize_array(smooth_weights),
        "seg_prefer_ratio": float(prefer_seg.mean()) if prefer_seg.size else None,
        "smooth_prefer_ratio": float(prefer_smooth.mean()) if prefer_smooth.size else None,
        "trend_components": {
            "low_freq_ratio": summarize_array(low_freq_ratio_vals),
            "direction_consistency": summarize_array(direction_consistency_vals),
            "linearity": summarize_array(linearity_vals),
        },
        "diagnostics": {
            "periodic_minus_trend_mean": float((periodic_scores - trend_scores).mean()) if periodic_scores.size else None,
            "periodic_gt_trend_ratio": float((periodic_scores >= trend_scores).mean()) if periodic_scores.size else None,
            "strong_seg_ratio_wseg_gt_0_8": float((seg_weights > 0.8).mean()) if seg_weights.size else None,
            "strong_smooth_ratio_wsmooth_gt_0_8": float((smooth_weights > 0.8).mean()) if smooth_weights.size else None,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--features", type=str, default="M", choices=["M", "MS", "S"])
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--periodicity", type=int, default=24)
    p.add_argument("--router_temperature", type=float, default=5.0)
    p.add_argument("--router_bias", type=float, default=0.0)
    p.add_argument("--router_eps", type=float, default=1e-6)

    p.add_argument("--trend_ma_window", type=int, default=0)
    p.add_argument("--trend_low_freq_weight", type=float, default=0.50)
    p.add_argument("--trend_direction_weight", type=float, default=0.30)
    p.add_argument("--trend_linearity_weight", type=float, default=0.20)

    p.add_argument("--normalize_num_features", type=int, default=None)
    p.add_argument("--normalize_eps", type=float, default=1e-5)
    p.add_argument("--normalize_affine", action="store_true")
    p.add_argument("--normalize_subtract_last", action="store_true")
    p.add_argument("--normalize_non_norm", action="store_true")

    p.add_argument("--output_json", type=str, required=True)
    args = p.parse_args()

    dataset_cls, loader_mode = get_dataset_class()

    results = {
        "dataset_name": args.dataset_name,
        "csv_path": args.csv_path,
        "loader_mode": loader_mode,
        "config": vars(args),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        results["splits"][split] = run_split(args, split, dataset_cls)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "dataset_name": args.dataset_name,
        "loader_mode": loader_mode,
        "output_json": str(output_path),
        "train_windows": results["splits"]["train"]["num_windows"],
        "val_windows": results["splits"]["val"]["num_windows"],
        "test_windows": results["splits"]["test"]["num_windows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
