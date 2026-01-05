import argparse
import json
import os
from types import SimpleNamespace
import torch
import time
from data_provider.data_factory import data_provider
from layers.meta_feature_v import batch_extract_meta_features_gpu


def build_args(dset, root_path, seq_len, label_len, pred_len, batch_size, num_workers):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data=dset,
        root_path=root_path,
        data_path=f"{dset}.csv",
        features="M",
        target="OT",
        freq="h",
        embed="timeF",
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        seasonal_patterns=None,
        percent=1.0,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_ratio=0,
        norm_const=0.4,

    )


def compute_stats_for_dataset(dset, args_ns, device):
    _, loader = data_provider(args_ns, flag="train")
    t0 = time.perf_counter()
    feats = []
    bidx = 0
    for batch_x, _, _, _ in loader:
        if device.type == "cuda":
            torch.cuda.synchronize()
        tb = time.perf_counter()
        x = batch_x.float().to(device)
        x, _, _ = normalize_input(x, args_ns)
        f = batch_extract_meta_features_gpu(x, args_ns.seq_len, args_ns.pred_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        te = time.perf_counter()
        print(f"[precompute_meta_features] {dset} batch {bidx}: {(te - tb):.4f}")
        bidx += 1
        feats.append(f.detach().cpu())
    if len(feats) == 0:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[precompute_meta_features] {dset} total: {(t1 - t0):.4f}")
    all_feats = torch.cat(feats, dim=0)
    print(f"all_feats_shape: {all_feats.shape}")
    mean = all_feats.mean(dim=0)
    std = all_feats.std(dim=0, unbiased=False)
    print(f"mean_shape: {mean.shape}, std_shape: {std.shape}")
    return {"mean": mean.numpy().tolist(), "std": std.numpy().tolist()}


def normalize_input(x,args_ns):
        # 计算均值
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        # 计算标准差
        # unbiased=False：使用无偏估计，计算样本方差
        # 1e-5：防止除零错误-接近零的方差会导致标准差接近零，在反向传播时可能引发梯度爆炸
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # 使用配置中的norm_const进一步缩放标准差
        stdev /= args_ns.norm_const
        x = x / stdev
        return x, means, stdev
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--dset", type=str, required=True)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ns = build_args(
        args.dset,
        "./dataset/",
        args.seq_len,
        48,
        args.pred_len,
        32,
        32,
    )
    ns.data = args.data
    stats = compute_stats_for_dataset(args.dset, ns, device)
    if stats is not None:
        result = {"mean": stats["mean"], "std": stats["std"]}
        out_dir = args.output
        if out_dir.endswith(".json"):
            out_dir = os.path.dirname(out_dir) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{args.dset}_{args.seq_len}_{args.pred_len}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
