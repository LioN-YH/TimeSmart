import numpy as np
import os
import glob
import re
import pandas as pd

# INTRO：分析 MoE Router 在每个 epoch 中的行为（熵、Top-1 平均概率、专家选择直方图）
# 目录路径
diag_dir = "/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_192_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl192_fs1.0_dm128_dp0.1_False_select_best_0/router_diag"

# Expert 名称
expert_names = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]

# 查找所有 epoch_X.npz 文件 (排除 _grad.npz)
files = glob.glob(os.path.join(diag_dir, "epoch_*.npz"))
# 过滤掉 _grad.npz
files = [f for f in files if "_grad.npz" not in f]


# 提取 epoch 编号并排序
def get_epoch(filename):
    match = re.search(r"epoch_(\d+).npz", filename)
    if match:
        return int(match.group(1))
    return -1


files = sorted(files, key=get_epoch)

results_metrics = []  # 存储每 epoch 的全局指标（熵、Top-1 概率）
results_hist = []  # 存储每 epoch 各专家被选中的频率直方图

print(f"Found {len(files)} behavior records.")

for f in files:
    epoch = get_epoch(f)
    try:
        data = np.load(f)

        # 1. 基础指标
        entropy = float(data["entropy_mean"]) if "entropy_mean" in data else np.nan
        top1 = float(data["top1_mean"]) if "top1_mean" in data else np.nan

        row_metrics = {"Epoch": epoch, "Entropy": entropy, "Top1_Prob": top1}
        results_metrics.append(row_metrics)

        # 2. 选择直方图 (每个 Expert 的被选频率)
        if "method_hist" in data:
            hist = data["method_hist"]
            row_hist = {"Epoch": epoch}
            for i, name in enumerate(expert_names):
                if i < len(hist):
                    row_hist[name] = hist[i]
            results_hist.append(row_hist)

    except Exception as e:
        print(f"Error reading {f}: {e}")

# 创建 DataFrame
df_metrics = pd.DataFrame(results_metrics).set_index("Epoch")
df_hist = pd.DataFrame(results_hist).set_index("Epoch")

# 保存结果到文件
output_dir = "/home/user10/TimeSmart/analysis"
report_path = os.path.join(output_dir, "router_behavior_report.txt")
metrics_csv_path = os.path.join(output_dir, "router_metrics.csv")
hist_csv_path = os.path.join(output_dir, "router_hist.csv")

# 保存 CSV
df_metrics.to_csv(metrics_csv_path)
df_hist.to_csv(hist_csv_path)
print(f"DataFrames saved to:\n- {metrics_csv_path}\n- {hist_csv_path}")

# 打开报告文件并输出
with open(report_path, "w") as f:

    def log(msg=""):
        print(msg)
        f.write(str(msg) + "\n")

    # 输出分析报告：熵与置信度
    log("\n" + "=" * 50)
    log("Router Uncertainty Analysis (Entropy & Top-1 Prob)")
    log("=" * 50)
    log(df_metrics)

    # 输出分析报告：专家选择分布
    log("\n" + "=" * 50)
    log("Expert Selection Frequency (Method Histogram)")
    log("=" * 50)
    log(df_hist)

    # 自动诊断
    log("\n" + "=" * 50)
    log("Automated Behavior Diagnosis")
    log("=" * 50)

    # 1. 检查是否坍塌 (Collapse)
    # 计算最后一个 Epoch 的分布
    if not df_hist.empty:
        last_epoch = df_hist.index.max()
        last_dist = df_hist.loc[last_epoch]

        log(f"Latest Epoch ({last_epoch}) Distribution:")
        log(last_dist)

        max_freq = last_dist.max()
        dominant_expert = last_dist.idxmax()

        if max_freq > 0.9:
            log(
                f"\n[CRITICAL] Mode Collapse detected! '{dominant_expert}' is selected {max_freq:.2%} of the time."
            )
        elif max_freq > 0.6:
            log(
                f"\n[WARNING] High Imbalance detected. '{dominant_expert}' dominates with {max_freq:.2%}."
            )
        else:
            log(
                f"\n[INFO] Selection is relatively distributed. Max usage: {max_freq:.2%} ({dominant_expert})."
            )

    # 2. 检查熵的变化趋势
    if len(df_metrics) >= 2:
        first_entropy = df_metrics.iloc[0]["Entropy"]
        last_entropy = df_metrics.iloc[-1]["Entropy"]
        log(f"\nEntropy Change: {first_entropy:.4f} -> {last_entropy:.4f}")

        if last_entropy < 0.1:
            log(
                "[INFO] Router has converged to a very deterministic policy (Low Entropy)."
            )
        elif last_entropy > 1.5:  # log(7) ≈ 1.94
            log(
                "[WARNING] Router is still very uncertain (High Entropy). May need more training or temperature adjustment."
            )

print(f"\nFull analysis report saved to: {report_path}")
