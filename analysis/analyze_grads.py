import numpy as np
import os
import glob
import re
import pandas as pd

# INTRO：分析 MoE Router 在每个 epoch 中的梯度（每个专家的平均梯度范数）

# 目录路径
diag_dir = "/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_336_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl336_fs1.0_dm128_dp0.1_False_select_best_0/router_diag"

# Expert 名称 (根据之前的分析)
expert_names = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]

# 查找所有 epoch_X_grad.npz 文件
files = glob.glob(os.path.join(diag_dir, "epoch_*_grad.npz"))


# 提取 epoch 编号并排序
def get_epoch(filename):
    match = re.search(r"epoch_(\d+)_grad.npz", filename)
    if match:
        return int(match.group(1))
    return -1


files = sorted(files, key=get_epoch)

results = []

print(f"Found {len(files)} gradient records.")

for f in files:
    epoch = get_epoch(f)
    try:
        data = np.load(f)
        # 优先使用预计算的均值，减少内存消耗
        if "grad_row_norm_mean" in data:
            mean_grads = data["grad_row_norm_mean"]
        elif "grad_row_norm" in data:
            grads = data["grad_row_norm"]
            mean_grads = np.mean(grads, axis=0)  # Shape: (Experts,)
        else:
            print(f"Warning: Gradient data not found in {f}")
            continue

        row = {"Epoch": epoch}
        for i, name in enumerate(expert_names):
            if i < len(mean_grads):
                row[name] = mean_grads[i]
        results.append(row)

    except Exception as e:
        print(f"Error reading {f}: {e}")

# 创建 DataFrame
df = pd.DataFrame(results)
df = df.set_index("Epoch")

# 保存结果到文件
output_dir = "/home/user10/TimeSmart/analysis"
report_path = os.path.join(output_dir, "router_grads_report.txt")
grads_csv_path = os.path.join(output_dir, "router_grads.csv")

# 保存 CSV
df.to_csv(grads_csv_path)
print(f"DataFrame saved to: {grads_csv_path}")

# 打开报告文件并输出
with open(report_path, "w") as f:

    def log(msg=""):
        print(msg)
        f.write(str(msg) + "\n")

    # 输出分析报告
    log("\n" + "=" * 50)
    log("Gradient Norm Analysis (Mean per Epoch)")
    log("=" * 50)
    log(df)

    # 计算每个 Expert 的总平均梯度
    log("\n" + "=" * 50)
    log("Overall Average Gradient per Expert")
    log("=" * 50)
    overall_mean = df.mean()
    log(overall_mean)

    # 检查不平衡性
    log("\n" + "=" * 50)
    log("Imbalance Analysis")
    log("=" * 50)
    max_grad = overall_mean.max()
    min_grad = overall_mean.min()
    ratio = max_grad / (min_grad + 1e-10)

    log(f"Max Gradient: {max_grad:.6f} ({overall_mean.idxmax()})")
    log(f"Min Gradient: {min_grad:.6f} ({overall_mean.idxmin()})")
    log(f"Max/Min Ratio: {ratio:.2f}")

    # 判断是否存在梯度饥饿
    # 阈值可以根据经验设定，例如比率超过 10 或 100，或者绝对值极小
    threshold_ratio = 10.0
    starved_experts = overall_mean[
        overall_mean < max_grad / threshold_ratio
    ].index.tolist()

    if starved_experts:
        log(
            f"\n[WARNING] Potential Gradient Starvation detected for: {starved_experts}"
        )
        log(
            "These experts receive significantly smaller gradients compared to the dominant one."
        )
    else:
        log("\n[INFO] No severe Gradient Starvation detected (Ratio < 10).")
        log("Gradients are relatively balanced across experts.")

    # 检查是否有 Expert 梯度趋近于 0
    dead_experts = overall_mean[overall_mean < 1e-6].index.tolist()
    if dead_experts:
        log(f"\n[CRITICAL] Dead Experts detected (Gradient ~ 0): {dead_experts}")

print(f"\nFull analysis report saved to: {report_path}")

# 保存结果
output_csv = os.path.join(diag_dir, "gradient_analysis.csv")
df.to_csv(output_csv)
print(f"\nAnalysis saved to {output_csv}")
