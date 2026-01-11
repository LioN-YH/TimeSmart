import os
import pandas as pd
import glob
import re

# 配置路径
source_dir = "/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_192_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl192_fs1.0_dm128_dp0.1_False_select_best_0/excel_records"
output_dir = "/home/user10/TimeSmart/analysis"
output_file = os.path.join(output_dir, "top1_selection_stats.csv")

# 定义方法名称映射 (根据之前的上下文确认的顺序)
method_names = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]


def get_epoch(filename):
    match = re.search(r"epoch_(\d+)_ts2imgWeights.xlsx", filename)
    if match:
        return int(match.group(1))
    return -1


def main():
    print(f"Scanning directory: {source_dir}")
    files = glob.glob(os.path.join(source_dir, "epoch_*_ts2imgWeights.xlsx"))
    files = sorted(files, key=get_epoch)

    if not files:
        print("No files found!")
        return

    all_stats = []

    for file_path in files:
        epoch = get_epoch(os.path.basename(file_path))
        print(f"Processing Epoch {epoch}...")

        try:
            # 读取 Excel 文件，sheet_name=None 读取所有 sheet
            xls = pd.read_excel(file_path, sheet_name=None)

            for sheet_name, df in xls.items():
                # sheet_name 格式通常为 "Variable_0", "Variable_1", etc.
                if not sheet_name.startswith("Variable_"):
                    continue

                var_idx = sheet_name.split("_")[1]

                # 假设列名是 0, 1, 2... 或者默认索引
                # 每一行是一个样本，找出最大值所在的列索引
                # idxmax(axis=1) 返回每行最大值的列名
                top1_series = df.idxmax(axis=1)

                # 统计频率
                counts = top1_series.value_counts().sort_index()

                # 将统计结果转换为字典
                for method_idx, count in counts.items():
                    # 确保索引在有效范围内
                    idx = int(method_idx)
                    if 0 <= idx < len(method_names):
                        method_name = method_names[idx]
                        all_stats.append(
                            {
                                "Epoch": epoch,
                                "Variable": f"Variable_{var_idx}",
                                "Method": method_name,
                                "Count": count,
                            }
                        )
                    else:
                        print(f"Warning: Unknown method index {idx} in {sheet_name}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 创建 DataFrame 并保存
    result_df = pd.DataFrame(all_stats)

    # 为了更清晰，可以透视表格
    # 行：Epoch, Variable
    # 列：Method
    # 值：Count
    pivot_df = result_df.pivot_table(
        index=["Epoch", "Variable"], columns="Method", values="Count", fill_value=0
    )

    # 重新排序列以匹配 method_names 的顺序
    existing_columns = [col for col in method_names if col in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]

    # 计算所有变量的汇总统计 (按 Epoch 求和)
    total_summary_df = pivot_df.groupby("Epoch").sum()

    print("\nAnalysis Result (Top 5 rows of Pivot):")
    print(pivot_df.head())

    print("\nTotal Summary (Top 5 rows):")
    print(total_summary_df.head())

    # 修改为保存 Excel，每个 Variable 一个 Sheet
    output_excel = os.path.join(output_dir, "top1_selection_stats.xlsx")

    print(f"\nSaving results to {output_excel}...")

    # 重置索引以便于处理
    flat_df = pivot_df.reset_index()

    # 获取所有唯一的变量名并排序
    variables = sorted(flat_df["Variable"].unique(), key=lambda x: int(x.split("_")[1]))

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        # 1. 首先写入总汇总 Sheet
        total_summary_df.to_excel(writer, sheet_name="Total_Summary")
        print(f"  - Wrote sheet: Total_Summary")

        # 2. 然后写入每个变量的 Sheet
        for var_name in variables:
            # 筛选当前变量的数据
            var_df = flat_df[flat_df["Variable"] == var_name].copy()

            # 移除 Variable 列（因为已经在 Sheet 名里了）
            var_df = var_df.drop(columns=["Variable"])

            # 设置 Epoch 为索引
            var_df = var_df.set_index("Epoch")

            # 写入 Sheet
            var_df.to_excel(writer, sheet_name=var_name)
            print(f"  - Wrote sheet: {var_name}")

    print(f"\nSuccessfully saved analysis result to: {output_excel}")


if __name__ == "__main__":
    main()
