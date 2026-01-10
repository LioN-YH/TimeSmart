import pandas as pd
import numpy as np
import os

# INTRO：分析ts2imgWeights.xlsx【selected_best】每个样本每个变量选中的方法
# 定义方法列表 (根据 layers/VE.py)
ts2img_methods = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]

# 文件路径
file_path = "/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_select_best_0/excel_records/epoch_13_ts2imgWeights.xlsx"

print(f"Reading file: {file_path}")

try:
    # 读取 Excel
    xl = pd.ExcelFile(file_path)

    # 结果字典
    results = {}
    summary = {}

    print(f"Found sheets: {xl.sheet_names}")

    # 遍历每个 Sheet
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)

        # 确保只处理数值列 (假设前7列是权重)
        # 如果列名是数字 0-6
        if list(df.columns[:7]) == [0, 1, 2, 3, 4, 5, 6]:
            data_df = df[[0, 1, 2, 3, 4, 5, 6]]
        else:
            # 尝试直接使用前7列
            data_df = df.iloc[:, :7]

        # 找到每行最大值的列索引
        # idxmax 返回的是列名，如果是整数 0-6，正好对应索引
        selected_indices = data_df.idxmax(axis=1)

        # 映射到方法名
        # 注意：如果列名不是 0-6 的整数，可能需要调整
        try:
            selected_methods = selected_indices.apply(lambda x: ts2img_methods[int(x)])
        except (ValueError, IndexError) as e:
            print(f"Error mapping indices in sheet {sheet_name}: {e}")
            print(f"Indices found: {selected_indices.unique()}")
            continue

        results[sheet_name] = selected_methods

        # 统计摘要
        summary[sheet_name] = selected_methods.value_counts()

    # 创建一个新的 DataFrame 保存结果
    # 结构：行是样本，列是变量，值是选择的方法
    result_df = pd.DataFrame(results)

    # 输出摘要
    print("\n" + "=" * 50)
    print("Summary of selections per variable (Method Counts):")
    print("=" * 50)
    for var, counts in summary.items():
        print(f"\n[{var}]:")
        print(counts)

    # 保存结果
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, "ts2img_selection_analysis.xlsx")
    result_df.to_excel(output_path, index_label="Sample_Index")
    print("\n" + "=" * 50)
    print(f"Detailed results saved to:\n{output_path}")
    print("=" * 50)

except Exception as e:
    print(f"An error occurred: {e}")
