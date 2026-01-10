import numpy as np
import pandas as pd
import os

# 定义方法列表
ts2img_methods = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]

# npy 文件路径
file_path = '/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_select_best_0/ts2img_weights/epoch_13.npy'

print(f"Loading file: {file_path}")

try:
    # 加载数据
    # 预期形状: [Samples, Variables, Methods]
    data = np.load(file_path)
    print(f"Data shape: {data.shape}")

    # 验证维度
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")
    
    num_samples, num_vars, num_methods = data.shape
    
    # 验证方法数量是否匹配
    if num_methods != len(ts2img_methods):
        print(f"Warning: Data has {num_methods} methods, but method list has {len(ts2img_methods)} names.")
        # 如果不匹配，可能需要处理，这里暂时假设前7个对应
        methods_to_use = ts2img_methods[:num_methods]
    else:
        methods_to_use = ts2img_methods

    # 找到每个样本每个变量选择的方法索引
    # argmax 在最后一个维度 (axis=2)
    selected_indices = np.argmax(data, axis=2) # Shape: [Samples, Variables]
    
    # 构建结果字典
    results = {}
    summary = {}
    
    for i in range(num_vars):
        var_name = f"Variable_{i}"
        
        # 获取该变量的所有样本选择索引
        var_indices = selected_indices[:, i]
        
        # 映射到方法名称
        var_methods = [methods_to_use[idx] for idx in var_indices]
        
        results[var_name] = var_methods
        
        # 统计
        summary[var_name] = pd.Series(var_methods).value_counts()
        
    # 创建 DataFrame
    result_df = pd.DataFrame(results)
    
    # 输出摘要
    print("\n" + "="*50)
    print("Summary of selections per variable (Method Counts):")
    print("="*50)
    for var, counts in summary.items():
        print(f"\n[{var}]:")
        print(counts)
        
    # 保存结果
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, 'ts2img_selection_analysis_npy.xlsx')
    result_df.to_excel(output_path, index_label='Sample_Index')
    
    print("\n" + "="*50)
    print(f"Detailed results saved to:\n{output_path}")
    print("="*50)

except Exception as e:
    print(f"An error occurred: {e}")
