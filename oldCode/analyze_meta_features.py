import pandas as pd
import os

FILE_PATH = "/home/user10/TimeSmart/Result_top3/results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_top3_stack_0/meta_tensors.xlsx"

def analyze_meta_features(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Reading file: {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return

    # Filter columns f0 to f19
    target_cols = [f"f{i}" for i in range(20)]
    
    # Check if columns exist
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        target_cols = [c for c in target_cols if c in df.columns]

    if not target_cols:
        print("No target columns (f0-f19) found.")
        return

    # Calculate statistics
    stats = df[target_cols].agg(['max', 'min', 'mean'])
    
    # Transpose for better readability: rows are features, cols are stats
    stats_T = stats.transpose()
    
    print("\n--- Meta Features Statistics (f0-f19) ---")
    print(stats_T)
    
    # Save to Excel
    output_xlsx = "meta_feature_stats.xlsx"
    stats_T.to_excel(output_xlsx)
    print(f"\nDetailed statistics saved to {output_xlsx}")

if __name__ == "__main__":
    analyze_meta_features(FILE_PATH)
