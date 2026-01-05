import numpy as np
import pandas as pd
import sys

# File path
file_path = "/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_select_best_0/ts2img_weights/epoch_1.npy"
output_excel = "/home/user10/TimeSmart/ts2img_weights_analysis.xlsx"

# Methods list
methods = ["seg", "gaf", "rp", "stft", "wavelet", "mel", "mtf"]

try:
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    print(f"Data shape: {data.shape}")

    if data.shape[-1] != len(methods):
        print(
            f"Error: Last dimension size ({data.shape[-1]}) does not match number of methods ({len(methods)})."
        )
        sys.exit(1)

    # Find the index of the maximum weight for each element
    # Axis 0: Time steps/Samples (2785)
    # Axis 1: Variables (7)
    # Axis 2: Method Weights (7)
    max_indices = np.argmax(data, axis=-1)

    # Map indices to method names
    # Create an empty array to hold strings
    result_names = np.empty(max_indices.shape, dtype=object)

    for i, method in enumerate(methods):
        result_names[max_indices == i] = method

    # Create DataFrame
    # Columns are variables
    columns = [f"Variable_{i+1}" for i in range(data.shape[1])]
    df = pd.DataFrame(result_names, columns=columns)

    print("DataFrame created.")
    print(df.head())

    # Save to Excel
    print(f"Saving to {output_excel}...")
    df.to_excel(output_excel, index=False)
    print("Done!")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
