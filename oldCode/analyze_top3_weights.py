import numpy as np
import pandas as pd
import sys

# File path
file_path = "Result_top3/results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_top3_stack_0/ts2img_weights.npy"
output_excel = "/home/user10/TimeSmart/ts2img_top3_weights_analysis.xlsx"

# Methods list
methods = np.array(["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"])

try:
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    print(f"Data shape: {data.shape}")

    if data.shape[-1] != len(methods):
        print(
            f"Error: Last dimension size ({data.shape[-1]}) does not match number of methods ({len(methods)})."
        )
        sys.exit(1)

    # Get indices of the top 3 weights
    # argsort sorts in ascending order, so we take the last 3 and reverse them
    top3_indices = np.argsort(data, axis=-1)[..., -3:][..., ::-1]

    # Map indices to method names
    # Result shape will be (2785, 7, 3)
    top3_names = methods[top3_indices]

    # Flatten the last two dimensions to create columns like:
    # Var1_Top1, Var1_Top2, Var1_Top3, Var2_Top1, ...
    num_samples = data.shape[0]
    num_vars = data.shape[1]

    # Reshape for DataFrame
    # We want rows to be samples
    # Columns to be Var1_Top1, Var1_Top2, Var1_Top3, Var2_Top1, ...

    column_names = []
    for i in range(num_vars):
        for rank in range(1, 4):
            column_names.append(f"Variable_{i+1}_Top{rank}")

    # Reshape data: (Samples, Vars, 3) -> (Samples, Vars * 3)
    reshaped_data = top3_names.reshape(num_samples, -1)

    # Create DataFrame
    df = pd.DataFrame(reshaped_data, columns=column_names)

    print("DataFrame created.")
    print(df.head())

    # Save to Excel
    print(f"Saving to {output_excel}...")
    df.to_excel(output_excel, index=False)
    print("Done!")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
