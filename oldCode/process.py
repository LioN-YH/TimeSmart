import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
TS2IMG_METHODS = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]
META_COLUMNS = [
    "min_val",
    "max_val",
    "skew_val",
    "kurt_val",
    "acf_lag1",
    "stationary",
    "roc_mean",
    "roc_std",
    "ar_coef",
    "resid_std",
    "freq_peak",
    "spec_entropy",
    "spec_skew",
    "spec_kurt",
    "slope",
]

RESULTS_DIR = "/home/user10/TimeSmart/1228/Result_top1/results"


def process_folder(folder_path):
    print(f"Processing {folder_path}...")

    # 1. Process meta_tensors.npy
    meta_path = os.path.join(folder_path, "meta_tensors.npy")
    if os.path.exists(meta_path):
        try:
            data = np.load(meta_path)
            # data shape: (B, D, 15)
            if data.ndim == 3:
                B, D, F = data.shape
                out_meta = os.path.join(folder_path, "meta_tensors.xlsx")

                with pd.ExcelWriter(out_meta, engine="openpyxl") as writer:
                    for d in range(D):
                        rows = []
                        for b in range(B):
                            rows.append([b] + list(data[b, d]))

                        cols = ["sample"] + META_COLUMNS
                        df_sheet = pd.DataFrame(rows, columns=cols)
                        df_sheet.to_excel(
                            writer, index=False, sheet_name=f"Variable_{d}"
                        )

                print(f"  Saved {out_meta}")
            else:
                print(f"  Skipping {meta_path}: Unexpected shape {data.shape}")
        except Exception as e:
            print(f"  Error processing meta_tensors.npy: {e}")
    else:
        print(f"  {meta_path} not found")

    # 2. Process ts2img_weights.npy
    weights_path = os.path.join(folder_path, "ts2img_weights.npy")
    if os.path.exists(weights_path):
        try:
            data = np.load(weights_path)
            # data shape: (B, D, M)
            if data.ndim == 3:
                B, D, M = data.shape
                valid_M = min(M, len(TS2IMG_METHODS))

                out_w = os.path.join(folder_path, "ts2img_weights.xlsx")
                out_best = os.path.join(folder_path, "ts2img_best_methods.xlsx")

                with pd.ExcelWriter(
                    out_w, engine="openpyxl"
                ) as writer_w, pd.ExcelWriter(
                    out_best, engine="openpyxl"
                ) as writer_best:

                    for d in range(D):
                        rows_w = []
                        rows_best = []

                        for b in range(B):
                            weights = data[b, d]
                            current_weights = weights[:valid_M]

                            # Part A: Weights values
                            rows_w.append([b] + list(current_weights))

                            # Part B: Best method
                            if np.isnan(current_weights).all():
                                best_method = "None"
                            else:
                                best_idx = np.nanargmax(current_weights)
                                best_method = TS2IMG_METHODS[best_idx]

                            rows_best.append([b, best_method])

                        # Save sheet for current variable
                        # Excel-1
                        cols_w = ["sample"] + TS2IMG_METHODS[:valid_M]
                        df_w = pd.DataFrame(rows_w, columns=cols_w)
                        df_w.to_excel(writer_w, index=False, sheet_name=f"Variable_{d}")

                        # Excel-2
                        cols_best = ["sample", "best_method"]
                        df_best = pd.DataFrame(rows_best, columns=cols_best)
                        df_best.to_excel(
                            writer_best, index=False, sheet_name=f"Variable_{d}"
                        )

                print(f"  Saved {out_w}")
                print(f"  Saved {out_best}")

            else:
                print(f"  Skipping {weights_path}: Unexpected shape {data.shape}")

        except Exception as e:
            print(f"  Error processing ts2img_weights.npy: {e}")
    else:
        print(f"  {weights_path} not found")


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return

    subdirs = [
        os.path.join(RESULTS_DIR, d)
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]

    print(f"Found {len(subdirs)} folders to process.")
    for folder in tqdm(subdirs):
        process_folder(folder)


if __name__ == "__main__":
    main()
