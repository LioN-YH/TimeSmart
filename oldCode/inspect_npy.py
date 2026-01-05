import os
import sys
import argparse
import numpy as np
import pandas as pd

DEFAULT_FILE = "Result_top3/results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_top3_stack_0/meta_tensors.npy"


def npy_to_dataframe(arr: np.ndarray) -> pd.DataFrame:
    if arr.ndim == 1:
        df = pd.DataFrame(arr.reshape(1, -1))
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        return df
    if arr.ndim == 2:
        rows = []
        for i in range(arr.shape[0]):
            row = {"sample_idx": i}
            for j in range(arr.shape[1]):
                row[f"f{j}"] = arr[i, j]
            rows.append(row)
        return pd.DataFrame(rows)
    if arr.ndim == 3:
        B, D, F = arr.shape
        rows = []
        for b in range(B):
            for d in range(D):
                row = {"sample_idx": b, "channel_idx": d}
                for f in range(F):
                    row[f"f{f}"] = arr[b, d, f]
                rows.append(row)
        return pd.DataFrame(rows)
    flat = arr.reshape(arr.shape[0], -1)
    df = pd.DataFrame(flat)
    df.columns = [f"f{i}" for i in range(df.shape[1])]
    return df


def save_dataframe(df: pd.DataFrame, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        try:
            df.to_excel(out_path, index=False)
            print(f"Saved Excel to: {out_path}")
        except Exception as e:
            csv_fallback = os.path.splitext(out_path)[0] + ".csv"
            df.to_csv(csv_fallback, index=False)
            print(f"Excel write failed ({e}); saved CSV to: {csv_fallback}")
    else:
        df.to_csv(out_path, index=False)
        print(f"Saved CSV to: {out_path}")


def inspect_npy(file_path: str):
    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        if data.ndim >= 2:
            sums = np.sum(data, axis=-1)
            print("\nSum of last dimension (first 5 samples):")
            print(sums[:5])
            is_sum_one = np.allclose(sums, 1.0, atol=1e-6)
            print(f"\nDo all rows sum to 1 (within tolerance 1e-6)? {is_sum_one}")
            if not is_sum_one:
                print(f"Min sum: {np.nanmin(sums)}")
                print(f"Max sum: {np.nanmax(sums)}")
        print("\nContent:")
        print(data)
    except Exception as e:
        print(f"Error loading .npy file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect or convert .npy to Excel/CSV")
    parser.add_argument("--input", "-i", default=DEFAULT_FILE, help="Path to .npy file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (.xlsx or .csv). Defaults to same folder/name with .xlsx",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Convert .npy to Excel/CSV instead of inspect",
    )
    args = parser.parse_args()

    in_path = os.path.normpath(args.input)
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        sys.exit(1)

    if args.excel:
        data = np.load(in_path)
        df = npy_to_dataframe(data)
        if args.output:
            out_path = os.path.normpath(args.output)
        else:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_dir = os.path.dirname(in_path)
            out_path = os.path.join(out_dir, base + ".xlsx")
        save_dataframe(df, out_path)
    else:
        inspect_npy(in_path)


if __name__ == "__main__":
    main()
