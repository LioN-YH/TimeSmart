import os
import sys
import numpy as np

src = r"/home/user10/TimeSmart/Result_top1/train_results/long_term_forecast_clip_ETTh1_512_96_TimeSmart_top3_ETTh1_ETTh1_ftM_ll48_sl512_pl96_fs1.0_dm128_dp0.1_False_select_best_0/ts2img_weights/epoch_1.npy"
path = src.replace("\\", "/")
data = np.load(path)

sys.path.append(os.path.abspath("."))
try:
    from layers.VE import ts2img_methods
except Exception:
    ts2img_methods = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]

out_dir = os.path.dirname(path)
out_xlsx = os.path.join(out_dir, "ts2img_weights.xlsx")
out_csv = os.path.join(out_dir, "ts2img_weights.csv")

try:
    import pandas as pd

    has_pd = True
except Exception:
    has_pd = False

if data.ndim == 3:
    B, D, M = data.shape
    cols = ["sample", "variable"] + ts2img_methods[:M]
    rows = []
    for b in range(B):
        for d in range(D):
            rows.append([b, d] + list(data[b, d, :M]))
    if has_pd:
        df = pd.DataFrame(rows, columns=cols)
    else:
        header = ",".join(cols)
        np.savetxt(out_csv, np.array(rows), delimiter=",", header=header, comments="")
        print(out_csv)
        sys.exit(0)
elif data.ndim == 2:
    B, M = data.shape
    cols = ["sample"] + ts2img_methods[:M]
    rows = []
    for b in range(B):
        rows.append([b] + list(data[b, :M]))
    if has_pd:
        df = pd.DataFrame(rows, columns=cols)
    else:
        header = ",".join(cols)
        np.savetxt(out_csv, np.array(rows), delimiter=",", header=header, comments="")
        print(out_csv)
        sys.exit(0)
else:
    if has_pd:
        flat = data.reshape(data.shape[0], -1)
        df = pd.DataFrame(flat)
    else:
        np.savetxt(out_csv, data.reshape(data.shape[0], -1), delimiter=",")
        print(out_csv)
        sys.exit(0)

if has_pd:
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="weights")
        print(out_xlsx)
    except Exception:
        df.to_csv(out_csv, index=False)
        print(out_csv)
