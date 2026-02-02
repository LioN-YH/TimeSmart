import pandas as pd
import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.cm as cm

# Add current directory to path so we can import layers
sys.path.append(os.getcwd())
from layers.VE import MT2VEncoder, ts2img_methods


class Config:
    def __init__(self):
        self.image_size = 224
        self.interpolation = "bilinear"
        self.compress_vars = False
        self.three_channel_image = False
        self.periodicity = 24
        self.gaf_method = "summation"
        self.rp_threshold = "point"
        self.rp_percentage = 10
        self.stft_window_size = 128
        self.stft_hop_length = 32
        self.use_log_scale = True
        self.wavelet_type = "morl"
        self.use_mel = True
        self.num_filters = 32
        self.mtf_downsample_threshold = 256
        self.use_fast_mode = False


def save_single_image(img_tensor, save_path):
    # img_tensor shape: (C, H, W)
    if img_tensor.shape[0] == 1:
        gray_img = img_tensor[0].cpu().numpy()
        colored_img = cm.viridis(gray_img)
        colored_img = colored_img[:, :, :3]
        colored_img = (colored_img * 255).astype(np.uint8)
        img = Image.fromarray(colored_img)
    elif img_tensor.shape[0] == 3:
        rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_img)
    else:
        # Fallback for other channel counts
        other_img = img_tensor.mean(dim=0).cpu().numpy()
        other_img = (other_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(other_img, mode="L")
    img.save(save_path)


def main():
    # Load data
    csv_path = "dataset/OT_trend.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Drop date if exists
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Config and Model
    config = Config()
    model = MT2VEncoder(config)

    # Prepare output directory
    output_dir = "variable_images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating images using methods: {ts2img_methods}")

    # Process each variable
    for col in df.columns:
        print(f"Processing variable: {col}...")
        # Get series
        series = df[col].values

        # Prepare tensor: (B, L, D) -> (1, L, 1)
        # We treat the single series as batch size 1, 1 feature.
        x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

        # Get images
        with torch.no_grad():
            # model(x) calls forward, which returns list of tensors corresponding to ts2img_methods
            img_tensors = model(x)

        # Save images
        for method, img_tensor in zip(ts2img_methods, img_tensors):
            # img_tensor shape from forward is (B, C, H, W) or (B, D, H, W) depending on logic
            # forward: if three_channel_image: repeat(1,3,1,1) -> (B, 3, H, W)
            # if not:
            #   seg -> (B, D, H, W) or (B, 1, H, W) if compress_vars
            #   We have B=1, D=1. compress_vars=False.
            #   So shape should be (1, 1, H, W).

            # Check shape
            if img_tensor.ndim == 4:
                # Take the first item in batch
                single_img = img_tensor[0]  # (C, H, W) or (D, H, W) -> (1, H, W)
            else:
                print(f"Unexpected shape for {method}: {img_tensor.shape}")
                continue

            save_path = os.path.join(output_dir, f"{col}_{method}.png")
            save_single_image(single_img, save_path)

    print(f"Done. Images saved in directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
