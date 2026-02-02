import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout

# Add project root to path
# Go up two levels: checkGradient -> analysis -> TimeSmart
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the model
from src.TimeApart.model import Model as TimeApartModel


class Config:
    seq_len = 96
    pred_len = 24
    enc_in = 7
    dropout = 0.1
    ts2img_method = "stft"  # Test with STFT first
    image_size = 224

    # Add necessary defaults for VE
    interpolation = "bilinear"
    compress_vars = False
    three_channel_image = False
    periodicity = 24


def check_gradient_flow(model_name, method="stft"):
    print(f"\n{'='*20} Checking Gradient Flow for method: {method} {'='*20}")

    # 1. Initialize Model
    config = Config()
    config.ts2img_method = method

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = TimeApartModel(config).to(device)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 2. Prepare Data
    batch_size = 2
    # Correctly create leaf tensor on device
    x = torch.randn(
        batch_size, config.seq_len, config.enc_in, device=device, requires_grad=True
    )
    target = torch.randn(batch_size, config.pred_len, config.enc_in, device=device)

    # 3. Forward Pass
    print("Running forward pass...")
    try:
        model.train()  # Ensure training mode
        output = model(x)
    except RuntimeError as e:
        print(f"[FATAL ERROR] Forward pass failed: {e}")
        if "numpy" in str(e):
            print(
                "Reason: The transformation likely uses numpy operations on tensors requiring gradients (non-differentiable)."
            )
        return False

    # 4. Backward Pass
    print("Running backward pass...")
    try:
        loss = nn.MSELoss()(output, target)
        loss.backward()
    except RuntimeError as e:
        print(f"[FATAL ERROR] Backward pass failed: {e}")
        return False

    # 5. Check Gradients
    print("\n--- Gradient Statistics ---")

    # Check gradients for key components
    components = {
        "Neck (Conv Layer)": model.neck[0].weight,
        "Adapter (Down Project)": model.adapter.down_project.weight,
        "Head (Output Layer)": model.head[-1].weight,
    }

    if model.revin.affine:
        components["RevIN (Input)"] = model.revin.affine_weight
    else:
        print(
            "[INFO] RevIN affine is False, skipping gradient check for RevIN weights."
        )

    # Robustly find the first trainable parameter in backbone
    backbone_first_layer = None
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            backbone_first_layer = param
            # print(f"Found backbone first layer: {name}")
            break
    components["Backbone (First Layer)"] = backbone_first_layer

    has_grad_issue = False

    for name, param in components.items():
        if param is None:
            print(f"[WARNING] Could not locate parameter for {name}")
            continue

        if param.grad is None:
            print(f"[ERROR] {name}: No gradient found! (grad is None)")
            has_grad_issue = True
        else:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            if grad_mean == 0:
                print(f"[WARNING] {name}: Zero gradient! (grad_mean=0)")
            else:
                print(
                    f"[OK] {name}: Mean Grad = {grad_mean:.6e}, Max Grad = {grad_max:.6e}"
                )

    # Special check: Does gradient flow back to input x?
    if x.grad is not None:
        print(
            f"\n[OK] Gradient flowed back to input X. Mean Grad: {x.grad.abs().mean().item():.6e}"
        )
    else:
        print(
            f"\n[CRITICAL WARNING] Gradient did NOT flow back to input X. The transformation '{method}' is effectively 'freezing' the input preprocessing (RevIN will NOT be updated)."
        )
        has_grad_issue = True

    return not has_grad_issue


if __name__ == "__main__":
    # Test multiple methods
    methods_to_test = ["stft", "gaf", "wavelet", "seg", "rp", "mel", "mtf"]

    for method in methods_to_test:
        try:
            check_gradient_flow(f"TimeApart-{method}", method)
        except Exception as e:
            print(f"Error testing method {method}: {e}")
            import traceback

            traceback.print_exc()
