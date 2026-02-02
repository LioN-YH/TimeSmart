import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
from layers.VE import MT2VEncoder, ts2img_methods

# INTRO：检查ts2img_methods是否可以正常完成梯度传导
class Config:
    def __init__(self):
        self.image_size = 64
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
        self.use_mel = False
        self.num_filters = 32
        self.mtf_downsample_threshold = 256
        self.use_fast_mode = False


def check_gpu_grad():
    print("Checking GPU compatibility and Gradient flow...")

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available, testing on CPU but checking for numpy/grad breaks."
        )
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Testing on device: {device}")

    # Config
    config = Config()
    model = MT2VEncoder(config).to(device)

    # Create dummy input with requires_grad=True
    B, L, D = 2, 200, 1
    x = torch.randn(B, L, D, device=device, requires_grad=True)

    # Test all methods
    for method in ts2img_methods:
        print(f"\nTesting method: {method}")
        try:
            # Forward pass
            output = model.get_ts2img_tensor(x, method)

            # Check if output is on the correct device
            if output.device.type != device.type:
                print(
                    f"FAILED: Output device {output.device} does not match input device {device}"
                )

            # Check gradient flow
            loss = output.sum()
            loss.backward(retain_graph=True)

            if x.grad is None:
                print(f"FAILED: No gradient at input for method {method}")
            else:
                grad_norm = x.grad.norm().item()
                if grad_norm == 0:
                    print(
                        f"WARNING: Gradient is zero for method {method} (might be expected for flat regions, but unusual)"
                    )
                else:
                    print(f"SUCCESS: Gradient flow confirmed (norm: {grad_norm:.4f})")

                # Zero grad for next iteration
                x.grad.zero_()

        except Exception as e:
            print(f"ERROR in method {method}: {e}")
            # print traceback
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    check_gpu_grad()
