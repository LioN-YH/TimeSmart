import torch
import torch.nn as nn
import sys
import os
import argparse
import datetime
from collections import defaultdict

# Add project root to path
# Go up two levels: checkGradient -> analysis -> TimeSmart
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Import the model
from src.TimeApart.model import Model as TimeApartModel


class Config:
    def __init__(self, method: str, finetune_vlm: bool):
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.dropout = 0.1
        self.ts2img_method = method
        self.image_size = 64
        self.finetune_vlm = finetune_vlm

        self.interpolation = "bilinear"
        self.compress_vars = False
        self.three_channel_image = False
        self.periodicity = 24


def _tensor_stats(t: torch.Tensor):
    t = t.detach()
    abs_t = t.abs()
    return {
        "mean_abs": abs_t.mean().item(),
        "max_abs": abs_t.max().item(),
        "norm": t.norm().item(),
        "finite": torch.isfinite(t).all().item(),
    }


def _shorten_name(name: str, max_len: int = 88):
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def check_gradient_flow(method="stft", finetune_vlm=True):
    print(
        f"\n{'='*20} Checking Gradient Flow | method={method} | finetune_vlm={finetune_vlm} {'='*20}"
    )

    # 1. Initialize Model
    config = Config(method=method, finetune_vlm=finetune_vlm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = TimeApartModel(config).to(device)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return False

    # 2. Prepare Data
    torch.manual_seed(0)
    batch_size = 1
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
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(
        f"\n--- Gradient Summary ---\ntrainable_params={len(trainable_params)} | loss={loss.item():.6e}"
    )

    has_grad_issue = False
    per_group = defaultdict(
        lambda: {
            "trainable": 0,
            "missing_grad": 0,
            "zero_grad": 0,
            "non_finite": 0,
            "mean_norm_sum": 0.0,
        }
    )
    missing_list = []
    zero_list = []
    non_finite_list = []

    for name, param in trainable_params:
        group = name.split(".", 1)[0]
        per_group[group]["trainable"] += 1

        grad = param.grad
        if grad is None:
            per_group[group]["missing_grad"] += 1
            missing_list.append(name)
            has_grad_issue = True
            continue

        stats = _tensor_stats(grad)
        if not stats["finite"]:
            per_group[group]["non_finite"] += 1
            non_finite_list.append((name, stats))
            has_grad_issue = True

        if stats["norm"] <= 0.0:
            per_group[group]["zero_grad"] += 1
            zero_list.append(name)

        per_group[group]["mean_norm_sum"] += stats["norm"]

    print(
        "\n[Group stats] name | trainable | missing | zero | non_finite | avg_grad_norm"
    )
    for group in sorted(per_group.keys()):
        s = per_group[group]
        denom = max(s["trainable"] - s["missing_grad"], 1)
        avg_norm = s["mean_norm_sum"] / denom
        print(
            f"{group:16s} | {s['trainable']:9d} | {s['missing_grad']:7d} | {s['zero_grad']:4d} | {s['non_finite']:10d} | {avg_norm:.6e}"
        )

    if missing_list:
        print("\n[ERROR] Missing gradients (first 30):")
        for n in missing_list[:30]:
            print(f"  - {_shorten_name(n)}")

    if non_finite_list:
        print("\n[ERROR] Non-finite gradients (first 10):")
        for n, st in non_finite_list[:10]:
            print(
                f"  - {_shorten_name(n)} | mean_abs={st['mean_abs']:.3e} max_abs={st['max_abs']:.3e} norm={st['norm']:.3e}"
            )

    if zero_list:
        print("\n[WARNING] Zero gradients (first 30):")
        for n in zero_list[:30]:
            print(f"  - {_shorten_name(n)}")

    # Special check: Does gradient flow back to input x?
    grad_x = x.grad
    if grad_x is None:
        print(
            "\n[CRITICAL] Input gradient is None: ts2img preprocessing likely breaks autograd."
        )
        has_grad_issue = True
    else:
        finite = torch.isfinite(grad_x).all().item()
        norm = grad_x.norm().item()
        mean_abs = grad_x.abs().mean().item()
        print(
            f"\n[Input grad] finite={finite} | norm={norm:.6e} | mean_abs={mean_abs:.6e}"
        )
        if not finite:
            has_grad_issue = True
        if norm <= 1e-12:
            print("[WARNING] Input grad norm is near zero.")

    return not has_grad_issue


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="analysis/checkGradient/gradient_report.txt",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f = open(out_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _Tee(old_stdout, f)
    sys.stderr = _Tee(old_stderr, f)
    print(f"[Report] {ts}")
    print(f"[Report] out={out_path}")

    # Test multiple methods
    methods_to_test = ["stft", "gaf", "wavelet", "cwt", "seg", "rp", "mel", "mtf", "st"]

    try:
        for method in methods_to_test:
            try:
                ok = check_gradient_flow(method=method, finetune_vlm=True)
                if not ok:
                    check_gradient_flow(method=method, finetune_vlm=False)
            except Exception as e:
                print(f"Error testing method {method}: {e}")
                import traceback

                traceback.print_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        f.close()
