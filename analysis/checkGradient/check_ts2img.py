import os
import sys
import traceback

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from layers.VE import MT2VEncoder, ts2img_methods


class Config:
    def __init__(self, compress_vars=False):
        self.image_size = 64
        self.interpolation = "bilinear"
        self.compress_vars = compress_vars
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


def run_method_grad_check(model, method, device, batch_size, seq_len, n_vars):
    x = torch.randn(batch_size, seq_len, n_vars, device=device, requires_grad=True)

    try:
        output = model.get_ts2img_tensor(x, method)

        if output.device.type != device.type:
            return "failed", f"输出设备错误: output={output.device}, input={device}"

        if not output.requires_grad:
            return "failed", "输出不在计算图中(output.requires_grad=False)"

        loss = output.mean()
        grad = torch.autograd.grad(loss, x, allow_unused=True)[0]

        if grad is None:
            return "failed", "输入梯度为 None（梯度断裂）"

        if not torch.isfinite(grad).all():
            return "failed", "输入梯度包含 NaN 或 Inf"

        grad_norm = grad.norm().item()
        if grad_norm <= 1e-12:
            return "warning", f"梯度范数接近 0: {grad_norm:.3e}"

        return (
            "passed",
            f"梯度正常, norm={grad_norm:.3e}, out_shape={tuple(output.shape)}",
        )

    except Exception as exc:
        return "failed", f"异常: {exc}\n{traceback.format_exc()}"


def run_suite(device, compress_vars, n_vars):
    suite_name = f"compress_vars={compress_vars}, D={n_vars}"
    print("\n" + "=" * 80)
    print(f"测试场景: {suite_name}")
    print("=" * 80)

    model = MT2VEncoder(Config(compress_vars=compress_vars)).to(device)
    model.train()

    results = {}
    for method in ts2img_methods:
        status, message = run_method_grad_check(
            model=model,
            method=method,
            device=device,
            batch_size=2,
            seq_len=200,
            n_vars=n_vars,
        )
        results[method] = (status, message)

        prefix = {
            "passed": "SUCCESS",
            "warning": "WARNING",
            "failed": "FAILED",
        }[status]
        print(f"[{prefix}] {method}: {message}")

    return results


def summarize(all_results):
    print("\n" + "#" * 80)
    print("最终汇总")
    print("#" * 80)

    total_passed = 0
    total_warning = 0
    total_failed = 0

    for suite_name, results in all_results.items():
        passed = [m for m, (s, _) in results.items() if s == "passed"]
        warning = [m for m, (s, _) in results.items() if s == "warning"]
        failed = [m for m, (s, _) in results.items() if s == "failed"]

        total_passed += len(passed)
        total_warning += len(warning)
        total_failed += len(failed)

        print(f"\n[{suite_name}]")
        print(f"  passed : {len(passed)} -> {passed}")
        print(f"  warning: {len(warning)} -> {warning}")
        print(f"  failed : {len(failed)} -> {failed}")

    print("\n" + "-" * 80)
    print(
        f"总计 -> passed: {total_passed}, warning: {total_warning}, failed: {total_failed}"
    )
    print("-" * 80)

    return total_failed == 0


def check_ts2img_grad_flow():
    print("Checking ts2img methods gradient flow...")
    print(f"Methods under test ({len(ts2img_methods)}): {ts2img_methods}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, testing on CPU")

    all_results = {}

    # 场景1：不压缩变量，单变量输入
    key1 = "univariate_no_compress"
    all_results[key1] = run_suite(device=device, compress_vars=False, n_vars=1)

    # 场景2：压缩变量，多变量输入
    key2 = "multivariate_compress"
    all_results[key2] = run_suite(device=device, compress_vars=True, n_vars=4)

    ok = summarize(all_results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(check_ts2img_grad_flow())
