"""Environment sanity check for the DG-DQA project.

Verifies the cu128 PyTorch wheel can actually launch a CUDA kernel on the
Blackwell sm_120 GPU (RTX 5060 Ti). The classic failure mode on a mismatched
build is a runtime "no kernel image is available for execution on the device"
error, which we catch and report explicitly. Exits non-zero on any failure.
"""

import sys


def main() -> int:
    """Print torch/CUDA info and run a small CUDA matmul. Return exit code."""
    try:
        import torch
    except ImportError as exc:  # torch must already be installed (cu128 wheel)
        print(f"[FAIL] could not import torch: {exc}")
        return 1

    print(f"torch version            : {torch.__version__}")
    print(f"torch.version.cuda        : {torch.version.cuda}")

    cuda_ok = torch.cuda.is_available()
    print(f"cuda.is_available()       : {cuda_ok}")
    if not cuda_ok:
        print("[FAIL] CUDA is not available — cannot run on GPU.")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"get_device_capability(0)  : {cap}   (expected (12, 0) for sm_120)")
    print(f"GPU name                  : {name}")

    if cap != (12, 0):
        # Not fatal in itself, but warn loudly — the project targets sm_120.
        print(f"[WARN] device capability is {cap}, expected (12, 0).")

    # Small CUDA matmul to confirm a kernel actually launches on this device.
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"CUDA matmul result        : shape={tuple(c.shape)}, "
              f"mean={c.mean().item():.6f}  [OK]")
    except RuntimeError as exc:
        msg = str(exc)
        print(f"[FAIL] CUDA matmul raised RuntimeError: {msg}")
        if "no kernel image" in msg.lower():
            print("       -> This is the sm_120 build-mismatch error. The "
                  "installed torch lacks a kernel for this GPU.")
            print("       -> Do NOT reinstall torch blindly; the cu128 wheel "
                  "is the intended build. Re-check the environment.")
        return 1

    print("\n[PASS] environment check succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
