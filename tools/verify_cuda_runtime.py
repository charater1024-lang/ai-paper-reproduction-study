"""Verify that the requested PyTorch CUDA wheel can train on the local GPU."""

from __future__ import annotations

import argparse

import torch


def normalized_cuda_version(version: str | None) -> str:
    """Convert a runtime such as ``13.0`` to the wheel-channel suffix ``130``."""

    return (version or "").replace(".", "")


def verify_cuda(expected_flavor: str) -> dict[str, str | float]:
    """Run a finite CUDA forward/backward pass and return device metadata."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected")

    expected_version = expected_flavor.removeprefix("cu")
    actual_version = normalized_cuda_version(torch.version.cuda)
    if not actual_version.startswith(expected_version):
        raise RuntimeError(
            f"expected CUDA {expected_flavor}, got runtime {torch.version.cuda}"
        )

    matrix = torch.randn(512, 512, device="cuda", requires_grad=True)
    loss = (matrix @ matrix).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError("CUDA smoke-test loss is not finite")

    properties = torch.cuda.get_device_properties(0)
    return {
        "PyTorch": torch.__version__,
        "CUDA runtime": torch.version.cuda or "unknown",
        "Device": properties.name,
        "VRAM GiB": round(properties.total_memory / 1024**3, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-flavor",
        required=True,
        choices=("cu126", "cu130", "cu132"),
    )
    args = parser.parse_args()

    for key, value in verify_cuda(args.expected_flavor).items():
        print(f"{key}: {value}")
    print("PASS: CUDA forward/backward")


if __name__ == "__main__":
    main()
