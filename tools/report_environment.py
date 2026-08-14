"""Print dependency versions and the PyTorch compute device for Windows setup."""

from __future__ import annotations

import argparse
import platform

import torch


def torch_summary() -> dict[str, str | bool]:
    """Collect the small runtime summary shown by the Windows launchers."""

    cuda_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "PyTorch": torch.__version__,
        "CUDA available": cuda_available,
        "Compute device": device,
    }


def dependency_summary() -> dict[str, str | bool]:
    """Collect versions of the packages required by the learning environment."""

    import jupyterlab
    import langchain_core
    import numpy
    import pandas
    import sklearn

    return {
        "Python": platform.python_version(),
        "JupyterLab": jupyterlab.__version__,
        "NumPy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "LangChain Core": langchain_core.__version__,
        **torch_summary(),
    }


def print_summary(summary: dict[str, str | bool]) -> None:
    """Print one stable key/value line per runtime property."""

    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-only",
        action="store_true",
        help="report only PyTorch and the selected compute device",
    )
    parser.add_argument(
        "--require-cuda-build",
        action="store_true",
        help="exit with status 1 unless PyTorch was compiled with a CUDA runtime",
    )
    args = parser.parse_args()
    if args.require_cuda_build and torch.version.cuda is None:
        raise SystemExit(1)
    summary = torch_summary() if args.torch_only else dependency_summary()
    print_summary(summary)


if __name__ == "__main__":
    main()
