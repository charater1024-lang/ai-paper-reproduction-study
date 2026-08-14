"""Exit successfully only for the repository's supported Python runtime."""

from __future__ import annotations

import struct
import sys

SUPPORTED_VERSION = (3, 12)


def is_supported_runtime(
    version: tuple[int, int],
    pointer_bits: int,
) -> bool:
    """Return whether a Python runtime satisfies the Windows install contract."""

    return version == SUPPORTED_VERSION and pointer_bits == 64


def main() -> None:
    version = sys.version_info[:2]
    pointer_bits = struct.calcsize("P") * 8
    if is_supported_runtime(version, pointer_bits):
        return

    print(
        "This project requires 64-bit Python 3.12; "
        f"detected Python {version[0]}.{version[1]} ({pointer_bits}-bit).",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
