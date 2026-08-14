"""Print and persist the hardware-aware training defaults used by the labs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_engineering_lab.hardware import HardwareProfile, recommend_training_profile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hardware = HardwareProfile.detect()
    training = recommend_training_profile(hardware)
    report = {
        "hardware": asdict(hardware),
        "recommended_training_profile": asdict(training),
    }
    output_path = ROOT / "artifacts" / "hardware_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
