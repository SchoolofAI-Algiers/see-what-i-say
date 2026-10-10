#!/usr/bin/env python3
"""Check the selected STARK VISION install profile without loading models."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def check_import(module_name: str, distribution_name: str | None = None) -> bool:
    try:
        importlib.import_module(module_name)
        version = importlib.metadata.version(distribution_name or module_name)
    except Exception as exc:  # Report native-library load errors as setup failures too.
        print(f"FAIL {module_name}: {exc}")
        return False
    print(f"OK   {module_name} {version}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also check model, audio, CUDA, and checkpoint requirements",
    )
    args = parser.parse_args()

    failures = 0
    for module_name, distribution in (("numpy", "numpy"), ("yaml", "PyYAML")):
        failures += not check_import(module_name, distribution)

    config_path = ROOT / "configs" / "model_paths.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL model config {config_path}: {exc}")
        return 1

    if args.full:
        full_imports = (
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("transformers", "transformers"),
            ("sam2", "SAM-2"),
            ("whisper", "openai-whisper"),
            ("scipy", "scipy"),
            ("sounddevice", "sounddevice"),
        )
        for module_name, distribution in full_imports:
            failures += not check_import(module_name, distribution)

        if shutil.which("ffmpeg"):
            print("OK   ffmpeg executable found")
        else:
            print("FAIL ffmpeg executable not found (install the OS package)")
            failures += 1

        try:
            import torch

            if torch.cuda.is_available():
                print(f"OK   CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                print("FAIL CUDA is unavailable to PyTorch; check the driver and CUDA wheel")
                failures += 1
        except Exception as exc:
            print(f"FAIL CUDA check: {exc}")
            failures += 1

        checkpoint_value = config.get("sam2", {}).get("checkpoint")
        if not checkpoint_value:
            print("FAIL configs/model_paths.yaml has no sam2.checkpoint value")
            failures += 1
        else:
            checkpoint_path = Path(checkpoint_value)
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint_path
            if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
                print(f"OK   SAM 2 checkpoint found: {checkpoint_path}")
            else:
                print(f"FAIL SAM 2 checkpoint missing or empty: {checkpoint_path}")
                print("     Run: bash scripts/download_models.sh")
                failures += 1

        sam2_config = config.get("sam2", {}).get("model_config")
        if sam2_config:
            try:
                sam2_module = importlib.import_module("sam2")
                package_dir = Path(next(iter(sam2_module.__path__)))
                config_parts = Path(sam2_config).parts
                if config_parts and config_parts[0] == "configs":
                    config_parts = config_parts[1:]
                resolved_config = package_dir / "configs" / Path(*config_parts)
                if resolved_config.is_file():
                    print(f"OK   SAM 2 config found: {resolved_config}")
                else:
                    print(f"FAIL SAM 2 config not found in installed package: {resolved_config}")
                    failures += 1
            except Exception as exc:
                print(f"FAIL resolving SAM 2 config {sam2_config}: {exc}")
                failures += 1
        else:
            print("FAIL configs/model_paths.yaml has no sam2.model_config value")
            failures += 1

    if failures:
        print(f"\nSetup check failed ({failures} issue(s)).")
        return 1

    profile = "full model" if args.full else "CPU/mock"
    print(f"\nSetup check passed for the {profile} profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
