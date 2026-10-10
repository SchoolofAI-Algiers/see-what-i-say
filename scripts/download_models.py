"""
    download models needed to run this offline ( checkpoints ) : whisper + sam2 + grounding dino
    caches them so we don't need to reinstall everytime
    uv run python scripts/download_models.py [--whisper small.en]
"""

from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

# we can change to a bigger model , now we're using tiny
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dino", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--whisper", default="small.en")
    p.add_argument("--skip-whisper", action="store_true")
    a = p.parse_args()

    ckpt = Path("checkpoints/sam2.1_hiera_tiny.pt")
    ckpt.parent.mkdir(exist_ok=True)
    if ckpt.exists():
        print(f"[ok] {ckpt} already present")
    else:
        print(f"[..] downloading {ckpt} ...")
        urllib.request.urlretrieve(SAM2_URL, ckpt)

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"[..] caching {a.dino}")
    AutoProcessor.from_pretrained(a.dino)
    AutoModelForZeroShotObjectDetection.from_pretrained(a.dino)
    if not a.skip_whisper:
        import whisper
        print(f"[..] caching whisper {a.whisper}")
        whisper.load_model(a.whisper)
    print("All models cached.")


if __name__ == "__main__":
    main()
