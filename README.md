# STARK VISION

**Tell it what you see.**

Real-time speech-guided referring video segmentation for a local webcam demo.

STARK VISION is designed around two cooperating loops: speech is parsed and grounded on a suitable camera frame to initialize a target, then SAM 2 tracks that target while effects and the HUD consume its mask. Grounding is for target acquisition; it is not run on every video frame.

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) and the committed `uv.lock` as its dependency source of truth. Python 3.11 or 3.12 is required.

### CPU development and mock pipeline

Install `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root run:

```bash
uv sync --group dev
uv run python scripts/mock_pipeline_demo.py
uv run pytest
```

This profile installs the application and test dependencies only. It does not install PyTorch, SAM 2, Whisper, or model weights, and does not require a camera or GPU.

### Full model environment (Linux, NVIDIA GPU)

The full profile installs speech and vision dependencies, including the SAM 2 implementation pinned in `pyproject.toml`, from the PyTorch CUDA 12.1 wheel index. Use a Linux x86_64 machine with a compatible NVIDIA driver; install the CUDA toolkit and `nvcc` if you want SAM 2's optional CUDA extension built. See the [SAM 2 installation requirements](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) before installing.

Install the system audio and video prerequisites first. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y curl ffmpeg libportaudio2 portaudio19-dev
```

Then install the full profile and download the default SAM 2.1 Small checkpoint:

```bash
uv sync --extra full-demo --group dev
bash scripts/download_models.sh
uv run python scripts/check_setup.py --full
```

The first grounding-model use downloads `IDEA-Research/grounding-dino-tiny` through Hugging Face Transformers into its normal cache. Whisper downloads its selected speech model the first time it is loaded. SAM 2 weights are downloaded separately into `checkpoints/`; large weights are ignored by Git.

Choose another official SAM 2.1 size with `SAM2_MODEL=tiny`, `small`, `base_plus`, or `large` before running the download script. Configure the matching checkpoint path in [`configs/model_paths.yaml`](configs/model_paths.yaml). The `small` checkpoint is the default; larger models need more GPU memory.

If SAM 2's optional CUDA extension cannot be built and you want the Python implementation without that extension, install with:

```bash
SAM2_BUILD_CUDA=0 uv sync --extra full-demo --group dev
```

The SAM 2 Python package still requires the compatible PyTorch and TorchVision pair declared by the project. Setting `SAM2_BUILD_CUDA=0` only skips compiling its optional extension; it does not make CPU inference real-time.

### Current runnable entry point

The checked-in demo is the deterministic mock pipeline:

```bash
uv run python scripts/mock_pipeline_demo.py
```

The full-model profile prepares the dependencies and model assets for integration. The current `src/stark_vision/main.py`, camera adapter, speech adapters, and SAM 2 wrapper are not implemented on this branch, so a live speech-to-mask webcam demo is not yet an available entry point. Do not use `scripts/run_demo.py` as the real-model launcher; it is also a mock demonstration.

## Dependency workflow

- Add shared Python dependencies to `pyproject.toml`; do not add subsystem-specific requirements or lock files.
- Regenerate and commit `uv.lock` with dependency changes (`uv lock`).
- Verify the lock before merging (`uv lock --check`).
- Use `uv sync --group dev` for CPU/mock development and `uv sync --extra full-demo --group dev` for the supported full-model profile.
- Keep model checkpoints, local caches, and machine-specific paths out of Git.

See [`docs/setup.md`](docs/setup.md) for clean-machine instructions, diagnostics, and troubleshooting. See [`docs/interfaces.md`](docs/interfaces.md) for the application contracts and [`docs/architecture.md`](docs/architecture.md) for the pipeline design.

## Repository map

```text
configs/                 Runtime defaults and model paths
checkpoints/             Downloaded weights (not committed)
docs/                    Setup, interfaces, architecture, and demo notes
scripts/                 Mock demo, model download, and setup checks
src/stark_vision/        Application modules and deterministic mocks
tests/                   Unit and integration tests
pyproject.toml           Canonical dependency declarations
uv.lock                  Resolved project-wide dependency lock
```

## References

- [SAM 2](https://github.com/facebookresearch/sam2) — promptable image/video segmentation
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — open-vocabulary grounding
- [Whisper](https://github.com/openai/whisper) — local speech recognition
- [UniPixel](https://github.com/PolyU-ChenLab/UniPixel) — future-extension reference (unified referring + reasoning)
