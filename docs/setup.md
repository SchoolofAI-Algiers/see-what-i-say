# Developer Setup

This guide starts from a fresh clone. It covers the CPU/mock workflow and the supported Linux/NVIDIA dependency and model-asset profile. The repository's committed `pyproject.toml` and `uv.lock` define Python dependencies; model code and large model weights are installed/downloaded separately from the application source.

## Requirements

- Python 3.11 or 3.12.
- `uv` installed and available on `PATH` ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
- Git for the pinned SAM 2 source dependency in the full profile.
- For live audio, FFmpeg and PortAudio development/runtime libraries.
- For the supported full inference setup, Linux x86_64, an NVIDIA GPU and a compatible driver. Install a CUDA toolkit compatible with the selected PyTorch build if compiling SAM 2's CUDA extension. SAM 2 recommends installing a matching PyTorch/TorchVision pair first and documents `SAM2_BUILD_CUDA=0` to skip that extension build: [official installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).

## 1. Clone and install `uv`

```bash
git clone <repository-url> see-what-i-say
cd see-what-i-say
```

Install `uv` with the command for your operating system from its official guide. Confirm the tools:

```bash
uv --version
python3 --version
```

`uv sync` creates the project's `.venv` if it does not exist, selects the Python requested by `requires-python` when available, installs the project, and synchronizes the selected dependency profile from the lock.

## 2. CPU/mock development

This is the fast setup for parser, state, effects, HUD, geometry, and mock pipeline work. It has no PyTorch or model checkpoint requirement.

```bash
uv sync --group dev
uv run python scripts/mock_pipeline_demo.py
uv run pytest
```

Expected result: the mock demo prints the recognized mock utterances, target changes, mask shape, and confirmation that grounding/tracker initialization happened once. This validates software contracts only; it makes no camera, model, or performance claim.

## 3. Full model dependencies on Linux/NVIDIA

Install OS audio requirements before Python packages. For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y curl ffmpeg libportaudio2 portaudio19-dev
```

Verify `ffmpeg -version` succeeds. `sounddevice` uses PortAudio for microphone input. OpenCV uses the camera and display provided by the host OS.

Install the full project profile:

```bash
uv sync --extra full-demo --group dev
```

This installs the pinned `facebookresearch/sam2` source revision along with compatible project-declared dependencies. The `torch` and `torchvision` packages are resolved from the configured PyTorch CUDA 12.1 index for this extra. Check the CUDA runtime visible to PyTorch:

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

If the CUDA extension build fails because `nvcc` or `CUDA_HOME` is unavailable, either install the matching CUDA toolkit and retry, or explicitly skip the optional extension:

```bash
SAM2_BUILD_CUDA=0 uv sync --extra full-demo --group dev
```

Skipping this extension does not switch PyTorch to CPU or guarantee real-time operation. It only disables the optional compiled operation in SAM 2.

## 4. Download and configure model assets

SAM 2 source code is installed as a pinned Python dependency by the full profile. Its checkpoint weights are a separate download and must not be committed.

From the project root:

```bash
bash scripts/download_models.sh
```

The default is `sam2.1_hiera_small.pt`, downloaded from Meta's official SAM 2.1 model files. Select `tiny`, `small`, `base_plus`, or `large` with `SAM2_MODEL`, for example:

```bash
SAM2_MODEL=tiny bash scripts/download_models.sh
```

The downloader writes under `checkpoints/` by default. Override the destination with `SAM2_CHECKPOINT_DIR=/path/to/checkpoints`. Update [`../configs/model_paths.yaml`](../configs/model_paths.yaml) to point to the downloaded checkpoint. The model configuration identifier must match the checkpoint size (`sam2.1_hiera_t`, `_s`, `_b+`, or `_l`).

Grounding DINO is currently configured through the Hugging Face Transformers API as `IDEA-Research/grounding-dino-tiny`; its processor and model weights are fetched from Hugging Face on first model load and cached there. Whisper's model weights are also downloaded on first load, by default under `~/.cache/whisper` or `$XDG_CACHE_HOME/whisper`. Both caches are external to this repository; use `HF_HOME` for Hugging Face files and `XDG_CACHE_HOME` for Whisper's cache location.

Validate dependencies, CUDA visibility, and local checkpoint paths:

```bash
uv run python scripts/check_setup.py --full
```

This check does not load model weights into GPU memory or claim that a webcam is available.

## 5. Run the available demo

The current runnable demo is the deterministic mock pipeline:

```bash
uv run python scripts/mock_pipeline_demo.py
```

On this branch, the production camera/STT/grounding/SAM 2 adapter files and the application main loop are still empty. Therefore the full-model installation and assets are preparatory; there is not yet a real webcam demo command to document. Integrate those adapters and define the real launcher before treating setup as end-to-end model validation.

## Dependency and branch policy

- `pyproject.toml` is the canonical dependency declaration. `uv.lock` is the single resolved lock for all declared extras and development groups.
- Do not add subsystem-local `requirements.txt`, setup files, or lock files. The empty root `requirements.txt` was removed in favor of the canonical project metadata.
- When changing dependencies, update `pyproject.toml`, regenerate `uv.lock` with `uv lock`, and include both in the same PR. Check it with `uv lock --check`.
- Keep SAM 2 pinned to the approved upstream commit. Do not point the dependency at `main` or an unpinned branch.
- Keep checkpoints and generated caches out of Git. Store machine-specific device settings and checkpoint paths in local config overrides or environment variables, not shared edits.

## Troubleshooting

### `uv sync` cannot find a Python version

Install Python 3.11/3.12 or let uv download one, then retry `uv sync`.

### PyTorch imports but `torch.cuda.is_available()` is false

Check that the NVIDIA driver is installed and visible (`nvidia-smi`), and that the environment installed the `full-demo` extra rather than only the default/mock profile. The lock's CUDA 12.1 wheel selection is for the Linux/NVIDIA profile.

### SAM 2 reports missing CUDA extension or `CUDA_HOME`

Install a CUDA toolkit with `nvcc` compatible with the selected PyTorch build and retry. For the optional extension, set `SAM2_BUILD_CUDA=0` during sync as shown above.

### Grounding DINO or Whisper cannot access its model

The first model load requires network access to its model host unless the weights have already been cached. Confirm the machine can reach Hugging Face for Grounding DINO and the Whisper model host. The SAM 2 checkpoint is downloaded separately using `scripts/download_models.sh`.

### Audio device errors

Confirm the microphone is available to the OS, PortAudio is installed, and the selected input device is not already in exclusive use. The project currently has no finalized cross-platform audio-device selection UI.

### Camera or OpenCV window is unavailable

Confirm the webcam is visible to the OS and that the current session supports GUI windows. The mock demo does not access camera hardware.
