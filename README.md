# Stark Vision : see-what-i-say

STARK VISION technical project for SoAI welcome day 26/27 season :

Real-time speech-guided referring video segmentation

**Mentors:** Rayan Ouali, Hibatallah Khadir

## Overview

### General Description

STARK VISION is a real-time computer vision and multimodal interaction system. A user stands in front of a webcam and speaks a natural-language description of a visual target (e.g. *"segment the guy wearing the red shirt"*). The system transcribes the speech, grounds the description in the current camera frame, produces a pixel-accurate segmentation mask, and tracks that target through subsequent frames — all without re-running the expensive language-grounding step on every frame.

### Architecture

The system is composed of two independent loops:

- **Command loop** (event-driven): mic → Whisper STT → command parsing → grounding model → SAM 2 initialization → target created/updated.
- **Vision loop** (continuous): webcam frame → SAM 2 tracking → mask update → effect applied → HUD rendered → displayed.

```
Speech -> Text -> Referring Expression -> Grounding -> Box -> SAM 2 Mask
                                                              |
                                                              v
                                                     Target State / ID
                                                              |
                                                              v
                                                     SAM 2 Video Tracking
                                                              |
                                                              v
                                                       Live Pixel Mask
                                                              |
                                                              v
                                                Effect Engine + HUD -> Screen
```
### Modular Division

| Workstream | components |
|---|---|
| **Vision / Models** | Grounding, SAM 2 segmentation & tracking |
| **Audio / Interaction** | Mic capture, Whisper STT, command parsing |
| **Systems / State** | Target manager, state machine, threading/queues |
| **Rendering / HUD** | Effects (highlight, recolor, blur, isolate), HUD overlay |
| **QA / Performance** | Tests, benchmarking, fallback controls |

Each workstream maps to a fixed set of modules below, communicating only through small, explicit interfaces (`grounder.find()`, `segmenter.initialize()`, `tracker.update()`), so backends can be swapped without touching the rest of the app.

---

## Project Structure

```
stark_vision/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── default.yaml               
│   └── model_paths.yaml            
│
├── checkpoints/
│   └── README.md                   
│
├── scripts/
│   ├── download_models.sh
│   ├── run_demo.py                 # entry point for demo
│   └── benchmark.py               
│
├── src/stark_vision/
│   ├── main.py                     
│   │
│   ├── camera/
│   │   └── capture.py
│   │
│   ├── speech/                    
│   │   ├── recorder.py
│   │   ├── stt.py
│   │   └── command_parser.py
│   │
│   ├── grounding/                  
│   │   └── grounder.py
│   │
│   ├── segmentation/              
│   │   ├── sam2_wrapper.py
│   │   └── tracker.py
│   │
│   ├── target/                     
│   │   ├── target_state.py
│   │   ├── target_manager.py
│   │   └── state_machine.py
│   │
│   ├── effects/                   
│   │   ├── base_effect.py
│   │   ├── highlight.py
│   │   ├── recolor.py
│   │   ├── blur.py
│   │   └── isolate.py
│   │
│   ├── hud/                        
│   │   └── overlay.py
│   │
│   ├── geometry/
│   │   └── coords.py              
│   │
│   └── utils/
│       ├── logging_setup.py
│       └── perf.py
│
├── tests/                          
│   ├── unit/
│   ├── integration/
│   └── data/sample_frames/
│
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   └── demo_script.md
│
└── notebooks/                      
    └── test.ipynb  #if anything was done in a notebooj
```
## Operator Controls
The interface of this project
| Key | Action |
|---|---|
| `SPACE` | listen / submit voice command |
| `M` | manual target selection (fallback) |
| `1` / `2` / `3` / `4` | target-lock / recolor / blur / isolate |
| `R` | reset target |
| `C` | clear all |
| `ESC` | emergency stop |


## References

- [Grounded-SAM-2](https://github.com/idea-research/grounded-sam-2) — grounding + SAM 2 integration, video tracking
- [Grounding DINO](https://github.com/idea-research/groundingdino) — open-set language-grounded detection
- [SAM 2](https://github.com/facebookresearch/sam2) — promptable image/video segmentation
- [Whisper](https://github.com/openai/whisper) — local speech recognition
- [UniPixel](https://github.com/PolyU-ChenLab/UniPixel) — future-extension reference (unified referring + reasoning)


## Running
some scripts that are needed
```powershell
uv sync --extra vision --extra speech --extra api
$env:SAM2_BUILD_CUDA = "0"; uv pip install --no-build-isolation -e ..\sam2   # SAM 2 (not in the lock)
uv run python scripts/download_models.py          # SAM 2 checkpoint + DINO + Whisper, once
uv run python -m stark_vision.main --no-speech     # type commands with T
uv run python -m stark_vision.main                 # push-to-talk with SPACE
uv run python scripts/run_on_video.py in.mp4 out.mp4 --command "segment the person in red"
```

Keys: SPACE talk, T type, M manual box, 1-4 effects, R/C reset, ESC quit. `uv sync` removes SAM 2;
use `uv sync --inexact` or reinstall it. Tests: `uv run pytest tests -q`.
