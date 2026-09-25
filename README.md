# Stark Vision : see-what-i-say
STARK VISION technical project for SoAI welcome day 26/27 season : 
Real-time speech-guided referring video segmentation
**Mentors:** Rayan Ouali, Hibatallah Khadir
---

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
│   ├── default.yaml                # thresholds, FPS targets, effect defaults
│   └── model_paths.yaml            # checkpoint paths, model variant selection
│
├── checkpoints/
│   └── README.md                   # download instructions only — no weights in git
│
├── scripts/
│   ├── download_models.sh
│   ├── run_demo.py                 # single entry point for the live demo
│   └── benchmark.py                # FPS / latency measurement harness
│
├── src/stark_vision/
│   ├── main.py                     # wires command loop + vision loop together
│   │
│   ├── camera/
│   │   └── capture.py
│   │
│   ├── speech/                     # Audio / Interaction
│   │   ├── recorder.py
│   │   ├── stt.py
│   │   └── command_parser.py
│   │
│   ├── grounding/                  # Vision / Models
│   │   └── grounder.py
│   │
│   ├── segmentation/               # Vision / Models
│   │   ├── sam2_wrapper.py
│   │   └── tracker.py
│   │
│   ├── target/                     # Systems / State
│   │   ├── target_state.py
│   │   ├── target_manager.py
│   │   └── state_machine.py
│   │
│   ├── effects/                    # Rendering / HUD
│   │   ├── base_effect.py
│   │   ├── highlight.py
│   │   ├── recolor.py
│   │   ├── blur.py
│   │   └── isolate.py
│   │
│   ├── hud/                        # Rendering / HUD
│   │   └── overlay.py
│   │
│   ├── geometry/
│   │   └── coords.py               # the one model <-> camera <-> display mapping
│   │
│   └── utils/
│       ├── logging_setup.py
│       └── perf.py
│
├── tests/                          # QA / Performance
│   ├── unit/
│   ├── integration/
│   └── data/sample_frames/
│
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   └── demo_script.md
│
└── notebooks/                      # exploration only, never imported by src/
    └── 00_sanity_checks.ipynb
```

---

## Getting Started

```bash
git clone <repo-url>
cd stark_vision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
python scripts/run_demo.py
```

---

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

---

## Team Workflow

- **Two-week build**, vertical milestones — a working end-to-end pipeline first, polish after.
- **Non-negotiable milestone (Day 5):** voice prompt → grounded target → live tracked mask, working end to end. If this isn't working, all new features pause until it is.
- See `docs/demo_script.md` for the scripted welcome-day showcase.

---

## References

- [Grounded-SAM-2](https://github.com/idea-research/grounded-sam-2) — grounding + SAM 2 integration, video tracking
- [Grounding DINO](https://github.com/idea-research/groundingdino) — open-set language-grounded detection
- [SAM 2](https://github.com/facebookresearch/sam2) — promptable image/video segmentation
- [Whisper](https://github.com/openai/whisper) — local speech recognition
- [UniPixel](https://github.com/PolyU-ChenLab/UniPixel) — future-extension reference (unified referring + reasoning)
