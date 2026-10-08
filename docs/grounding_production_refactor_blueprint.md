# Grounding Engine Architecture and Refactoring Notes

Technical overview of the grounding subsystem refactor in `src/stark_vision/grounding/` for real-time referring expression tracking.

## Core Problems with the MVP

The initial prototype in `scripts/test_grounder_live.py` had four issues that broke down in practice:

1. **Static 0.25 threshold.** Fixed confidence cutoffs fail because Grounding DINO cross-attention distributes score mass across words. Simple single-word prompts ("person") score 0.60+, while descriptive prompts ("the laptop on the wooden desk") peak around 0.28-0.31. A fixed 0.35 threshold dropped genuine long queries, while 0.20 let in background noise.
2. **Raw box jitter.** Frame-to-frame detection shifts 5 to 15 pixels due to lighting and sensor noise. Passing unfilitered coordinates into SAM 2 corrupts its point and box prompts, causing mask drift.
3. **Overlapping box ambiguity.** DINO frequently fires multiple boxes on the same target (e.g. torso box and full-body box). Naive candidate counts flagged this as multiple distinct people, causing false ambiguity prompts.
4. **Main loop starvation.** Running 85ms DINO inference synchronously in the video capture loop capped camera throughput to 10 FPS and caused frame drop.

## Pipeline Architecture

We decoupled detection and tracking into two async paths:

```text
Camera Thread (30+ FPS) ──> Ring Buffer ──> Video Feed / UI Canvas
                                  │
                       (On Query Change)
                                  ▼
                         Grounding DINO (Base, FP16)
                                  │ (85ms latency)
                                  ▼
                    Post-Processing & Filtering
                    ├── Dynamic Threshold (DAT)
                    ├── IoU Cluster Deduplication
                    └── 1-Euro Centroid Smoothing
                                  │
                                  ▼
                       WebSocket Telemetry / SAM 2 Handoff
```

The vision loop runs camera capture on a background thread so the video feed never stutters when the detector runs. Grounding only fires on new prompts or recovery, keeping the GPU free for SAM 2 tracking at 30 FPS.

## The 4 Algorithmic Modifications

### 1. Dynamic Adaptive Thresholding (`adaptive_threshold.py`)
Computes cutoffs dynamically based on query length and relative score distribution:
- Base threshold lowers from 0.28 down to 0.20 as token count and descriptive modifiers increase.
- Relative falloff filter: drops any candidate whose score is below 65% of the top candidate's confidence (`score < 0.65 * top_score`).
- Keeps the floor at 0.20 to reject background clutter.

### 2. 1-Euro Centroid Smoothing (`temporal_filter.py`)
Applies adaptive low-pass filtering to bounding box coordinates:
- Centroid `(cx, cy)` runs through a 1-Euro filter: minimum cutoff frequency `fc = 1.0 Hz` suppresses stationary hand jitter, while speed coefficient `beta = 0.007` eliminates lag during rapid movement.
- Width and height are smoothed in log-space (`ln w`, `ln h`) to prevent box distortion when the camera zooms.

### 3. Spatial Disambiguation (`disambiguation.py`)
Resolves candidate ambiguities before triggering voice clarification:
- IoU clustering: boxes with IoU >= 0.45 or containment ratio >= 0.70 are merged into one candidate.
- Direction parser: phrases like "on the left", "on the right", "nearest", or "largest" sort candidates spatially. For example, "the one on the left" picks the lowest `cx` without prompting the user.
- Ambiguity trigger: only activates if two distinct, non-overlapping objects have a score gap smaller than 0.15.

### 4. Rate-Decoupled Target State Machine (`target/state_machine.py`)
Enforces five operational states: `IDLE`, `SEARCHING`, `TRACKING`, `LOST`, and `RECOVERY`. Grounding DINO only runs during `SEARCHING` and `RECOVERY`. Once locked, state transitions to `TRACKING` and hands off coordinates to the tracker.

## Contract Interoperability (`contracts.py`)

All outputs conform to the shared contracts:
- `Grounder.find()` accepts either Rayan's `Frame` dataclass (RGB uint8) or raw OpenCV NumPy arrays.
- `GroundingCandidate.box` returns a `BoundingBox` instance with `.center`, `.xyxy`, and supports sequence indexing (`box[0]`) for backwards compatibility with existing UI callers.
- `HudRenderer` implements `render(frame_rgb, snapshot)` returning `RenderedFrame`.

## Hardware Benchmarks (RTX 4060 Laptop GPU)

- Inference time: 84 to 88 ms per query (Base model with FP16 autocast).
- VRAM footprint: ~470 MB allocated.
- Camera throughput: 37 FPS steady at 1280x720 resolution.
