# Grounding Subsystem Upgrades Guide

Quick reference for the four core grounding upgrades and how to explain them to teammates.

## 1. Dynamic Adaptive Threshold (DAT)

Replaces the hardcoded 0.25 cutoff with dynamic scoring:
- `query complexity`: long descriptive queries distribute cross-attention scores across multiple words, resulting in lower peak probabilities. The threshold scales down dynamically from 0.28 to 0.20 based on word count and modifiers.
- `relative score falloff`: keeps candidates that fall within 65% of the top detection score, cutting off long-tail background noise.
- File: `src/stark_vision/grounding/adaptive_threshold.py`

## 2. Temporal Smoothing Filter

Removes coordinate jitter before feeding the bounding box to SAM 2:
- `centroid`: filtered via 1-Euro adaptive filter (`fc=1.0 Hz`, `beta=0.007`). Stays rock solid when stationary and tracks smoothly without latency when the target moves.
- `dimensions`: box width and height are smoothed in log space to avoid aspect ratio popping under zoom.
- File: `src/stark_vision/grounding/temporal_filter.py`

## 3. Spatial Disambiguation

Handles multiple candidate detections:
- `IoU deduplication`: merges duplicate boxes predicting parts of the same object (IoU >= 0.45 or containment >= 0.70).
- `spatial direction resolution`: parses "left", "right", "nearest", and "top" directly from the query, auto-selecting the right candidate without asking for clarification.
- `ambiguity trigger`: flags true ambiguity only when separate objects have a score delta < 0.15.
- File: `src/stark_vision/grounding/disambiguation.py`

## 4. Rate-Decoupled Scheduler

Decouples detection from video playback:
- Grounding DINO runs at 1-2 Hz on demand (new prompt or re-acquisition).
- Video feed and reticle rendering run at 30+ FPS via background capture thread.
- Five lifecycle states: `IDLE`, `SEARCHING`, `TRACKING`, `LOST`, `RECOVERY`.
- File: `src/stark_vision/target/state_machine.py` and `target_manager.py`
