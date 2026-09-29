# Grounding Engine Notes

Notes and implementation details for the vision grounding part of STARK VISION.

## What this does

This module handles taking a video frame from the webcam along with a text prompt (like "guy in red shirt" or "laptop on desk") and finding the bounding box coordinates for that object so it can be passed to SAM2 for mask tracking.

Standard detection like YOLO only works for fixed classes from COCO (person, chair, etc), so it cant tell which person is wearing what. Grounding DINO allows open vocabulary queries using text attention over visual features.

## Main Components

### Grounder class (`src/stark_vision/grounding/grounder.py`)
Loads `IDEA-Research/grounding-dino-tiny` using HuggingFace transformers on CUDA.

The main function is `find(frame_bgr, text_prompt)` (also aliased to `find_candidates`). 
It takes:
- `frame_bgr`: raw opencv frame
- `text_prompt`: natural language description
- `box_threshold`: minimum box confidence (default 0.30)
- `text_threshold`: minimum text alignment similarity (default 0.25)

Returns a list of `GroundingCandidate` objects sorted by confidence score.

### GroundingCandidate
Simple dataclass holding the prediction:
- `box`: absolute pixel coords `(x1, y1, x2, y2)` matching opencv frame dimensions
- `score`: float confidence between 0 and 1
- `label`: matched phrase text
- helper properties: `width`, `height`, `center`, `area`

### Ambiguity check
`is_ambiguous(candidates, delta_threshold=0.15)` checks if the top two detected candidates have scores that are too close to each other. If score difference is below 0.15, we flag it so the system can ask user to clarify which object they meant.

## Implementation details and gotchas

- OpenCV captures in BGR, but transformers expect RGB with imagenet normalization. If you dont convert with `cv2.cvtColor`, red and blue channels get swapped and color based prompts fail.
- Grounding DINO tokenizer expects queries to end with a dot `.` to seperate phrases cleanly. The helper `_normalize_prompt` handles this automatically.
- Model runs inside `torch.no_grad()` to avoid building gradient graph in memory during inference, which keeps VRAM usage around 660MB on RTX 4060 laptop.
- The raw output from model is normalized `[cx, cy, w, h]` in 0 to 1 range. The processor scales this back to actual camera width and height so downstream doesnt need manual coordinate transforms.

## Testing and Demo

Unit tests are in `tests/unit/test_grounder.py`:
```bash
uv run pytest tests/unit/test_grounder.py -v
```

To run interactive demo on webcam:
```bash
uv run python scripts/test_grounder_live.py
```
Keybindings:
- SPACE: run grounding on current frame
- P: type new text prompt in terminal
- S: save frame snapshot
- Q or ESC: exit
