# Grounding Engine Notes

Notes and implementation details for the vision grounding part of STARK VISION.

## What this does

This module handles taking a video frame from the webcam along with a text prompt (like "guy in red shirt" or "laptop on desk") and finding the bounding box coordinates for that object so it can be passed to SAM2 for mask tracking.

Standard detection like YOLO only works for fixed classes from COCO (person, chair, etc), so it cant tell which person is wearing what. Grounding DINO allows open vocabulary queries using text attention over visual features.

## Main Components

### Grounder class (`src/stark_vision/grounding/grounder.py`)
Loads `IDEA-Research/grounding-dino-base` (or `tiny`) using HuggingFace transformers on CUDA.

The main function is `find(frame_bgr, text_prompt)` (also aliased to `find_candidates`). 
It takes:
- `frame_bgr`: raw opencv frame
- `text_prompt`: natural language description
- `box_threshold`: minimum box confidence (default 0.25)
- `text_threshold`: minimum text alignment similarity (default 0.20)

Returns a list of `GroundingCandidate` objects sorted by confidence score.

### GroundingCandidate
Dataclass holding the prediction (from `stark_vision.contracts`):
- `box`: `BoundingBox` instance with `.center`, `.xyxy`, and pixel coordinates `(x1, y1, x2, y2)`
- `score`: float confidence between 0 and 1
- `label`: matched phrase text
- helper properties: `x1`, `y1`, `x2`, `y2`, `width`, `height`, `center`, `area`

### Ambiguity check
`is_ambiguous(candidates, delta_threshold=0.15)` checks if the top two detected candidates have scores that are too close to each other. If score difference is below 0.15, we flag it so the system can ask user to clarify which object they meant.

### JarvisHUD Tactical AR Overlay (`src/stark_vision/hud/overlay.py`)
Renders the Iron Man / Mark LXXXV heads-up display directly on the video feed:
- Rotating Arc Reactor gyroscopic gauge with concentric segmented rings and tick marks
- Dynamic acoustic voice waveform equalizer bars (JARVIS speech pulse)
- Tactical corner L-brackets with sweeping laser line, target crosshairs, and range estimates
- Cyber-chamfered command console dock with live on-screen keyboard search bar
- Diagnostic 2.5-second boot sequence with non-blocking spoken audio greeting via `src/stark_vision/speech/jarvis_voice.py`

## Implementation details and gotchas

- OpenCV captures in BGR, but transformers expect RGB with imagenet normalization. If you dont convert with `cv2.cvtColor`, red and blue channels get swapped and color based prompts fail.
- Grounding DINO tokenizer expects queries to end with a dot `.` to seperate phrases cleanly. The helper `_normalize_prompt` handles this automatically, cleaning extra commas, slashes, and spaces.
- Model runs inside `torch.no_grad()` to avoid building gradient graph in memory during inference, keeping VRAM usage around 892MB for base model on RTX 4060 laptop.
- The raw output from model is normalized `[cx, cy, w, h]` in 0 to 1 range. The processor scales this back to actual camera width and height so downstream doesnt need manual coordinate transforms.
- Terminal `input()` calls freeze the OpenCV GUI event loop. All query editing is handled directly on-screen through ASCII key codes in `cv2.waitKey(1)` to maintain a locked 30 FPS display.

## Testing and Demo

Unit tests are in `tests/unit/`:
```bash
uv run pytest tests/unit/ -v
```

To run interactive demo on webcam:
```bash
uv run python scripts/test_grounder_live.py
```
Keybindings (directly inside OpenCV window):
- Typing any text: edits search query live on-screen
- ENTER: runs grounding on current frame with active query
- BACKSPACE: deletes character from query
- SPACE: rescans active query (or types space while editing)
- C: clears all active target boxes and search buffer
- + / -: tunes detection threshold sensitivity live
- S: saves frame snapshot to `tests/data/sample_frames/`
- ESC: clean exit
