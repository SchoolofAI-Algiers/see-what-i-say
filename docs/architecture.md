# Architecture
Two loops (see the project spec):

- **Command loop** (runs on speech command): SPACE/T -> `AudioRecorder` -> `SpeechToText` (Whisper)
  -> `CommandParser` -> vision loop queue -> `GroundingWorker` (Grounding DINO) -> `TargetManager`
  -> `SAM2Tracker.initialize`.
- **Vision loop** (every frame): `Camera` thread -> `FrameBuffer` -> `SAM2Tracker.update` ->
  `TargetManager.observe` -> `EffectEngine` -> `HudRenderer` -> `cv2.imshow`.

Rules: only the vision loop (`App.process_frame` / `handle_command`) mutates `TargetManager` and the
tracker. Workers exchange immutable requests/results through queues. Follow-up commands
("blur him", "follow her") never re-ground. Operator fallback: key `M` draws a box and enters the same
`stage -> tracker.initialize -> promote` path.

Modules: `main.py` (App + runtime), `video.py` (offline runner), `segmentation/sam2_wrapper.py`
(windowed SAM 2), `grounding/` (DINO + filtering), `target/` (state), `effects/engine.py`, `hud/`.
Data contracts live in `contracts.py`; protocols in `docs/interfaces.md`.
