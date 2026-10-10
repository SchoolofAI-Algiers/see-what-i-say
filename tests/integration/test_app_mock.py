"""test real parser with mock grounding and segmentation"""
import time
from stark_vision.contracts import BoundingBox, EffectMode, GroundingCandidate
from stark_vision.main import App
from stark_vision.mocks.adapters import MockGrounder, MockTracker, make_scene_frames

def _run_until(app, frames, pred, start=1):
    for frame in frames[start:]:
        app.on_frame(frame)
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_voice_flow_without_regrounding():
    frames = make_scene_frames(40)
    grounder = MockGrounder((GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.9, "person in red"),))
    app = App(grounder, MockTracker())
    try:
        app.on_frame(frames[0])
        app.handle_command(app.parser.parse("segment the person in red"))
        assert _run_until(app, frames, lambda: app.manager.active is not None)
        assert app.manager.active.prompt == "the person in red"
        assert len(grounder.calls) == 1

        app.handle_command(app.parser.parse("blur him"))
        assert app.manager.effect.mode is EffectMode.BLUR
        app.handle_command(app.parser.parse("follow her"))
        assert app.manager.message.startswith("Following")
        assert len(grounder.calls) == 1  # follow-ups never re-ground

        app.handle_command(app.parser.parse("clear everything"))
        assert app.manager.active is None
    finally:
        app.close()


def test_manual_box_fallback_skips_grounding():
    frames = make_scene_frames(5)
    grounder = MockGrounder(())
    app = App(grounder, MockTracker())
    try:
        app.on_frame(frames[0])
        app.select_manual_box(BoundingBox(10, 14, 30, 48))
        assert app.manager.active is not None
        assert grounder.calls == []
    finally:
        app.close()


def test_unknown_and_empty_transcript_messages():
    app = App(MockGrounder(()), MockTracker())
    try:
        app.handle_command(app.parser.parse(""))
        assert app.manager.message == "Didn't catch that"
        app.handle_command(app.parser.parse("dance"))
        assert app.manager.message.startswith("Unknown command")
    finally:
        app.close()
