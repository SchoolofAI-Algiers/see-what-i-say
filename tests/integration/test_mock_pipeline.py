from __future__ import annotations

import unittest

import numpy as np

from stark_vision.camera.frame_buffer import FrameBuffer
from stark_vision.contracts import (
    BoundingBox,
    CandidateSelector,
    CandidateSelectorKind,
    CommandType,
    EffectMode,
    EffectSpec,
    Frame,
    GroundingCandidate,
    TrackingResult,
    TrackingStatus,
)
from stark_vision.effects.engine import EffectEngine
from stark_vision.mocks.pipeline import make_default_mock_pipeline
from stark_vision.mocks.adapters import MockSTT
from stark_vision.speech.command_parser import CommandParser
from stark_vision.target.target_manager import TargetManager


class CommandParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = CommandParser()

    def test_selection_effect_clarification_stop_and_unknown(self) -> None:
        self.assertEqual(self.parser.parse("STARK, segment the person in red").prompt, "the person in red")
        self.assertEqual(self.parser.parse("make him blue").effect.color_rgb, (0, 80, 255))
        self.assertEqual(self.parser.parse("blur her").effect.mode, EffectMode.BLUR)
        left = self.parser.parse("the one on the left")
        self.assertEqual(left.type, CommandType.RESOLVE_CANDIDATE)
        self.assertEqual(left.selector.value, "left")
        self.assertEqual(self.parser.parse("number two").selector.value, 2)
        self.assertEqual(self.parser.parse("stop tracking").type, CommandType.STOP_TRACKING)
        self.assertEqual(self.parser.parse("what is he doing?").type, CommandType.UNKNOWN)


class TargetManagerTests(unittest.TestCase):
    def test_ambiguous_candidates_resolve_by_spatial_side(self) -> None:
        manager = TargetManager(score_margin=0.15)
        candidates = (
            GroundingCandidate(BoundingBox(55, 2, 80, 35), 0.78, "right"),
            GroundingCandidate(BoundingBox(5, 2, 25, 35), 0.72, "left"),
        )
        manager.begin_acquisition(7, "person in red", 0)
        self.assertIsNone(manager.receive_candidates(7, candidates))
        self.assertTrue(manager.pending.ambiguous)
        selected = manager.resolve_candidate(CandidateSelector(CandidateSelectorKind.SIDE, "left"))
        self.assertEqual(selected.label, "left")

    def test_failed_replacement_keeps_active_target(self) -> None:
        manager = TargetManager()
        initial = GroundingCandidate(BoundingBox(2, 2, 8, 8), 0.9, "old")
        manager.begin_acquisition(1, "old", 0)
        manager.receive_candidates(1, [initial])
        first_id = manager.pending_target_id()
        mask = np.ones((10, 10), dtype=np.bool_)
        manager.promote(TrackingResult(first_id, 0, initial.box, mask, 0.9, TrackingStatus.TRACKING))

        manager.begin_acquisition(2, "new", 1)
        manager.receive_candidates(2, [GroundingCandidate(BoundingBox(1, 1, 7, 7), 0.9, "new")])
        manager.abort_pending("initialization failed")
        self.assertEqual(manager.active.prompt, "old")


class FrameBufferTests(unittest.TestCase):
    def test_missing_seed_or_gap_requests_reacquisition(self) -> None:
        buffer = FrameBuffer(max_frames=3)
        for frame_id in range(4):
            buffer.append(Frame(frame_id, (frame_id + 1) * 10, np.zeros((4, 4, 3), dtype=np.uint8)))
        self.assertIsNone(buffer.after(0))
        self.assertEqual([frame.id for frame in buffer.after(1)], [2, 3])


class EffectTests(unittest.TestCase):
    def test_all_effects_return_new_rgb_frame_and_do_not_mutate_input(self) -> None:
        frame = np.full((12, 14, 3), 100, dtype=np.uint8)
        original = frame.copy()
        mask = np.zeros((12, 14), dtype=np.bool_)
        mask[3:9, 4:10] = True
        engine = EffectEngine()
        specs = (
            EffectSpec(EffectMode.RECOLOR, (0, 0, 255)),
            EffectSpec(EffectMode.BLUR),
            EffectSpec(EffectMode.ISOLATE),
            EffectSpec(EffectMode.HIGHLIGHT),
        )
        for spec in specs:
            output = engine.apply(frame, mask, spec)
            self.assertEqual(output.shape, frame.shape)
            self.assertEqual(output.dtype, np.uint8)
            self.assertFalse(np.shares_memory(output, frame))
        np.testing.assert_array_equal(frame, original)


class MockPipelineTests(unittest.TestCase):
    def test_speech_to_tracking_effect_replacement_and_stop(self) -> None:
        pipeline = make_default_mock_pipeline()
        pipeline.capture_next()
        pipeline.listen_and_dispatch()
        self.assertTrue(pipeline.manager.pending.ambiguous)
        self.assertEqual(len(pipeline.grounder.calls), 1)

        for _ in range(4):
            pipeline.capture_next()
        pipeline.listen_and_dispatch()
        self.assertIsNotNone(pipeline.manager.active)
        self.assertEqual(pipeline.manager.active.prompt, "the person in red")
        self.assertEqual(pipeline.manager.active.frame_id, 4)
        self.assertEqual(pipeline.manager.active.mask.shape, (64, 96))
        self.assertEqual(pipeline.manager.active.mask.dtype, np.bool_)
        self.assertEqual(len(pipeline.grounder.calls), 1)

        pipeline.capture_next()
        pipeline.listen_and_dispatch()
        self.assertEqual(pipeline.manager.effect.mode, EffectMode.RECOLOR)
        self.assertEqual(len(pipeline.grounder.calls), 1)
        recolored = pipeline.last_rendered.image
        self.assertEqual(tuple(recolored[20, 20]), (0, 80, 255))

        pipeline.capture_next()
        pipeline.listen_and_dispatch()
        self.assertEqual(pipeline.manager.effect.mode, EffectMode.BLUR)

        pipeline.capture_next()
        pipeline.listen_and_dispatch()
        self.assertIsNone(pipeline.manager.active)
        self.assertIsNone(pipeline.manager.pending)
        self.assertEqual(pipeline.manager.effect.mode, EffectMode.NONE)
        self.assertEqual(pipeline.tracker.reset_count, 1)
        self.assertEqual(len(pipeline.grounder.calls), 1)

    def test_ambiguous_retarget_keeps_old_target_until_commit(self) -> None:
        pipeline = make_default_mock_pipeline()
        pipeline.capture_next()
        # Narrow the first result to an unambiguous candidate.
        pipeline.grounder.candidates = (GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.9, "first"),)
        pipeline.dispatch(pipeline.parser.parse("segment the first person"))
        old_id = pipeline.manager.active.target_id

        pipeline.grounder.candidates = (
            GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.78, "left"),
            GroundingCandidate(BoundingBox(58, 13, 82, 50), 0.72, "right"),
        )
        pipeline.dispatch(pipeline.parser.parse("segment another person"))
        self.assertEqual(pipeline.manager.active.target_id, old_id)
        self.assertTrue(pipeline.manager.pending.ambiguous)
        pipeline.dispatch(pipeline.parser.parse("the one on the right"))
        self.assertNotEqual(pipeline.manager.active.target_id, old_id)
        self.assertIn(old_id, pipeline.tracker.removed_ids)

    def test_expired_seed_is_regrounded_before_tracker_initialization(self) -> None:
        pipeline = make_default_mock_pipeline()
        pipeline.stt = MockSTT((
            "segment the person in red",
            "the one on the left",
            "the one on the left",
        ))
        pipeline.frames = FrameBuffer(max_frames=2)
        pipeline.capture_next()
        pipeline.listen_and_dispatch()
        for _ in range(4):
            pipeline.capture_next()

        # The first clarification chooses a candidate, but its seed has expired.
        # The harness must ground again on the newest retained frame and ask again
        # if the refreshed result is still ambiguous.
        pipeline.listen_and_dispatch()
        self.assertEqual(len(pipeline.grounder.calls), 2)
        self.assertIsNone(pipeline.manager.active)
        self.assertTrue(pipeline.manager.pending.ambiguous)
        self.assertEqual(pipeline.manager.pending.source_frame_id, 4)

        pipeline.listen_and_dispatch()
        self.assertIsNotNone(pipeline.manager.active)
        self.assertEqual(pipeline.manager.active.frame_id, 4)
        self.assertEqual(pipeline.tracker.initialize_count, 1)


if __name__ == "__main__":
    unittest.main()
