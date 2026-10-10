"""Synchronous harness that exercises real app logic with mocked adapters."""

from __future__ import annotations

from dataclasses import dataclass

from stark_vision.camera.frame_buffer import FrameBuffer
from stark_vision.contracts import (
    BoundingBox,
    CommandEvent,
    CommandType,
    EffectMode,
    Frame,
    GroundingCandidate,
    GroundingRequest,
    GroundingResult,
    RenderedFrame,
    RuntimeSnapshot,
    TrackingStatus,
)
from stark_vision.effects.engine import EffectEngine
from stark_vision.hud.overlay import HudRenderer
from stark_vision.mocks.adapters import (
    MockAudioRecorder,
    MockCamera,
    MockGrounder,
    MockSTT,
    MockTracker,
    make_scene_frames,
)
from stark_vision.speech.command_parser import CommandParser
from stark_vision.target.target_manager import TargetManager


@dataclass(frozen=True)
class PipelineStep:
    transcript: str | None
    command: CommandEvent | None
    rendered: RenderedFrame | None


class MockPipeline:
    """Run a complete deterministic speech-to-render sequence without model I/O."""

    def __init__(
        self,
        camera: MockCamera,
        recorder: MockAudioRecorder,
        stt: MockSTT,
        grounder: MockGrounder,
        tracker: MockTracker,
        *,
        frame_buffer: FrameBuffer | None = None,
        manager: TargetManager | None = None,
    ) -> None:
        self.camera = camera
        self.recorder = recorder
        self.stt = stt
        self.grounder = grounder
        self.tracker = tracker
        self.frames = frame_buffer or FrameBuffer()
        self.manager = manager or TargetManager()
        self.parser = CommandParser()
        self.effects = EffectEngine()
        self.hud = HudRenderer()
        self._next_request_id = 1
        self.last_rendered: RenderedFrame | None = None

    def capture_next(self) -> RenderedFrame | None:
        frame = self.camera.read()
        if frame is None:
            return None
        self.frames.append(frame)
        results = self.tracker.update(frame)
        self.manager.observe(results)
        self.last_rendered = self._render(frame)
        return self.last_rendered

    def listen_and_dispatch(self) -> PipelineStep:
        self.recorder.start()
        audio = self.recorder.stop()
        transcript = self.stt.transcribe(audio)
        command = self.parser.parse(transcript.text)
        self.dispatch(command)
        return PipelineStep(transcript.text, command, self.last_rendered)

    def dispatch(self, command: CommandEvent) -> None:
        if command.type is CommandType.SELECT_TARGET:
            self._select_target(command)
        elif command.type is CommandType.RESOLVE_CANDIDATE:
            candidate = self.manager.resolve_candidate(command.selector) if command.selector else None
            if candidate is not None:
                self._stage_and_promote(candidate)
        elif command.type is CommandType.SET_EFFECT:
            if command.effect is not None:
                self.manager.set_effect(command.effect)
        elif command.type is CommandType.FOLLOW:
            if self.manager.active is None:
                self.manager.message = "No active target to follow"
            else:
                self.manager.message = f"Following {self.manager.active.prompt}"
        elif command.type is CommandType.STOP_TRACKING:
            self.tracker.reset()
            self.manager.stop()
        else:
            self.manager.message = "Command not recognized"
        latest = self.frames.latest
        if latest is not None:
            self.last_rendered = self._render(latest)

    def _select_target(self, command: CommandEvent) -> None:
        latest = self.frames.latest
        if latest is None or not command.prompt:
            self.manager.message = "Camera frame or target prompt is unavailable"
            return
        if self.manager.pending is not None:
            old_pending_id = self.manager.pending_target_id()
            if old_pending_id is not None:
                self.tracker.remove(old_pending_id)
        request_id = self._next_request_id
        self._next_request_id += 1
        self.manager.begin_acquisition(request_id, command.prompt, latest.id)
        request = GroundingRequest(request_id, command.prompt, latest)
        candidates = self.grounder.find(request.frame, request.prompt)
        result = GroundingResult(request.request_id, request.frame.id, tuple(candidates))
        selected = self.manager.receive_grounding_result(result)
        if selected is not None:
            self._stage_and_promote(selected)

    def _stage_and_promote(self, candidate: GroundingCandidate) -> None:
        pending = self.manager.pending
        if pending is None:
            return
        seed = self.frames.get(pending.source_frame_id)
        buffered = self.frames.after(pending.source_frame_id)
        if seed is None or buffered is None:
            newest = self.frames.latest
            if newest is None:
                self.manager.abort_pending("No frame is available to retry acquisition")
                return
            self.manager.update_seed_frame(pending.request_id, newest.id)
            request = GroundingRequest(pending.request_id, pending.prompt, newest)
            selected = self.manager.receive_grounding_result(
                GroundingResult(
                    request.request_id,
                    request.frame.id,
                    tuple(self.grounder.find(request.frame, request.prompt)),
                )
            )
            if selected is None:
                return
            candidate = selected
            seed = newest
            buffered = ()

        target_id = self.manager.pending_target_id()
        if target_id is None:
            return
        result = self.tracker.initialize(seed, candidate.box, target_id)
        if result.mask is None or result.status is TrackingStatus.LOST:
            self.manager.abort_pending("Tracker could not initialize the selected candidate")
            self.tracker.remove(target_id)
            return

        pending_result = result
        for frame in buffered:
            results = self.tracker.update(frame)
            self.manager.observe(results)
            pending_result = next(
                (item for item in results if item.target_id == target_id),
                pending_result,
            )
        old_target_id = self.manager.promote(pending_result)
        if self.manager.pending is not None:
            failed_target_id = self.manager.pending_target_id()
            self.manager.abort_pending("New target was lost before promotion")
            if failed_target_id is not None:
                self.tracker.remove(failed_target_id)
        if old_target_id is not None:
            self.tracker.remove(old_target_id)

    def _render(self, frame: Frame) -> RenderedFrame:
        active = self.manager.active
        mask = active.mask if active and active.frame_id == frame.id else None
        effected = self.effects.apply(frame.image, mask, self.manager.effect)
        snapshot = RuntimeSnapshot(
            frame_id=frame.id,
            active_target=active,
            pending_prompt=self.manager.pending.prompt if self.manager.pending else None,
            pending_candidates=self.manager.pending.candidates if self.manager.pending else (),
            pending_is_ambiguous=bool(self.manager.pending and self.manager.pending.ambiguous),
            effect=self.manager.effect,
            message=self.manager.message,
        )
        return self.hud.render(effected, snapshot)


def make_default_mock_pipeline() -> MockPipeline:
    candidates = (
        GroundingCandidate(BoundingBox(10, 14, 30, 48), 0.78, "person in red, left"),
        GroundingCandidate(BoundingBox(58, 13, 82, 50), 0.72, "person in red, right"),
    )
    return MockPipeline(
        camera=MockCamera(make_scene_frames(12)),
        recorder=MockAudioRecorder(),
        stt=MockSTT((
            "segment the person in red",
            "the one on the left",
            "make him blue",
            "blur him",
            "stop tracking",
        )),
        grounder=MockGrounder(candidates),
        tracker=MockTracker(),
    )
