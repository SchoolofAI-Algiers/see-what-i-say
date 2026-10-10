"""Push-to-talk microphone recorder implementing the AudioRecorder protocol."""

from __future__ import annotations

import threading

import numpy as np

from stark_vision.contracts import AudioBuffer


class AudioRecorder:
    def __init__(self, sample_rate_hz: int = 16_000, device: int | str | None = None) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.device = device  # None -> system default input
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self.level = 0.0  # RMS of the latest audio chunk (for a live meter)

    def _callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
        with self._lock:
            self._chunks.append(chunk)

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("recording is already active")
        import sounddevice as sd  # lazy: keeps import side-effect free

        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> AudioBuffer:
        if self._stream is None:
            raise RuntimeError("recording has not started")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.level = 0.0
        with self._lock:
            chunks, self._chunks = self._chunks, []
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return AudioBuffer(samples.astype(np.float32), self.sample_rate_hz)


def list_input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
