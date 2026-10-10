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

    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

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
        with self._lock:
            chunks, self._chunks = self._chunks, []
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return AudioBuffer(samples.astype(np.float32), self.sample_rate_hz)
