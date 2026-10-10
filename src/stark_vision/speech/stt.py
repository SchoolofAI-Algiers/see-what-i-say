"""Local Whisper speech recognition implementing the SpeechRecognizer protocol."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from stark_vision.contracts import AudioBuffer, Transcript

WHISPER_RATE_HZ = 16_000


class SpeechToText:
    def __init__(self, model_name: str = "small.en", language: str = "en") -> None:
        import whisper  # lazy: heavy import

        self.language = language
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        samples = audio.samples
        if samples.size == 0:
            return Transcript("", self.language)
        if audio.sample_rate_hz != WHISPER_RATE_HZ:
            from math import gcd

            g = gcd(WHISPER_RATE_HZ, audio.sample_rate_hz)
            samples = resample_poly(samples, WHISPER_RATE_HZ // g, audio.sample_rate_hz // g)
        samples = samples.astype(np.float32)
        result = self.model.transcribe(samples, language=self.language, temperature=0, fp16=False)
        return Transcript(result["text"].strip(), self.language)
