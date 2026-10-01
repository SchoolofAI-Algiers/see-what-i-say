"""
STARK VISION — JARVIS Voice Feedback System.

Provides asynchronous, non-blocking text-to-speech audio feedback for the
STARK VISION Mark LXXXV HUD using native Linux speech-dispatcher (spd-say)
or system speech synthesizers.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading

logger = logging.getLogger("stark_vision.speech.jarvis")


def speak_jarvis_async(
    text: str,
    rate: int = -5,
    pitch: int = -10,
    voice_type: str = "male1",
) -> None:
    """
    Speak text asynchronously in a background daemon thread using spd-say.

    Fails silently and gracefully if audio output or spd-say is unavailable,
    ensuring zero impact on video frame rates or application stability.

    Args:
        text: The phrase to speak.
        rate: Speech rate (-100 to 100, default: -5 for deliberate AI tone).
        pitch: Voice pitch (-100 to 100, default: -10 for deep British AI tone).
        voice_type: Voice type descriptor (default: 'male1').
    """

    def _worker() -> None:
        try:
            if shutil.which("spd-say"):
                subprocess.run(
                    [
                        "spd-say",
                        "-t",
                        voice_type,
                        "-p",
                        str(pitch),
                        "-r",
                        str(rate),
                        text,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=8.0,
                    check=False,
                )
            elif shutil.which("espeak-ng"):
                subprocess.run(
                    ["espeak-ng", "-v", "en-gb", "-p", "35", "-s", "145", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=8.0,
                    check=False,
                )
            elif shutil.which("espeak"):
                subprocess.run(
                    ["espeak", "-v", "en-uk", "-p", "35", "-s", "145", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=8.0,
                    check=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Speech synthesis failed or timed out: %s", exc)

    thread = threading.Thread(target=_worker, daemon=True, name="jarvis_tts")
    thread.start()


def play_startup_greeting() -> None:
    """
    Play the iconic JARVIS startup greeting upon Mark LXXXV HUD initialization.
    """
    greeting = (
        "JARVIS online. Mark 85 neural interface calibrated. "
        "All systems nominal. Welcome back, Mr. Stark."
    )
    speak_jarvis_async(greeting, rate=-4, pitch=-8)


def play_target_locked(label: str) -> None:
    """
    Announce target acquisition confirmation.
    """
    phrase = f"Target acquired: {label}."
    speak_jarvis_async(phrase, rate=0, pitch=-8)


def play_shutdown_farewell() -> None:
    """
    Play farewell confirmation on application shutdown.
    """
    farewell = "Powering down Mark 85 systems. Goodbye, Mr. Stark."
    speak_jarvis_async(farewell, rate=-5, pitch=-8)
