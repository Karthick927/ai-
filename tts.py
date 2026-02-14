import asyncio
import inspect
import io
from typing import Callable, Optional

import edge_tts
import pygame
from edge_tts.exceptions import NoAudioReceived

pygame.mixer.init()

PRIMARY_VOICE = "en-IE-EmilyNeural"
FALLBACK_VOICES = [
    "en-US-AriaNeural",
    "en-US-JennyNeural",
]
SPEECH_RATE = "+5%"
SPEECH_VOLUME = "+0%"
SPEECH_PITCH = "+0Hz"


async def _emit_lip(
    lip_callback: Optional[Callable[[float], object]],
    value: float,
) -> None:
    if lip_callback is None:
        return
    maybe_awaitable = lip_callback(value)
    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


async def _stream_audio_and_word_boundaries(
    text: str,
    voice: str,
) -> tuple[bytes, list[float]]:
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=SPEECH_RATE,
        volume=SPEECH_VOLUME,
        pitch=SPEECH_PITCH,
        boundary="WordBoundary",  # Required for lip sync timing
        connect_timeout=5,
        receive_timeout=10,
    )
    audio_data = b""
    word_offsets_sec = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            # Edge offset is in 100-ns ticks.
            offset_ticks = chunk.get("offset")
            if isinstance(offset_ticks, int):
                word_offsets_sec.append(offset_ticks / 10_000_000.0)
    return audio_data, word_offsets_sec


async def _drive_lip_from_word_boundaries(
    word_offsets_sec: list[float],
    lip_callback: Optional[Callable[[float], object]],
) -> None:
    if lip_callback is None:
        return
    if not word_offsets_sec:
        while pygame.mixer.get_busy():
            await _emit_lip(lip_callback, 0.08)
            await asyncio.sleep(0.03)
        await _emit_lip(lip_callback, 0.0)
        return

    loop = asyncio.get_running_loop()
    last_index = 0
    start = loop.time()
    while pygame.mixer.get_busy():
        now = loop.time() - start
        while last_index + 1 < len(word_offsets_sec) and word_offsets_sec[last_index + 1] <= now:
            last_index += 1

        nearest = word_offsets_sec[last_index]
        delta = abs(now - nearest)
        if delta < 0.10:
            lip_value = max(0.2, 1.0 - (delta / 0.10))
        else:
            lip_value = 0.06
        await _emit_lip(lip_callback, lip_value)
        await asyncio.sleep(0.03)

    await _emit_lip(lip_callback, 0.0)


async def speak_realtime(
    text: str,
    lip_callback: Optional[Callable[[float], object]] = None,
):
    """Async TTS with fallback voices, safe failure handling, and optional lip callback."""
    last_error = None

    for voice in [PRIMARY_VOICE, *FALLBACK_VOICES]:
        try:
            audio_data, word_offsets_sec = await _stream_audio_and_word_boundaries(text, voice)
            if not audio_data:
                raise NoAudioReceived("No audio bytes returned from TTS stream.")

            sound = pygame.mixer.Sound(io.BytesIO(audio_data))
            sound.play()
            await _drive_lip_from_word_boundaries(word_offsets_sec, lip_callback)
            return
        except NoAudioReceived as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

    if last_error:
        print(f"[TTS warning] Could not synthesize speech: {last_error}")


def speak(text: str, lip_callback: Optional[Callable[[float], object]] = None):
    """Synchronous wrapper for main.py."""
    if not text or not text.strip():
        return
    try:
        asyncio.run(speak_realtime(text, lip_callback=lip_callback))
    except RuntimeError:
        # Handles event-loop conflicts on some Python environments.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(speak_realtime(text, lip_callback=lip_callback))
        finally:
            loop.close()


if __name__ == "__main__":
    speak("Hello karthick, I'm ready to talk!")
