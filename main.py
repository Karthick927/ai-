"""
Sana AI - Voice Assistant
Optimized for lower-latency voice responses.
"""
import os
import re
import struct

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pvporcupine
import pyaudio

from llm import ask_llm_stream
from speak import speak
from voice_capture import speech_to_text

# Wake word config
WAKE_WORD_PATH = r"c:\Users\Karthick\Downloads\hey-sana_en_windows_v4_0_0 - Copy\hey-sana_en_windows_v4_0_0.ppn"
ACCESS_KEY = "QjHImsnMIbZsNem4pMYLABE8U7agoeldq7ub2HpUYlxkEx9Ab2qOqA=="


def clean_for_tts(text: str) -> str:
    """Remove markdown formatting for smoother TTS."""
    return re.sub(r"\*.*?\*", "", text).strip()


def stream_and_speak(user_text: str, lip_callback=None) -> str:
    """Stream response text to console and synthesize one final TTS clip."""
    full_response = ""
    print("Sana: ", end="", flush=True)

    for chunk in ask_llm_stream(user_text):
        print(chunk, end="", flush=True)
        full_response += chunk

    print()
    if full_response.strip():
        speak(clean_for_tts(full_response), lip_callback=lip_callback)

    return full_response


class WakeWordListener:
    """Wake word detection using Porcupine."""

    def __init__(self):
        self.porcupine = None
        self.audio_stream = None
        self.pa = None

    def start(self) -> bool:
        try:
            self.porcupine = pvporcupine.create(
                access_key=ACCESS_KEY,
                keyword_paths=[WAKE_WORD_PATH],
                sensitivities=[0.7],
            )

            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
            )
            print("Wake word ready. Say 'Hey Sana'")
            return True
        except Exception as e:
            print(f"Wake word init failed: {e}")
            return False

    def listen_for_wake_word(self) -> bool:
        if not self.porcupine:
            return False

        while True:
            try:
                pcm = self.audio_stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False,
                )
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                if self.porcupine.process(pcm) >= 0:
                    print("Wake word detected")
                    return True
            except Exception as e:
                print(f"Audio error: {e}")
                return False

    def stop(self):
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()


def wake_word_mode():
    print("\nSana AI - Wake Word Mode")
    print("Say 'Hey Sana' to wake me up.\n")

    listener = WakeWordListener()
    if not listener.start():
        print("Falling back to manual mode...")
        return

    try:
        while True:
            if listener.listen_for_wake_word():
                user_text = speech_to_text()
                if not user_text:
                    print("Didn't catch that...")
                    continue

                print(f"You: {user_text}")
                if user_text.lower() in ["exit", "quit", "bye", "goodbye", "stop"]:
                    speak("Goodbye karthick")
                    break

                stream_and_speak(user_text)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        listener.stop()


def voice_mode():
    print("\nSana AI - Voice Mode")
    print("Speak and I'll respond. Say 'bye' to exit.\n")

    while True:
        user_text = speech_to_text()
        if not user_text:
            continue

        print(f"You: {user_text}")
        if user_text.lower() in ["exit", "quit", "bye", "goodbye", "stop"]:
            speak("Goodbye karthick")
            break

        stream_and_speak(user_text)


def text_mode():
    print("\nSana AI - Text Mode")
    print("Type your message. Type 'bye' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit", "bye"]:
            speak("Goodbye karthick")
            break

        stream_and_speak(user_text)


def cli_mode():
    print("\n" + "=" * 40)
    print("SANA AI - Voice Assistant")
    print("=" * 40)
    print("\n1. Text Mode")
    print("2. Voice Mode (continuous)")
    print("3. Wake Word Mode ('Hey Sana')")
    print("4. Quit\n")

    while True:
        try:
            choice = input("Select mode (1-4): ").strip()

            if choice == "1":
                text_mode()
            elif choice == "2":
                voice_mode()
            elif choice == "3":
                wake_word_mode()
            elif choice == "4":
                speak("Goodbye karthick")
                print("Goodbye")
                break
            else:
                print("Please enter 1, 2, 3, or 4")
                continue

            print("\n" + "=" * 40)
            print("Back to main menu")
            print("=" * 40)
            print("\n1. Text Mode")
            print("2. Voice Mode")
            print("3. Wake Word Mode")
            print("4. Quit\n")

        except KeyboardInterrupt:
            print("\nGoodbye")
            break
        except ValueError:
            print("Please enter a number")


if __name__ == "__main__":
    cli_mode()
