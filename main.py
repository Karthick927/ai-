"""
Sana AI - Voice Assistant with 3D Avatar
Run app.py to start the web interface with 3D avatar
Or use this CLI version below
"""
import os

from matplotlib import text
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# rest of your imports
from stt import speech_to_text
from llm import ask_llm
from tts import speak_with_emotion
import re
import threading

def speak_async(text):
    threading.Thread(
        target=speak_with_emotion,
        args=(text,),
        daemon=True
    ).start()


def clean_for_tts(text: str) -> str:
    return re.sub(r"\*.*?\*", "", text).strip()

def cli_mode():
    """CLI mode for testing without web interface"""
    print("🎭 Sana AI - CLI Mode")
    print("Type 'exit', 'quit', or 'bye' to stop\n")

    choice  = int(input("Choose input method(1-Text,2-Speech and 3-quit) : "))
    while True:
        if choice == 1:
            user_text_1 = input("You: ")
            user_text = clean_for_tts(user_text_1)
            if user_text.lower() in ["exit", "quit", "bye"]:
                speak_with_emotion("Goodbye senpai!", emotion="sad")
                print("👋 Goodbye senpai!")
                break
            ai_text = ask_llm(user_text)
            speak_async(ai_text)
            print("Sana:", ai_text)
        elif choice == 2:
            user_text_1 = speech_to_text()
            user_text = clean_for_tts(user_text_1)
            if user_text.lower() in ["exit", "quit", "bye"]:
                speak_with_emotion("Goodbye senpai!", emotion="sad")
                print("👋 Goodbye senpai!")
                break
            ai_text = ask_llm(user_text)
            speak_async(ai_text)
            print("Sana:", ai_text)
        elif choice == 3:
            print("👋 Goodbye senpai!")
            speak_with_emotion("Goodbye senpai!", emotion="sad")
            break
        else:
            print("Invalid choice, please select 1, 2, or 3.")
            continue
if __name__ == "__main__":
    cli_mode()
