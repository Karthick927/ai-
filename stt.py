import vosk
import pyaudio
import json
import sys
import os

# Download model first from: https://alphacephei.com/vosk/models
# Extract it to a folder, e.g., "model" in the same directory

MODEL_PATH = "vosk-model-small-en-us-0.15"  # Path to your Vosk model folder

# Initialize Vosk model
if not os.path.exists(MODEL_PATH):
    print(f"Please download the model from https://alphacephei.com/vosk/models")
    print(f"Extract it and place in '{MODEL_PATH}' folder")
    sys.exit(1)

model = vosk.Model(MODEL_PATH)


def speech_to_text(timeout_seconds: float = 5.0) -> str:
    """
    Listen for speech and return the recognized text.
    Returns empty string if nothing recognized within timeout.
    """
    recognizer = vosk.KaldiRecognizer(model, 16000)
    
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=16000,
                      input=True,
                      frames_per_buffer=8192)
    stream.start_stream()
    
    print("🎤 Listening...")
    
    result_text = ""
    silence_frames = 0
    max_silence = int(16000 / 4096 * 1.5)  # ~1.5 seconds of silence to end
    heard_speech = False
    
    try:
        import time
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds and not heard_speech:
                break
                
            data = stream.read(4096, exception_on_overflow=False)
            
            if recognizer.AcceptWaveform(data):
                # Final result (complete sentence)
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                if text:
                    result_text = text
                    break
            else:
                # Partial result
                partial = json.loads(recognizer.PartialResult())
                text = partial.get('partial', '')
                if text:
                    heard_speech = True
                    print(f"⏳ {text}", end='\r')
                    silence_frames = 0
                else:
                    if heard_speech:
                        silence_frames += 1
                        if silence_frames > max_silence:
                            # Get final result after silence
                            final = json.loads(recognizer.FinalResult())
                            result_text = final.get('text', '')
                            break
                            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()
    
    if result_text:
        print(f"✅ {result_text}")
    
    return result_text


if __name__ == "__main__":
    print("🎤 Voice Capture Test - Press Ctrl+C to stop")
    print("-" * 50)
    
    while True:
        try:
            text = speech_to_text()
            if text:
                print(f"You said: {text}")
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            break
