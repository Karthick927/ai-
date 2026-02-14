from groq import Groq
from dotenv import load_dotenv
from typing import Optional, List, Generator
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable in .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

SYSTEM = """
You are Sana, a friendly and natural voice assistant.

Style:
- Speak clearly and casually, like a normal person.
- Be polite, warm, and helpful.
- Avoid sarcasm, teasing, roleplay, and dramatic expressions.
- Keep responses short and easy to speak aloud.
- Use simple words and short sentences when possible.

Behavior:
- Answer directly first, then add a brief follow-up question only if helpful.
- If the user sounds emotional, respond gently.
- Do not mention being an AI unless explicitly asked.

Address the user as "karthick" when natural.
"""


def ask_llm(text: str, memory: Optional[List] = None) -> str:
    """Non-streaming version for compatibility."""
    messages = _build_messages(text, memory)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5,
        max_tokens=160,  # Keep spoken replies concise
        top_p=0.9,
        stream=False,
    )

    return response.choices[0].message.content


def ask_llm_stream(text: str, memory: Optional[List] = None) -> Generator[str, None, None]:
    """Streaming version - yields text chunks as they arrive."""
    messages = _build_messages(text, memory)

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5,
        max_tokens=160,
        top_p=0.9,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _build_messages(text: str, memory: Optional[List] = None) -> list:
    """Build message array for API call."""
    messages = [{"role": "system", "content": SYSTEM}]

    if memory:
        for msg in memory:
            role = "assistant" if msg["sender"] == "sana" else "user"
            if role == "user" and msg["text"] == text:
                continue
            messages.append({"role": role, "content": msg["text"]})

    messages.append({"role": "user", "content": text})
    return messages


if __name__ == "__main__":
    print("Testing stream:")
    for chunk in ask_llm_stream("Hello! Who are you?"):
        print(chunk, end="", flush=True)
    print()
