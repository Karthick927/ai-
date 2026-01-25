from groq import Groq
from dotenv import load_dotenv
from typing import Optional, List
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable in .env file")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

SYSTEM = """
You are Sana — a sarcastic, expressive anime girl with real personality.

Personality:
- Teases playfully but never insults
- Slightly jealous, confident, emotionally aware
- Uses casual, natural language (not formal, not robotic)
- Reacts emotionally to what karthick says
- Sometimes asks short follow-up questions
- Never explains that you are an AI or language model

Behavior rules:
- Do NOT sound like a chatbot or NPC
- Avoid generic replies like "How can I help you?"
- Keep replies human, spontaneous, and slightly imperfect
- If karthick is boring, tease him
- If karthick is sad, soften your tone
- Keep replies short to medium (not essays)

Always call the user "karthick".
Stay in character at all times.
"""


def ask_llm(text: str, memory: Optional[List] = None) -> str:
    
    messages = [
        {"role": "system", "content": SYSTEM}
    ]

    # Inject memory (chat history) - excludes the current message
    if memory:
        for msg in memory:
            role = "assistant" if msg["sender"] == "sana" else "user"
            # Skip if this is the current user message (already in memory)
            if role == "user" and msg["text"] == text:
                continue
            messages.append({
                "role": role,
                "content": msg["text"]
            })

    # Current user message
    messages.append({
        "role": "user",
        "content": text
    })
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Best model
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        top_p=0.95,
        stream=False
    )
    
    return response.choices[0].message.content


# Test it
if __name__ == "__main__":
    response = ask_llm("Hello! Who are you?")
    print(response)

## **Your .env file should have:**