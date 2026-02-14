import asyncio
import json
from pathlib import Path

import numpy as np
import vosk
from aiohttp import WSMsgType, web

from main import stream_and_speak
from voice_capture import model as vosk_model

BASE_DIR = Path(__file__).resolve().parent


def create_recognizer() -> vosk.KaldiRecognizer:
    recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    recognizer.SetWords(False)
    return recognizer


def compute_lip_value(pcm_bytes: bytes) -> float:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    normalized = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    return max(0.0, min(1.0, rms * 8.0))


async def run_ai_pipeline(ws: web.WebSocketResponse, user_text: str) -> None:
    """Use the same LLM+TTS path as main.py."""
    try:
        loop = asyncio.get_running_loop()

        def lip_callback(value: float) -> None:
            if ws.closed:
                return
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "lip", "value": float(value)}),
                loop,
            )

        await ws.send_json({"type": "status", "state": "thinking"})
        assistant_text = await asyncio.to_thread(stream_and_speak, user_text, lip_callback)
        await ws.send_json({"type": "assistant_final", "text": assistant_text})
        await ws.send_json({"type": "status", "state": "listening"})
    except Exception as exc:
        await ws.send_json({"type": "status", "state": "error"})
        await ws.send_json({"type": "error", "message": str(exc)})


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)

    recognizer = create_recognizer()
    listening = False
    last_partial = ""

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                event_type = payload.get("type")
                if event_type == "start_listen":
                    recognizer = create_recognizer()
                    listening = True
                    last_partial = ""
                    await ws.send_json({"type": "status", "state": "listening"})
                elif event_type == "stop_listen":
                    listening = False
                    final_text = json.loads(recognizer.FinalResult()).get("text", "").strip()
                    if final_text:
                        await ws.send_json({"type": "final", "text": final_text})
                        await run_ai_pipeline(ws, final_text)
                    await ws.send_json({"type": "status", "state": "idle"})

            elif msg.type == WSMsgType.BINARY:
                if not listening:
                    continue

                audio_bytes = msg.data
                await ws.send_json({"type": "lip", "value": compute_lip_value(audio_bytes)})

                if recognizer.AcceptWaveform(audio_bytes):
                    final_text = json.loads(recognizer.Result()).get("text", "").strip()
                    if final_text:
                        await ws.send_json({"type": "final", "text": final_text})
                        listening = False
                        await run_ai_pipeline(ws, final_text)
                        recognizer = create_recognizer()
                        listening = True
                    last_partial = ""
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip()
                    if partial and partial != last_partial:
                        last_partial = partial
                        await ws.send_json({"type": "partial", "text": partial})

            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        await ws.close()

    return ws


async def index_handler(_: web.Request) -> web.FileResponse:
    return web.FileResponse(BASE_DIR / "index.html")


async def file_handler(request: web.Request) -> web.StreamResponse:
    rel_path = request.match_info.get("path", "")
    if rel_path == "":
        return web.FileResponse(BASE_DIR / "index.html")

    candidate = (BASE_DIR / rel_path).resolve()
    if not str(candidate).startswith(str(BASE_DIR.resolve())):
        raise web.HTTPForbidden()
    if not candidate.exists() or not candidate.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(candidate)


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/", index_handler)
    app.router.add_get("/{path:.*}", file_handler)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=8000)
