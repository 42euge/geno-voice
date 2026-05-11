#!/usr/bin/env python3
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import config as cfg
from stt import get_engine as get_stt_engine
from tts import get_engine as get_tts_engine
from session.activation import ActivationTracker
from session.triggers import filter_noise, detect_triggers
from session.notes import SessionNoteProcessor

log = logging.getLogger("geno-voice")

stt_engine = None
tts_engine = None
activation_tracker = ActivationTracker()
session_notes = None


def _init_stt():
    global stt_engine
    stt_cfg = cfg.get("stt")
    engine_name = stt_cfg.get("engine", "whisper")
    if engine_name == "whisper":
        stt_engine = get_stt_engine("whisper", model_repo=stt_cfg.get("model"))
    elif engine_name == "gemma4":
        g = stt_cfg.get("gemma4", {})
        stt_engine = get_stt_engine("gemma4", **g)
    else:
        stt_engine = get_stt_engine(engine_name)
    log.info("STT engine: %s", stt_engine.name)


def _init_tts():
    global tts_engine
    tts_cfg = cfg.get("tts")
    tts_engine = get_tts_engine(
        tts_cfg.get("engine", "kokoro"),
        language=tts_cfg.get("language", "a"),
    )
    log.info("TTS engine: %s", tts_engine.name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_notes
    _init_stt()
    _init_tts()

    from datetime import datetime
    session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    session_dir = str(Path.home() / ".mindreflect" / "sessions" / session_id)
    session_notes = SessionNoteProcessor(session_dir=session_dir, model="gemma4:e2b")
    log.info("Session notes: %s", session_dir)

    log.info(
        "geno-voice ready on %s:%s",
        cfg.get("server", "host"),
        cfg.get("server", "port"),
    )
    yield
    if session_notes:
        session_notes._update_meta()
    log.info("Shutting down geno-voice")


app = FastAPI(title="geno-voice", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_engine": stt_engine.name if stt_engine else None,
        "tts_engine": tts_engine.name if tts_engine else None,
    }


@app.get("/voices")
async def voices():
    return {"voices": tts_engine.list_voices()}


@app.get("/config")
async def get_config():
    return cfg.load_config()


@app.post("/config")
async def update_config(request: Request):
    body = await request.json()
    new_cfg = cfg.update(body)

    stt_changed = "stt" in body
    tts_changed = "tts" in body
    if stt_changed:
        _init_stt()
    if tts_changed:
        _init_tts()

    return new_cfg


@app.post("/stt/transcribe")
async def stt_transcribe(request: Request):
    wav_bytes = await request.body()
    if not wav_bytes:
        return JSONResponse({"error": "empty body"}, status_code=400)

    loop = asyncio.get_event_loop()
    text, elapsed = await loop.run_in_executor(None, stt_engine.transcribe, wav_bytes)

    if text is None:
        return JSONResponse({"error": "transcription failed", "elapsed": elapsed}, status_code=500)

    activation_tracker.process_chunk(wav_bytes)

    filtered = filter_noise(text)
    if filtered is None:
        return {"text": "", "elapsed": elapsed, "filtered": True}

    trigger = detect_triggers(filtered)
    trigger_info = None
    if trigger.triggered:
        trigger_info = {
            "type": trigger.trigger_type.value,
            "hint": trigger.hint.value,
            "confidence": trigger.confidence,
        }

    return {"text": filtered, "elapsed": elapsed, "trigger": trigger_info}


@app.get("/activation")
async def get_activation():
    s = activation_tracker.state
    return {
        "score": round(s.score, 3),
        "fast_ema": round(s.fast_ema, 3),
        "slow_ema": round(s.slow_ema, 3),
        "trajectory": round(s.trajectory, 3),
        "is_elevated": s.is_elevated,
        "is_crying": s.is_crying,
        "chunks": s.chunks_processed,
    }


@app.post("/notes/process")
async def process_note(request: Request):
    """Process a transcript chunk in the background via Ollama tool use."""
    if not session_notes:
        return JSONResponse({"error": "no session"}, status_code=503)
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return {"status": "skipped"}

    session_notes.chunk_index += 1
    asyncio.create_task(session_notes._process_chunk(text, session_notes.chunk_index))
    return {"status": "processing", "chunk": session_notes.chunk_index}


@app.get("/notes/themes")
async def get_themes():
    """Get current session themes and summary."""
    if not session_notes:
        return {"themes": [], "summary": ""}
    return {
        "themes": session_notes.active_themes,
        "summary": session_notes.running_summary,
        "chunks": session_notes.chunk_index,
    }


@app.get("/cue/{cue_type}")
async def get_cue(cue_type: str):
    """Return a random backchannel cue WAV for the given type."""
    import random
    cue_dir = Path(__file__).parent / "session" / "cues" / cue_type
    if not cue_dir.exists():
        return JSONResponse({"error": f"unknown cue type: {cue_type}"}, status_code=404)
    wavs = list(cue_dir.glob("*.wav"))
    if not wavs:
        return JSONResponse({"error": "no cues available"}, status_code=404)
    chosen = random.choice(wavs)
    return Response(content=chosen.read_bytes(), media_type="audio/wav")


@app.post("/tts/synthesize")
async def tts_synthesize(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    tts_cfg = cfg.get("tts")
    voice = body.get("voice", tts_cfg.get("voice", "af_heart"))
    speed = body.get("speed", tts_cfg.get("speed", 1.0))

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, tts_engine.synthesize, text, voice, speed)

    return Response(content=wav_bytes, media_type="audio/wav")


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            import json
            body = json.loads(data)
            text = body.get("text", "")
            if not text:
                continue

            tts_cfg = cfg.get("tts")
            voice = body.get("voice", tts_cfg.get("voice", "af_heart"))
            speed = body.get("speed", tts_cfg.get("speed", 1.0))

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, lambda: list(tts_engine.stream(text, voice, speed)))

            for chunk in chunks:
                await ws.send_bytes(chunk)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("WebSocket error: %s", e)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    server_cfg = cfg.get("server")
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 5111)

    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(uv_config)

    def handle_signal(sig, frame):
        log.info("Received %s, shutting down...", signal.Signals(sig).name)
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    server.run()


if __name__ == "__main__":
    main()
