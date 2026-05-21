"""Speech-to-text routes: a batch POST (whole clip, then transcribe -- see
transcription.py) and a streaming WebSocket (audio chunks in, live
transcript deltas out -- see realtime_stt.py) that shows the user text
while they're still speaking instead of only after they stop.
"""

import asyncio
import contextlib
import json
import uuid
import wave
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logger import get_logger
from app.core.task_bridge import wait_and_cancel_rest
from app.services.realtime_stt import INPUT_SAMPLE_RATE, RealtimeApiTranscriber, StreamingTranscriber
from app.services.transcription import transcribe_audio

logger = get_logger(__name__)
router = APIRouter()

_DEBUG_CAPTURE_DIR = Path(__file__).resolve().parent.parent.parent / "tmp"


@router.post("/transcribe")
async def transcribe(request: Request) -> dict:
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio body")

    content_type = request.headers.get("content-type", "audio/webm")
    if "webm" in content_type:
        filename = "audio.webm"
    elif "wav" in content_type:
        filename = "audio.wav"
    else:
        filename = "audio.mp4"

    try:
        text = await transcribe_audio(audio, filename=filename)
    except Exception as exc:
        logger.error("transcription failed: %s", exc, extra={"node_name": "transcribe"}, exc_info=True)
        raise HTTPException(status_code=502, detail="transcription failed") from exc

    logger.info("transcribed", extra={"node_name": "transcribe"})
    return {"text": text}


async def _relay_browser_audio(
    websocket: WebSocket,
    transcriber: StreamingTranscriber,
    capture_sink: bytearray | None = None,
) -> None:
    """Reads raw PCM16 binary frames from the browser and forwards them to
    the transcriber, until either an explicit {"type":"stop"} control
    message or a disconnect. Manual receive() loop (not iter_bytes()) since
    this socket carries both binary audio frames and one JSON control
    message -- iter_bytes() assumes bytes-only and breaks on a text frame.

    If capture_sink is provided (debug only), every received audio frame is
    also appended to it so the raw browser audio can be written to a WAV.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))

        audio = message.get("bytes")
        if audio is not None:
            if capture_sink is not None:
                capture_sink.extend(audio)
            await transcriber.send_audio(audio)
            continue

        text = message.get("text")
        if text is not None:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("type") == "stop":
                return


def _write_debug_capture(pcm: bytes, session_id: str) -> None:
    _DEBUG_CAPTURE_DIR.mkdir(exist_ok=True)
    path = _DEBUG_CAPTURE_DIR / f"stt_capture_{session_id}.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(INPUT_SAMPLE_RATE)
        wav_file.writeframes(pcm)
    logger.info(
        "stt debug capture written: %s (%d bytes)",
        path,
        len(pcm),
        extra={"node_name": "transcribe_ws", "session_id": session_id},
    )


async def _relay_transcripts(websocket: WebSocket, transcriber: StreamingTranscriber) -> None:
    async for event in transcriber.events():
        await websocket.send_json({"type": event.kind, "text": event.text})


async def _run_transcription(
    websocket: WebSocket,
    transcriber: StreamingTranscriber,
    session_id: str,
    capture_sink: bytearray | None = None,
) -> None:
    uplink = asyncio.create_task(
        _relay_browser_audio(websocket, transcriber, capture_sink), name=f"stt-uplink-{session_id}"
    )
    downlink = asyncio.create_task(_relay_transcripts(websocket, transcriber), name=f"stt-downlink-{session_id}")
    # FIRST_COMPLETED, not FIRST_EXCEPTION: an explicit "stop" ends the
    # uplink task with no exception, and that alone should end the turn
    # (mirrors the pipeline-vs-watcher pairing in voice_routes.py, not its
    # producer-vs-consumer pairing, which intentionally waits for both).
    await wait_and_cancel_rest((uplink, downlink), asyncio.FIRST_COMPLETED)


@router.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid.uuid4().hex
    log_extra = {"node_name": "transcribe_ws", "session_id": session_id}

    capture_sink: bytearray | None = bytearray() if settings.stt_debug_capture else None

    logger.info("streaming transcription session started", extra=log_extra)
    try:
        async with RealtimeApiTranscriber() as transcriber:
            await _run_transcription(websocket, transcriber, session_id, capture_sink)
        await websocket.send_json({"type": "done"})
        logger.info("streaming transcription session completed", extra=log_extra)
    except WebSocketDisconnect:
        logger.info("client disconnected mid-session", extra=log_extra)
    except Exception as exc:
        logger.error("streaming transcription session failed: %s", exc, extra=log_extra, exc_info=True)
        await _safe_send_error(websocket, str(exc))
    finally:
        if capture_sink:
            with contextlib.suppress(Exception):
                _write_debug_capture(bytes(capture_sink), session_id)
        await _safe_close(websocket)


async def _safe_send_error(websocket: WebSocket, detail: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "detail": detail})


async def _safe_close(websocket: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await websocket.close()
