"""
FastAPI app — TWO dedicated summarization endpoints.

  POST /summarize/call   CALL summarization  — upload audio → Whisper → summary
  POST /summarize/chat   CHAT summarization  — JSON dialogue text → summary
  GET  /health           service + model status

Both return the same SummarizeResponse (summary + notes + tasks).
Interactive docs: GET /docs
"""
import asyncio
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .config import (
    DEVICE, DIARIZE, GEMMA_MODEL, LOG_DIR, SAMPLE_RATE, WHISPER_MODEL, log,
)
from .diarization import diarize_and_label
from .logging_utils import log_interaction
from .pipeline import process_conversation
from .postprocessing import deduplicate_tasks
from .schemas import ChatRequest, SummarizeResponse, TaskItem

# Downstream service the chat summary is pushed to (override via env).
CHATBUCKET_URL = os.getenv(
    "CHATBUCKET_URL", "https://test-server.chatbucket.chat/v1/unread-summaries"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] loading models ...")
    models.load_all()
    log.info("[STARTUP] all models ready ✅")
    yield
    log.info("[SHUTDOWN] done")


app = FastAPI(
    title="Call & Chat Summarizer",
    version="1.0.0",
    description=(
        "Turn **voice calls** and **chat threads** into a structured "
        "**summary + notes + action items**.\n\n"
        "- `POST /summarize/call` — audio upload → Whisper transcription → summary\n"
        "- `POST /summarize/chat` — dialogue text → summary"
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["status"], summary="Service & model status")
def health():
    return {
        "status": "ok" if (models.GEMMA and models.WHISPER) else "loading",
        "gemma_model": GEMMA_MODEL,
        "whisper_model": WHISPER_MODEL,
        "device": DEVICE,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Logs — the saved input+output record of every request (logs/requests.jsonl)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/logs", tags=["status"], summary="Recent request logs (input + output)")
def get_logs(
    limit: int = 50,
    source: Optional[str] = None,        # filter: "chat" or "call"
    request_id: Optional[str] = None,    # fetch one request by id
):
    """Return the saved input+output records, newest first.
    Each record has: request_id, timestamp, source, elapsed_s, input, output, error."""
    path = LOG_DIR / "requests.jsonl"
    if not path.exists():
        return {"count": 0, "total": 0, "logs": []}

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if source and rec.get("source") != source:
                continue
            if request_id and rec.get("request_id") != request_id:
                continue
            entries.append(rec)

    total = len(entries)
    limit = max(1, min(limit, 1000))
    return {"count": min(limit, total), "total": total, "logs": entries[-limit:][::-1]}


# ═══════════════════════════════════════════════════════════════════════════
#  Shared: run the summary pipeline and build the response
# ═══════════════════════════════════════════════════════════════════════════
async def _summarize(
    dialogue_text: str,
    source: str,
    input_meta: dict,
    request_id: str,
    t0: datetime,
    transcript_text: Optional[str] = None,
) -> SummarizeResponse:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_conversation, dialogue_text)
    sorted_tasks = deduplicate_tasks(result["all_tasks"])
    elapsed = (datetime.now() - t0).total_seconds()

    output = {
        "summary": result["summary"],
        "notes": result["notes"],
        "tasks": sorted_tasks,
        "total_tasks": len(sorted_tasks),
        "speakers": result["speakers"],
        "retried": result["retried"],
    }
    log_interaction(request_id, source, input_meta, output, elapsed)  # logs INPUT + OUTPUT

    return SummarizeResponse(
        source=source,
        summary=result["summary"],
        notes=result["notes"],
        tasks=[TaskItem(**t) for t in sorted_tasks],
        total_tasks=len(sorted_tasks),
        speakers=result["speakers"],
        retried=result["retried"],
        transcript=transcript_text,
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  1) CALL summarization  — audio → transcript → summary
# ═══════════════════════════════════════════════════════════════════════════
@app.post(
    "/summarize/call",
    response_model=SummarizeResponse,
    tags=["summarize"],
    summary="Summarize a voice call",
    description=(
        "Upload an audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.opus`, ...). "
        "It is transcribed with Whisper, then summarized into a summary, notes, "
        "and action items. The `transcript` is included in the response.\n\n"
        "Send as `multipart/form-data` with a `file` field. Optionally pass "
        "`language` (ISO code such as `hi`, `ta`, `en`); omit it for auto-detect."
    ),
)
async def summarize_call(
    file: UploadFile = File(..., description="Audio file of the call"),
    language: Optional[str] = Form(None, description="ISO language code (e.g. hi, ta, en). Omit for auto-detect."),
    num_speakers: Optional[int] = Form(None, description="Number of speakers on the call, if known. Omit to auto-detect."),
):
    request_id = uuid.uuid4().hex[:12]
    t0 = datetime.now()
    loop = asyncio.get_event_loop()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    audio_bytes = await file.read()
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    lang = (language or "").strip().lower() or None
    if lang in ("auto", ""):
        lang = None

    diarized = DIARIZE and models.EMBEDDER is not None
    log.info(f"[CALL][{request_id}] file={file.filename} bytes={len(audio_bytes)} "
             f"lang={lang or 'auto'} diarize={diarized} num_speakers={num_speakers or 'auto'}")
    speakers_detected = None
    try:
        if diarized:
            tr = await loop.run_in_executor(None, models.WHISPER.transcribe_segments, tmp_path, lang)
            raw_text = tr["text"]
            dia = await loop.run_in_executor(
                None, diarize_and_label, tr["wav"], tr["segments"],
                models.EMBEDDER, models.REGISTRY, num_speakers,
            )
            labelled = dia["dialogue"]
            speakers_detected = dia["speakers"]
            # labelled transcript drives the pipeline (so tasks get speaker names);
            # fall back to plain text if diarization produced nothing.
            transcript_text = labelled or raw_text
            dialogue_text = labelled or raw_text
        else:
            tr = await loop.run_in_executor(None, models.WHISPER.transcribe, tmp_path, lang)
            transcript_text = dialogue_text = tr["text"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript_text:
        err = "Whisper returned an empty transcript"
        log_interaction(request_id, "call", {"filename": file.filename, "language": lang},
                        {}, (datetime.now() - t0).total_seconds(), error=err)
        raise HTTPException(status_code=422, detail=err)

    input_meta = {
        "filename": file.filename, "language": lang, "bytes": len(audio_bytes),
        "num_speakers": num_speakers, "speakers_detected": speakers_detected,
        "transcript": transcript_text, "chars": len(transcript_text),
    }
    return await _summarize(dialogue_text, "call", input_meta, request_id, t0,
                            transcript_text=transcript_text)


# ═══════════════════════════════════════════════════════════════════════════
#  2) CHAT summarization  — dialogue text → summary
# ═══════════════════════════════════════════════════════════════════════════
@app.post(
    "/summarize/chat",
    tags=["summarize"],
    summary="Summarize a chat conversation",
    description=(
        "Send a JSON body `{\"dialogue\": \"...\", \"conversationId\": \"...\", "
        "\"userId\": \"...\"}`. Speaker labels (`Name: message` per line) improve task "
        "extraction. Returns the summary/notes/tasks and forwards them to chatbucket.\n\n"
        "**Latency:** no size limit — large conversations are chunked and processed. "
        "Processing is synchronous and scales ~linearly (~13s per 1000 chars, e.g. ~260s "
        "for 20000 chars, ~520s for 40000). Clients MUST set a generous read timeout sized "
        "to the input length and should NOT auto-retry on timeout."
    ),
)
def summarize_chat(request: ChatRequest):
    request_id = uuid.uuid4().hex[:12]
    t0 = datetime.now()

    # No hard size cap — large conversations are chunked by the pipeline and
    # processed. Latency scales ~linearly with length (~13s / 1000 chars).
    log.info(f"[CHAT][{request_id}] words={len(request.dialogue.split())} "
             f"chars={len(request.dialogue)} conv={request.conversationId}")

    result = process_conversation(request.dialogue)
    sorted_tasks = deduplicate_tasks(result["all_tasks"])
    elapsed = (datetime.now() - t0).total_seconds()

    summary_data = {
        "source": "chat",
        "conversationId": request.conversationId,
        "userId": request.userId,
        "summary": result["summary"],
        "notes": result["notes"],
        "tasks": sorted_tasks,
        "total_tasks": len(sorted_tasks),
        "speakers": result["speakers"],
        "retried": result["retried"],
        "transcript": request.dialogue,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
    }
    log_interaction(request_id, "chat",
                    {"dialogue": request.dialogue, "chars": len(request.dialogue),
                     "conversationId": request.conversationId, "userId": request.userId},
                    summary_data, elapsed)

    # ── forward to chatbucket (a downstream outage must NOT drop the summary) ──
    forwarded = {"ok": False, "status_code": None, "response": None, "error": None}
    try:
        resp = requests.post(
            CHATBUCKET_URL,
            json=summary_data,
            headers={"accept": "*/*", "Content-Type": "application/json"},
            timeout=30,
        )
        forwarded["status_code"] = resp.status_code
        resp.raise_for_status()
        forwarded["ok"] = True
        forwarded["response"] = resp.json() if resp.content else None
    except requests.RequestException as e:
        forwarded["error"] = str(e)
        log.warning(f"[CHAT][{request_id}] forward to chatbucket failed: {e}")

    log.info(f"[CHAT][{request_id}] {elapsed:.1f}s tasks={len(sorted_tasks)} "
             f"forwarded={forwarded['ok']}")
    return {**summary_data, "forwarded": forwarded}


# ═══════════════════════════════════════════════════════════════════════════
#  Voice enrollment — register a person's voiceprint so calls name them
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/voices", tags=["voices"], summary="List enrolled voices")
def list_voices():
    if models.REGISTRY is None:
        raise HTTPException(status_code=503, detail="voice recognition is disabled (DIARIZE=0)")
    return {"enrolled": sorted(models.REGISTRY.voiceprints.keys())}


@app.post("/voices/enroll", tags=["voices"], summary="Enroll a speaker's voice")
async def enroll_voice(
    name: str = Form(..., description="Person's name (becomes their task-assignment label)"),
    file: UploadFile = File(..., description="A clean ~5-15s voice sample of this person"),
):
    """Register one person's voiceprint. After enrolling, calls where this voice
    appears get labelled with `name` instead of 'Speaker N', so tasks are assigned
    to them. The sample is also saved to voices/<name> for reload on restart."""
    import librosa

    if models.REGISTRY is None:
        raise HTTPException(status_code=503, detail="voice recognition is disabled (DIARIZE=0)")
    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="audio file is required")

    audio_bytes = await file.read()
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        wav, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        models.REGISTRY.enroll(name.strip(), wav.astype("float32"))
        # persist for restart
        from .config import VOICES_DIR
        (VOICES_DIR / f"{name.strip()}{suffix}").write_bytes(audio_bytes)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"enrolled": name.strip(), "total_voices": len(models.REGISTRY.voiceprints)}
