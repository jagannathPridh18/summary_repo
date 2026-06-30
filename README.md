# Call & Chat Summarizer

One service that turns **voice calls** and **chat threads** into a structured
**summary + notes + action items**, via two dedicated endpoints.

- **`POST /summarize/call`** → audio → [Whisper](https://hf.co/openai/whisper-large-v3) transcribes → **diarization + voice recognition** labels each speaker → summary pipeline
- **`POST /summarize/chat`** → dialogue text (JSON) → summary pipeline
- **Models** (loaded in-process, fp16 on GPU):
  - `google/gemma-4-E2B-it` — summary / notes / tasks
  - `openai/whisper-large-v3` — multilingual transcription
  - `speechbrain/spkrec-ecapa-voxceleb` — speaker voice embeddings (who is speaking)

### Calls: who is speaking?

A raw call is just audio with no speaker labels, so tasks couldn't be assigned to
people. The call path solves this in two steps:

1. **Diarization** — each Whisper segment is embedded with SpeechBrain ECAPA and
   clustered, so turns are grouped by voice → `Speaker 1 / Speaker 2 / …`.
2. **Voice recognition** — each cluster's voiceprint is matched against people you
   **enrolled** (`POST /voices/enroll`). A match becomes their real name, so the
   labelled transcript reads `Mike: …`, `Sarah: …` and tasks are assigned by name —
   exactly like chat. Unmatched voices stay `Speaker N`.

## Project layout

```
summar_call&text/
├── run.py                  # entry point  →  python run.py
├── requirements.txt
├── summar_call_text/       # virtual environment (created locally)
├── logs/
│   ├── app.log             # human-readable runtime log
│   └── requests.jsonl      # one JSON line per request: INPUT + OUTPUT
├── samples/                # example audio for testing
├── voices/                 # enrolled voiceprints  (Name.wav)  → speaker names on calls
└── summarizer/             # the package
    ├── config.py           # settings + logging (all env-overridable)
    ├── logging_utils.py    # input/output request logging
    ├── models.py           # Gemma-4 + Whisper + SpeechBrain wrappers (singletons)
    ├── voiceid.py          # ECAPA voice embeddings + enrolment/recognition registry
    ├── diarization.py      # cluster segments by voice → labelled transcript with names
    ├── preprocessing.py    # dedup / clean / chunk dialogue
    ├── prompts.py          # task-extraction + summary prompts
    ├── parser.py           # parse TASK[...] blocks + summary JSON
    ├── postprocessing.py   # filter / merge / prioritise tasks, clean notes
    ├── pipeline.py         # process_conversation()  (shared by call + chat)
    ├── schemas.py          # pydantic request/response models
    └── api.py              # FastAPI app + endpoints
```

## Setup

```bash
cd summar_call&text

# create the venv inside the folder
uv venv summar_call_text            #  or:  python -m venv summar_call_text
source summar_call_text/bin/activate

# install (PyTorch first — see requirements.txt header for the CUDA index)
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

## Run

```bash
source summar_call_text/bin/activate
python run.py                       # serves on 0.0.0.0:8077
```

First start downloads the models (~13 GB) and loads them on the GPU.
Check readiness:  `curl localhost:8077/health`

## Use

Two endpoints — one per input type (interactive docs at `GET /docs`):

```bash
# 1) CALL — POST /summarize/call : upload audio (multipart), optional language
curl -X POST localhost:8077/summarize/call \
     -F 'file=@samples/Hindi.mp3' -F 'language=hi'

# 2) CHAT — POST /summarize/chat : JSON body with the dialogue text
curl -X POST localhost:8077/summarize/chat \
     -H 'Content-Type: application/json' \
     -d '{"dialogue": "Sarah: Mike fix the login crash by EOD?\nMike: Sure, I will."}'
```

Optional on calls: pass `num_speakers=<n>` if you know how many people are on the
line (improves diarization); otherwise it auto-detects.

### Enroll voices (so calls show real names)

```bash
# register each person once with a clean ~5-15s voice sample
curl -X POST localhost:8077/voices/enroll -F 'name=Mike'  -F 'file=@mike.wav'
curl -X POST localhost:8077/voices/enroll -F 'name=Sarah' -F 'file=@sarah.wav'
curl localhost:8077/voices            # list enrolled names
```

You can also just drop `voices/Mike.wav`, `voices/Sarah.wav` into the folder before
starting the server — they're loaded automatically. Without enrollment, calls still
work but speakers are labelled `Speaker 1 / Speaker 2 / …`.

Both return the same shape:

```json
{
  "source": "call",
  "summary": "...",
  "notes": ["- ...", "- ..."],
  "tasks": [{"title": "...", "assigned_to": "...", "assigned_from": "...",
             "due": "...", "message": "...", "priority": "high"}],
  "total_tasks": 1,
  "speakers": "...",
  "retried": false,
  "transcript": "...",          // null for chat
  "request_id": "…",
  "timestamp": "…"
}
```

## Configuration (environment variables)

| Var | Default | Purpose |
|-----|---------|---------|
| `GEMMA_MODEL` | `google/gemma-4-E2B-it` | summarization model |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | transcription model |
| `DEVICE` | `cuda` if available | `cuda` / `cpu` |
| `PORT` | `8077` | HTTP port |
| `HF_TOKEN` | – | only if a gated model is used |
| `MAX_WORDS` / `CHUNK_OVERLAP` | `400` / `50` | dialogue chunking |

## Logs

Every `/summarize` request is recorded in `logs/requests.jsonl` with **both**
the input (filename/language/dialogue/transcript) and the output
(summary/notes/tasks) — a complete, replayable audit trail.
