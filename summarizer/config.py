"""
Configuration + base logging.

All tunables are read from environment variables so the same code runs
on different machines without edits.
"""
import logging
import os
from pathlib import Path

# Reduce CUDA fragmentation (must be set before torch initialises CUDA) — lets us
# coexist with other GPU processes on a shared card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── models ─────────────────────────────────────────────────────────────────
GEMMA_MODEL   = os.getenv("GEMMA_MODEL",   "google/gemma-4-E2B-it")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
DEVICE        = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN      = os.getenv("HF_TOKEN") or None

# ── generation / chunking ──────────────────────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1200"))   # task-extraction pass
SUMMARY_TOKENS = int(os.getenv("SUMMARY_TOKENS", "600"))    # summary/notes pass
MAX_WORDS      = int(os.getenv("MAX_WORDS", "400"))         # words per chunk
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_CHARS      = int(os.getenv("MAX_CHARS", "20000"))      # chat input ceiling

# ── diarization + voice recognition (calls) ────────────────────────────────
DIARIZE                = os.getenv("DIARIZE", "1") == "1"   # enable speaker separation on calls
EMBED_MODEL            = os.getenv("EMBED_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
VOICES_DIR             = ROOT_DIR / "voices"               # enrolled voiceprints (Name.wav)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
DIARIZE_DISTANCE       = float(os.getenv("DIARIZE_DISTANCE", "0.65"))   # cluster merge cutoff (1 - cosine)
RECOGNIZE_THRESHOLD    = float(os.getenv("RECOGNIZE_THRESHOLD", "0.45"))  # min cosine sim to name a speaker
SAMPLE_RATE            = 16000

# ── server ─────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8077"))                       # 8000 is often taken

# ── base logger (app.log + console) ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
# Quiet the very chatty HTTP/transfer logs from huggingface_hub/httpx.
for noisy in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("summarizer")
