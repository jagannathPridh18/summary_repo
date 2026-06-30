"""
Voice recognition with SpeechBrain ECAPA embeddings.

  • SpeechBrainEmbedder — turn a waveform into a 192-d speaker embedding.
  • VoiceRegistry       — enrolled voiceprints (Name → embedding); match an
                          unknown embedding to a known person by cosine similarity.

Enrolled voices live in  voices/<Name>.wav  (one clean sample per person).
They are loaded at startup; new ones can be added via POST /voices/enroll.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import (
    DEVICE, EMBED_MODEL, RECOGNIZE_THRESHOLD, ROOT_DIR, SAMPLE_RATE, VOICES_DIR, log,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class SpeechBrainEmbedder:
    """Wraps speechbrain ECAPA-TDNN; embeds 16 kHz mono waveforms."""

    def __init__(self, model_id: str = EMBED_MODEL, device: str = DEVICE):
        from speechbrain.inference.speaker import EncoderClassifier

        log.info(f"[VOICEID] loading {model_id} on {device} ...")
        t0 = datetime.now()
        self.device = device
        self.encoder = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=str(ROOT_DIR / "summar_call_text" / "sb_ecapa"),
            run_opts={"device": device},
        )
        log.info(f"[VOICEID] ready in {(datetime.now()-t0).total_seconds():.1f}s")

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """wav: 1-D float32 numpy @ 16 kHz → L2-normalised 192-d embedding."""
        if wav.size < SAMPLE_RATE // 10:          # < 0.1s → too short to embed
            return np.zeros(192, dtype=np.float32)
        t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            emb = self.encoder.encode_batch(t.to(self.device)).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb) or 1e-9
        return (emb / norm).astype(np.float32)


class VoiceRegistry:
    """Known speakers: name → embedding. Matches unknown embeddings to names."""

    def __init__(self, embedder: SpeechBrainEmbedder, threshold: float = RECOGNIZE_THRESHOLD):
        self.embedder = embedder
        self.threshold = threshold
        self.voiceprints: dict[str, np.ndarray] = {}

    # ── enrollment ──────────────────────────────────────────────
    def load_dir(self, voices_dir: Path = VOICES_DIR) -> None:
        """Load every voices/<Name>.<ext> as an enrolled voiceprint."""
        import librosa
        exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
        count = 0
        for p in sorted(voices_dir.glob("*")):
            if p.suffix.lower() not in exts:
                continue
            try:
                wav, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
                self.voiceprints[p.stem] = self.embedder.embed(wav.astype(np.float32))
                count += 1
            except Exception as e:
                log.warning(f"[VOICEID] could not enroll {p.name}: {e}")
        log.info(f"[VOICEID] enrolled {count} voice(s): {list(self.voiceprints)}")

    def enroll(self, name: str, wav: np.ndarray) -> None:
        self.voiceprints[name] = self.embedder.embed(wav)
        log.info(f"[VOICEID] enrolled '{name}' (total={len(self.voiceprints)})")

    # ── recognition ─────────────────────────────────────────────
    def match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """Return (name, score) for the best enrolled match above threshold, else (None, score)."""
        if not self.voiceprints or not np.any(emb):
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, ref in self.voiceprints.items():
            s = _cosine(emb, ref)
            if s > best_score:
                best_name, best_score = name, s
        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score
