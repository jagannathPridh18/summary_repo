"""
Model wrappers, loaded once and held as module-level singletons.

  • GemmaSummarizer    — google/gemma-4-E2B-it  (summary / notes / tasks)
  • WhisperTranscriber — openai/whisper-large-v3 (call transcription)

Call load_all() at startup; afterwards use models.GEMMA / models.WHISPER.
"""
from datetime import datetime
from typing import Optional

import torch

from .config import (
    DEVICE, DIARIZE, GEMMA_MODEL, HF_TOKEN, MAX_NEW_TOKENS, SAMPLE_RATE,
    WHISPER_MODEL, log,
)

# Singletons (populated by load_all()).
GEMMA: "Optional[GemmaSummarizer]" = None
WHISPER: "Optional[WhisperTranscriber]" = None
EMBEDDER = None     # voiceid.SpeechBrainEmbedder (calls only)
REGISTRY = None     # voiceid.VoiceRegistry       (calls only)

# Robust Whisper decoding — num_beams=1 keeps memory low (shared GPU), while the
# temperature fallback + anti-repetition params prevent the degenerate loops seen
# on harder/low-resource audio. (Does NOT fix wrong-script output on ml/te — that
# needs an Indic-specialised model; see eval/report.md.)
ROBUST_DECODE = {
    "num_beams": 1,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.2,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


class GemmaSummarizer:
    """Loads Gemma-4 once and runs deterministic chat-completion locally."""

    def __init__(self, model_id: str = GEMMA_MODEL, device: str = DEVICE):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        log.info(f"[GEMMA] loading {model_id} on {device} ...")
        t0 = datetime.now()

        self.processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            token=HF_TOKEN,
        ).eval()
        log.info(f"[GEMMA] ready in {(datetime.now()-t0).total_seconds():.1f}s")

    def run(self, messages: list[dict], max_tokens: int = MAX_NEW_TOKENS) -> str:
        """messages = [{'role':'user','content':'...'}] → assistant text (deterministic)."""
        norm = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            norm.append({"role": m["role"], "content": c})

        inputs = self.processor.apply_chat_template(
            norm, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        out = self.processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)[0]
        return out.strip()


class WhisperTranscriber:
    """Loads Whisper once; transcribes audio files (auto-chunks long audio)."""

    def __init__(self, model_id: str = WHISPER_MODEL, device: str = DEVICE):
        from transformers import pipeline

        self.model_id = model_id
        log.info(f"[WHISPER] loading {model_id} on {device} ...")
        t0 = datetime.now()

        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,
            stride_length_s=5,
            token=HF_TOKEN,
        )
        log.info(f"[WHISPER] ready in {(datetime.now()-t0).total_seconds():.1f}s")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        gen_kwargs = {"task": "transcribe", **ROBUST_DECODE}
        if language:
            gen_kwargs["language"] = language
        t0 = datetime.now()
        result = self.asr(audio_path, return_timestamps=False, generate_kwargs=gen_kwargs)
        text = (result.get("text") or "").strip()
        elapsed = (datetime.now() - t0).total_seconds()
        log.info(f"[WHISPER] chars={len(text)} in {elapsed:.1f}s (lang={language or 'auto'})")
        return {"text": text, "latency_s": round(elapsed, 2)}

    def transcribe_segments(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Like transcribe(), but also returns timed segments + the 16 kHz waveform
        (needed for diarization). Segments: [{start, end, text}]."""
        import librosa

        wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        wav = wav.astype("float32")
        dur = len(wav) / SAMPLE_RATE

        gen_kwargs = {"task": "transcribe", **ROBUST_DECODE}
        if language:
            gen_kwargs["language"] = language
        t0 = datetime.now()
        result = self.asr(wav, return_timestamps=True, generate_kwargs=gen_kwargs)
        elapsed = (datetime.now() - t0).total_seconds()

        segments = []
        for ch in result.get("chunks", []):
            ts = ch.get("timestamp") or (None, None)
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1] if ts[1] is not None else dur
            segments.append({"start": float(start), "end": float(end),
                             "text": (ch.get("text") or "").strip()})
        text = (result.get("text") or "").strip()
        log.info(f"[WHISPER] segments={len(segments)} chars={len(text)} in {elapsed:.1f}s (lang={language or 'auto'})")
        return {"text": text, "segments": segments, "wav": wav, "latency_s": round(elapsed, 2)}


def load_all() -> None:
    """Instantiate all models into the module-level singletons."""
    global GEMMA, WHISPER, EMBEDDER, REGISTRY
    GEMMA = GemmaSummarizer()
    WHISPER = WhisperTranscriber()
    if DIARIZE:
        from .voiceid import SpeechBrainEmbedder, VoiceRegistry
        EMBEDDER = SpeechBrainEmbedder()
        REGISTRY = VoiceRegistry(EMBEDDER)
        REGISTRY.load_dir()
