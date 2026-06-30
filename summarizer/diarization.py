"""
Speaker diarization via embedding clustering (no gated models).

Pipeline for a call:
  1. Whisper gives timed segments  [{start, end, text}, ...]
  2. ECAPA embeds each segment's audio slice          (voiceid.SpeechBrainEmbedder)
  3. Agglomerative clustering groups segments by voice → Speaker 0/1/2 …
  4. Each cluster's mean embedding is matched to an enrolled name (voiceid.VoiceRegistry)
  5. Segments are merged into a labelled transcript:  "Mike: ...", "Speaker 2: ..."

The labelled transcript then flows into the normal summary pipeline, so tasks
get assigned to real people — exactly like chat.
"""
import numpy as np

from .config import DIARIZE_DISTANCE, SAMPLE_RATE, log
from .voiceid import SpeechBrainEmbedder, VoiceRegistry


def _slice(wav: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * SAMPLE_RATE))
    b = min(len(wav), int(end * SAMPLE_RATE))
    return wav[a:b]


def _cluster(embeddings: np.ndarray, num_speakers: int | None) -> np.ndarray:
    """Cluster row-wise embeddings → integer labels. Auto-detects count if num_speakers is None."""
    n = len(embeddings)
    if n == 1:
        return np.zeros(1, dtype=int)

    from sklearn.cluster import AgglomerativeClustering

    if num_speakers and num_speakers >= 1:
        model = AgglomerativeClustering(
            n_clusters=min(num_speakers, n), metric="cosine", linkage="average",
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=DIARIZE_DISTANCE,
            metric="cosine", linkage="average",
        )
    return model.fit_predict(embeddings)


def diarize_and_label(
    wav: np.ndarray,
    segments: list[dict],
    embedder: SpeechBrainEmbedder,
    registry: VoiceRegistry,
    num_speakers: int | None = None,
) -> dict:
    """
    wav: full call audio, 1-D float32 @ 16 kHz.
    segments: Whisper segments [{start, end, text}].
    Returns {"dialogue": labelled text, "speakers": [...], "segments": [...]}.
    """
    segments = [s for s in segments if (s.get("text") or "").strip()]
    if not segments:
        return {"dialogue": "", "speakers": [], "segments": []}

    # 1) embed each segment
    embs, valid = [], []
    for s in segments:
        e = embedder.embed(_slice(wav, s["start"], s["end"]))
        embs.append(e)
        valid.append(bool(np.any(e)))
    embs = np.vstack(embs)

    # 2) cluster the segments that produced a real embedding
    labels = np.full(len(segments), -1, dtype=int)
    valid_idx = [i for i, v in enumerate(valid) if v]
    if len(valid_idx) >= 1:
        sub_labels = _cluster(embs[valid_idx], num_speakers)
        for k, i in enumerate(valid_idx):
            labels[i] = int(sub_labels[k])
    # segments too short to embed inherit the previous segment's speaker
    last = 0
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = last
        else:
            last = labels[i]

    # 3) name each cluster from its mean embedding (best match wins; dupes fall back)
    cluster_ids = sorted(set(labels.tolist()))
    scored = []
    for cid in cluster_ids:
        members = embs[[i for i in range(len(labels)) if labels[i] == cid and valid[i]]]
        mean = members.mean(axis=0) if len(members) else np.zeros(192, dtype=np.float32)
        name, score = registry.match(mean)
        scored.append((cid, name, score))

    cluster_name: dict[int, str] = {}
    used: set[str] = set()
    # assign recognised names highest-confidence first, no duplicates
    for cid, name, score in sorted(scored, key=lambda x: -x[2]):
        if name and name not in used:
            cluster_name[cid] = name
            used.add(name)
    # remaining clusters get stable "Speaker N" labels
    n = 1
    for cid in cluster_ids:
        if cid not in cluster_name:
            cluster_name[cid] = f"Speaker {n}"
            n += 1

    # 4) merge consecutive same-speaker segments into a labelled transcript
    lines, seg_out = [], []
    cur_spk, cur_txt = None, []
    for s, lab in zip(segments, labels):
        spk = cluster_name[lab]
        txt = s["text"].strip()
        seg_out.append({"speaker": spk, "start": round(s["start"], 2),
                        "end": round(s["end"], 2), "text": txt})
        if spk == cur_spk:
            cur_txt.append(txt)
        else:
            if cur_spk is not None:
                lines.append(f"{cur_spk}: {' '.join(cur_txt)}")
            cur_spk, cur_txt = spk, [txt]
    if cur_spk is not None:
        lines.append(f"{cur_spk}: {' '.join(cur_txt)}")

    speakers = [cluster_name[c] for c in cluster_ids]
    log.info(f"[DIARIZE] segments={len(segments)} speakers={speakers}")
    return {"dialogue": "\n".join(lines), "speakers": speakers, "segments": seg_out}
