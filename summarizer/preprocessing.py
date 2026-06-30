"""Pre-processing: dedup repeated lines, strip greetings/timestamps, chunk."""
import re

from .config import CHUNK_OVERLAP, MAX_WORDS

GREETING_PHRASES = {
    "morning", "hi", "hello", "hey", "morning!", "hi!", "ready", "ready here",
    "hi all", "hi everyone", "good morning", "good morning!", "morning all",
    "morning everyone", "yeah lets go", "lets go",
}


def dedup_dialogue(raw: str) -> str:
    """Collapse repeated identical lines; keep extra speakers in [also: ...]."""
    seen, order = {}, []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, activity = line.partition(":")
        name, act_key = name.strip(), activity.strip().lower()
        if not act_key:
            continue
        if act_key not in seen:
            seen[act_key] = {"original": activity.strip(), "names": []}
            order.append(act_key)
        if name and name not in seen[act_key]["names"]:
            seen[act_key]["names"].append(name)

    deduped = []
    for key in order:
        entry = seen[key]
        names, act = entry["names"], entry["original"]
        if len(names) == 1:
            deduped.append(f"{names[0]}: {act}")
        elif names:
            deduped.append(f"{names[0]}: {act} [also: {', '.join(names[1:])}]")
        else:
            deduped.append(act)
    return "\n".join(deduped)


def clean_dialogue(raw: str) -> str:
    lines, cleaned, speaker = raw.strip().split("\n"), [], None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ts = re.match(r'^(.+?)(\d{1,2}:\d{2}\s*(?:AM|PM))$', line)
        if ts:
            speaker = ts.group(1).strip()
        else:
            if speaker:
                msg = line.lower().rstrip(".,!?")
                if msg not in GREETING_PHRASES and len(line.split()) > 1:
                    cleaned.append(f"{speaker}: {line}")
            else:
                cleaned.append(line)
    return "\n".join(cleaned)


def extract_speakers(text: str) -> str:
    names = []
    for line in text.split("\n"):
        m = re.match(r'^([^:\[]+):', line)
        if m:
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
    if not names:
        return "the participants"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def chunk_dialogue(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + MAX_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP
    return chunks


# ── narrative-prose detection / cleanup ────────────────────────────────────
# A "labelled" line looks like  "Name: ..."  (the format the pipeline expects).
_LABEL_LINE_RE = re.compile(r"^\s*[A-Za-z][\w .'-]{0,30}:\s*\S")
# A strict "Name:" label (1-3 capitalised words) — used to clean speakerizer output.
_NAME_LABEL_STRICT = re.compile(r"^[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2}:\s*\S")
# "Jagan suggested ...", "Rahul requested ..." → named speaker in prose.
_REPORTING_VERBS = (
    "said|asked|replied|mentioned|suggested|offered|requested|added|told|wanted|"
    "responded|confirmed|shared|started|joined|agreed|noted|explained|complained|"
    "reminded|promised|announced|proposed|warned|brought|pointed|insisted|asked"
)
_NAMED_ACTION_RE = re.compile(rf"\b[A-Z][a-z]+\b\s+(?:{_REPORTING_VERBS})\b")


def is_prose_transcript(text: str) -> bool:
    """True when the input is narrative prose that names speakers ("Jagan suggested …")
    rather than a labelled transcript ("Jagan: …"). Such text loses speaker attribution
    in the normal pipeline, so it should be speakerized first."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    labeled = sum(1 for ln in lines if _LABEL_LINE_RE.match(ln))
    if labeled >= max(2, 0.5 * len(lines)):
        return False  # already a labelled transcript — leave it alone
    return bool(_NAMED_ACTION_RE.search(text))


def labeled_lines_only(text: str) -> str:
    """Keep only clean 'Name: utterance' lines (drops any model preamble/commentary)."""
    return "\n".join(
        ln.strip() for ln in text.splitlines() if _NAME_LABEL_STRICT.match(ln.strip())
    )
