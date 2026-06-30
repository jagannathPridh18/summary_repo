"""Parse model output: TASK[...] blocks, and the JSON summary/notes pass."""
import json
import re

_FIELD_LABELS = r'title|assigned_to|assigned_from|due|message|priority'


def parse_field(block: str, field: str) -> str:
    # Non-greedy: stop at the next field label (single-line blocks) or newline.
    m = re.search(
        rf'{field}:\s*(.+?)(?=\s*(?:{_FIELD_LABELS}):|\n|\]|$)',
        block, re.IGNORECASE,
    )
    val = m.group(1).strip().rstrip(".,;:") if m else ""

    if field == "due" and val.lower() in ["n/a", "", "none", "null"]:
        title_m = re.search(r'title:\s*(.+)',   block, re.IGNORECASE)
        msg_m   = re.search(r'message:\s*(.+)', block, re.IGNORECASE)
        search_text = (
            (title_m.group(1) if title_m else "") + " " +
            (msg_m.group(1)   if msg_m   else "")
        ).lower()
        date_match = re.search(
            r'(by\s+\w+day|before noon|eod|today|tomorrow|this week|next week|'
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'next\s+\w+day|end of\s+\w+)',
            search_text,
        )
        if date_match:
            val = date_match.group(1).strip().title()
    return val


def parse_output(decoded: str, speakers: str) -> dict:
    """Parse the task-extraction pass (TASK[...] blocks + optional summary/notes)."""
    full = "TASKS:\n" + decoded

    task_blocks = re.findall(r'TASK\[(.*?)\]', full, re.DOTALL)
    tasks = []
    for block in task_blocks:
        title         = parse_field(block, "title")
        assigned_to   = parse_field(block, "assigned_to")
        assigned_from = parse_field(block, "assigned_from")
        due           = parse_field(block, "due")
        message       = parse_field(block, "message")
        priority      = parse_field(block, "priority").lower()
        if title or assigned_to:
            if priority not in ["high", "medium", "low"]:
                priority = "medium"
            tasks.append({
                "title": title, "assigned_to": assigned_to,
                "assigned_from": assigned_from, "due": due,
                "message": message, "priority": priority,
            })

    summary = ""
    s_match = re.search(r'Summary:\s*(.+?)(?=\n\s*Notes:|\Z)', full, re.DOTALL)
    if s_match:
        summary = re.sub(r'\*+', '', s_match.group(1).strip().replace('\n', ' ')).strip()

    notes, current_theme = [], None
    n_match = re.search(r'Notes:\s*(.*?)(?=\Z)', full, re.DOTALL)
    if n_match:
        for line in n_match.group(1).strip().split('\n'):
            line = line.strip()
            if not line or line.lower() in ["none", "n/a"]:
                continue
            tm = re.match(r'\*{0,2}([^\*\n]+?)\*{0,2}:\s*$', line)
            if tm and len(line) < 60:
                current_theme = tm.group(1).strip()
                notes.append(f"**{current_theme}**")
            else:
                clean = re.sub(r'^[-•*]\s*', '', line).strip()
                clean = re.sub(r'\*+', '', clean).strip()
                if clean:
                    notes.append(f"  - {clean}" if current_theme else f"- {clean}")

    return {
        "summary": summary or "",
        "notes": notes or ["No key notes found."],
        "tasks": tasks,
    }


def parse_summary_json(raw: str) -> tuple[str, list[str]]:
    """Parse the summary-pass output. Prefers JSON; falls back to headers/prose."""
    # 1) Try to locate and parse a JSON object.
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            summary = str(obj.get("summary", "")).strip()
            notes_raw = obj.get("notes", []) or []
            if isinstance(notes_raw, str):
                notes_raw = [notes_raw]
            notes = [f"- {str(n).strip()}" for n in notes_raw if str(n).strip()]
            if summary:
                return summary, (notes or ["No key notes found."])
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    # 2) Truncated/partial JSON: pull fields out by regex even if it won't parse.
    sm = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if sm:
        summary = sm.group(1).encode().decode('unicode_escape', 'ignore').strip()
        notes = [
            f"- {n.encode().decode('unicode_escape', 'ignore').strip()}"
            for n in re.findall(r'"((?:[^"\\]|\\.)*)"', raw.split('"notes"', 1)[-1])
            if n.strip()
        ]
        if summary:
            return summary, (notes or ["No key notes found."])

    # 3) Fallback: split on a NOTES header (handles "SUMMARY ... NOTES ..." prose).
    text = re.sub(r'\*+', '', raw).strip()
    parts = re.split(r'(?i)\bnotes?\b\s*:?', text, maxsplit=1)
    summary = re.sub(r'(?i)^\s*summary\s*:?\s*', '', parts[0].strip()).strip()
    notes = []
    if len(parts) > 1:
        for seg in re.split(r'(?:\n|•|\s-\s|(?<=[.])\s{2,})', parts[1]):
            seg = re.sub(r'^[-•*\s]+', '', seg).strip()
            if seg and not re.match(r'(?i)^[a-z &]+:$', seg):
                notes.append(f"- {seg}")
    return summary, (notes or ["No key notes found."])
