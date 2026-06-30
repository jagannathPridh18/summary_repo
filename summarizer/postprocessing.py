"""Post-processing: filter completed/noise tasks, dedup/merge, fix priorities, clean notes."""
import re
from difflib import SequenceMatcher

from .config import log

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

COMPLETED_SIGNALS = [
    "has been fixed", "has been deployed", "has been pushed", "has been merged",
    "has been resolved", "has been completed", "was fixed", "was deployed",
    "was pushed", "was merged", "was resolved", "is fixed", "is done",
    "is complete", "is resolved", "is deployed", "already fixed", "already done",
    "fixed and confirmed", "pushed for review", "pushed to main",
    "deployed to", "merged to main", "completed successfully",
    "smoke testing completed", "rollback complete", "service restored",
    "confirmed on my end", "confirmed working", "all done", "its done", "its fixed",
    "running clean", "fully resolved", "already resolved", "back up",
    "fully working", "is working", "are working", "is live", "is back up",
]
LOW_SIGNALS = [
    "discuss", "after standup", "after scrum", "update tracker",
    "post blockers", "split work", "coordinate", "api contract",
    "flag blockers", "update jira", "update ticket", "in jira",
]
HIGH_SIGNALS = [
    "production release", "release tonight", "release today",
    "blocker", "crash", "hotfix", "regression", "verify fix",
    "verify hotfix", "database migration", "fix login",
    "fix android", "fix crash", "qa verification", "security",
]
NOISE_TASKS = [
    "flag blockers", "post blockers", "update jira",
    "let's make", "lets make", "make this demo",
]
BANNED_NOTE_LABELS = {"updates:", "tasks:", "updates", "tasks", "]", "[", "", "update:", "task:"}
TASK_LIKE_RE = re.compile(
    r'assigned to|will (fix|deploy|update|verify|complete|push|review|send|check)'
    r'|needs to|TASK\[|title:|assigned_to:|due:|message:|priority:'
    r'|\bfix\b.{0,30}\bby\b|\bwill\b.{0,20}\bby\b',
    re.IGNORECASE,
)


def is_noise_task(task: dict) -> bool:
    return any(n in task.get("title", "").lower() for n in NOISE_TASKS)


def is_completed_task(task: dict) -> bool:
    text = (task.get("title", "") + " " + task.get("message", "")).lower()
    return any(s in text for s in COMPLETED_SIGNALS)


def correct_priority(task: dict) -> str:
    text = (task.get("title", "") + " " + task.get("message", "")).lower()
    for kw in HIGH_SIGNALS:
        if kw in text:
            return "high"
    for kw in LOW_SIGNALS:
        if kw in text:
            return "low"
    return task.get("priority", "medium")


# Collective assignees are always valid (not real people to validate).
COLLECTIVE_NAMES = {
    "all", "everyone", "everybody", "team", "the team", "all team", "all members",
    "anyone", "someone", "us", "we",
}


def validate_names(tasks: list, raw_dialogue: str) -> list:
    real_names: set[str] = set()
    # 1) speakers — names that appear as "Name:" at line start
    for line in raw_dialogue.split("\n"):
        if ":" in line:
            name = line.split(":")[0].strip().lower()
            if name:
                real_names.add(name)
                real_names.add(name.split()[0])
    # 2) addressed-but-silent people — a person delegated to ("Mike please …",
    #    "Priya, can you …") is a valid assignee even if they never speak. Accept
    #    any capitalised first-name-like token that actually occurs in the text.
    for tok in re.findall(r"\b([A-Z][a-z]{1,19})\b", raw_dialogue):
        real_names.add(tok.lower())

    for task in tasks:
        for field in ["assigned_to", "assigned_from"]:
            val = task.get(field, "")
            if not val or val.lower() in ["n/a", "unassigned", ""]:
                continue
            ok = []
            for name in re.split(r"[,&/]", val):
                name = name.strip()
                if not name:
                    continue
                n0 = name.lower().split()[0] if name else ""
                if (name.lower() in COLLECTIVE_NAMES or n0 in COLLECTIVE_NAMES
                        or n0 in real_names or name.lower() in real_names):
                    ok.append(name)
                else:
                    log.warning(f"[VALIDATE] hallucinated name removed: '{name}' from {field}")
            task[field] = ", ".join(ok) if ok else ("N/A" if field == "assigned_from" else "unassigned")

        af = task.get("assigned_from", "N/A").strip()
        if af != "N/A":
            parts = [p.strip() for p in af.split(",")]
            valid_parts = [p for p in parts if len(p) > 2 and not re.match(r"^[A-Z]$", p.strip())]
            if not valid_parts:
                task["assigned_from"] = "N/A"
            elif len(valid_parts) < len(parts):
                task["assigned_from"] = ", ".join(valid_parts)
    return tasks


def deduplicate_tasks(tasks: list) -> list:
    seen, unique = [], []
    for task in tasks:
        key = task["title"].lower().strip()
        if key not in seen:
            seen.append(key)
            unique.append(task)
    unique.sort(key=lambda t: PRIORITY_ORDER.get(t.get("priority", "medium").lower(), 1))
    return unique


def _ids(title: str) -> list:
    """Numeric identifiers in a title — 'Fix bug 13' → ['13']."""
    return sorted(re.findall(r"\d+", title or ""))


def merge_similar_tasks(tasks: list, threshold: float = 0.8) -> list:
    merged, used = [], set()
    for i, t1 in enumerate(tasks):
        if i in used:
            continue
        group = [t1]
        for j, t2 in enumerate(tasks):
            if j <= i or j in used:
                continue
            # Distinct numeric ids (bug 1 vs bug 13) are different tasks — never merge,
            # no matter how similar the surrounding text is.
            if _ids(t1["title"]) != _ids(t2["title"]):
                continue
            sim = SequenceMatcher(None, t1["title"].lower(), t2["title"].lower()).ratio()
            if sim >= threshold:
                group.append(t2)
                used.add(j)
        if len(group) > 1:
            all_to, all_from = set(), set()
            for t in group:
                for n in re.split(r"[,&/]", t.get("assigned_to", "")):
                    n = n.strip()
                    if n and n.lower() not in ["n/a", "unassigned", ""]:
                        all_to.add(n)
                for n in re.split(r"[,&/]", t.get("assigned_from", "")):
                    n = n.strip()
                    if n and n.lower() not in ["n/a", ""]:
                        all_from.add(n)
            best = min(group, key=lambda t: PRIORITY_ORDER.get(t.get("priority", "medium").lower(), 1))
            best["assigned_to"]   = ", ".join(sorted(all_to))   if all_to   else "unassigned"
            best["assigned_from"] = ", ".join(sorted(all_from)) if all_from else "N/A"
            merged.append(best)
        else:
            merged.append(t1)
    return merged


def enforce_summary(summary: str, max_sentences: int = 8) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
    unique = []
    for s in sentences:
        if not any(SequenceMatcher(None, s.lower(), u.lower()).ratio() > 0.65 for u in unique):
            unique.append(s)
    return " ".join(unique[:max_sentences]) if unique else summary


def clean_notes(notes: list[str]) -> list[str]:
    cleaned, skip = [], False
    for line in notes:
        s = line.strip()
        if s.lower() in BANNED_NOTE_LABELS:
            continue
        if re.search(r'TASK\[|title:|assigned_to:|due:|message:|priority:', line, re.IGNORECASE):
            skip = True
        if skip:
            if s == "]":
                skip = False
            continue
        # A note attributed to a speaker ("Mike: I will fix the bug") is a legitimate
        # observation — don't let the task-leak guard ("will fix"/"needs to") drop it.
        named = re.match(r'^[-•*\s]*\*{0,2}[A-Z][\w .]{0,25}:\s', line) is not None
        if TASK_LIKE_RE.search(line) and not s.startswith("**") and not named:
            continue
        line = re.sub(r'\s+by\s+[A-Z][a-z]+(?:[,\s]+(?:and\s+)?[A-Z][a-z]+)*\.?$', '.', line)
        line = re.sub(r'\.\.+', '.', line).strip()
        cleaned.append(line)
    return cleaned if cleaned else ["No key notes found."]


def post_process(tasks: list) -> list:
    pending = []
    for task in tasks:
        if is_noise_task(task) or is_completed_task(task):
            continue
        if task.get("assigned_to", "").lower() in ["unassigned", ""]:
            hint = re.search(r'(?:from|with)\s+([A-Z][a-z]+)', task.get("message", ""))
            if hint:
                task["assigned_to"] = hint.group(1)
        at = task.get("assigned_to", "").lower()
        if at in ["everyone", "all team", "the team", "team", "anyone", "someone", "all members"]:
            task["assigned_to"] = "All"
        pending.append(task)

    deduped = []
    for task in pending:
        is_dup = False
        for kept in deduped:
            t1, t2 = task["title"].lower(), kept["title"].lower()
            # substring match must NOT collapse different numbered tasks
            # ('fix bug 1' is a substring of 'fix bug 13' but they are distinct)
            same_ids = _ids(t1) == _ids(t2)
            if t1 == t2 or (same_ids and (t1 in t2 or t2 in t1)):
                if task["assigned_to"] and task["assigned_to"] not in kept["assigned_to"]:
                    kept["assigned_to"] += ", " + task["assigned_to"]
                is_dup = True
                break
        if not is_dup:
            deduped.append(task)

    for task in deduped:
        task["priority"] = correct_priority(task)
    deduped.sort(key=lambda t: PRIORITY_ORDER.get(t.get("priority", "medium").lower(), 1))
    return deduped
