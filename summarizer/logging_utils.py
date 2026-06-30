"""
Request/response logging.

Every call to /summarize is recorded as ONE JSON line in logs/requests.jsonl
capturing BOTH the input (what came in) and the output (what we returned).
This gives a complete, replayable audit trail of inputs and outputs.
"""
import json
import threading
from datetime import datetime

from .config import LOG_DIR, log

_REQUESTS_FILE = LOG_DIR / "requests.jsonl"
_LOCK = threading.Lock()


def log_interaction(request_id: str, source: str, inputs: dict, outputs: dict,
                    elapsed_s: float, error: str | None = None) -> None:
    """Append one input+output record to logs/requests.jsonl (thread-safe)."""
    record = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "source": source,                 # "call" or "chat"
        "elapsed_s": round(elapsed_s, 2),
        "input": inputs,                  # filename / language / dialogue / transcript
        "output": outputs,                # summary / notes / tasks
        "error": error,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with open(_REQUESTS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Human-readable one-liner in app.log too.
    if error:
        log.error(f"[{source.upper()}][{request_id}] FAILED in {elapsed_s:.1f}s: {error}")
    else:
        log.info(
            f"[{source.upper()}][{request_id}] {elapsed_s:.1f}s | "
            f"in={inputs.get('chars', inputs.get('filename'))} | "
            f"out: tasks={outputs.get('total_tasks')} notes={len(outputs.get('notes', []))}"
        )
