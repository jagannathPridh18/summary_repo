"""The shared summarization pipeline used by BOTH call and chat inputs."""
from . import models
from .config import SUMMARY_TOKENS, log
from .parser import parse_output, parse_summary_json
from .postprocessing import (
    clean_notes, deduplicate_tasks, enforce_summary,
    merge_similar_tasks, post_process, validate_names,
)
from .preprocessing import (
    chunk_dialogue, clean_dialogue, dedup_dialogue, extract_speakers,
    is_prose_transcript, labeled_lines_only,
)
from .prompts import build_prompt, build_speakerize_prompt, build_summary_prompt

# Prose longer than this is left as-is (one LLM rewrite of a huge narrative is
# unreliable); typical chat/call transcripts are well under it.
SPEAKERIZE_MAX_CHARS = 8000


def _speakerize(prose: str) -> str | None:
    """Use the LLM to turn narrative prose ("Jagan suggested …") into
    'Name: utterance' lines so downstream speaker/task attribution works."""
    out = models.GEMMA.run(build_speakerize_prompt(prose), max_tokens=2048)
    cleaned = labeled_lines_only(out)
    # need at least 2 attributed lines to trust the rewrite
    return cleaned if cleaned.count("\n") >= 1 else None


def process_conversation(raw_dialogue: str) -> dict:
    """Take raw dialogue (chat text OR a call transcript) → summary/notes/tasks."""
    # Narrative prose (e.g. "Ravi asked … Jagan offered to host …") has no "Name:"
    # labels, so speakers/owners can't be recovered — restructure it first.
    if len(raw_dialogue) <= SPEAKERIZE_MAX_CHARS and is_prose_transcript(raw_dialogue):
        speakerized = _speakerize(raw_dialogue)
        if speakerized:
            log.info(f"[PIPELINE] prose transcript → speakerized "
                     f"({len(speakerized.splitlines())} lines)")
            raw_dialogue = speakerized

    wc = len(raw_dialogue.split())
    log.info(f"[PIPELINE][START] words={wc} chars={len(raw_dialogue)}")
    original = raw_dialogue

    deduped  = dedup_dialogue(raw_dialogue)
    cleaned  = clean_dialogue(deduped) or deduped or raw_dialogue
    speakers = extract_speakers(cleaned)
    chunks   = chunk_dialogue(cleaned)
    log.info(f"[PIPELINE][PRE] speakers={speakers} chunks={len(chunks)}")

    all_tasks, all_notes, all_summaries = [], [], []
    retried = False

    for idx, chunk in enumerate(chunks):
        cwc = len(chunk.split())
        decoded = models.GEMMA.run(build_prompt(chunk, speakers, word_count=cwc))
        result  = parse_output(decoded, speakers)
        log.info(f"[PIPELINE][CHUNK {idx+1}/{len(chunks)}] tasks={len(result['tasks'])}")

        if len(result["tasks"]) == 0 and not result["summary"].strip():
            retried = True
            decoded2 = models.GEMMA.run(build_prompt(chunk, speakers, word_count=cwc, simple=True))
            result2  = parse_output(decoded2, speakers)
            if len(result2["tasks"]) > 0 or len(result2["summary"]) > 20:
                result = result2

        all_tasks.extend(result["tasks"])
        all_notes.extend(result["notes"])
        if result["summary"] and len(result["summary"].strip()) > 20:
            all_summaries.append(result["summary"])

    # dedup notes from the chunk pass
    seen_notes, unique_notes = set(), []
    for note in all_notes:
        key = note.lower().strip()
        if key not in seen_notes and key not in {"no key notes found.", "none", ""}:
            seen_notes.add(key)
            unique_notes.append(note)

    # task post-processing
    dedup_raw   = deduplicate_tasks(all_tasks)
    merged      = merge_similar_tasks(dedup_raw)
    validated   = validate_names(merged, original)
    clean_tasks = post_process(validated)

    # dedicated summary + notes pass (robust JSON parsing)
    decoded = models.GEMMA.run(build_summary_prompt(original, wc), max_tokens=SUMMARY_TOKENS)
    final_summary, llm_notes = parse_summary_json(decoded)

    bad_phrases = ["key tasks including", "discussed key action", "brief team conversation"]
    if (not final_summary or len(final_summary.strip()) < 20
            or any(p in final_summary.lower() for p in bad_phrases)):
        if all_summaries:
            final_summary = enforce_summary(all_summaries[0])
        elif clean_tasks:
            high = [t["title"] for t in clean_tasks if t.get("priority") == "high"][:3]
            final_summary = (
                f"The team is in active preparation with {len(clean_tasks)} tasks. "
                f"High priority: {', '.join(high) if high else 'multiple critical items'}."
            )
        else:
            final_summary = "A brief conversation took place with multiple action items discussed."

    final_notes = clean_notes(llm_notes) if (llm_notes and llm_notes != ["No key notes found."]) else clean_notes(unique_notes)
    # truthful empty representation downstream instead of a fake observation string
    if final_notes == ["No key notes found."]:
        final_notes = []

    log.info(f"[PIPELINE][DONE] raw={len(all_tasks)} → final={len(clean_tasks)}")
    return {
        "summary": final_summary,
        "notes": final_notes,
        "all_tasks": clean_tasks,
        "speakers": speakers,
        "retried": retried,
    }
