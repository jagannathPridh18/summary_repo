"""Prompt builders for the task-extraction pass and the summary/notes pass."""


def build_prompt(dialogue: str, speakers: str, word_count: int = 0, simple: bool = False) -> list[dict]:
    if not simple:
        return [{"role": "user", "content": f"""You are an expert meeting analyst. Process this conversation between {speakers}.

Output in EXACTLY this order:

━━━━ 1. TASKS ━━━━
Extract EVERY pending action item. Read VERY carefully.

ASSIGNED_TO RULES (most important):
- assigned_to = the person who is ASKED or RESPONSIBLE to do the work
- "Sarah: Mike can you fix X?" → assigned_to = Mike (not Sarah)
- "Sarah: Everyone update Jira" → assigned_to = All
- "Mike: I will fix X by Wednesday" → assigned_to = Mike
- "Sarah: Jess please fix the bug" → assigned_to = Jess
- "everyone"/"anyone"/"someone"/"all"/"team" → assigned_to = All
- NEVER assign to the speaker just because they spoke — only if they said "I will do X"
- If unclear who will do it → assigned_to = unassigned

ASSIGNED_FROM RULES:
- assigned_from = who ASKED or DELEGATED the task
- "Sarah: Mike please fix X" → assigned_from = Sarah
- "Mike: I will fix X" (self-assigned) → assigned_from = N/A
- If no clear delegation → assigned_from = N/A

DUE DATE RULES:
- Extract ANY time reference: "by Wednesday", "today", "before noon", "next Friday", "EOD", "Tuesday", "this weekend"
- Only use N/A if absolutely no time is mentioned

OTHER RULES:
- Never create tasks for things already done (fixed, resolved, working, complete)
- One task per activity — no duplicates
- Priority: HIGH=production bugs/blockers/crashes, MEDIUM=features/reviews/testing, LOW=docs/coordination

TASK[
title: <specific action>
assigned_to: <person or people>
assigned_from: <delegator or N/A>
due: <deadline or N/A>
message: <detail>
priority: <high/medium/low>
]

━━━━ 2. SUMMARY ━━━━
Write EXACTLY 4 sentences — no more, no less:
Sentence 1: What phase or status the team is in right now
Sentence 2: What specific activities or work items were discussed
Sentence 3: What blockers or dependencies exist (or "No blockers identified")
Sentence 4: What the immediate next steps are

FORBIDDEN phrases: "the team discussed", "key action items", "productive meeting", "collaborative effort"
Use concrete nouns and specific verbs only.

━━━━ 3. NOTES ━━━━
Write observations as plain bullet points. NO theme headers. No names. No actions. No tasks.

Example:
- Multiple critical bugs are in active resolution ahead of the demo
- Database migration is blocked on DevOps approval

Stop after NOTES. Do not add extra text.

━━━━ EXAMPLE ━━━━
Chat:
Sarah: Gaurav can you fix the notification badge bug by tomorrow?
Gaurav: Sure will fix it by tomorrow EOD.
Sarah: Kumar please fix the login bug by EOD today.
Kumar: On it.
Sarah: Everyone update Jira by end of day.

TASKS:
TASK[
title: Fix notification badge bug
assigned_to: Gaurav
assigned_from: Sarah
due: Tomorrow EOD
message: fix the notification badge bug
priority: high
]
TASK[
title: Fix login bug
assigned_to: Kumar
assigned_from: Sarah
due: EOD today
message: fix the login bug
priority: high
]
TASK[
title: Update Jira
assigned_to: All
assigned_from: Sarah
due: EOD today
message: update Jira with current task status
priority: low
]

Summary: The team is in pre-release phase with a client demo scheduled for tomorrow. Frontend is fixing the login bug, and regression testing is scheduled for 5 PM. DevOps approval for database migration is the only blocker. Next steps are completing the login fix and running regression tests by EOD.

Notes:
- Notification badge bug has been confirmed resolved
- Login bug resolution is in progress

━━━━ NOW PROCESS ━━━━
[{word_count} words]

{dialogue}

TASKS:"""}]

    return [{"role": "user", "content": f"""Conversation ({word_count} words) between {speakers}.

TASKS:
TASK[title: <> assigned_to: <> assigned_from: <> due: <> message: <> priority: <high/medium/low>]

Summary: Write 4 sentences: phase, activities, blockers, next steps. No names.

Notes: Plain bullet points, no theme headers.

{dialogue}

TASKS:"""}]


def build_speakerize_prompt(prose: str) -> list[dict]:
    """Turn a narrative description of a conversation into a 'Name: utterance' transcript."""
    return [{"role": "user", "content": f"""Rewrite this narrative description of a group conversation into a clean speaker-labelled transcript.

Rules:
- Output ONLY lines in the form  Name: what they said  — one statement per line.
- Attribute every statement to the correct named person. Resolve pronouns (he/she/they) to the right name using context.
- Keep requests, offers and commitments explicit (who asked whom to do what, who volunteered).
- Do NOT invent names or facts. Do NOT add commentary, numbering, or headers.

Narrative:
{prose}

Transcript:"""}]


def build_summary_prompt(dialogue: str, word_count: int) -> list[dict]:
    length_instruction = (
        "2 sentences" if word_count < 100
        else "3-4 sentences" if word_count < 500
        else "5-6 sentences covering all themes"
    )
    return [{"role": "user", "content": f"""You are a meeting analyst. Read the conversation and reply with ONLY a JSON object — no markdown fences, no commentary before or after.

Format EXACTLY:
{{"summary": "<narrative, {length_instruction}, no speaker names, covering current phase / key activities / blockers / next steps>",
 "notes": ["<Name>: <observation>", "<Name>: <observation>", "..."]}}

Rules:
- summary: never say "Conversation between..." and never list names.
- notes: EACH note MUST begin with the name of the person who raised or is the source of that
  point, taken from the speaker labels in the conversation, e.g. "Sana: the payment gateway
  timeout still reproduces". If a point is shared by the whole group, use "Team". Keep them
  short observations — no action items, no tasks. Use [] if none.

Conversation ({word_count} words):
{dialogue}

JSON:"""}]
