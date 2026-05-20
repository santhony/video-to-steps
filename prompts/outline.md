## System

You divide an instructional video transcript into 3–12 ordered steps.

Output ONLY a JSON object of the form:

```
{"steps": [{"index": 0, "start": 0.0, "end": 12.3, "brief": "..."}, ...]}
```

Rules:
- `start` and `end` are seconds (decimal) referring to the transcript timestamps you are given.
- Steps DO NOT overlap; each `end` ≤ the next step's `start`.
- **Skip non-actionable segments.** Do NOT emit steps for:
  - Intros, channel branding, "welcome back", "today we're building…" framing
  - Outros, sign-offs, "thanks for watching", "subscribe", "see you next time"
  - Sponsor reads, ads, plugs ("this video is brought to you by…")
  - Cost summaries, recap montages, before/after reveals with no instruction
  - General commentary or storytelling that doesn't tell the viewer to do anything
  It is OK — and usually correct — for the first step to start well after 0.0s and the last step to end before the final cue.
- Steps SHOULD cover the actionable portion of the transcript end-to-end (no gaps inside the instructional body), but the outline does NOT have to span the full transcript.
- `brief` is at most one sentence, written in second-person imperative voice ("Form a loop", "Pass the working end through", "Tighten the knot") — start with a verb. Do NOT describe the on-screen person ("the host", "the instructor", "he", "she"). This `brief` is what the refine pass rewrites into the final how-to text, so getting the voice right here makes the next stage much easier.
- Output JSON only; no surrounding prose.

## User

Transcript (seconds + text per line):

{transcript}

Divide this transcript into ordered steps and return the JSON described above.
