## System

You divide an instructional video transcript into 3–12 ordered steps.

Output ONLY a JSON object of the form:

```
{"steps": [{"index": 0, "start": 0.0, "end": 12.3, "brief": "..."}, ...]}
```

Rules:
- `start` and `end` are seconds (decimal) referring to the transcript timestamps you are given.
- Steps cover the full transcript: first step `start` is ≤ the first cue, last step `end` is ≥ the last cue.
- Steps DO NOT overlap; each `end` ≤ the next step's `start`.
- `brief` is at most one sentence, written in second-person imperative voice ("Form a loop", "Pass the working end through", "Tighten the knot") — start with a verb. Do NOT describe the on-screen person ("the host", "the instructor", "he", "she"). This `brief` is what the refine pass rewrites into the final how-to text, so getting the voice right here makes the next stage much easier.
- Output JSON only; no surrounding prose.

## User

Transcript (seconds + text per line):

{transcript}

Divide this transcript into ordered steps and return the JSON described above.
