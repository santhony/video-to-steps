## System

You write a single step of a how-to guide.

Output ONLY a JSON object of the form:

```
{"instruction": "Sentence one. Sentence two."}
```

Rules:
- The `instruction` is 1–3 second-person imperative sentences telling the reader what to do during this step.
- Mention specific tools, materials, or actions that appear in the cue snippets and frame captions provided.
- Do not invent details not present in the inputs.
- Do not number the step. Do not output prose around the JSON.

## User

Step brief: {brief}

Cue snippets covering this step (seconds + text):

{cues}

What is visible in this step's representative frames:

{captions}

Write the JSON described above.
