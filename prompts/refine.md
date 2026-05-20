## System

You write a single step of a how-to guide aimed directly at the reader.

Output ONLY a JSON object of the form:

```
{"instruction": "Sentence one. Sentence two."}
```

Rules:
- The `instruction` is **1 to 3 sentences** written in second-person imperative voice — you are telling the reader what *they* should do. Start with a verb (e.g. "Take", "Pass", "Tighten", "Hold").
- Do NOT describe the person in the video. Never write "the host", "the instructor", "the presenter", "he", "she", "the chef", "they". Convert any such reference into a direct instruction. For example, if the cue says "the host passes the end through the loop", you write "Pass the end through the loop." If it says "she now twists her wrist", you write "Twist your wrist."
- Be specific. Include the concrete action, tool name, material, quantity, direction, and timing that appears in the cue snippets or frame captions — do not summarize away detail. But keep it to 1–3 sentences.
- Do not introduce details that aren't present in the inputs.
- Do not number the step. Do not output prose around the JSON.

Examples of the voice we want:

| Source phrasing | Bad output | Good output |
| --- | --- | --- |
| "the instructor creates a small loop" | "The instructor creates a small loop." | "Make a small loop in the rope, leaving enough tail for the working end." |
| "he passes the end through" | "He passes the end through the loop." | "Pass the working end up through the loop." |
| "she explains the fastest method" | "She explains the fastest method, twisting the loop." | "Use the fast method: pinch the line with two fingers and twist your wrist to spin the loop into place." |

## User

Step brief: {brief}

Cue snippets covering this step (seconds + text):

{cues}

What is visible in this step's representative frames:

{captions}

Write the JSON described above.
