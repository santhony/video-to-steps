"""VTT parsing and rolling-repeat dedup for YouTube auto-captions.

`parse_vtt` is a thin wrapper over `webvtt.read` that converts string
timestamps to float seconds and yields `Cue` instances in file order.

`dedupe_rolling` collapses YouTube's two-line karaoke-style auto-caption
pattern, where the same text appears across consecutive overlapping cues
because the player needs to redraw a "current line" each tick.
"""

from __future__ import annotations

from pathlib import Path

import webvtt

from .types import Cue


def _ts_to_seconds(ts: str) -> float:
    # webvtt-python timestamps look like "00:01:23.456" or "01:23:45.678".
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
    elif len(parts) == 2:
        h, m, rest = "0", parts[0], parts[1]
    else:
        raise ValueError(f"unrecognized VTT timestamp: {ts!r}")
    s, _, ms = rest.partition(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms or "0") / 1000.0)


def parse_vtt(path: str | Path) -> list[Cue]:
    """Parses a .vtt file into a list of Cue, in file/temporal order.

    Newlines inside cue text are collapsed to spaces so downstream consumers
    see a single line per cue.
    """
    path = Path(path)
    cues: list[Cue] = []
    for v in webvtt.read(str(path)):
        text = " ".join(v.text.splitlines()).strip()
        if not text:
            continue
        cues.append(Cue(start=_ts_to_seconds(v.start), end=_ts_to_seconds(v.end), text=text))
    return cues


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_rolling(cues: list[Cue]) -> list[Cue]:
    """Collapses YouTube rolling-repeat auto-captions into stable cues.

    Adjacent cues where one's normalized text is a prefix of the other's
    are merged: the longer text wins, and the merged cue's `end` is the
    later of the two `end`s.
    """
    if not cues:
        return []

    out: list[Cue] = [cues[0]]
    for cue in cues[1:]:
        last = out[-1]
        a, b = _norm(last.text), _norm(cue.text)
        if a == b or a.startswith(b) or b.startswith(a):
            # Same content; keep the longer text and extend the time range.
            keep_text = last.text if len(last.text) >= len(cue.text) else cue.text
            out[-1] = Cue(start=last.start, end=max(last.end, cue.end), text=keep_text)
        else:
            out.append(cue)
    return out
