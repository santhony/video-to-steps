"""Shared prompt loading utilities for LLM passes.

Private module (leading underscore indicates project-internal use only).
"""

from __future__ import annotations

import re
from pathlib import Path


def load_system_user(prompt_path: Path) -> tuple[str, str]:
    """Load and parse a prompt template file.

    Returns:
        (system_prompt, user_template) — both are stripped strings.
        The system_prompt is the content before the '## User' heading;
        the user_template is the content after.

    Raises:
        RuntimeError: if the file does not contain exactly one '## User' heading.
    """
    text = prompt_path.read_text(encoding="utf-8")
    # Split on "## User" heading; everything before is system, after is user template.
    parts = re.split(r"^## User\s*$", text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise RuntimeError(f"{prompt_path} must contain a '## User' heading")
    sys_part = re.sub(r"^## System\s*$", "", parts[0], flags=re.MULTILINE).strip()
    user_part = parts[1].strip()
    return sys_part, user_part
