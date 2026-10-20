from __future__ import annotations
import re
from typing import Any

def to_snake_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub('[^a-z0-9]+', '_', text)
    return text.strip('_') or None
