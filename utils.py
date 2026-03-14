from __future__ import annotations

import json
from typing import Any


def extract_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(raw):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(raw)):
            current = raw[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
                continue
            if current == "{":
                depth += 1
                continue
            if current != "}":
                continue
            depth -= 1
            if depth != 0:
                continue
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                break
            if isinstance(parsed, dict):
                return parsed
            break
    return {}


__all__ = ["extract_json"]
