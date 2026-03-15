from __future__ import annotations

import json

from soul.agent.types import AgentEvent
from soul.config import AgentConfig


class ScratchpadStore:
    def __init__(self, config: AgentConfig) -> None:
        self._path = config.scratchpad_path

    def _ensure_ready(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    def append(self, event: AgentEvent) -> None:
        self._ensure_ready()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")

    def reset(self) -> None:
        self._ensure_ready()
        self._path.write_text("", encoding="utf-8")

    def recent(self, limit: int = 12) -> list[AgentEvent]:
        self._ensure_ready()
        events: list[AgentEvent] = []
        for line in self._path.read_text(encoding="utf-8").splitlines()[-limit:]:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = payload.get("kind", "planning")
            events.append(
                AgentEvent(
                    kind=kind if isinstance(kind, str) else "planning",
                    title=str(payload.get("title", "")),
                    detail=str(payload.get("detail", "")),
                    created_at=str(payload.get("created_at", "")),
                )
            )
        return events
