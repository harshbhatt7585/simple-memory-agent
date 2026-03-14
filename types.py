from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class AgentEvent:
    kind: str
    title: str
    detail: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class RunResult:
    reply: str
    events: list[AgentEvent] = field(default_factory=list)
    iterations: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = ["AgentEvent", "RunResult"]
