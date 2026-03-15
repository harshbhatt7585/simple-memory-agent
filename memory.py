from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4

from soul.config import AgentConfig

TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9 ]")
MULTISPACE_RE = re.compile(r"\s+")

IGNORED_WORKSPACE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".soul",
}
IGNORED_WORKSPACE_SUFFIXES = {
    ".pyc",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".db",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _query_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in TOKEN_RE.findall(text.lower()):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _normalize_for_dedupe(text: str) -> str:
    cleaned = NON_ALNUM_SPACE_RE.sub("", text.lower())
    return MULTISPACE_RE.sub(" ", cleaned).strip()


@dataclass(slots=True)
class MemoryEntry:
    id: str
    kind: str
    text: str
    tags: list[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class IndexedMemoryRecord:
    entry: MemoryEntry
    source_path: str
    source_priority: int


@dataclass(slots=True)
class FileMemoryMatch:
    path: str
    excerpt: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "excerpt": self.excerpt,
            "score": self.score,
        }


class MemoryStore:
    def __init__(self, config: AgentConfig) -> None:
        self._path = config.memory_path
        self._daily_dir = config.daily_memory_dir
        self._index_path = config.memory_index_path
        self._legacy_path = self._path.with_name("memory.jsonl")
        self._legacy_markdown_path = self._path.with_name("memory.md")
        self._workspace_root = config.workspace_root
        self._max_excerpt_chars = config.max_excerpt_chars

    def _ensure_ready(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._daily_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def append(self, *, text: str, kind: str, tags: list[str]) -> MemoryEntry:
        entry = MemoryEntry(
            id=str(uuid4()),
            kind=kind,
            text=text,
            tags=tags,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._ensure_ready()
        with self._daily_path().open("a", encoding="utf-8") as handle:
            handle.write(entry.text.strip() + "\n\n")
        self._refresh_index(self._memory_records())
        return entry

    def reset(self) -> None:
        self._ensure_ready()
        self._path.write_text("", encoding="utf-8")
        for path in self._daily_dir.glob("*.md"):
            path.unlink()
        if self._index_path.exists():
            self._index_path.unlink()

    def all(self) -> list[MemoryEntry]:
        return [record.entry for record in self._memory_records()]

    def search(self, *, query: str, limit: int) -> list[MemoryEntry]:
        match_query = self._build_match_query(query)
        if not match_query:
            return []

        records = self._memory_records()
        if not records:
            return []

        self._refresh_index(records)
        records_by_rowid = {index: record for index, record in enumerate(records, start=1)}
        candidates = self._search_index(
            match_query=match_query,
            limit=max(limit * 8, limit),
            records_by_rowid=records_by_rowid,
        )
        ranked = self._rank_candidates(query=query, candidates=candidates)
        return self._dedupe_entries([entry for _, _, _, entry in ranked], limit)

    def search_workspace(self, *, query: str, limit: int) -> list[FileMemoryMatch]:
        query_terms = set(_query_terms(query))
        if not query_terms:
            return []

        matches: list[FileMemoryMatch] = []
        for path in self._iter_workspace_files():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            normalized = " ".join(text.split())
            if not normalized:
                continue

            score = len(query_terms & set(_query_terms(normalized)))
            if score == 0:
                continue

            excerpt = self._build_excerpt(normalized, query_terms)
            matches.append(
                FileMemoryMatch(
                    path=str(path.relative_to(self._workspace_root)),
                    excerpt=excerpt,
                    score=score,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def _memory_records(self) -> list[IndexedMemoryRecord]:
        self._ensure_ready()
        records: list[IndexedMemoryRecord] = []
        records.extend(self._records_for_plaintext(self._path, kind="curated", source_priority=6))
        for path in sorted(self._daily_dir.glob("*.md")):
            records.extend(self._records_for_plaintext(path, kind="daily", source_priority=3))
        records.extend(self._records_for_plaintext(self._legacy_markdown_path, kind="legacy_markdown", source_priority=2))
        for entry in self._load_legacy_jsonl_entries():
            priority = 5 if entry.kind == "preference" else 2
            records.append(
                IndexedMemoryRecord(
                    entry=entry,
                    source_path=str(self._legacy_path),
                    source_priority=priority,
                )
            )
        return records

    def _records_for_plaintext(self, path: Path, *, kind: str, source_priority: int) -> list[IndexedMemoryRecord]:
        return [
            IndexedMemoryRecord(
                entry=entry,
                source_path=str(path),
                source_priority=source_priority,
            )
            for entry in self._load_plaintext_entries(path, kind=kind)
        ]

    def _refresh_index(self, records: list[IndexedMemoryRecord]) -> None:
        with self._connect_index() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(
                    text,
                    kind UNINDEXED,
                    source_path UNINDEXED,
                    source_priority UNINDEXED,
                    tokenize = 'porter unicode61'
                )
                """
            )
            connection.execute("DELETE FROM memory_fts")
            connection.executemany(
                """
                INSERT INTO memory_fts(rowid, text, kind, source_path, source_priority)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        rowid,
                        record.entry.text,
                        record.entry.kind,
                        record.source_path,
                        record.source_priority,
                    )
                    for rowid, record in enumerate(records, start=1)
                ],
            )
            connection.commit()

    def _search_index(
        self,
        *,
        match_query: str,
        limit: int,
        records_by_rowid: dict[int, IndexedMemoryRecord],
    ) -> list[tuple[float, IndexedMemoryRecord]]:
        with self._connect_index() as connection:
            rows = connection.execute(
                """
                SELECT rowid, bm25(memory_fts) AS rank
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank ASC, CAST(source_priority AS INTEGER) DESC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()

        matches: list[tuple[float, IndexedMemoryRecord]] = []
        for row in rows:
            record = records_by_rowid.get(int(row["rowid"]))
            if record is None:
                continue
            matches.append((float(row["rank"]), record))
        return matches

    def _rank_candidates(
        self,
        *,
        query: str,
        candidates: list[tuple[float, IndexedMemoryRecord]],
    ) -> list[tuple[int, float, int, MemoryEntry]]:
        normalized_query = _normalize_text(query)
        ranked: list[tuple[int, float, int, MemoryEntry]] = []
        for rank, record in candidates:
            normalized_text = _normalize_text(record.entry.text)
            exact_phrase = int(bool(normalized_query) and normalized_query in normalized_text)
            ranked.append(
                (
                    exact_phrase,
                    rank,
                    record.source_priority,
                    record.entry,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1], -item[2]))
        return ranked

    def _build_match_query(self, query: str) -> str:
        terms = _query_terms(query)
        if not terms:
            return ""
        return " OR ".join(f'"{term}"' for term in terms)

    def _connect_index(self) -> sqlite3.Connection:
        self._ensure_ready()
        connection = sqlite3.connect(self._index_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _iter_workspace_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self._workspace_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_WORKSPACE_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in IGNORED_WORKSPACE_SUFFIXES:
                continue
            files.append(path)
        return files

    def _build_excerpt(self, text: str, query_terms: set[str]) -> str:
        words = text.split()
        for idx, word in enumerate(words):
            token = NON_ALNUM_RE.sub("", word.lower())
            if token in query_terms:
                start = max(0, idx - 20)
                end = min(len(words), idx + 20)
                excerpt = " ".join(words[start:end])
                return excerpt[: self._max_excerpt_chars]
        return text[: self._max_excerpt_chars]

    def _daily_path(self) -> Path:
        date_key = datetime.now().strftime("%Y-%m-%d")
        return self._daily_dir / f"{date_key}.md"

    def _load_plaintext_entries(self, path: Path, *, kind: str) -> list[MemoryEntry]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        entries: list[MemoryEntry] = []
        for block in re.split(r"\n\s*\n+", raw):
            entry = self._parse_plaintext_block(block, kind=kind)
            if entry is not None:
                entries.append(entry)
        return entries

    def _parse_plaintext_block(self, block: str, *, kind: str) -> MemoryEntry | None:
        content = block.strip()
        if not content:
            return None
        return MemoryEntry(
            id=str(uuid4()),
            kind=kind,
            text=content,
            tags=[],
            created_at="",
        )

    def _dedupe_entries(self, entries: list[MemoryEntry], limit: int) -> list[MemoryEntry]:
        deduped: list[MemoryEntry] = []
        seen: set[str] = set()
        for entry in entries:
            normalized = _normalize_for_dedupe(entry.text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(entry)
            if len(deduped) >= limit:
                break
        return deduped

    def _load_legacy_jsonl_entries(self) -> list[MemoryEntry]:
        if not self._legacy_path.exists():
            return []

        entries: list[MemoryEntry] = []
        for line in self._legacy_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            raw_tags = payload.get("tags", [])
            tags = self._coerce_tags(raw_tags)
            entries.append(
                MemoryEntry(
                    id=str(payload.get("id", "")),
                    kind=str(payload.get("kind", "note")).strip() or "note",
                    text=text,
                    tags=tags,
                    created_at=str(payload.get("created_at", "")),
                )
            )
        return entries

    def _coerce_tags(self, raw_tags: object) -> list[str]:
        if not isinstance(raw_tags, list):
            return []
        tags: list[str] = []
        for raw_tag in raw_tags:
            tag = str(raw_tag).strip()
            if tag:
                tags.append(tag)
        return tags


__all__ = ["FileMemoryMatch", "MemoryEntry", "MemoryStore"]
