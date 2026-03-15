from __future__ import annotations

from abc import ABC, abstractmethod
from html.parser import HTMLParser
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from soul.agent.memory import MemoryStore
from soul.config import AgentConfig


def _parse_result_limit(value: Any, *, upper_bound: int) -> int:
    try:
        return max(1, min(int(value), upper_bound))
    except (TypeError, ValueError):
        return upper_bound


def _function_schema(
    *,
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class Tools(ABC):
    description = ""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        raise NotImplementedError


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_tag_stack.append(tag.lower())

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._ignored_tag_stack and self._ignored_tag_stack[-1] == lowered:
            self._ignored_tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._ignored_tag_stack:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


class _HTMLMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._text_extractor = _HTMLTextExtractor()
        self._title_parts: list[str] = []
        self._links: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._text_extractor.handle_starttag(tag, attrs)
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
        if lowered == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)

    def handle_endtag(self, tag: str) -> None:
        self._text_extractor.handle_endtag(tag)
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        self._text_extractor.handle_data(data)
        if self._in_title:
            text = " ".join(data.split())
            if text:
                self._title_parts.append(text)

    def extract(self) -> dict[str, Any]:
        title = " ".join(self._title_parts).strip()
        return {
            "title": title,
            "text": self._text_extractor.text(),
            "links": self._links,
        }


class MemoryRecallAgentTool(Tools):
    description = "Recall relevant saved memories and related details from workspace files."

    def __init__(self, config: AgentConfig) -> None:
        super().__init__("memory_recall")
        self._config = config
        self._store = MemoryStore(config)

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "tool": self.name, "error": "missing query"}

        max_results = _parse_result_limit(args.get("limit", self._config.search_limit), upper_bound=self._config.search_limit)

        matches = self._store.search(query=query, limit=max_results)
        file_matches = [] if matches else self._store.search_workspace(query=query, limit=max_results)
        return {
            "ok": True,
            "tool": self.name,
            "query": query,
            "memories": [entry.to_dict() for entry in matches],
            "memory_count": len(matches),
            "file_memories": [match.to_dict() for match in file_matches],
            "file_memory_count": len(file_matches),
        }

    def schema(self) -> dict[str, Any]:
        return _function_schema(
            name=self.name,
            description=self.description,
            properties={
                "query": {"type": "string", "description": "What memory to search for."},
                "limit": {"type": "integer", "description": "Maximum number of memories to return."},
            },
            required=["query"],
        )


class MemoryWriteAgentTool(Tools):
    description = "Write a note, preference, or outcome into local memory."

    def __init__(self, config: AgentConfig) -> None:
        super().__init__("memory_write")
        self._store = MemoryStore(config)

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", "")).strip()
        if not text:
            return {"ok": False, "tool": self.name, "error": "missing text"}

        kind = str(args.get("kind", "note")).strip().lower() or "note"
        raw_tags = args.get("tags", [])
        tags = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()] if isinstance(raw_tags, list) else []
        entry = self._store.append(text=text, kind=kind, tags=tags)
        return {
            "ok": True,
            "tool": self.name,
            "memory": entry.to_dict(),
        }

    def schema(self) -> dict[str, Any]:
        return _function_schema(
            name=self.name,
            description=self.description,
            properties={
                "text": {"type": "string", "description": "Memory text to store."},
                "kind": {"type": "string", "description": "Memory kind such as note or preference."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for later recall.",
                },
            },
            required=["text"],
        )


class WebSearchAgentTool(Tools):
    description = "Search the web with Tavily and return structured results."

    def __init__(self, config: AgentConfig) -> None:
        super().__init__("web_search")
        self._config = config

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "tool": self.name, "error": "missing query"}
        if not self._config.tavily_api_key:
            return {"ok": False, "tool": self.name, "error": "missing TAVILY_API_KEY", "query": query}

        max_results = _parse_result_limit(args.get("limit", self._config.search_limit), upper_bound=self._config.search_limit)

        topic = str(args.get("topic", "general")).strip().lower() or "general"
        if topic not in {"general", "news"}:
            topic = "general"

        payload = json.dumps(
            {
                "query": query,
                "topic": topic,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._config.tavily_api_key}",
                "Content-Type": "application/json",
                "User-Agent": self._config.user_agent,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._config.request_timeout_seconds) as response:
                body = response.read(self._config.max_document_bytes).decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "tool": self.name,
                "error": f"HTTP {exc.code} while searching Tavily",
                "query": query,
                "detail": detail[: self._config.max_excerpt_chars],
            }
        except URLError as exc:
            return {"ok": False, "tool": self.name, "error": f"network error while searching Tavily: {exc}", "query": query}

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "tool": self.name,
                "error": "invalid JSON returned by Tavily",
                "query": query,
            }

        raw_results = parsed.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[dict[str, Any]] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "snippet": str(item.get("content", "")).strip()[: self._config.max_excerpt_chars],
                }
            )

        response_payload = {
            "ok": True,
            "tool": self.name,
            "query": query,
            "topic": topic,
            "results": results,
            "result_count": len(results),
        }
        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip():
            response_payload["answer"] = answer.strip()[: self._config.max_excerpt_chars]
        return response_payload

    def schema(self) -> dict[str, Any]:
        return _function_schema(
            name=self.name,
            description=self.description,
            properties={
                "query": {"type": "string", "description": "Search query."},
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Whether the query is general web search or news-focused.",
                },
                "limit": {"type": "integer", "description": "Maximum number of results to return."},
            },
            required=["query"],
        )


class WebFetchAgentTool(Tools):
    description = "Fetch a web page and convert it into a readable excerpt."

    def __init__(self, config: AgentConfig) -> None:
        super().__init__("web_fetch")
        self._config = config

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url", "")).strip()
        if not url:
            return {"ok": False, "tool": self.name, "error": "missing url"}

        request = Request(url, headers={"User-Agent": self._config.user_agent}, method="GET")
        try:
            with urlopen(request, timeout=self._config.request_timeout_seconds) as response:
                raw = response.read(self._config.max_document_bytes)
                content_type = response.headers.get("Content-Type", "")
        except HTTPError as exc:
            return {"ok": False, "tool": self.name, "error": f"HTTP {exc.code} while fetching {url}"}
        except URLError as exc:
            return {"ok": False, "tool": self.name, "error": f"network error while fetching {url}: {exc}"}

        text = raw.decode("utf-8", errors="replace")
        if "html" in content_type.lower():
            parser = _HTMLMetadataParser()
            parser.feed(text)
            text = parser.extract()["text"]

        excerpt = " ".join(text.split())[: self._config.max_excerpt_chars]
        return {
            "ok": True,
            "tool": self.name,
            "url": url,
            "content_type": content_type,
            "excerpt": excerpt,
        }

    def schema(self) -> dict[str, Any]:
        return _function_schema(
            name=self.name,
            description=self.description,
            properties={
                "url": {"type": "string", "description": "URL to fetch."},
            },
            required=["url"],
        )


class HTMLPraserAgentTool(Tools):
    description = "Parse raw HTML into plain text and simple metadata."

    def __init__(self, config: AgentConfig) -> None:
        super().__init__("html_praser")
        self._config = config

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        html = str(args.get("html", ""))
        if not html.strip():
            return {"ok": False, "tool": self.name, "error": "missing html"}

        parser = _HTMLMetadataParser()
        parser.feed(html)
        parsed = parser.extract()
        text = " ".join(parsed["text"].split())[: self._config.max_excerpt_chars]
        links = parsed["links"][: self._config.search_limit]

        return {
            "ok": True,
            "tool": self.name,
            "title": parsed["title"],
            "text": text,
            "links": links,
            "link_count": len(parsed["links"]),
        }

    def schema(self) -> dict[str, Any]:
        return _function_schema(
            name=self.name,
            description=self.description,
            properties={
                "html": {"type": "string", "description": "Raw HTML to parse."},
            },
            required=["html"],
        )


def build_default_tools(config: AgentConfig) -> list[Tools]:
    return [
        MemoryRecallAgentTool(config),
        MemoryWriteAgentTool(config),
        WebSearchAgentTool(config),
        WebFetchAgentTool(config),
        HTMLPraserAgentTool(config),
    ]


def get_tools() -> list[str]:
    return [
        f"memory_recall: {MemoryRecallAgentTool.description}",
        f"memory_write: {MemoryWriteAgentTool.description}",
        f"web_search: {WebSearchAgentTool.description}",
        f"web_fetch: {WebFetchAgentTool.description}",
        f"html_praser: {HTMLPraserAgentTool.description}",
    ]


def get_tool_usage_guide() -> list[str]:
    return [
        "memory_recall args: {\"query\": string, \"limit\": integer optional}. Use for saved preferences, past decisions, repo facts, and local file memory.",
        "memory_write args: {\"text\": string, \"kind\": string optional, \"tags\": string[] optional}. Use to save durable preferences, decisions, and important facts.",
        "web_search args: {\"query\": string, \"topic\": \"general\"|\"news\" optional, \"limit\": integer optional}. Use for current facts, finance, news, and external information.",
        "web_fetch args: {\"url\": string}. Use to fetch and verify a specific page after search.",
        "html_praser args: {\"html\": string}. Use to parse raw HTML into readable text and links.",
    ]


def build_ollama_tools(tools: list[Tools]) -> list[dict[str, Any]]:
    return [tool.schema() for tool in tools]


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=True)


__all__ = [
    "Tools",
    "MemoryRecallAgentTool",
    "MemoryWriteAgentTool",
    "WebSearchAgentTool",
    "WebFetchAgentTool",
    "HTMLPraserAgentTool",
    "build_default_tools",
    "build_ollama_tools",
    "format_tool_result",
    "get_tools",
    "get_tool_usage_guide",
]
