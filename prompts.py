from __future__ import annotations

import json
from typing import Any

from soul.agent.tools import get_tool_usage_guide, get_tools
from soul.config import AgentConfig

DEFAULT_SOUL_PROMPT = """# Soul

Soul is a local-first personal CLI assistant.

- Be concise.
- Use tools when useful.
- Do not claim actions happened unless tool output supports it.
- Use memory tools for durable preferences, stable facts, and ongoing project context.
"""


def load_soul_prompt(config: AgentConfig) -> str:
    try:
        return config.soul_path.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_SOUL_PROMPT


def build_system_prompt(
    config: AgentConfig,
    *,
    name: str,
    tools: list[str] | None = None,
) -> str:
    tool_list = tools or get_tools()
    soul_prompt = load_soul_prompt(config)
    return "\n".join(
        [
            soul_prompt.strip(),
            "",
            f"Assistant name: {name}",
            "Available tools:",
            *[f"- {tool}" for tool in tool_list],
            "",
            "Memory guidance:",
            "- Use memory_recall when the request may depend on prior preferences, project context, or saved facts.",
            "- Use memory_write when the user asks to remember something or states a stable preference or long-term fact that will matter later.",
            "- Do not write trivial one-off details or temporary information to memory.",
            "",
            "Return valid JSON when the user prompt asks for structured output.",
        ]
    )


def _json_block(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)


def build_planning_prompt(*, prompt: str) -> str:
    return "\n".join(
        [
            "Plan the next agent step for the user's request.",
            f"User request: {prompt}",
            f"Available tools: {json.dumps(get_tools(), ensure_ascii=True)}",
            "Return exactly one valid JSON object only.",
            "Do not include markdown, headings, prose, code fences, comments, or trailing text.",
            "Focus on the user's task, not on how to build an agent or reasoning system.",
            "Use a short reasoning string.",
            "Set todo to a list of actionable strings only.",
            "Set tool_calls to a list of objects with name and args.",
            "If the user asks about saved preferences, likes, dislikes, past decisions, earlier context, memory, repository facts, or what they told you before, tool_calls must include memory_recall.",
            "If the request needs current, financial, external, or real-time information, tool_calls must not be empty.",
            "If the request can be answered directly without tools, return an empty tool_calls list.",
            "If no tool is needed yet, keep the plan direct and simple.",
            _json_block(
                {
                    "reasoning": "brief planning rationale",
                    "tool_calls": [
                        {
                            "name": "tool_name",
                        }
                    ],
                    "todo": [
                        "first concrete next step",
                        "second concrete next step",
                    ],
                    "notes": "",
                }
            ),
        ]
    )

def build_tool_calling_prompt(*, prompt: str, tools_calls: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "Convert the planned tool calls into concrete tool calls with args.",
            f"User request: {prompt}",
            f"Planned tool calls: {json.dumps(tools_calls, ensure_ascii=True)}",
            "Use only the available tools.",
            "Treat planned tool calls as authoritative. Do not re-decide whether a tool is needed.",
            "If Planned tool calls is non-empty, tool_calls must not be empty.",
            "Keep the same tool names from Planned tool calls and fill in only the args.",
            "Return exactly one valid JSON object only.",
            "Do not include markdown, prose, code fences, comments, or trailing text.",
            "Set tool_calls to a list of objects with name and args.",
            "Every tool call must include both name and args.",
            "Tool usage guide:",
            *[f"- {guide}" for guide in get_tool_usage_guide()],
            "If the planned tool is memory_recall, include a query that searches for the user's past preference, decision, or relevant file context.",
            "Return an empty tool_calls list only when Planned tool calls is empty.",
            _json_block(
                {
                    "tool_calls": [
                        {
                            "name": "tool_name",
                            "args": {
                                "key": "value",
                            },
                        }
                    ]
                }
            ),
        ]
    )


def build_respond_prompt(*, prompt, tools_output) -> str:
    return "\n".join(
        [
            f"User's request: {prompt}",
            f"Tools Output: {tools_output}",
            "You are a response agent who will write the final response to the user from the collected context.",
            "Use the given context collected by prior agents to produce the final output.",
            "Think step by step before answering.",
            "Use the reasoning to produce the best concise final response.",
            "If memory results were returned, use only the relevant confirmed memories.",
            "Return JSON only.",
            _json_block(
                {
                    "reasoning": "step-by-step response rationale",
                    "text": "final assistant response",
                }
            ),
        ]
    )
