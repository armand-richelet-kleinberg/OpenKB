"""Chat messages <-> one prompt for a CLI, and the CLI's answer back into a chat reply.

An agent CLI driven as a model receives one text and returns one text. This module
keeps OpenKB's message-based calls working on top of that:

* ``render``: system prompt, conversation, tool results and the tool catalogue become
  one prompt (the system part separately, when the CLI accepts a system prompt file).
* Tool calls are emulated with a small JSON protocol the model is told to follow; the
  reply is parsed back into OpenAI-shaped ``tool_calls`` so the agents SDK runs the tool
  exactly as it would with a native function-calling endpoint.
* ``response_format: json_object`` becomes an instruction plus a tolerant extraction.

Nothing here runs a process; it is pure text in, text out, and fully unit-tested.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

TOOL_PROTOCOL = """\
## Tools
You may call the tools listed below. To call one or more tools, reply with ONLY this JSON
object and nothing else (no prose, no code fence):
{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments matching its parameters>}}]}
When you have what you need, reply with your final answer as plain text instead.
Never invent a tool result: results arrive in the next message as [tool result ...].
"""

JSON_ONLY = "Reply with one valid JSON object only: no prose before or after it, no code fence."


@dataclass
class Rendered:
    system: str
    prompt: str
    tool_names: set[str] = field(default_factory=set)
    wants_json: bool = False


def _text(content: Any) -> str:
    """Flatten OpenAI content (string or list of parts) to text. Images cannot travel through
    a text CLI; they are named so the model knows something was there."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, dict):
            kind = part.get("type")
            if kind in ("text", "input_text", "output_text"):
                parts.append(str(part.get("text", "")))
            elif kind in ("image_url", "input_image", "image"):
                parts.append("[image omitted: this CLI backend receives text only]")
            elif "text" in part:
                parts.append(str(part["text"]))
        else:
            parts.append(str(part))
    return "\n".join(p for p in parts if p)


def _tool_catalogue(tools: list[dict] | None) -> tuple[str, set[str]]:
    if not tools:
        return "", set()
    lines, names = [TOOL_PROTOCOL, "Available tools:"], set()
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = str(fn.get("name", "")).strip()
        if not name:
            continue
        names.add(name)
        params = json.dumps(fn.get("parameters") or {}, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"- {name}: {str(fn.get('description', '')).strip()}\n"
            f"  parameters (JSON schema): {params}"
        )
    return "\n".join(lines), names


def render(
    messages: list[dict],
    tools: list[dict] | None = None,
    response_format: Any = None,
    system_inline: bool = False,
) -> Rendered:
    """One prompt from a message list. ``system_inline`` folds the system text into the prompt
    for CLIs that have no system-prompt option."""
    system_parts: list[str] = []
    convo: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        if role in ("system", "developer"):
            system_parts.append(_text(message.get("content")))
        elif role == "assistant":
            text = _text(message.get("content"))
            if text:
                convo.append(f"[assistant]\n{text}")
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                convo.append(
                    f"[assistant called tool {fn.get('name', '')} id={call.get('id', '')}]\n"
                    f"{fn.get('arguments', '')}"
                )
        elif role == "tool":
            convo.append(
                f"[tool result id={message.get('tool_call_id', '')}]\n"
                f"{_text(message.get('content'))}"
            )
        else:
            convo.append(f"[user]\n{_text(message.get('content'))}")
    catalogue, names = _tool_catalogue(tools)
    wants_json = isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    )
    if wants_json and response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            catalogue += "\n\nThe JSON object must match this schema: " + json.dumps(
                schema, ensure_ascii=False, sort_keys=True
            )
    system = "\n\n".join(p for p in system_parts if p.strip())
    tail = []
    if catalogue:
        tail.append(catalogue)
    if wants_json:
        tail.append(JSON_ONLY)
    body = "\n\n".join(convo)
    if len(convo) > 1 or catalogue:
        body = (
            "# Conversation so far\n\n"
            + body
            + "\n\n# Your turn\nReply as the assistant to the last message."
        )
    prompt_parts = (
        ([f"# Instructions\n\n{system}"] if system_inline and system else []) + [body] + tail
    )
    return Rendered(
        system="" if system_inline else system,
        prompt="\n\n".join(prompt_parts),
        tool_names=names,
        wants_json=wants_json,
    )


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _first_json_object(text: str) -> Any:
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except ValueError:
            try:
                from json_repair import repair_json

                return json.loads(repair_json(candidate[start : end + 1]))
            except Exception:
                return None
    return None


@dataclass
class Parsed:
    content: str
    tool_calls: list[dict]

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


def parse(answer: str, rendered: Rendered) -> Parsed:
    """The CLI's text as a chat reply: tool calls when it followed the protocol with known
    tool names, a clean JSON object when JSON was asked for, plain text otherwise."""
    text = (answer or "").strip()
    if rendered.tool_names:
        data = _first_json_object(text) if "tool_calls" in text else None
        calls = data.get("tool_calls") if isinstance(data, dict) else None
        if isinstance(calls, list) and calls:
            out: list[dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict) or call.get("name") not in rendered.tool_names:
                    out = []
                    break
                args = call.get("arguments", {})
                out.append(
                    {
                        "id": "call_" + uuid.uuid4().hex[:24],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": args
                            if isinstance(args, str)
                            else json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            if out:
                return Parsed(content="", tool_calls=out)
    if rendered.wants_json:
        data = _first_json_object(text)
        if isinstance(data, (dict, list)):
            return Parsed(content=json.dumps(data, ensure_ascii=False), tool_calls=[])
    return Parsed(content=text, tool_calls=[])


def estimate_tokens(text: str) -> int:
    """A visible estimate (about 4 characters per token) when the CLI reports no usage."""
    return max(1, len(text) // 4)
