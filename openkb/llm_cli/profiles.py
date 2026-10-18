"""CLI backend profiles: how to drive one agent CLI as a plain text model.

A profile is data. The built-in ones cover Claude Code (``claude``) and the GitHub
Copilot CLI (``copilot``); a knowledge base can override any field, or declare a new
profile for another CLI its organisation has approved (an enterprise gateway, a
regional deployment), under ``cli_backends:`` in ``.openkb/config.yaml``::

    model: claude_cli/sonnet
    cli_backends:
      claude_cli:
        env: {CLAUDE_CODE_USE_BEDROCK: "1", AWS_REGION: eu-central-1}
      copilot_cli:
        host: mycorp.ghe.com
      acme_cli:                      # a new profile, used as model "acme_cli/<model>"
        executable: acme-llm
        args: ["run", "--model", "{model}", "--quiet"]
        prompt: stdin
        output: text

Placeholders in ``args``: ``{model}``, ``{system_file}``, ``{workdir}``. An argument
that contains a placeholder whose value is empty is dropped together with the flag
right before it when that flag is listed in ``optional_flags``.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field, replace
from typing import Any

PROMPT_MODES = ("stdin", "file")
OUTPUT_MODES = ("text", "claude-json")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

# Environment variables whose PRESENCE (never value) tells which deployment a CLI
# talks to. Only names are listed; values are read only for region/host facts.
CLAUDE_ENTERPRISE_SWITCHES = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
REGION_VARIABLES = (
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "CLOUD_ML_REGION",
    "ANTHROPIC_FOUNDRY_REGION",
)


@dataclass(frozen=True)
class CliProfile:
    """One CLI, driven headlessly: no tools, prompt on stdin (or a file), answer on stdout."""

    name: str
    executable: str
    args: tuple[str, ...]
    prompt: str = "stdin"
    output: str = "text"
    system: str = "file"  # "file": passed through {system_file}; "inline": folded into the prompt
    optional_flags: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 900.0
    host: str = ""
    login_args: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)
    default_model: str = ""

    def resolve_executable(self) -> str:
        found = shutil.which(self.executable)
        if not found:
            raise FileNotFoundError(
                f"CLI backend '{self.name}': executable '{self.executable}' was not found on PATH. "
                "Install it, sign in once, or set cli_backends.<name>.executable to its full path."
            )
        return found

    def command(self, model: str, system_file: str = "", workdir: str = "") -> list[str]:
        """The argv after the executable. Values are separate entries: no shell, no quoting."""
        values = {
            "model": model if model not in ("", "default", "auto") else "",
            "system_file": system_file,
            "workdir": workdir,
        }
        out: list[str] = []
        for arg in self.args:
            names = re.findall(r"\{(\w+)\}", arg)
            if names and any(not values.get(n, "") for n in names):
                if out and out[-1] in self.optional_flags:
                    out.pop()
                continue
            out.append(re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), ""), arg))
        return out

    def environment(self) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update({k: str(v) for k, v in self.env.items()})
        return merged


BUILTIN: dict[str, CliProfile] = {
    # Claude Code, print mode. The request travels on stdin (documented, up to 10 MB);
    # the short argument only says so. Tools are disabled so the CLI acts as a model,
    # and the default coding-agent system prompt is replaced by OpenKB's own.
    # Personal subscription: sign in once with `claude`. Enterprise: Claude Team/Enterprise
    # sign-in, or route through Bedrock / Vertex AI / Foundry with the CLI's own variables
    # (set them under cli_backends.claude_cli.env) to pin the processing region.
    "claude_cli": CliProfile(
        name="claude_cli",
        executable="claude",
        args=(
            "-p",
            "Answer the request given on standard input. Output only the answer.",
            "--output-format",
            "json",
            "--model",
            "{model}",
            "--system-prompt-file",
            "{system_file}",
            "--tools",
            "",
            "--max-turns",
            "1",
        ),
        output="claude-json",
        optional_flags=("--model", "--system-prompt-file"),
        version_args=("--version",),
        default_model="sonnet",
    ),
    # GitHub Copilot CLI. Piped stdin is the prompt (documented; -p would ignore stdin and
    # caps long prompts), -s prints only the answer, and every tool kind is denied.
    # Personal account: `copilot login`. GitHub Enterprise Cloud with data residency:
    # set cli_backends.copilot_cli.host to <name>.ghe.com and run `openkb backend login`.
    "copilot_cli": CliProfile(
        name="copilot_cli",
        executable="copilot",
        args=(
            "-s",
            "--no-ask-user",
            "--model",
            "{model}",
            "--deny-tool=shell,write,read,url,memory",
        ),
        system="inline",
        optional_flags=("--model",),
        login_args=("login",),
        version_args=("--version",),
        default_model="",
    ),
}


def _validated(name: str, data: dict[str, Any], base: CliProfile | None) -> CliProfile:
    if not _NAME.match(name):
        raise ValueError(
            f"cli_backends: invalid profile name '{name}' (lowercase letters, digits, underscore)."
        )
    known = {
        "executable",
        "args",
        "prompt",
        "output",
        "system",
        "optional_flags",
        "env",
        "timeout",
        "host",
        "login_args",
        "default_model",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"cli_backends.{name}: unknown key(s) {sorted(unknown)}.")
    if base is None and not {"executable", "args"} <= set(data):
        raise ValueError(f"cli_backends.{name}: a new profile needs 'executable' and 'args'.")
    fields: dict[str, Any] = {}
    for key in ("executable", "prompt", "output", "system", "host", "default_model"):
        if key in data:
            fields[key] = str(data[key])
    for key in ("args", "optional_flags", "login_args"):
        if key in data:
            if not isinstance(data[key], list) or not all(
                isinstance(v, (str, int, float)) for v in data[key]
            ):
                raise ValueError(f"cli_backends.{name}.{key} must be a list of strings.")
            fields[key] = tuple(str(v) for v in data[key])
    if "env" in data:
        if not isinstance(data["env"], dict):
            raise ValueError(
                f"cli_backends.{name}.env must be a mapping of variable names to values."
            )
        for key in data["env"]:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
                raise ValueError(
                    f"cli_backends.{name}.env: '{key}' is not an environment variable name."
                )
        fields["env"] = {str(k): str(v) for k, v in data["env"].items()}
    if "timeout" in data:
        fields["timeout"] = float(data["timeout"])
    profile = replace(base, **fields) if base else CliProfile(name=name, **fields)
    if profile.prompt not in PROMPT_MODES:
        raise ValueError(f"cli_backends.{name}.prompt must be one of {PROMPT_MODES}.")
    if profile.output not in OUTPUT_MODES:
        raise ValueError(f"cli_backends.{name}.output must be one of {OUTPUT_MODES}.")
    if profile.system not in ("file", "inline"):
        raise ValueError(f"cli_backends.{name}.system must be 'file' or 'inline'.")
    if profile.host and not re.match(r"^[A-Za-z0-9.-]+$", profile.host):
        raise ValueError(
            f"cli_backends.{name}.host must be a bare hostname such as mycorp.ghe.com."
        )
    return profile


def load_profiles(config: dict[str, Any] | None) -> dict[str, CliProfile]:
    """Built-in profiles overlaid with the knowledge base's ``cli_backends`` block."""
    profiles = dict(BUILTIN)
    block = (config or {}).get("cli_backends") or {}
    if not isinstance(block, dict):
        raise ValueError("cli_backends must be a mapping of profile name to settings.")
    for name, data in block.items():
        profiles[name] = _validated(
            str(name), data or {}, BUILTIN.get(str(name)) or profiles.get(str(name))
        )
    return profiles


def split_model(model: str) -> tuple[str, str]:
    """``claude_cli/sonnet`` -> (``claude_cli``, ``sonnet``)."""
    provider, _, rest = model.strip().partition("/")
    return provider, rest
