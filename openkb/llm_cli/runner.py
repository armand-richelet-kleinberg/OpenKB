"""Runs one CLI call: argv without a shell, prompt on stdin (or in a file), answer parsed.

Each call runs in a fresh empty working directory, so a project's own CLI settings,
hooks or instruction files are never picked up by accident, and the system prompt file
(when the profile uses one) lives there and disappears with it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from openkb.llm_cli.profiles import CliProfile


class CliBackendError(RuntimeError):
    """The CLI did not produce an answer. The message names the profile and the reason,
    never the prompt."""


@dataclass
class CliResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    duration: float
    exit_code: int
    cost_usd: float | None = None


def _prepare(
    profile: CliProfile, model: str, system: str, prompt: str, workdir: Path
) -> tuple[list[str], bytes | None]:
    system_file = ""
    if system and profile.system == "file":
        path = workdir / "system-prompt.md"
        path.write_text(system, encoding="utf-8")
        system_file = str(path)
    argv = [
        profile.resolve_executable(),
        *profile.command(model, system_file=system_file, workdir=str(workdir)),
    ]
    if profile.prompt == "file":
        path = workdir / "request.md"
        path.write_text(prompt, encoding="utf-8")
        argv = [a.replace("{prompt_file}", str(path)) for a in argv]
        return argv, None
    return argv, prompt.encode("utf-8")


def _interpret(
    profile: CliProfile, code: int, stdout: bytes, stderr: bytes, duration: float
) -> CliResult:
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace").strip()
    if profile.output == "claude-json":
        data = None
        for line in reversed(out.strip().splitlines()):
            try:
                data = json.loads(line)
                break
            except ValueError:
                continue
        if data is None:
            try:
                data = json.loads(out)
            except ValueError:
                raise CliBackendError(
                    f"{profile.name}: no JSON result (exit {code}). {err[-400:]}"
                ) from None
        if not isinstance(data, dict):
            raise CliBackendError(f"{profile.name}: unexpected result shape (exit {code}).")
        if data.get("is_error") or code != 0 or data.get("result") is None:
            reason = str(data.get("result") or data.get("subtype") or err or "error")
            raise CliBackendError(f"{profile.name}: {reason[:400]} (exit {code}).")
        usage = data.get("usage") or {}
        cost = data.get("total_cost_usd")
        return CliResult(
            text=str(data["result"]),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            duration=duration,
            exit_code=code,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        )
    if code != 0:
        raise CliBackendError(f"{profile.name}: exit {code}. {err[-400:] or out[-400:]}")
    if not out.strip():
        raise CliBackendError(f"{profile.name}: empty answer (exit 0). {err[-400:]}")
    return CliResult(
        text=out.strip(), input_tokens=None, output_tokens=None, duration=duration, exit_code=code
    )


def run(
    profile: CliProfile, model: str, system: str, prompt: str, timeout: float | None = None
) -> CliResult:
    with tempfile.TemporaryDirectory(prefix="openkb-cli-") as tmp:
        workdir = Path(tmp)
        argv, stdin = _prepare(profile, model, system, prompt, workdir)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                cwd=workdir,
                env=profile.environment(),
                timeout=timeout or profile.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise CliBackendError(
                f"{profile.name}: no answer within {int(timeout or profile.timeout)} s."
            ) from None
        return _interpret(
            profile,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.monotonic() - started,
        )


async def arun(
    profile: CliProfile, model: str, system: str, prompt: str, timeout: float | None = None
) -> CliResult:
    with tempfile.TemporaryDirectory(prefix="openkb-cli-") as tmp:
        workdir = Path(tmp)
        argv, stdin = _prepare(profile, model, system, prompt, workdir)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir),
            env=profile.environment(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin), timeout=timeout or profile.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise CliBackendError(
                f"{profile.name}: no answer within {int(timeout or profile.timeout)} s."
            ) from None
        return _interpret(
            profile, process.returncode or 0, stdout, stderr, time.monotonic() - started
        )
