"""CLI model backends: profiles, transcript protocol, process runner, LiteLLM provider.

Real CLIs are replaced by small Python programs that behave like them (prompt on stdin,
JSON or text on stdout), so every rule is exercised end to end without a network.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import litellm
import pytest

from openkb import llm_cli
from openkb.llm_cli import runner, transcript
from openkb.llm_cli.profiles import BUILTIN, CliProfile, load_profiles, split_model

FAKE_CLAUDE = textwrap.dedent("""
    import json, os, sys
    args = sys.argv[1:]
    prompt = sys.stdin.read()
    system = ""
    if "--system-prompt-file" in args:
        with open(args[args.index("--system-prompt-file") + 1], encoding="utf-8") as fh:
            system = fh.read()
    if os.environ.get("FAKE_FAIL"):
        failure = {"type": "result", "subtype": "error_during_execution", "is_error": True}
        print(json.dumps(dict(failure, result="Not logged in")))
        sys.exit(1)
    if "Available tools:" in prompt and "[tool result id=" not in prompt:
        out = json.dumps({"tool_calls": [{"name": "read_file", "arguments": {"path": "index.md"}}]})
    elif "JSON object only" in prompt:
        body = json.dumps({"system": system, "files_in_cwd": os.listdir(".")})
        out = "```json\\n" + body + "\\n```"
    else:
        marker = "after-tool " if "[tool result id=" in prompt else ""
        out = "ANSWER " + marker + json.dumps({"args": args, "system": system, "prompt": prompt})
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": out,
                      "usage": {"input_tokens": 12, "output_tokens": 5}, "total_cost_usd": 0.0}))
""")

FAKE_COPILOT = textwrap.dedent("""
    import sys
    prompt = sys.stdin.read()
    if "-p" in sys.argv:
        sys.exit("prompt must come on stdin")
    print("COPILOT " + " ".join(sys.argv[1:]) + " || " + prompt)
""")


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _claude(tmp_path: Path, **overrides) -> CliProfile:
    base = BUILTIN["claude_cli"]
    script = _script(tmp_path, "fake_claude", FAKE_CLAUDE)
    from dataclasses import replace

    return replace(base, executable=sys.executable, args=(str(script), *base.args), **overrides)


def _copilot(tmp_path: Path) -> CliProfile:
    base = BUILTIN["copilot_cli"]
    script = _script(tmp_path, "fake_copilot", FAKE_COPILOT)
    from dataclasses import replace

    return replace(base, executable=sys.executable, args=(str(script), *base.args))


# --- profiles -------------------------------------------------------------------------


def test_claude_profile_disables_tools_and_drops_empty_optional_flags():
    argv = BUILTIN["claude_cli"].command("", system_file="")
    assert "--model" not in argv and "--system-prompt-file" not in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    full = BUILTIN["claude_cli"].command("opus", system_file="/t/s.md")
    assert (
        full[full.index("--model") + 1] == "opus"
        and full[full.index("--system-prompt-file") + 1] == "/t/s.md"
    )


def test_copilot_profile_reads_stdin_denies_every_tool_and_omits_default_model():
    argv = BUILTIN["copilot_cli"].command("auto")
    assert "-p" not in argv and "--model" not in argv
    assert "--deny-tool=shell,write,read,url,memory" in argv and "-s" in argv


def test_config_overlays_builtins_declares_new_profiles_and_rejects_bad_shapes():
    profiles = load_profiles(
        {
            "cli_backends": {
                "claude_cli": {
                    "env": {"CLAUDE_CODE_USE_VERTEX": "1", "CLOUD_ML_REGION": "europe-west4"}
                },
                "copilot_cli": {"host": "acme.ghe.com"},
                "acme_cli": {
                    "executable": "acme",
                    "args": ["run", "--model", "{model}"],
                    "optional_flags": ["--model"],
                },
            }
        }
    )
    assert profiles["claude_cli"].env["CLOUD_ML_REGION"] == "europe-west4"
    assert profiles["claude_cli"].args == BUILTIN["claude_cli"].args
    assert profiles["copilot_cli"].host == "acme.ghe.com"
    assert profiles["acme_cli"].command("") == ["run"]
    for bad in (
        {"x_cli": {"args": []}},
        {"claude_cli": {"nope": 1}},
        {"claude_cli": {"env": {"bad name": "1"}}},
        {"copilot_cli": {"host": "https://acme.ghe.com"}},
        {"Bad": {"executable": "a", "args": []}},
    ):
        with pytest.raises(ValueError):
            load_profiles({"cli_backends": bad})
    assert split_model("claude_cli/sonnet") == ("claude_cli", "sonnet")


def test_missing_executable_is_named():
    from dataclasses import replace

    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        replace(
            BUILTIN["claude_cli"], executable="definitely-not-installed-cli"
        ).resolve_executable()


# --- transcript -------------------------------------------------------------------------


def test_render_folds_conversation_tools_results_and_images():
    rendered = transcript.render(
        [
            {"role": "system", "content": "SYS"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Q1"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "A-CONTENT"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert rendered.system == "SYS" and "SYS" not in rendered.prompt
    assert (
        "[image omitted" in rendered.prompt and "[tool result id=c1]\nA-CONTENT" in rendered.prompt
    )
    assert "Available tools:" in rendered.prompt and rendered.tool_names == {"read_file"}
    inline = transcript.render(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
        system_inline=True,
    )
    assert inline.system == "" and inline.prompt.startswith("# Instructions\n\nSYS")


def test_parse_tool_calls_only_for_known_tools_and_json_extraction():
    rendered = transcript.render(
        [{"role": "user", "content": "q"}], tools=[{"function": {"name": "read_file"}}]
    )
    parsed = transcript.parse(
        '{"tool_calls": [{"name": "read_file", "arguments": {"path": "x"}}]}', rendered
    )
    assert parsed.finish_reason == "tool_calls" and json.loads(
        parsed.tool_calls[0]["function"]["arguments"]
    ) == {"path": "x"}
    unknown = transcript.parse('{"tool_calls": [{"name": "rm_rf", "arguments": {}}]}', rendered)
    assert unknown.tool_calls == [] and "rm_rf" in unknown.content
    as_json = transcript.render(
        [{"role": "user", "content": "q"}], response_format={"type": "json_object"}
    )
    assert transcript.parse('Sure:\n```json\n{"a": 1}\n```', as_json).content == '{"a": 1}'
    assert transcript.parse('{"a": 1,}', as_json).content == '{"a": 1}'


# --- runner ---------------------------------------------------------------------------


def test_runner_sends_prompt_on_stdin_system_in_a_file_in_an_empty_workdir(tmp_path):
    result = runner.run(_claude(tmp_path), "sonnet", "SYSTEM TEXT", "PROMPT TEXT")
    payload = json.loads(result.text.split(" ", 1)[1])
    assert payload["prompt"] == "PROMPT TEXT" and payload["system"] == "SYSTEM TEXT"
    assert (result.input_tokens, result.output_tokens) == (12, 5)
    asked_json = runner.run(
        _claude(tmp_path), "sonnet", "S", "reply with one valid JSON object only"
    )
    assert json.loads(asked_json.text.strip("`\njson"))["files_in_cwd"] == ["system-prompt.md"]


def test_runner_reports_cli_errors_without_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(runner.CliBackendError) as error:
        runner.run(_claude(tmp_path), "sonnet", "", "SECRET PROMPT")
    assert "Not logged in" in str(error.value) and "SECRET PROMPT" not in str(error.value)


def test_async_runner_and_timeout(tmp_path):
    result = asyncio.run(runner.arun(_claude(tmp_path), "", "", "hello"))
    assert result.text.startswith("ANSWER")
    from dataclasses import replace

    slow = replace(_claude(tmp_path), args=("-c", "import time; time.sleep(5)"))
    with pytest.raises(runner.CliBackendError, match="no answer within"):
        runner.run(slow, "", "", "x", timeout=0.5)


def test_copilot_text_output_inline_system(tmp_path):
    result = runner.run(
        _copilot(tmp_path),
        "gpt-5.4",
        "",
        transcript.render(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}],
            system_inline=True,
        ).prompt,
    )
    assert (
        result.text.startswith("COPILOT")
        and "--model gpt-5.4" in result.text
        and "SYS" in result.text
        and result.input_tokens is None
    )


# --- provider through LiteLLM and the agents SDK ----------------------------------------


@pytest.fixture
def registered(tmp_path, monkeypatch):
    profiles = {"claude_cli": _claude(tmp_path), "copilot_cli": _copilot(tmp_path)}
    monkeypatch.setattr("openkb.llm_cli.load_profiles", lambda config: profiles)
    llm_cli.configure({"model": "claude_cli/sonnet"})
    yield profiles
    monkeypatch.undo()
    llm_cli.configure()


def test_litellm_completion_json_mode_and_usage(registered):
    response = litellm.completion(
        model="claude_cli/sonnet",
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "x"}],
        response_format={"type": "json_object"},
    )
    assert json.loads(response.choices[0].message.content)["system"] == "S"
    assert response.usage.total_tokens == 17


def test_agents_sdk_tool_loop_and_streaming_run_through_the_cli(registered):
    from agents import Agent, Runner, function_tool, set_tracing_disabled

    set_tracing_disabled(True)
    seen = []

    @function_tool
    def read_file(path: str) -> str:
        """Read a wiki file."""
        seen.append(path)
        return "CONTENT OF " + path

    agent = Agent(
        name="q", instructions="Use the wiki.", tools=[read_file], model="litellm/claude_cli/sonnet"
    )
    result = asyncio.run(Runner.run(agent, "What is in the index?", max_turns=4))
    assert seen == ["index.md"] and result.final_output.startswith("ANSWER after-tool"), (
        result.final_output[:600]
    )

    async def streamed():
        run = Runner.run_streamed(agent, "again", max_turns=4)
        async for _ in run.stream_events():
            pass
        return run.final_output

    assert asyncio.run(streamed()).startswith("ANSWER after-tool")


def test_cli_models_are_recognised():
    assert llm_cli.is_cli_model("claude_cli/sonnet") and llm_cli.is_cli_model("copilot_cli/gpt-5.4")
    assert not llm_cli.is_cli_model("anthropic/claude-sonnet-4-6")
