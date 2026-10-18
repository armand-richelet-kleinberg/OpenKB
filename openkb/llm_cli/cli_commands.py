"""``openkb backend``: inspect, verify and sign in to the CLI backend of a knowledge base."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import click

from openkb.llm_cli import governance
from openkb.llm_cli.profiles import CliProfile, load_profiles, split_model


def _kb_config(ctx: click.Context) -> tuple[Path | None, dict]:
    from openkb.cli import _find_kb_dir
    from openkb.config import load_config

    kb_dir = _find_kb_dir((ctx.obj or {}).get("kb_dir_override"))
    if kb_dir is None:
        return None, {}
    config = load_config(kb_dir / ".openkb" / "config.yaml")
    # The KB's own profiles and policy govern every call made from here, as in any command.
    from openkb import llm_cli

    llm_cli.configure(config, kb_dir)
    return kb_dir, config


def _profile(config: dict, name: str | None) -> tuple[CliProfile, str]:
    profiles = load_profiles(config)
    provider, model = split_model(str(config.get("model", "")))
    chosen = name or provider
    if chosen not in profiles:
        raise click.ClickException(
            f"'{chosen or config.get('model', '')}' is not a CLI backend. "
            f"Known: {', '.join(sorted(profiles))}."
        )
    profile = profiles[chosen]
    return profile, (model if chosen == provider else "") or profile.default_model


@click.group()
def backend() -> None:
    """Inspect, verify and sign in to the CLI backend (Claude Code, Copilot CLI, others)."""


@backend.command()
@click.option(
    "--profile", "name", default=None, help="Profile to inspect (default: the one in the KB model)."
)
@click.pass_context
def show(ctx: click.Context, name: str | None) -> None:
    """Show the profile, the observed deployment and the data_governance decision (no call)."""
    kb_dir, config = _kb_config(ctx)
    profile, model = _profile(config, name)
    policy = governance.load_policy(config)
    facts = governance.observe(profile, policy)
    reasons = governance.evaluate(policy, facts)
    click.echo(
        json.dumps(
            {
                "profile": profile.name,
                "executable": profile.executable,
                "model": model or "(CLI default)",
                "argv": profile.command(model, system_file="<system-file>", workdir="<tmp>"),
                **governance.receipt(policy, facts, reasons),
            },
            indent=2,
        )
    )
    if kb_dir is not None:
        governance.write_receipt(kb_dir, governance.receipt(policy, facts, reasons))
    if reasons:
        raise SystemExit(2)


@backend.command()
@click.option(
    "--profile", "name", default=None, help="Profile to verify (default: the one in the KB model)."
)
@click.option(
    "--no-call",
    is_flag=True,
    help="Check the installation and flags only; send nothing to a model.",
)
@click.pass_context
def doctor(ctx: click.Context, name: str | None, no_call: bool) -> None:
    """Verify the CLI end to end: installed, version, flags, policy, and one real answer."""
    import litellm

    kb_dir, config = _kb_config(ctx)
    profile, model = _profile(config, name)
    ok = True
    try:
        exe = profile.resolve_executable()
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from None
    click.echo(f"executable    : {exe}")
    try:
        version = subprocess.run(
            [exe, *profile.version_args],
            capture_output=True,
            text=True,
            timeout=60,
            env=profile.environment(),
            stdin=subprocess.DEVNULL,
        )
        click.echo(
            "version       : "
            + (((version.stdout or version.stderr).strip().splitlines() or ["(none)"])[0])
        )
        helptext = subprocess.run(
            [exe, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            env=profile.environment(),
            stdin=subprocess.DEVNULL,
        )
        helpout = (helptext.stdout or "") + (helptext.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as error:
        raise click.ClickException(f"{profile.name}: the CLI did not start ({error}).") from None
    flags = sorted(
        {
            re.split(r"=", a)[0]
            for a in profile.command(model or "x", system_file="f", workdir="w")
            if a.startswith("-")
        }
    )
    missing = [f for f in flags if f not in helpout]
    click.echo(f"flags used    : {' '.join(flags)}")
    if missing:
        ok = False
        click.echo(
            f"flags missing : {' '.join(missing)}  <- not listed by '--help' on this build; "
            f"adjust cli_backends.{profile.name}.args"
        )
    policy = governance.load_policy(config)
    facts = governance.observe(profile, policy)
    reasons = governance.evaluate(policy, facts)
    click.echo(
        f"deployment    : {facts.deployment}"
        + (f" (region {facts.region}, {facts.region_source})" if facts.region else "")
    )
    click.echo(f"governance    : {'allowed' if not reasons else 'REFUSED - ' + ' '.join(reasons)}")
    if kb_dir is not None:
        governance.write_receipt(kb_dir, governance.receipt(policy, facts, reasons))
    if reasons:
        raise SystemExit(2)
    if no_call:
        raise SystemExit(0 if ok else 1)
    try:
        answer = litellm.completion(
            model=f"{profile.name}/{model}" if model else f"{profile.name}/",
            messages=[{"role": "user", "content": "Reply with exactly the word READY."}],
        )
        text = (answer.choices[0].message.content or "").strip()
        click.echo(f"answer        : {text[:80]}")
        ok = ok and "READY" in text.upper()
    except Exception as error:  # the provider names the reason, never the prompt
        ok = False
        click.echo(f"answer        : FAILED - {str(error)[:400]}")
    raise SystemExit(0 if ok else 1)


@backend.command()
@click.option(
    "--profile", "name", default=None, help="Profile to sign in (default: the one in the KB model)."
)
@click.pass_context
def login(ctx: click.Context, name: str | None) -> None:
    """Run the CLI's own sign-in here (on the enterprise host when the profile names one)."""
    _, config = _kb_config(ctx)
    profile, _ = _profile(config, name)
    if not profile.login_args:
        click.echo(
            f"{profile.name}: sign in with the CLI itself (for Claude Code: run "
            f"`{profile.executable}` once and follow its login), then run `openkb backend doctor`."
        )
        return
    argv = [profile.resolve_executable(), *profile.login_args]
    if profile.host:
        argv += ["--host", profile.host]
    click.echo("running       : " + " ".join(argv))
    raise SystemExit(subprocess.call(argv, env=profile.environment()))
