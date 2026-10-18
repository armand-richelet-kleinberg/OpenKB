"""Agent CLIs as OpenKB model backends (Claude Code, GitHub Copilot CLI, declared others).

``configure(config, kb_dir)`` registers one LiteLLM custom provider per profile, so a
knowledge base whose ``model`` is ``claude_cli/sonnet`` or ``copilot_cli/gpt-5.4`` runs
every LLM step through that CLI and the account it is signed in to. The knowledge base's
``data_governance`` policy is checked before each call; see ``docs/cli-backends.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openkb.llm_cli.governance import Policy, evaluate, load_policy, observe, receipt, write_receipt
from openkb.llm_cli.profiles import BUILTIN, CliProfile, load_profiles, split_model

__all__ = [
    "configure",
    "is_cli_model",
    "cli_provider_names",
    "active_profile",
    "BUILTIN",
    "CliProfile",
]

_registered: dict[str, Any] = {}


def cli_provider_names() -> set[str]:
    return set(BUILTIN) | set(_registered)


def is_cli_model(model: str) -> bool:
    return split_model(model or "")[0] in cli_provider_names()


def configure(
    config: dict[str, Any] | None = None, kb_dir: Path | None = None
) -> dict[str, CliProfile]:
    """(Re)register the CLI providers from the built-in profiles and the KB config.

    Idempotent; replaces only entries this module registered, so other custom providers in
    ``litellm.custom_provider_map`` are kept. When the configured model is a CLI model, the
    governance receipt is written to ``.openkb/llm-backend.json``.
    """
    import litellm

    from openkb.llm_cli.provider import CliBackend

    profiles = load_profiles(config)
    policy = load_policy(config)
    ours = set(_registered) | set(profiles)
    kept = [
        entry for entry in (litellm.custom_provider_map or []) if entry.get("provider") not in ours
    ]
    _registered.clear()
    for name, profile in profiles.items():
        handler = CliBackend(profile, policy, kb_dir)
        _registered[name] = handler
        kept.append({"provider": name, "custom_handler": handler})
    litellm.custom_provider_map = kept
    from litellm.utils import custom_llm_setup

    custom_llm_setup()
    model = str((config or {}).get("model", ""))
    provider, _ = split_model(model)
    if provider in profiles:
        facts = observe(profiles[provider], policy)
        write_receipt(kb_dir, receipt(policy, facts, evaluate(policy, facts)))
    return profiles


def active_profile(model: str) -> CliProfile | None:
    handler = _registered.get(split_model(model or "")[0])
    return handler.profile if handler else None


__all__ += ["Policy", "load_policy"]
