"""Data governance for CLI backends: which account and deployment process the knowledge base.

An organisation subject to the EU AI Act, the GDPR or a data-localisation rule needs to
know, and to prove, where its documents were processed and under which account. This
module states those facts and enforces the policy the knowledge base declares::

    data_governance:
      account: enterprise            # personal | enterprise (what the operator declares)
      require_enterprise: true       # refuse a backend that is not an enterprise deployment
      allowed_regions: [eu-central-1, eu-west-3, europe-west4, EU]
      region: EU                     # declared region, used when the CLI exposes none
      require_observed_region: false # true: a declared region is not enough
      audit_log: true                # one line per call in .openkb/llm-audit.jsonl (no content)
      purpose: "Compile the IS knowledge base for internal agents"

Facts are labelled ``observed`` (read from the CLI's own configuration: its enterprise
switches, region variables, enterprise host) or ``declared`` (stated in this file). The
receipt never claims compliance; it records what was observed, what was declared and
what the policy decided, so a reviewer can judge. No secret is read or written.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openkb.llm_cli.profiles import CLAUDE_ENTERPRISE_SWITCHES, REGION_VARIABLES, CliProfile

RECEIPT_NAME = "llm-backend.json"
AUDIT_NAME = "llm-audit.jsonl"


class GovernanceError(RuntimeError):
    """The declared data-governance policy refuses this backend; nothing was sent."""


@dataclass(frozen=True)
class Policy:
    account: str = ""
    require_enterprise: bool = False
    allowed_regions: tuple[str, ...] = ()
    region: str = ""
    require_observed_region: bool = False
    audit_log: bool = False
    purpose: str = ""


def load_policy(config: dict[str, Any] | None) -> Policy:
    block = (config or {}).get("data_governance") or {}
    if not isinstance(block, dict):
        raise ValueError("data_governance must be a mapping.")
    known = {
        "account",
        "require_enterprise",
        "allowed_regions",
        "region",
        "require_observed_region",
        "audit_log",
        "purpose",
    }
    unknown = set(block) - known
    if unknown:
        raise ValueError(f"data_governance: unknown key(s) {sorted(unknown)}.")
    account = str(block.get("account", "")).strip().lower()
    if account not in ("", "personal", "enterprise"):
        raise ValueError("data_governance.account must be 'personal' or 'enterprise'.")
    regions = block.get("allowed_regions") or []
    if not isinstance(regions, list):
        raise ValueError("data_governance.allowed_regions must be a list.")
    return Policy(
        account=account,
        require_enterprise=bool(block.get("require_enterprise", False)),
        allowed_regions=tuple(str(r).strip() for r in regions if str(r).strip()),
        region=str(block.get("region", "")).strip(),
        require_observed_region=bool(block.get("require_observed_region", False)),
        audit_log=bool(block.get("audit_log", False)),
        purpose=str(block.get("purpose", "")).strip()[:500],
    )


@dataclass
class Facts:
    profile: str
    deployment: str
    enterprise_observed: bool
    region: str = ""
    region_source: str = ""
    host: str = ""
    switches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "deployment": self.deployment,
            "enterpriseObserved": self.enterprise_observed,
            "region": self.region,
            "regionSource": self.region_source,
            "host": self.host,
            "switches": self.switches,
        }


def observe(profile: CliProfile, policy: Policy, environ: dict[str, str] | None = None) -> Facts:
    """What the CLI's own configuration says about where requests go. Names of switches and
    region values only - never a credential."""
    env = dict(environ if environ is not None else os.environ)
    env.update(profile.env)
    region, source = "", ""
    for name in REGION_VARIABLES:
        if env.get(name):
            region, source = env[name].strip(), f"observed:{name}"
            break
    switches = [
        name
        for name in CLAUDE_ENTERPRISE_SWITCHES
        if env.get(name, "").strip() not in ("", "0", "false")
    ]
    if profile.name == "claude_cli" or profile.executable.lower().startswith("claude"):
        if "CLAUDE_CODE_USE_BEDROCK" in switches:
            deployment = "anthropic-model-via-amazon-bedrock"
        elif "CLAUDE_CODE_USE_VERTEX" in switches:
            deployment = "anthropic-model-via-google-vertex-ai"
        elif "CLAUDE_CODE_USE_FOUNDRY" in switches:
            deployment = "anthropic-model-via-microsoft-foundry"
        elif _gateway_host(env.get("ANTHROPIC_BASE_URL", "")):
            deployment = "anthropic-compatible-gateway:" + _gateway_host(
                env.get("ANTHROPIC_BASE_URL", "")
            )
        else:
            deployment = "anthropic-direct"
            region, source = (region, source) if switches else ("", "")
        enterprise = bool(switches) or deployment.startswith("anthropic-compatible-gateway")
        host = ""
    elif profile.name == "copilot_cli" or profile.executable.lower().startswith("copilot"):
        host = profile.host
        deployment = (
            f"github-enterprise-cloud-data-residency:{host}"
            if host.endswith(".ghe.com")
            else ("github-enterprise-server:" + host if host else "github.com")
        )
        enterprise = bool(host)
        region, source = "", ""
    else:
        host = profile.host
        deployment = f"custom-cli:{profile.name}"
        enterprise = False
    if not region and policy.region:
        region, source = policy.region, "declared"
    return Facts(
        profile=profile.name,
        deployment=deployment,
        enterprise_observed=enterprise,
        region=region,
        region_source=source,
        host=host,
        switches=switches,
    )


def _gateway_host(url: str) -> str:
    """The host of a non-default Anthropic base URL (a corporate gateway), or ''."""
    from urllib.parse import urlparse

    host = (urlparse(url.strip()).hostname or "").lower() if url.strip() else ""
    return "" if host in ("", "api.anthropic.com") else host


def evaluate(policy: Policy, facts: Facts) -> list[str]:
    """Reasons to refuse, empty when the policy allows the backend."""
    reasons: list[str] = []
    if (
        policy.require_enterprise
        and not facts.enterprise_observed
        and policy.account != "enterprise"
    ):
        reasons.append(
            f"require_enterprise: '{facts.deployment}' is not an observed enterprise deployment "
            "and the account is not declared enterprise."
        )
    if policy.require_enterprise and policy.account == "personal":
        reasons.append("require_enterprise: the account is declared personal.")
    if policy.allowed_regions:
        if not facts.region:
            reasons.append(
                "allowed_regions: the processing region is unknown (set a region variable for "
                "the CLI, or declare data_governance.region)."
            )
        elif facts.region not in policy.allowed_regions:
            reasons.append(
                f"allowed_regions: region '{facts.region}' is not in "
                f"{list(policy.allowed_regions)}."
            )
    if policy.require_observed_region and not facts.region_source.startswith("observed"):
        reasons.append(
            "require_observed_region: the region is declared, not observed from the CLI's "
            "configuration."
        )
    return reasons


def receipt(policy: Policy, facts: Facts, reasons: list[str]) -> dict[str, Any]:
    return {
        "schema": "OpenKB.CliBackendReceipt.v1",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "facts": facts.as_dict(),
        "declared": {"account": policy.account, "region": policy.region, "purpose": policy.purpose},
        "policy": {
            "requireEnterprise": policy.require_enterprise,
            "allowedRegions": list(policy.allowed_regions),
            "requireObservedRegion": policy.require_observed_region,
            "auditLog": policy.audit_log,
        },
        "decision": "refused" if reasons else "allowed",
        "reasons": reasons,
        "note": (
            "Facts are observed from the CLI configuration or declared by the operator; this "
            "receipt is evidence for a reviewer, not a compliance certificate."
        ),
    }


def write_receipt(kb_dir: Path | None, data: dict[str, Any]) -> None:
    if kb_dir is None:
        return
    from openkb.locks import atomic_write_text

    target = Path(kb_dir) / ".openkb" / RECEIPT_NAME
    if target.parent.is_dir():
        atomic_write_text(target, json.dumps(data, indent=2, sort_keys=True) + "\n")


def audit(
    kb_dir: Path | None,
    policy: Policy,
    facts: Facts,
    model: str,
    prompt: str,
    answer: str,
    duration: float,
    exit_code: int,
) -> None:
    """One line per call - who processed what size of input where - without any content."""
    if kb_dir is None or not policy.audit_log:
        return
    target = Path(kb_dir) / ".openkb" / AUDIT_NAME
    if not target.parent.is_dir():
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": facts.profile,
        "model": model,
        "deployment": facts.deployment,
        "region": facts.region,
        "regionSource": facts.region_source,
        "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "promptChars": len(prompt),
        "answerChars": len(answer),
        "durationMs": int(duration * 1000),
        "exitCode": exit_code,
    }
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
