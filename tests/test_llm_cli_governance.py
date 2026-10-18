"""Data governance of CLI backends: observed vs declared facts, policy, receipt, audit."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openkb.llm_cli import governance
from openkb.llm_cli.profiles import BUILTIN

CLAUDE = BUILTIN["claude_cli"]
COPILOT = BUILTIN["copilot_cli"]


def _facts(profile, policy=governance.Policy(), env=None):
    return governance.observe(profile, policy, environ=env or {})


def test_claude_deployments_are_observed_from_the_cli_switches():
    assert (
        _facts(CLAUDE).deployment == "anthropic-direct" and not _facts(CLAUDE).enterprise_observed
    )
    bedrock = _facts(CLAUDE, env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-central-1"})
    assert (
        bedrock.deployment,
        bedrock.region,
        bedrock.region_source,
        bedrock.enterprise_observed,
    ) == ("anthropic-model-via-amazon-bedrock", "eu-central-1", "observed:AWS_REGION", True)
    vertex = _facts(
        replace(CLAUDE, env={"CLAUDE_CODE_USE_VERTEX": "1", "CLOUD_ML_REGION": "europe-west4"})
    )
    assert (
        vertex.deployment == "anthropic-model-via-google-vertex-ai"
        and vertex.region == "europe-west4"
    )
    assert _facts(CLAUDE, env={"CLAUDE_CODE_USE_BEDROCK": "0"}).deployment == "anthropic-direct"
    # The public API address is not a gateway; a corporate one is.
    assert (
        _facts(CLAUDE, env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}).deployment
        == "anthropic-direct"
    )
    gateway = _facts(CLAUDE, env={"ANTHROPIC_BASE_URL": "https://llm-gw.corp.example/anthropic"})
    assert (
        gateway.deployment == "anthropic-compatible-gateway:llm-gw.corp.example"
        and gateway.enterprise_observed
    )
    # A region variable alone does not make a direct subscription regional.
    assert _facts(CLAUDE, env={"AWS_REGION": "eu-west-1"}).region == ""


def test_copilot_enterprise_host_is_observed_and_region_is_declared():
    ghe = _facts(replace(COPILOT, host="acme.ghe.com"), governance.Policy(region="EU"))
    assert (
        ghe.deployment == "github-enterprise-cloud-data-residency:acme.ghe.com"
        and ghe.enterprise_observed
    )
    assert (ghe.region, ghe.region_source) == ("EU", "declared")
    assert _facts(COPILOT).deployment == "github.com" and not _facts(COPILOT).enterprise_observed


def test_policy_refuses_personal_unknown_region_wrong_region_and_declared_only_region():
    strict = governance.Policy(require_enterprise=True, allowed_regions=("eu-central-1",))
    assert any(
        "require_enterprise" in r for r in governance.evaluate(strict, _facts(CLAUDE, strict))
    )
    assert any("unknown" in r for r in governance.evaluate(strict, _facts(CLAUDE, strict)))
    ok = _facts(CLAUDE, strict, {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-central-1"})
    assert governance.evaluate(strict, ok) == []
    wrong = _facts(CLAUDE, strict, {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"})
    assert any("not in" in r for r in governance.evaluate(strict, wrong))
    declared = governance.Policy(
        account="enterprise", require_enterprise=True, region="EU", allowed_regions=("EU",)
    )
    assert governance.evaluate(declared, _facts(CLAUDE, declared)) == []
    observed_only = replace(declared, require_observed_region=True)
    assert any(
        "declared, not observed" in r
        for r in governance.evaluate(observed_only, _facts(CLAUDE, observed_only))
    )
    personal = governance.Policy(account="personal", require_enterprise=True)
    assert any(
        "declared personal" in r
        for r in governance.evaluate(
            personal, _facts(replace(COPILOT, host="acme.ghe.com"), personal)
        )
    )


def test_policy_shape_is_validated():
    for bad in ({"account": "team"}, {"allowed_regions": "EU"}, {"unknown": 1}):
        with pytest.raises(ValueError):
            governance.load_policy({"data_governance": bad})
    assert governance.load_policy({}).require_enterprise is False


def test_receipt_and_audit_carry_facts_never_content(tmp_path):
    (tmp_path / ".openkb").mkdir()
    policy = governance.Policy(audit_log=True, account="enterprise", purpose="compile")
    facts = _facts(CLAUDE, policy, {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-central-1"})
    governance.write_receipt(tmp_path, governance.receipt(policy, facts, []))
    data = json.loads((tmp_path / ".openkb" / governance.RECEIPT_NAME).read_text())
    assert (
        data["decision"] == "allowed"
        and data["facts"]["region"] == "eu-central-1"
        and "not a compliance certificate" in data["note"]
    )
    governance.audit(
        tmp_path, policy, facts, "sonnet", "CONFIDENTIAL PROMPT", "CONFIDENTIAL ANSWER", 1.5, 0
    )
    line = (tmp_path / ".openkb" / governance.AUDIT_NAME).read_text()
    assert "CONFIDENTIAL" not in line and json.loads(line)["promptChars"] == len(
        "CONFIDENTIAL PROMPT"
    )
    governance.audit(tmp_path, governance.Policy(), facts, "m", "p", "a", 0.1, 0)
    assert len((tmp_path / ".openkb" / governance.AUDIT_NAME).read_text().splitlines()) == 1


def test_a_refused_backend_sends_nothing(tmp_path, monkeypatch):
    import litellm

    for name in (
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        monkeypatch.delenv(name, raising=False)
    from openkb import llm_cli

    calls = []
    monkeypatch.setattr("openkb.llm_cli.runner.run", lambda *a, **k: calls.append(a))
    llm_cli.configure(
        {"model": "claude_cli/sonnet", "data_governance": {"require_enterprise": True}}, tmp_path
    )
    try:
        with pytest.raises(Exception, match="refused by data_governance"):
            litellm.completion(
                model="claude_cli/sonnet", messages=[{"role": "user", "content": "x"}]
            )
        assert calls == []
    finally:
        llm_cli.configure()
