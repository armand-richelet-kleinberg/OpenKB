# CLI backends: Claude Code, GitHub Copilot CLI and other approved CLIs

OpenKB normally calls a model through LiteLLM with an API key. A CLI backend runs every
LLM step through an agent CLI the operator is already signed in to instead:

| Model string | CLI | Account |
|---|---|---|
| `claude_cli/<model>` (e.g. `claude_cli/sonnet`) | Claude Code (`claude`) | Claude Pro/Max, Team or Enterprise sign-in; or Amazon Bedrock, Google Vertex AI, Microsoft Foundry, or a corporate gateway configured for Claude Code |
| `copilot_cli/<model>` (e.g. `copilot_cli/gpt-5.4`, `copilot_cli/auto`) | GitHub Copilot CLI (`copilot`) | Personal Copilot, Copilot Business/Enterprise, or GitHub Enterprise Cloud with data residency (`*.ghe.com`) |
| `<name>/<model>` | any CLI declared under `cli_backends` | whatever that CLI authenticates with |

No key is stored by OpenKB. The CLI's own sign-in is the authentication.

## How it works

`openkb/llm_cli` registers one LiteLLM custom provider per profile. The compiler, PageIndex
and the agents SDK (query, chat, lint, skills) therefore run unchanged:

- the request travels on **stdin**, never as a command-line argument, so a long prompt has
  no length ceiling and no quoting or injection surface;
- each call runs in a fresh **empty working directory**, so a project's own CLI settings,
  hooks or instruction files are never picked up;
- **tools are disabled**: Claude Code with `--tools ""`, Copilot with every tool kind denied.
  The CLI acts as a model, not as an agent on your machine;
- **tool calls are emulated**: OpenKB's tools are described in the prompt with a small JSON
  protocol and the reply is turned back into native tool calls, so the agents SDK runs
  them as usual. Only tools OpenKB offered are ever executed;
- `response_format: json_object` becomes an explicit instruction plus a tolerant extraction;
- images cannot travel through a text CLI; they are replaced by a visible placeholder.

## Configuration (`.openkb/config.yaml`)

```yaml
model: claude_cli/sonnet

cli_backends:              # optional: override a built-in profile or declare a new one
  claude_cli:
    env:                   # Claude Code's own variables; values here are not secrets
      CLAUDE_CODE_USE_BEDROCK: "1"
      AWS_REGION: eu-central-1
    timeout: 900
  copilot_cli:
    host: mycorp.ghe.com   # GitHub Enterprise Cloud with data residency
  acme_cli:                # another approved CLI, used as model "acme_cli/<model>"
    executable: acme-llm
    args: ["run", "--quiet", "--model", "{model}"]
    optional_flags: ["--model"]
    prompt: stdin          # stdin | file  (file: "{prompt_file}" placeholder in args)
    output: text           # text | claude-json
    system: inline         # inline | file  (file: "{system_file}" placeholder in args)

data_governance:           # optional: enforced before every call
  account: enterprise      # personal | enterprise (declared)
  require_enterprise: true
  allowed_regions: [eu-central-1, eu-west-3, europe-west4, EU]
  region: EU               # declared region, used when the CLI exposes none
  require_observed_region: false
  audit_log: true          # .openkb/llm-audit.jsonl, one line per call, no content
  purpose: "Internal knowledge base for engineering agents"
```

## Data governance (EU AI Act, GDPR, data localisation)

A regulated organisation needs to show where documents were processed and under which
account. Before every call OpenKB:

1. **observes** the deployment from the CLI's own configuration: Claude Code's
   `CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX` / `CLAUDE_CODE_USE_FOUNDRY`
   switches and region variables (`AWS_REGION`, `CLOUD_ML_REGION`, ...), a non-default
   `ANTHROPIC_BASE_URL` gateway, or the Copilot enterprise host. Only variable names and
   region or host values are read, never a credential;
2. combines them with what the operator **declares** (`account`, `region`);
3. applies the policy: `require_enterprise`, `allowed_regions`, `require_observed_region`.
   A refused backend sends nothing.

`.openkb/llm-backend.json` records the observed facts, the declared ones, the policy and
the decision. With `audit_log: true`, `.openkb/llm-audit.jsonl` adds one line per call:
time, profile, model, deployment, region, prompt SHA-256 and sizes, duration, exit code.
No prompt or answer text is stored.

These records are **evidence for a reviewer, not a compliance certificate**. Whether a
deployment meets the EU AI Act, the GDPR or a localisation rule depends on contracts and
settings outside OpenKB (the provider's data-processing terms, the tenant's residency
settings, retention). A region that the CLI does not expose (a Claude Team plan, a
github.com Copilot licence) can only be declared; set `require_observed_region: true`
to refuse anything that is not observed.

## Commands

```bash
openkb backend login     # the CLI's own sign-in (Copilot: copilot login [--host <ghe host>])
openkb backend show      # profile, argv, observed deployment, policy decision; no model call
openkb backend doctor    # installed? version? the flags this profile uses? policy? one real answer
openkb backend doctor --no-call
```

`doctor` checks every flag the profile passes against the CLI's own `--help` and names
any that this build does not list. CLI flags change between releases: run it after each
CLI upgrade, and adjust `cli_backends.<name>.args` if a flag moved.

## Known limits

- One model answer per call: no token streaming (the answer is delivered as one stream
  chunk, which OpenKB's streaming paths accept).
- Tool calling is emulated by instruction; a model that ignores the protocol answers in
  plain text instead, and the agent loop ends with that answer.
- Token usage is the CLI's own report when it gives one (Claude Code), otherwise an
  estimate of about four characters per token.
- Copilot CLI: no documented way to disable every tool exists; the profile denies each
  documented tool kind. Verify with `openkb backend doctor` on your build.
