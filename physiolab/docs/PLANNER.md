# Pipeline-generation agent (PHY-14 prototype)

## What it does

1. A **provider** reads the user's prompt and returns a JSON plan.
2. `AgentPlanner` checks it: known fields only, schema-valid `Pipeline`, then `validate_pipeline`.
3. If the check fails, the error is sent back to the provider for one retry. If it fails again,
   the result becomes `unsupported` with the errors listed. **Nothing invalid is ever returned
   as runnable, and nothing a provider writes is executed.**
4. Every result carries `provenance`: provider, model version, prompt hash, time, attempts and
   validation errors.

Files: `orchestration/agent.py` (planner, `PlanResult`), `orchestration/providers.py`
(providers), `scripts/eval_planner.py` (scoring), `tests/test_agent_planner.py`.

## Providers

| Provider | Status | Purpose |
| --- | --- | --- |
| `RuleBasedProvider` | Working, offline | Deterministic keyword baseline; no API key; tests the harness |
| `CallableProvider` | Working wrapper | Plug in Claude Opus, Gemini or Codex from a script |

All providers receive identical instructions from `build_instructions()` (catalog, parameter
bounds, output format), so comparisons are fair.

### Comparing an LLM (outside the repo; never commit keys)

```python
# my_llms.py, kept outside the repo, key read from an environment variable
from physiolab.orchestration.providers import CallableProvider

def claude():
    import anthropic
    client = anthropic.Anthropic()
    model = "PINNED-MODEL-ID"
    def call(instructions, prompt):
        reply = client.messages.create(model=model, max_tokens=2000, system=instructions,
                                       messages=[{"role": "user", "content": prompt}])
        return reply.content[0].text
    return CallableProvider("anthropic", model, call)
```

```sh
uv run --directory backend python ../scripts/eval_planner.py --provider my_llms:claude --proposed
```

## Scoring (`scripts/eval_planner.py`)

| Metric | Meaning |
| --- | --- |
| status | Planner status equals the golden status |
| pipeline | Runnable pipeline equals the golden pipeline |
| missing recall | Share of golden missing layers the planner also named |
| run | Runnable plan executes in the simulation and produces inferences |
| deploy | The web export renders those inferences |
| intent | **Pending**: needs a separate judge model with a pinned version |

Baseline today: 100% status accuracy, 0.81 missing-layer recall on 14 fixtures.

## Honest limits

- The rule-based baseline was written with the fixtures in view, so its 100% score shows the
  harness works, not that it generalises. Real models are needed for the comparison.
- Validation catches invalid pipelines, not wrong intent. Example found during testing: a
  Garmin prompt produced a valid EMG demo; fixed in the baseline, but this is exactly what the
  intent judge must catch for LLM providers.
- Not wired into `api.py` yet. `AgentPlanner.plan()` keeps the `PipelinePlanner` interface, but
  showing refusals in the UI needs `PlanResult` in `contracts.py` plus `make schema`: a contract
  change for team review.
- The runtime only runs the EMG chain, so only one pipeline shape can be runnable today.
