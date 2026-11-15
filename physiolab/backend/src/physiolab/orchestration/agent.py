"""PHY-14 prototype: a provider-configurable planner that can only emit validated pipelines.

Flow: prompt -> provider (rule-based baseline or any LLM) -> JSON candidate -> schema check ->
validate_pipeline -> PlanResult with provenance. A candidate that fails validation is retried
with the error as feedback, then downgraded to an explicit "unsupported" result. Nothing a
provider returns is executed; only validated Pipeline data reaches the runtime.

PlanResult lives here, not in contracts.py, so this prototype needs no API/schema change.
Moving it into contracts.py (and exposing it through the API) is a reviewed follow-up.
"""

import hashlib
import json
from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field, ValidationError

from physiolab.contracts import Catalog, Contract, Pipeline, ProjectCreate, now
from physiolab.devices.catalog import CATALOG
from physiolab.orchestration.validation import validate_pipeline

Status = Literal[
    "runnable",
    "partial",
    "unsupported",
    "needs_clarification",
    "requires_review",
    "invalid_request",
    "refused",
]


class Missing(Contract):
    layer: str
    capability: str
    blocked_by: str
    reason: str


class Provenance(Contract):
    planner: str = "agent-v0"
    provider: str
    model: str
    prompt_sha256: str
    created_at: datetime = Field(default_factory=now)
    attempts: int
    validation_errors: list[str] = Field(default_factory=list)


class PlanResult(Contract):
    status: Status
    pipeline: Pipeline | None = None
    missing: list[Missing] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    suggestion: dict[str, float] | None = None
    user_message: str
    provenance: Provenance


class GenerationProvider(Protocol):
    """Anything that turns (instructions, prompt) into a JSON string. Swap to compare models."""

    name: str
    model: str

    def generate(self, instructions: str, prompt: str, feedback: str | None = None) -> str: ...


class PlanNotRunnable(Exception):
    def __init__(self, result: PlanResult):
        super().__init__(result.user_message)
        self.result = result


OUTPUT_FIELDS = ["status", "pipeline", "missing", "questions", "suggestion", "user_message"]


def build_instructions(catalog: Catalog = CATALOG) -> str:
    """Identical instructions for every provider, so comparisons are fair."""
    ready = [d.id for d in catalog.devices if d.status == "ready"]
    planned = [f"{d.id} ({d.status}, {d.delivery})" for d in catalog.devices if d.status != "ready"]
    return f"""You plan physiological sensing pipelines. Reply with ONE JSON object only.

Available today (the ONLY implementations you may put in a runnable pipeline):
- devices: {", ".join(ready)}
- processing: {", ".join(catalog.processors)}
- models: {", ".join(catalog.models)}
- applications: {", ".join(catalog.targets)}
Planned, NOT available: {"; ".join(planned)}

Runnable pipelines must be exactly: device -> bandpass -> sliding-window -> model -> application,
with parameters bandpass {{low_hz, high_hz}} (0 < low < high < 50), sliding-window
{{size, stride}} (integers, 1 <= stride <= size <= 1000), demo-emg-rms {{threshold}} (0 < t <= 2).

Fields: {", ".join(OUTPUT_FIELDS)}.
- status: runnable | partial | unsupported | needs_clarification | requires_review |
  invalid_request | refused
- pipeline: only when status is runnable;
  {{"schema_version": "1.0", "nodes": [...], "edges": [...]}}
- missing: list of {{layer, capability, blocked_by, reason}} for anything unavailable
- questions: when the prompt does not name a device or target, ask; never choose for the user
- suggestion: valid parameter values when a requested value is impossible
- user_message: one or two plain sentences for the user
Never invent implementations. Never claim validation the models do not have. Refuse requests
to run custom code. Clinical/patient requests are requires_review."""


def _fallback(errors: list[str]) -> dict:
    return {
        "status": "unsupported",
        "missing": [
            {
                "layer": "planner",
                "capability": "valid pipeline",
                "blocked_by": "provider output",
                "reason": "; ".join(errors)[-500:],
            }
        ],
        "user_message": "I couldn't produce a pipeline that passes validation, so nothing was run.",
    }


class AgentPlanner:
    """PipelinePlanner implementation. plan() keeps the existing interface; plan_detailed()
    returns the full PlanResult including refusals and provenance."""

    def __init__(self, provider: GenerationProvider, catalog: Catalog = CATALOG, attempts=2):
        self.provider, self.catalog, self.max_attempts = provider, catalog, attempts
        self.instructions = build_instructions(catalog)

    def _check(self, raw: str) -> dict:
        candidate = json.loads(raw)
        if not isinstance(candidate, dict):
            raise ValueError("Provider output must be a JSON object")
        unknown = set(candidate) - set(OUTPUT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown fields: {sorted(unknown)}")
        status = candidate.get("status")
        if status == "runnable":
            pipeline = Pipeline.model_validate(candidate.get("pipeline"))
            validate_pipeline(pipeline)
            candidate["pipeline"] = pipeline.model_dump()
        else:
            candidate.pop("pipeline", None)  # never keep an executable graph on a refusal
        return candidate

    def plan_detailed(self, prompt: str) -> PlanResult:
        errors: list[str] = []
        candidate, feedback, attempt = None, None, 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.provider.generate(self.instructions, prompt, feedback)
                candidate = self._check(raw)
                PlanResult.model_validate({**candidate, "provenance": self._provenance(prompt, 1)})
                break
            except (ValueError, ValidationError, TypeError) as error:
                message = str(error).splitlines()[0][:300]
                errors.append(f"attempt {attempt}: {message}")
                feedback, candidate = f"Your last answer was rejected: {message}", None
        provenance = self._provenance(prompt, attempt, errors)
        return PlanResult.model_validate(
            {**(candidate or _fallback(errors)), "provenance": provenance}
        )

    def plan(self, request: ProjectCreate) -> Pipeline:
        result = self.plan_detailed(request.prompt)
        if result.status != "runnable" or result.pipeline is None:
            raise PlanNotRunnable(result)
        return result.pipeline

    def _provenance(self, prompt: str, attempts: int, errors=()) -> Provenance:
        return Provenance(
            provider=self.provider.name,
            model=self.provider.model,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            attempts=attempts,
            validation_errors=list(errors),
        )
