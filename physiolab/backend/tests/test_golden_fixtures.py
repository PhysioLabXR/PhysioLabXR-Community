"""PHY-13: golden prompt fixtures must agree with the contracts and the validator.

Runnable fixtures must validate. Target pipelines for future capabilities must be well-formed
graphs, yet still be rejected by today's validator, so nothing unbuilt can execute.
"""

import json
from pathlib import Path

import pytest

from physiolab.contracts import Pipeline
from physiolab.orchestration.validation import validate_pipeline

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
GOLDEN = EXAMPLES / "golden"
CORE = sorted(GOLDEN.glob("*.json"))
ALL = sorted(GOLDEN.rglob("*.json"))
STATUSES = {
    "runnable",
    "partial",
    "unsupported",
    "needs_clarification",
    "requires_review",
    "invalid_request",
    "refused",
}
NEEDS_TARGET = {"partial", "unsupported", "requires_review"}
NEEDS_MISSING = NEEDS_TARGET | {"refused"}
REQUIRED = {
    "fixture_version",
    "id",
    "original_prompt",
    "prompt",
    "categories",
    "requested_target",
    "requirements",
    "expected_behavior",
    "constraints",
    "ambiguities",
    "result",
    "evaluation",
}
COVERAGE = {"wearable", "web", "phone", "quest", "clinician-dashboard", "research-bci"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def assert_well_formed_graph(pipeline: Pipeline):
    """Unique ids, known endpoints, no cycles, devices feed in, an application comes out."""
    ids = [n.id for n in pipeline.nodes]
    assert len(ids) == len(set(ids))
    kinds = {n.id: n.kind for n in pipeline.nodes}
    incoming = {i: set() for i in ids}
    outgoing = {i: set() for i in ids}
    for e in pipeline.edges:
        assert e.source in kinds and e.target in kinds
        outgoing[e.source].add(e.target)
        incoming[e.target].add(e.source)
    for i, kind in kinds.items():
        if kind == "device":
            assert not incoming[i] and outgoing[i]
        elif kind == "application":
            assert incoming[i] and not outgoing[i]
        else:
            assert incoming[i] and outgoing[i]
    remaining, ready = dict(incoming), [i for i in ids if not incoming[i]]
    while ready:
        current = ready.pop()
        for nxt in outgoing[current]:
            remaining[nxt] = remaining[nxt] - {current}
            if not remaining[nxt]:
                ready.append(nxt)
        remaining.pop(current)
    assert not remaining, "Graph contains a cycle"


def test_ten_core_fixtures_cover_categories_and_ambiguity():
    assert len(CORE) == 10
    fixtures = [load(p) for p in CORE]
    assert COVERAGE <= set().union(*(f["categories"] for f in fixtures))
    assert any(f["result"]["status"] == "needs_clarification" for f in fixtures)
    assert any(f["result"]["status"] == "runnable" for f in fixtures)


def test_prompts_json_lists_the_core_fixtures():
    prompts = {p["id"]: p["prompt"] for p in load(EXAMPLES / "prompts.json")}
    assert prompts == {f["id"]: f["prompt"] for f in map(load, CORE)}


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_fixture_shape(path):
    fixture = load(path)
    assert set(fixture) == REQUIRED
    assert fixture["id"] == path.stem
    result = fixture["result"]
    status = result["status"]
    assert status in STATUSES
    assert result["user_message"]
    if status in NEEDS_MISSING:
        assert result["missing"], "Non-runnable results must name what is missing"
    for item in result.get("missing", []):
        assert set(item) == {"layer", "capability", "blocked_by", "reason"}
    if status == "needs_clarification":
        assert result["questions"]
    if status == "invalid_request":
        assert result["suggestion"] and "requested_pipeline" in result
    if status != "runnable":
        assert "pipeline" not in result, "Only runnable fixtures carry an executable pipeline"
    assert ("target_pipeline" in result) == (status in NEEDS_TARGET)


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_pipelines_validate_or_are_rejected(path):
    result = load(path)["result"]
    if result["status"] == "runnable":
        assert result["provenance"] in {"simulated", "replayed", "live"}
        validate_pipeline(Pipeline.model_validate(result["pipeline"]))
    if "target_pipeline" in result:
        assert result["target_pipeline"]["executable"] is False
        future = Pipeline.model_validate(result["target_pipeline"]["pipeline"])
        assert_well_formed_graph(future)
        with pytest.raises(ValueError):
            validate_pipeline(future)
    if "requested_pipeline" in result:
        with pytest.raises(ValueError):
            validate_pipeline(Pipeline.model_validate(result["requested_pipeline"]))


def test_emg_golden_matches_checked_in_example():
    golden = load(GOLDEN / "emg-gesture.json")["result"]["pipeline"]
    assert golden == load(EXAMPLES / "emg-pipeline.json")
