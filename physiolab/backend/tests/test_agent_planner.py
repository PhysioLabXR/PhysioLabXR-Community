"""PHY-14 prototype: the planner may only emit validated pipelines and must explain refusals."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from physiolab.api import create_app
from physiolab.contracts import ProjectCreate
from physiolab.devices.catalog import CATALOG
from physiolab.orchestration.agent import AgentPlanner, PlanNotRunnable, build_instructions
from physiolab.orchestration.providers import CallableProvider, RuleBasedProvider

GOLDEN = Path(__file__).resolve().parents[2] / "examples" / "golden"
FIXTURES = sorted(GOLDEN.rglob("*.json"))


class Scripted:
    """Returns canned responses in order, like a misbehaving LLM."""

    name, model = "scripted", "test"

    def __init__(self, *responses):
        self.responses, self.feedback = list(responses), []

    def generate(self, instructions, prompt, feedback=None):
        self.feedback.append(feedback)
        return self.responses.pop(0)


def golden_emg() -> dict:
    return json.loads((GOLDEN / "emg-gesture.json").read_text())["result"]["pipeline"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_rule_based_baseline_matches_golden_status(path):
    fixture = json.loads(path.read_text())
    result = AgentPlanner(RuleBasedProvider()).plan_detailed(fixture["prompt"])
    assert result.status == fixture["result"]["status"]
    if result.status == "runnable":
        assert result.pipeline.model_dump() == fixture["result"]["pipeline"]
    else:
        assert result.pipeline is None


@pytest.mark.parametrize(
    "prompt",
    [
        "Show my heart rate from my Garmin on the web.",
        "Build a web app.",
        "Show my Muse headband focus on a web page.",
        "Detect a gesture from my real EMG armband in the web preview.",
    ],
)
def test_unrecognized_or_real_devices_never_fall_back_to_the_demo(prompt):
    result = AgentPlanner(RuleBasedProvider()).plan_detailed(prompt)
    assert result.status != "runnable" and result.pipeline is None


def test_hallucinated_device_is_never_returned_as_runnable():
    bad = golden_emg()
    bad["nodes"][0]["implementation"] = "apple-watch"
    provider = Scripted(
        *[json.dumps({"status": "runnable", "pipeline": bad, "user_message": "ok"})] * 2
    )
    result = AgentPlanner(provider).plan_detailed("Use my Apple Watch")
    assert result.status == "unsupported"
    assert result.pipeline is None
    assert result.provenance.attempts == 2
    assert len(result.provenance.validation_errors) == 2
    assert provider.feedback[1].startswith("Your last answer was rejected")


def test_retry_with_feedback_can_recover():
    good = json.dumps({"status": "runnable", "pipeline": golden_emg(), "user_message": "ok"})
    result = AgentPlanner(Scripted("not json", good)).plan_detailed("simulated EMG demo")
    assert result.status == "runnable"
    assert result.provenance.attempts == 2


def test_out_of_range_parameters_and_unknown_fields_are_rejected():
    bad = golden_emg()
    bad["nodes"][1]["parameters"]["high_hz"] = 120.0
    responses = [
        json.dumps({"status": "runnable", "pipeline": bad, "user_message": "ok"}),
        json.dumps({"status": "runnable", "pipeline": golden_emg(), "user_message": "ok", "x": 1}),
    ]
    result = AgentPlanner(Scripted(*responses)).plan_detailed("filter 80-120 Hz")
    assert result.status == "unsupported" and result.pipeline is None


def test_refusals_cannot_smuggle_a_pipeline():
    response = {"status": "unsupported", "pipeline": golden_emg(), "user_message": "no"}
    result = AgentPlanner(Scripted(json.dumps(response))).plan_detailed("anything")
    assert result.status == "unsupported" and result.pipeline is None


def test_plan_keeps_the_pipeline_planner_interface():
    planner = AgentPlanner(RuleBasedProvider())
    request = ProjectCreate(prompt="Show a gesture when the simulated EMG signal activates.")
    assert planner.plan(request).model_dump() == golden_emg()
    with pytest.raises(PlanNotRunnable) as error:
        planner.plan(ProjectCreate(prompt="Show my Oura sleep on my phone."))
    assert error.value.result.missing


def test_provenance_and_callable_provider():
    seen = {}

    def fake_llm(instructions, prompt):
        seen["instructions"] = instructions
        return json.dumps(
            {
                "status": "needs_clarification",
                "questions": ["Which device?"],
                "user_message": "Which device?",
            }
        )

    result = AgentPlanner(CallableProvider("fake", "fake-1", fake_llm)).plan_detailed("stress app")
    assert result.provenance.provider == "fake" and result.provenance.model == "fake-1"
    assert len(result.provenance.prompt_sha256) == 64
    assert result.questions == ["Which device?"]
    for name in CATALOG.processors + CATALOG.models + CATALOG.targets:
        assert name in seen["instructions"]


def test_instructions_list_planned_devices_as_unavailable():
    text = build_instructions()
    assert "apple-watch (planned" in text and "simulated-emg" in text


def test_plan_endpoint_previews_without_creating_anything(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        ok = client.post("/api/plan", json={"prompt": "Show a gesture from the simulated EMG."})
        assert ok.status_code == 200 and ok.json()["status"] == "runnable"
        assert ok.json()["pipeline"] == golden_emg()
        no = client.post("/api/plan", json={"prompt": "Show my Oura sleep on my phone."}).json()
        assert no["status"] == "unsupported" and no["pipeline"] is None
        assert {m["blocked_by"] for m in no["missing"]} >= {"PHY-7", "PHY-15"}
        assert no["provenance"]["provider"] == "rule-based"
        assert client.post("/api/plan", json={"prompt": ""}).status_code == 422
        assert client.get("/api/projects").json() == []
