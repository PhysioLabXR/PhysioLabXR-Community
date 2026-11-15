"""Generation providers for AgentPlanner.

RuleBasedProvider is an offline, deterministic baseline: no API key, no network. It exists so the
planner, validation and evaluation harness can be tested before any LLM is wired in. It is
keyword matching, not language understanding, and is expected to lose to real models on new
prompts.

CallableProvider wraps any function (instructions, prompt) -> str, so Claude, Gemini or Codex
can be plugged in from a script or notebook without adding SDKs or secrets to this repo.
"""

import json
import os
import re
from collections.abc import Callable

from physiolab.contracts import Catalog
from physiolab.devices.catalog import CATALOG

# capability -> (layer, blocked_by, reason)
BLOCKERS = {
    "apple-watch": ("device", "PHY-6", "Apple Watch connector is planned only."),
    "oura": ("device", "PHY-7", "Oura V2 sync is planned only (batch delivery)."),
    "legacy-lsl-eeg": ("device", "PHY-8", "EEG bridge is not yet validated."),
    "legacy-eye-tracker": ("device", "PHY-8", "Eye-tracker bridge is not yet validated."),
    "phone-motion": (
        "device",
        "unassigned (after PHY-5)",
        "Phone motion connector is planned only.",
    ),
    "real-emg": ("device", "PHY-8", "No live EMG source exists."),
    "focus-estimator": ("model", "PHY-10/PHY-11", "No validated focus model."),
    "cognitive-load-estimator": ("model", "PHY-10/PHY-11", "No validated cognitive-load model."),
    "stress-estimator": ("model", "PHY-10/PHY-11", "No validated stress model."),
    "p300-classifier": ("model", "PHY-11/PHY-12", "No P300 model."),
    "rep-counter": ("model", "unassigned", "No repetition counter."),
    "descriptive-summary": ("processing", "PHY-9", "No aggregation modules."),
    "phone": ("application", "PHY-15", "No phone target."),
    "quest": ("application", "PHY-15", "No Quest target."),
    "web-game": ("application", "PHY-15", "No game target."),
    "web-speller": ("application", "PHY-15", "No speller target."),
    "clinician-dashboard": ("application", "PHY-15", "No dashboard target."),
    "web-session-export": ("application", "PHY-15", "No multi-stream export."),
}

DEVICES = [
    (r"apple watch|\bwatch\b", "apple-watch"),
    (r"\boura\b", "oura"),
    (r"\beeg\b", "legacy-lsl-eeg"),
    (r"eye.?track", "legacy-eye-tracker"),
    (r"accelerometer|phone motion", "phone-motion"),
]
MODELS = [
    (r"\bfocus\b", "focus-estimator"),
    (r"cognitive load", "cognitive-load-estimator"),
    (r"\bstress\b", "stress-estimator"),
    (r"\bp300\b", "p300-classifier"),
    (r"repetition|\breps?\b", "rep-counter"),
    (r"summary|summari[sz]e", "descriptive-summary"),
]
TARGETS = [
    (r"\bquest\b", "quest"),
    (r"dashboard.*clinician|clinician.*dashboard", "clinician-dashboard"),
    (r"\bphone\b(?!'s accelerometer)", "phone"),
    (r"dinosaur|\bgame\b", "web-game"),
    (r"speller", "web-speller"),
    (r"export", "web-session-export"),
    (r"web preview|\bweb\b", "web-preview"),
]
REVIEW = r"patient|clinician|clinical|discharged"
CODE = r"python|script|run (this|my) code"


def _miss(capability: str) -> dict:
    layer, blocked_by, reason = BLOCKERS[capability]
    return {"layer": layer, "capability": capability, "blocked_by": blocked_by, "reason": reason}


def _emg_pipeline(threshold=0.25, low=10.0, high=40.0) -> dict:
    nodes = [
        {"id": "sensor", "kind": "device", "implementation": "simulated-emg", "parameters": {}},
        {
            "id": "filter",
            "kind": "processing",
            "implementation": "bandpass",
            "parameters": {"low_hz": low, "high_hz": high},
        },
        {
            "id": "window",
            "kind": "processing",
            "implementation": "sliding-window",
            "parameters": {"size": 100.0, "stride": 50.0},
        },
        {
            "id": "model",
            "kind": "model",
            "implementation": "demo-emg-rms",
            "parameters": {"threshold": threshold},
        },
        {"id": "app", "kind": "application", "implementation": "web-preview", "parameters": {}},
    ]
    edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return {"schema_version": "1.0", "nodes": nodes, "edges": edges}


class RuleBasedProvider:
    name = "rule-based"
    model = "keyword-rules-v0"

    def __init__(self, catalog: Catalog = CATALOG):
        self.catalog = catalog

    def generate(self, instructions: str, prompt: str, feedback: str | None = None) -> str:
        return json.dumps(self._plan(prompt.lower()))

    def _plan(self, text: str) -> dict:
        if re.search(CODE, text):
            return {
                "status": "refused",
                "missing": [
                    {
                        "layer": "processing",
                        "capability": "arbitrary code execution",
                        "blocked_by": "safety boundary",
                        "reason": "Only registered modules run.",
                    }
                ],
                "user_message": "I can't run custom code. I can only use the built-in modules.",
            }

        emg = re.search(r"\bemg\b|muscle", text)
        simulated = "simulated" in text
        devices = [cap for pattern, cap in DEVICES if re.search(pattern, text)]
        if emg:
            devices.append("simulated-emg" if simulated else "real-emg")
        models = [cap for pattern, cap in MODELS if re.search(pattern, text)]
        targets = [cap for pattern, cap in TARGETS if re.search(pattern, text)]
        if "web-preview" in targets and len(targets) > 1:
            targets.remove("web-preview")  # a more specific web target was named

        if not devices:
            # Never default to the EMG demo: an unrecognized or unnamed device means ask.
            questions = ["Which device should I use? I don't recognize a supported one."]
            if not targets:
                questions.append("Should the app run on phone, web or Quest?")
            questions.append("What exactly should the app show?")
            return {
                "status": "needs_clarification",
                "questions": questions,
                "missing": [_miss(m) for m in models],
                "user_message": "Before I plan this, which device do you have? I can only run "
                "the simulated EMG demo today.",
            }

        band = re.search(r"(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*(\d+(?:\.\d+)?)\s*hz", text)
        if band and devices == ["simulated-emg"]:
            low, high = float(band.group(1)), float(band.group(2))
            if not 0 < low < high < 50:
                return {
                    "status": "invalid_request",
                    "suggestion": {"low_hz": 10.0, "high_hz": 40.0},
                    "user_message": "The simulated signal is sampled at 100 Hz, so filters must "
                    "stay below 50 Hz. Want 10-40 Hz instead?",
                }

        missing = [_miss(c) for c in devices + models + targets if c in BLOCKERS]
        if len([d for d in devices if d != "simulated-emg"]) > 1 or ("p300-classifier" in models):
            missing.append(
                {
                    "layer": "runtime",
                    "capability": "multi-stream graph",
                    "blocked_by": "contract extension (team decision)",
                    "reason": "The runtime runs a single chain only.",
                }
            )

        if re.search(REVIEW, text):
            return {
                "status": "requires_review",
                "missing": missing,
                "user_message": "This needs clinical and privacy review before I can plan it.",
            }

        if not missing:
            threshold = 0.25
            if "more sensitive" in text:
                threshold = 0.15
            elif "less sensitive" in text:
                threshold = 0.35
            pipeline = _emg_pipeline(threshold)
            if band:
                pipeline = _emg_pipeline(threshold, float(band.group(1)), float(band.group(2)))
            return {
                "status": "runnable",
                "pipeline": pipeline,
                "user_message": "Built a simulated EMG gesture demo for the web preview "
                f"(threshold {threshold}). It uses a fake signal, not a real sensor.",
            }

        runnable_layers = devices == ["simulated-emg"] and not models
        status = "partial" if runnable_layers else "unsupported"
        names = ", ".join(m["capability"] for m in missing)
        return {
            "status": status,
            "missing": missing,
            "user_message": f"I can't build this yet. Missing: {names}.",
        }


class CallableProvider:
    """Wrap any LLM call. Example:

    def ask_claude(instructions, prompt):
        return client.messages.create(model=MODEL, system=instructions, ...).content[0].text

    planner = AgentPlanner(CallableProvider("anthropic", MODEL, ask_claude))
    """

    def __init__(self, name: str, model: str, call: Callable[[str, str], str]):
        self.name, self.model, self.call = name, model, call

    def generate(self, instructions: str, prompt: str, feedback: str | None = None) -> str:
        if feedback:
            prompt = f"{prompt}\n\n{feedback}\nReturn corrected JSON only."
        return self.call(instructions, prompt)


def provider_from_env() -> RuleBasedProvider:
    """Default provider. LLM providers are configured in scripts, never with committed keys."""
    name = os.environ.get("PHYSIOLAB_PLANNER_PROVIDER", "rule-based")
    if name != "rule-based":
        raise ValueError(
            f"Provider '{name}' is not configured in the server. Wrap it with CallableProvider "
            "in scripts/eval_planner.py, or set PHYSIOLAB_PLANNER_PROVIDER=rule-based."
        )
    return RuleBasedProvider()
