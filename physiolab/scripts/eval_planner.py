"""Score planner providers against the PHY-13 golden fixtures.

    uv run --directory backend python ../scripts/eval_planner.py
    uv run --directory backend python ../scripts/eval_planner.py --provider my_llms:claude

--provider takes module:function returning a provider (e.g. a CallableProvider wrapping an LLM
SDK). Keep that module and its API keys outside the repo. The intent score needs a separate
judge model with a pinned version and is reported as pending until one is configured.
"""

import argparse
import asyncio
import importlib
import json
import sys
import tempfile
from pathlib import Path

from physiolab.applications.export import export_preview
from physiolab.contracts import Project
from physiolab.orchestration.agent import AgentPlanner, PlanResult
from physiolab.orchestration.providers import RuleBasedProvider
from physiolab.orchestration.runtime import Runtime
from physiolab.storage import Store

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "examples" / "golden"


def load_provider(spec: str | None):
    if not spec:
        return RuleBasedProvider()
    module, function = spec.split(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), function)()


async def run_and_deploy(result: PlanResult, prompt: str) -> tuple[bool, bool]:
    """Run: the simulation produces inferences. Deploy: the web export renders them."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp))
        project = Project(name="eval", prompt=prompt, pipeline=result.pipeline)
        store.save_project(project)
        runtime = Runtime(store, interval=0, max_batches=5)
        session = runtime.start(project)
        await runtime.tasks[session.id]
        session = store.session(session.id)
        ran = session.status == "completed" and len(session.inferences) > 0
        html = export_preview(project, session) if ran else ""
        return ran, "Replay session" in html and session.inferences[0].label in html


def score(fixture: dict, result: PlanResult) -> dict:
    expected = fixture["result"]
    row = {
        "id": fixture["id"],
        "expected": expected["status"],
        "got": result.status,
        "status_match": result.status == expected["status"],
        "attempts": result.provenance.attempts,
    }
    if expected["status"] == "runnable":
        row["pipeline_match"] = (
            result.pipeline is not None and result.pipeline.model_dump() == expected["pipeline"]
        )
    wanted = {m["layer"] for m in expected.get("missing", [])}
    if wanted:
        found = {m.layer for m in result.missing}
        row["missing_recall"] = round(len(wanted & found) / len(wanted), 2)
    if expected["status"] == "needs_clarification":
        row["asked"] = bool(result.questions)
    return row


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", help="module:function returning a provider")
    parser.add_argument("--proposed", action="store_true", help="include proposed fixtures")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args()

    planner = AgentPlanner(load_provider(args.provider))
    paths = sorted(GOLDEN.glob("*.json"))
    if args.proposed:
        paths += sorted((GOLDEN / "proposed").glob("*.json"))

    rows, results = [], []
    for path in paths:
        fixture = json.loads(path.read_text())
        result = planner.plan_detailed(fixture["prompt"])
        row = score(fixture, result)
        if result.status == "runnable":
            row["run"], row["deploy"] = await run_and_deploy(result, fixture["prompt"])
        rows.append(row)
        results.append({"fixture": fixture["id"], "result": result.model_dump(mode="json")})

    p = planner.provider
    print(f"Provider: {p.name} / {p.model}   fixtures: {len(rows)}\n")
    print("| fixture | expected | got | status | pipeline | missing recall | run | deploy |")
    print("|---|---|---|---|---|---|---|---|")
    mark = {True: "yes", False: "NO", None: "-"}
    for r in rows:
        print(
            f"| {r['id']} | {r['expected']} | {r['got']} | {mark[r['status_match']]} | "
            f"{mark[r.get('pipeline_match')]} | {r.get('missing_recall', '-')} | "
            f"{mark[r.get('run')]} | {mark[r.get('deploy')]} |"
        )
    status_rate = sum(r["status_match"] for r in rows) / len(rows)
    recalls = [r["missing_recall"] for r in rows if "missing_recall" in r]
    print(f"\nStatus accuracy: {status_rate:.0%}")
    if recalls:
        print(f"Mean missing-layer recall: {sum(recalls) / len(recalls):.2f}")
    print("Intent score: pending (needs a separate, pinned judge model)")
    if args.json:
        payload = {"provider": p.name, "model": p.model, "rows": rows, "results": results}
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
