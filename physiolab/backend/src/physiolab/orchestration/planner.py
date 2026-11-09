"""Marvin: replace the explicit template planner behind this contract with a validated agent.

The user's prompt is preserved as intent; this starter does not claim to interpret it.
"""

from typing import Protocol

from physiolab.contracts import Edge, Node, Pipeline, ProjectCreate


class PipelinePlanner(Protocol):
    def plan(self, request: ProjectCreate) -> Pipeline: ...


class TemplatePlanner:
    def plan(self, request: ProjectCreate) -> Pipeline:
        return Pipeline(
            nodes=[
                Node(id="sensor", kind="device", implementation="simulated-emg"),
                Node(
                    id="filter",
                    kind="processing",
                    implementation="bandpass",
                    parameters={"low_hz": 10.0, "high_hz": 40.0},
                ),
                Node(
                    id="window",
                    kind="processing",
                    implementation="sliding-window",
                    parameters={"size": 100, "stride": 50},
                ),
                Node(
                    id="model",
                    kind="model",
                    implementation="demo-emg-rms",
                    parameters={"threshold": 0.25},
                ),
                Node(id="app", kind="application", implementation="web-preview"),
            ],
            edges=[
                Edge(source=a, target=b)
                for a, b in [
                    ("sensor", "filter"),
                    ("filter", "window"),
                    ("window", "model"),
                    ("model", "app"),
                ]
            ],
        )
