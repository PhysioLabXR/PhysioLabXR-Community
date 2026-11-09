"""Public contracts. Generate web types from app.openapi(); do not duplicate these shapes."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, json_schema_serialization_defaults_required=True
    )


class Channel(Contract):
    name: str
    unit: str


class StreamDescriptor(Contract):
    id: str
    device_id: str
    modality: str
    sample_rate_hz: float = Field(gt=0)
    channels: list[Channel] = Field(min_length=1)
    delivery: Literal["live", "batch", "simulated"]
    clock: Literal["unix_seconds"] = "unix_seconds"


class SampleBatch(Contract):
    stream: StreamDescriptor
    timestamps: list[float] = Field(min_length=1)
    samples: list[list[float]] = Field(min_length=1)
    sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def dimensions(self):
        if len(self.samples) != len(self.timestamps):
            raise ValueError("One timestamp is required per sample row")
        if any(len(row) != len(self.stream.channels) for row in self.samples):
            raise ValueError("Sample columns must match stream channels")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Timestamps must increase strictly within a batch")
        return self


class Node(Contract):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["device", "processing", "model", "application"]
    implementation: str
    parameters: dict[str, float | str | bool] = Field(default_factory=dict)


class Edge(Contract):
    source: str
    target: str


class Pipeline(Contract):
    schema_version: Literal["1.0"] = "1.0"
    nodes: list[Node] = Field(min_length=1, max_length=50)
    edges: list[Edge] = Field(max_length=100)


class ProjectCreate(Contract):
    name: str = Field(default="EMG gesture demo", min_length=1, max_length=100)
    prompt: str = Field(
        default="Show a gesture when the simulated muscle signal activates.", max_length=2000
    )
    template: Literal["emg-demo"] = "emg-demo"


class Project(Contract):
    id: UUID = Field(default_factory=uuid4)
    name: str
    prompt: str
    pipeline: Pipeline
    created_at: datetime = Field(default_factory=now)


class Inference(Contract):
    timestamp: float
    model_id: str
    label: Literal["rest", "gesture"]
    value: float
    unit: Literal["rms"] = "rms"
    simulated: Literal[True] = True


class Session(Contract):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    pipeline: Pipeline
    status: Literal["running", "completed", "stopped", "failed", "interrupted"] = "running"
    started_at: datetime = Field(default_factory=now)
    ended_at: datetime | None = None
    raw: list[SampleBatch] = Field(default_factory=list)
    inferences: list[Inference] = Field(default_factory=list)
    error: str | None = None
    simulated: Literal[True] = True


class DeviceInfo(Contract):
    id: str
    name: str
    delivery: Literal["live", "batch", "simulated", "unverified"]
    status: Literal["ready", "planned", "legacy-adapter"]
    notes: str


class Catalog(Contract):
    devices: list[DeviceInfo]
    processors: list[str]
    models: list[str]
    targets: list[str]
