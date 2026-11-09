import asyncio
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from physiolab.api import create_app
from physiolab.contracts import Project, ProjectCreate, SampleBatch, Session
from physiolab.devices.legacy import LegacyDeviceAdapter
from physiolab.devices.simulated import SimulatedEMG
from physiolab.models.demo import DemoEMGModel
from physiolab.orchestration.planner import TemplatePlanner
from physiolab.orchestration.runtime import Runtime
from physiolab.orchestration.validation import validate_pipeline
from physiolab.processing.modules import Bandpass, SlidingWindow
from physiolab.storage import Store


def test_signal_processing_and_deterministic_labels():
    source = SimulatedEMG(1000)
    source.start()
    bp, window, model = Bandpass(10, 40, 100), SlidingWindow(100, 50), DemoEMGModel()
    results = []
    for _ in range(20):
        results.extend(model.infer(w) for w in window.process(bp.process(source.read())))
    assert len(results) == 7  # 400 samples; 100-sample windows with stride 50
    assert results[0].label == "rest"
    assert results[-1].label == "gesture"
    assert results[-1].timestamp == pytest.approx(1003.99)
    assert all(r.simulated for r in results)
    source.stop()
    with pytest.raises(RuntimeError):
        source.read()


def test_bandpass_attenuates_out_of_band_energy_and_is_chunk_continuous():
    stream = SimulatedEMG.descriptor
    times = np.arange(1000) / 100
    values = np.sin(2 * np.pi * 2 * times) + np.sin(2 * np.pi * 20 * times)
    full = SampleBatch(
        stream=stream, timestamps=times.tolist(), samples=values[:, None].tolist(), sequence=0
    )
    expected = np.array(Bandpass(10, 40, 100).process(full).samples)[:, 0]
    bp, actual = Bandpass(10, 40, 100), []
    for i in range(0, 1000, 20):
        part = full.model_copy(
            update={"timestamps": full.timestamps[i : i + 20], "samples": full.samples[i : i + 20]}
        )
        actual.extend(row[0] for row in bp.process(part).samples)
    np.testing.assert_allclose(actual, expected)
    fft = np.abs(np.fft.rfft(expected[200:]))
    frequencies = np.fft.rfftfreq(800, 0.01)
    assert fft[np.argmin(abs(frequencies - 20))] > 100 * fft[np.argmin(abs(frequencies - 2))]


def test_sample_contract_rejects_misaligned_data():
    with pytest.raises(ValidationError):
        SampleBatch(stream=SimulatedEMG.descriptor, timestamps=[1, 2], samples=[[1]], sequence=0)
    with pytest.raises(ValidationError):
        SampleBatch(
            stream=SimulatedEMG.descriptor, timestamps=[1, 1], samples=[[1], [2]], sequence=0
        )
    with pytest.raises(ValidationError):
        SampleBatch(
            stream=SimulatedEMG.descriptor, timestamps=[1], samples=[[float("nan")]], sequence=0
        )


@pytest.mark.parametrize(
    "mutation", ["cycle", "duplicate", "unknown", "nyquist", "fractional", "boolean"]
)
def test_invalid_graphs_cannot_execute(mutation):
    pipeline = TemplatePlanner().plan(ProjectCreate())
    if mutation == "cycle":
        pipeline.edges[0].source = "app"
    elif mutation == "duplicate":
        pipeline.nodes[-1].id = pipeline.nodes[0].id
    elif mutation == "unknown":
        pipeline.nodes[0].implementation = "apple-watch"
    elif mutation == "nyquist":
        pipeline.nodes[1].parameters["high_hz"] = 50
    elif mutation == "fractional":
        pipeline.nodes[2].parameters["size"] = 2.5
    else:
        pipeline.nodes[3].parameters["threshold"] = True
    with pytest.raises(ValueError):
        validate_pipeline(pipeline)


def test_legacy_adapter_transposes_and_converts_clock():
    class Device:
        def start_stream(self):
            pass

        def stop_stream(self):
            pass

        def process_frames(self):
            return [[1, 2, 3]], [1, 2, 3], ["info"]

    adapter = LegacyDeviceAdapter(Device(), SimulatedEMG.descriptor, lambda t: t + 1000)
    batch = adapter.read()
    assert batch.samples == [[1], [2], [3]]
    assert batch.timestamps == [1001, 1002, 1003]


def test_stop_immediately_and_shutdown_are_terminal(tmp_path):
    async def scenario():
        store = Store(tmp_path)
        runtime = Runtime(store, interval=0.001)
        project = Project(
            name="demo", prompt="gesture", pipeline=TemplatePlanner().plan(ProjectCreate())
        )
        session = runtime.start(project)
        with pytest.raises(ValueError):
            runtime.start(project)
        stopped = await runtime.stop(session.id)
        assert stopped.status == "stopped"
        assert stopped.ended_at is not None
        assert not runtime.tasks and not runtime.active
        assert (await runtime.stop(session.id)).status == "stopped"
        runtime.start(project)
        await runtime.close()
        assert not runtime.tasks and not runtime.active

    asyncio.run(scenario())


def test_api_run_record_export_and_restart(tmp_path):
    import time

    app = create_app(tmp_path, interval=0.001, max_batches=20)
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "<script>alert(1)</script>"}).json()
        id = project["id"]
        invalid = deepcopy(project["pipeline"])
        invalid["nodes"][1]["parameters"]["high_hz"] = 70
        assert client.put(f"/api/projects/{id}/pipeline", json=invalid).status_code == 422
        session = client.post(f"/api/projects/{id}/sessions").json()
        sid = session["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = client.get(f"/api/sessions/{sid}").json()
            if result["status"] != "running":
                break
            time.sleep(0.01)
        assert result["status"] == "completed"
        assert len(result["raw"]) == 20
        assert {i["label"] for i in result["inferences"]} == {"rest", "gesture"}
        assert client.get(f"/api/sessions/{sid}/export").json() == result
        html = client.get(f"/api/sessions/{sid}/app").text
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "Replay session" in html
        assert client.get("/api/sessions/not-a-uuid").status_code == 422
    with TestClient(create_app(tmp_path)) as client:
        assert len(client.get("/api/projects").json()) == 1
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "completed"


def test_restart_marks_unfinished_recording_interrupted(tmp_path):
    store = Store(tmp_path)
    project = Project(name="demo", prompt="", pipeline=TemplatePlanner().plan(ProjectCreate()))
    session = Session(project_id=project.id, pipeline=project.pipeline)
    store.save_session(session)
    recovered = Store(tmp_path).session(session.id)
    assert recovered.status == "interrupted"
    assert isinstance(recovered.ended_at, datetime)
