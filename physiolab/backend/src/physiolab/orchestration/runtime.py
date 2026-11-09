import asyncio

from physiolab.contracts import Project, Session, now
from physiolab.devices.simulated import SimulatedEMG
from physiolab.models.demo import DemoEMGModel
from physiolab.orchestration.validation import validate_pipeline
from physiolab.processing.modules import Bandpass, SlidingWindow
from physiolab.storage import Store


class Runtime:
    def __init__(self, store: Store, interval: float = 0.2, max_batches: int = 300):
        self.store, self.interval, self.max_batches = store, interval, max_batches
        self.tasks: dict = {}
        self.active: dict = {}

    def start(self, project: Project) -> Session:
        validate_pipeline(project.pipeline)
        if any(s.project_id == project.id for s in self.active.values()):
            raise ValueError("This project already has a running session")
        if len(self.active) >= 4:
            raise ValueError("This development runtime allows four concurrent sessions")
        session = Session(project_id=project.id, pipeline=project.pipeline.model_copy(deep=True))
        self.store.save_session(session)
        self.active[session.id] = session
        self.tasks[session.id] = asyncio.create_task(self._run(session))
        return session

    async def _run(self, session: Session):
        nodes = session.pipeline.nodes
        device = SimulatedEMG(session.started_at.timestamp())
        try:
            bandpass = Bandpass(**nodes[1].parameters, rate_hz=100)
            window = SlidingWindow(**{k: int(v) for k, v in nodes[2].parameters.items()})
            model = DemoEMGModel(**nodes[3].parameters)
            device.start()
            for _ in range(self.max_batches):
                batch = device.read()
                session.raw.append(batch)
                session.inferences.extend(
                    model.infer(w) for w in window.process(bandpass.process(batch))
                )
                self.store.save_session(session)
                await asyncio.sleep(self.interval)
            session.status = "completed"
        except asyncio.CancelledError:
            session.status = "stopped"
        except Exception as error:
            session.status, session.error = "failed", str(error)
        finally:
            device.stop()
            session.ended_at = now()
            self.store.save_session(session)
            self.active.pop(session.id, None)
            self.tasks.pop(session.id, None)

    async def stop(self, id):
        task = self.tasks.get(id)
        if task:
            # Let a newly-created task enter its try/finally before cancellation.
            await asyncio.sleep(0)
            task.cancel()
            await task
        return self.store.session(id)

    async def close(self):
        for id in list(self.tasks):
            await self.stop(id)
