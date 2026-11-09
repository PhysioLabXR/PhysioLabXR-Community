import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from physiolab.applications.export import export_preview
from physiolab.contracts import Catalog, Pipeline, Project, ProjectCreate, Session
from physiolab.devices.catalog import CATALOG
from physiolab.orchestration.planner import TemplatePlanner
from physiolab.orchestration.runtime import Runtime
from physiolab.orchestration.validation import validate_pipeline
from physiolab.storage import Store


def create_app(data_dir: Path | None = None, interval: float = 0.2, max_batches: int = 300):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(data_dir or Path(os.getenv("PHYSIOLAB_DATA_DIR", ".data")))
        app.state.runtime = Runtime(app.state.store, interval=interval, max_batches=max_batches)
        yield
        await app.state.runtime.close()

    app = FastAPI(
        title="PhysioLab",
        version="0.1.0",
        lifespan=lifespan,
        description="Local development API. Simulated data only; no multi-user authentication yet.",
    )

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": "Resource not found"})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "simulation", "version": "0.1.0"}

    @app.get("/api/catalog", response_model=Catalog)
    def catalog():
        return CATALOG

    @app.get("/api/projects", response_model=list[Project])
    def projects():
        return app.state.store.projects()

    @app.post("/api/projects", response_model=Project, status_code=201)
    def create_project(request: ProjectCreate):
        project = Project(
            name=request.name, prompt=request.prompt, pipeline=TemplatePlanner().plan(request)
        )
        app.state.store.save_project(project)
        return project

    @app.get("/api/projects/{id}", response_model=Project)
    def project(id: UUID):
        return app.state.store.project(id)

    @app.put("/api/projects/{id}/pipeline", response_model=Project)
    def update_pipeline(id: UUID, pipeline: Pipeline):
        project = app.state.store.project(id)
        if any(s.project_id == id for s in app.state.runtime.active.values()):
            raise HTTPException(409, "Stop the run before changing the pipeline")
        try:
            validate_pipeline(pipeline)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        project.pipeline = pipeline
        app.state.store.save_project(project)
        return project

    @app.post("/api/projects/{id}/sessions", response_model=Session, status_code=201)
    async def start(id: UUID):
        try:
            return app.state.runtime.start(app.state.store.project(id))
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/projects/{id}/sessions", response_model=list[Session])
    def sessions(id: UUID):
        app.state.store.project(id)
        return sorted(
            (s for s in app.state.store.sessions() if s.project_id == id),
            key=lambda s: s.started_at,
            reverse=True,
        )

    @app.get("/api/sessions/{id}", response_model=Session)
    def session(id: UUID):
        return app.state.store.session(id)

    @app.post("/api/sessions/{id}/stop", response_model=Session)
    async def stop(id: UUID):
        return await app.state.runtime.stop(id)

    @app.get("/api/sessions/{id}/export")
    def export(id: UUID):
        return JSONResponse(
            app.state.store.session(id).model_dump(mode="json"),
            headers={"Content-Disposition": f'attachment; filename="{id}.json"'},
        )

    @app.get("/api/sessions/{id}/app", response_class=HTMLResponse)
    def application(id: UUID):
        session = app.state.store.session(id)
        project = app.state.store.project(session.project_id)
        return HTMLResponse(
            export_preview(project, session),
            headers={"Content-Disposition": f'attachment; filename="physiolab-{id}.html"'},
        )

    # Production Docker build copies web/dist alongside the backend; API routes take precedence.
    web_dir = Path(os.getenv("PHYSIOLAB_WEB_DIR", "../web/dist"))
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = create_app()
