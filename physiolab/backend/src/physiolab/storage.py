"""Atomic local JSON storage for one process. Replace with a tenant-scoped database for hosting."""

from pathlib import Path
from uuid import UUID

from physiolab.contracts import Project, Session, now


class Store:
    def __init__(self, root: Path):
        self.root = root
        for kind in ["projects", "sessions"]:
            (root / kind).mkdir(parents=True, exist_ok=True)
        for session in self.sessions():
            if session.status == "running":
                session.status, session.ended_at = "interrupted", now()
                session.error = "Server restarted before the run finished"
                self.save_session(session)

    def _save(self, kind, value):
        path = self.root / kind / f"{value.id}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(value.model_dump_json(indent=2))
        temp.replace(path)

    def save_project(self, project: Project):
        self._save("projects", project)

    def save_session(self, session: Session):
        self._save("sessions", session)

    def projects(self) -> list[Project]:
        return [
            Project.model_validate_json(p.read_text())
            for p in sorted((self.root / "projects").glob("*.json"))
        ]

    def sessions(self) -> list[Session]:
        return [
            Session.model_validate_json(p.read_text())
            for p in sorted((self.root / "sessions").glob("*.json"))
        ]

    def project(self, id: UUID) -> Project:
        return Project.model_validate_json((self.root / "projects" / f"{id}.json").read_text())

    def session(self, id: UUID) -> Session:
        return Session.model_validate_json((self.root / "sessions" / f"{id}.json").read_text())
