import { useCallback, useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Play,
  Plus,
  Stop,
  WaveSine,
} from "@phosphor-icons/react";
import {
  api,
  type Catalog,
  type Pipeline,
  type Project,
  type Session,
} from "./lib/api";
import { PipelineCanvas } from "./components/PipelineCanvas";
import { Inspector } from "./components/Inspector";
import { SessionPanel } from "./components/SessionPanel";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [selectedNode, setSelectedNode] = useState("sensor");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(
    "Show a gesture when the simulated muscle signal activates.",
  );
  const [name, setName] = useState("EMG gesture demo");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"flow" | "devices">("flow");
  const active = sessions.find((s) => s.status === "running");
  const session =
    sessions.find((s) => s.id === selectedSession) ?? sessions[0] ?? null;

  const action = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const loadProjects = useCallback(
    async () => setProjects(await api<Project[]>("/projects")),
    [],
  );
  useEffect(() => {
    Promise.all([
      loadProjects(),
      api<Catalog>("/catalog").then(setCatalog),
    ]).catch((e) => setError(e.message));
  }, [loadProjects]);
  useEffect(() => {
    if (!project) return;
    let live = true;
    const refresh = async () => {
      try {
        const result = await api<Session[]>(`/projects/${project.id}/sessions`);
        if (live) setSessions(result);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void refresh();
    const timer = setInterval(refresh, 700);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [project?.id]);
  const open = (p: Project) => {
    setProject(p);
    setSessions([]);
    setSelectedSession(null);
    setSelectedNode(p.pipeline.nodes[0].id);
    setTab("flow");
    setError("");
    setNotice("");
  };
  const save = async (parameters: Record<string, number>) =>
    action(async () => {
      if (!project) return;
      const pipeline: Pipeline = {
        ...project.pipeline,
        nodes: project.pipeline.nodes.map((n) =>
          n.id === selectedNode ? { ...n, parameters } : n,
        ),
      };
      const updated = await api<Project>(
        `/projects/${project.id}/pipeline`,
        "PUT",
        pipeline,
      );
      setProject(updated);
      await loadProjects();
      setNotice("Parameters saved");
    });

  return (
    <>
      <header className="topbar">
        <button className="brand" onClick={() => setProject(null)}>
          <WaveSine size={26} />
          PhysioLab
        </button>
        <span className="breadcrumb">
          {project ? `Projects / ${project.name}` : "Builder workspace"}
        </span>
        <div className="actions">
          <a
            className="btn quiet"
            href="https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab"
            target="_blank"
            rel="noreferrer"
          >
            <BookOpen /> Developer guide
          </a>
          {project && (
            <button
              className="btn primary"
              disabled={busy}
              onClick={() =>
                void action(async () => {
                  if (active) {
                    const s = await api<Session>(
                      `/sessions/${active.id}/stop`,
                      "POST",
                    );
                    setSessions((old) =>
                      old.map((x) => (x.id === s.id ? s : x)),
                    );
                  } else {
                    const s = await api<Session>(
                      `/projects/${project.id}/sessions`,
                      "POST",
                    );
                    setSessions((old) => [s, ...old]);
                    setSelectedSession(s.id);
                  }
                })
              }
            >
              {active ? <Stop weight="fill" /> : <Play weight="fill" />}
              {active ? "Stop run" : "Run simulation"}
            </button>
          )}
        </div>
      </header>
      {error && (
        <div role="alert" className="banner error">
          {error}
          <button
            className="text-button"
            onClick={() => void action(loadProjects)}
          >
            Retry connection
          </button>
        </div>
      )}
      {notice && (
        <div role="status" className="banner">
          {notice}
        </div>
      )}
      {!project ? (
        <main className="landing">
          <span className="kicker">PHYSIOLOGICAL SIGNALS, CONNECTED</span>
          <h1>
            Build an app that
            <br />
            <em>responds to you.</em>
          </h1>
          <p className="intro">
            Connect your sensors, shape the signal, and bring it to life.
            <br />
            Start with a working simulation, then make each layer your own.
          </p>
          <form
            className="create-card"
            onSubmit={(e) => {
              e.preventDefault();
              void action(async () => {
                const p = await api<Project>("/projects", "POST", {
                  name,
                  prompt,
                  template: "emg-demo",
                });
                await loadProjects();
                open(p);
              });
            }}
          >
            <label className="field">
              Project name
              <input
                required
                maxLength={100}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <label className="field">
              What would you like to build?
              <textarea
                maxLength={2000}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>
            <div className="create-footer">
              <div>
                <span className="tag">EMG gesture template</span>
                <p className="muted">
                  Your intent is saved. This starter runs the selected template.
                </p>
              </div>
              <button className="btn primary" disabled={busy} type="submit">
                Create project <ArrowRight />
              </button>
            </div>
          </form>
          <section>
            <div className="section-title">
              <h2>Your projects</h2>
              <span className="muted">{projects.length} projects</span>
            </div>
            <div className="project-grid">
              {projects.map((p) => (
                <button
                  className="project-card"
                  key={p.id}
                  onClick={() => open(p)}
                >
                  <WaveSine size={24} />
                  <strong>{p.name}</strong>
                  <span className="muted">{p.prompt}</span>
                  <span className="project-bottom">
                    {p.pipeline.nodes.length} components <ArrowRight />
                  </span>
                </button>
              ))}
              {!projects.length && (
                <div className="project-card empty">
                  <Plus size={24} />
                  Create your first project above.
                </div>
              )}
            </div>
          </section>
        </main>
      ) : (
        <main className="workspace">
          <nav className="workspace-nav">
            <button className="text-button" onClick={() => setProject(null)}>
              <ArrowLeft /> Projects
            </button>
            <div className="tabs">
              <button
                aria-pressed={tab === "flow"}
                onClick={() => setTab("flow")}
              >
                Pipeline
              </button>
              <button
                aria-pressed={tab === "devices"}
                onClick={() => setTab("devices")}
              >
                Device catalog
              </button>
            </div>
            <span className="muted">
              <span className={`dot ${active ? "live" : ""}`} />
              {active ? "Simulation running" : "Ready to run"}
            </span>
          </nav>
          {tab === "flow" ? (
            <>
              <div className="builder">
                <PipelineCanvas
                  pipeline={project.pipeline}
                  selected={selectedNode}
                  onSelect={setSelectedNode}
                  running={!!active}
                />
                <Inspector
                  key={`${project.id}:${selectedNode}`}
                  node={project.pipeline.nodes.find(
                    (n) => n.id === selectedNode,
                  )!}
                  disabled={busy || !!active}
                  save={save}
                />
              </div>
              <SessionPanel
                sessions={sessions}
                selected={session}
                onSelect={setSelectedSession}
              />
            </>
          ) : (
            <section className="device-section">
              <span className="kicker">DEVICE LAYER · SERENA</span>
              <h2>Connector library</h2>
              <p className="muted">
                Delivery and readiness are explicit for each connector.
              </p>
              <div className="project-grid">
                {catalog?.devices.map((d) => (
                  <article className="project-card" key={d.id}>
                    <strong>{d.name}</strong>
                    <div className="actions">
                      <span className="tag">{d.status}</span>
                      <span className="tag">{d.delivery}</span>
                    </div>
                    <p>{d.notes}</p>
                  </article>
                ))}
              </div>
            </section>
          )}
        </main>
      )}
      <footer>
        PhysioLab · development workspace{" "}
        <span>Devices / Signal processing / Models / Applications</span>
      </footer>
    </>
  );
}
