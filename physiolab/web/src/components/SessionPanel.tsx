import { DownloadSimple } from "@phosphor-icons/react";
import type { Session } from "../lib/api";
export function SessionPanel({
  sessions,
  selected,
  onSelect,
}: {
  sessions: Session[];
  selected: Session | null;
  onSelect: (id: string) => void;
}) {
  const latest = selected?.inferences?.at(-1);
  const values =
    selected?.raw?.flatMap((b) => b.samples.map((row) => row[0])).slice(-200) ??
    [];
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${i * 3},${55 - v * 45}`)
    .join(" ");
  return (
    <section className="session-panel">
      <div className="section-title">
        <div>
          <span className="kicker">LIVE PREVIEW</span>
          <h2>From signal to action</h2>
        </div>
        <span className="tag">Simulated data</span>
      </div>
      <div className="preview-grid">
        <div className="signal-card">
          <span className="muted">EMG · arbitrary units · 100 Hz</span>
          <svg viewBox="0 0 600 110" role="img" aria-label="EMG waveform">
            <path
              d={path}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
        </div>
        <div className="inference-card">
          <span className="kicker">GESTURE OUTPUT</span>
          <strong data-testid="inference">
            {latest?.label ?? "Waiting for samples"}
          </strong>
          <span className="muted">
            {latest
              ? `${latest.value.toFixed(3)} RMS`
              : "Run the pipeline to begin"}
          </span>
        </div>
      </div>
      <div className="section-title">
        <h3>Run sessions</h3>
        {selected && (
          <div className="actions">
            <a className="btn" href={`/api/sessions/${selected.id}/export`}>
              <DownloadSimple /> Raw data + results
            </a>
            <a className="btn" href={`/api/sessions/${selected.id}/app`}>
              <DownloadSimple /> Web app
            </a>
          </div>
        )}
      </div>
      {sessions.length ? (
        <table>
          <thead>
            <tr>
              <th>Started</th>
              <th>Status</th>
              <th>Samples</th>
              <th>Inferences</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr
                key={s.id}
                className={selected?.id === s.id ? "active-row" : ""}
              >
                <td>
                  <button
                    className="text-button"
                    onClick={() => onSelect(s.id)}
                  >
                    {new Date(s.started_at).toLocaleString()}
                  </button>
                </td>
                <td>{s.status}</td>
                <td>{s.raw?.reduce((n, b) => n + b.samples.length, 0)}</td>
                <td>{s.inferences?.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="empty">Your recorded runs will appear here.</p>
      )}
      {selected?.error && <p role="alert">{selected.error}</p>}
    </section>
  );
}
