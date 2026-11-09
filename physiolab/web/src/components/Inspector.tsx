import { useState } from "react";
import type { Pipeline } from "../lib/api";
export function Inspector({
  node,
  disabled,
  save,
}: {
  node: Pipeline["nodes"][number];
  disabled: boolean;
  save: (parameters: Record<string, number>) => Promise<void>;
}) {
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(
      Object.entries(node.parameters ?? {}).map(([k, v]) => [k, Number(v)]),
    ),
  );
  const owners = {
    device: "Serena",
    processing: "Haosen",
    model: "Haosen",
    application: "Marvin",
  };
  return (
    <aside className="inspector">
      <span className="kicker">COMPONENT INSPECTOR</span>
      <h3>{node.implementation}</h3>
      <p className="muted">Owner · {owners[node.kind]}</p>
      <dl>
        <dt>Node ID</dt>
        <dd>{node.id}</dd>
        <dt>Layer</dt>
        <dd>{node.kind}</dd>
      </dl>
      {node.kind === "device" && (
        <p>
          This development source produces a deterministic signal. Real sensor
          connectors appear in the device catalog.
        </p>
      )}
      {node.kind === "model" && (
        <p>
          RMS threshold baseline for simulated EMG. This model is not promptable
          and does not estimate focus or cognitive load.
        </p>
      )}
      {node.kind === "application" && (
        <p>
          Replay and export a recorded session as a standalone web app. Phone
          and Quest deployment adapters are next.
        </p>
      )}
      {Object.entries(values).map(([key, value]) => (
        <label className="field" key={key}>
          {key}
          <input
            type="number"
            step="any"
            aria-label={key}
            value={value}
            disabled={disabled}
            onChange={(e) =>
              setValues({ ...values, [key]: Number(e.target.value) })
            }
          />
        </label>
      ))}
      {!!Object.keys(values).length && (
        <button
          className="btn primary"
          disabled={
            disabled || Object.values(values).some((v) => !Number.isFinite(v))
          }
          onClick={() => void save(values)}
        >
          Save parameters
        </button>
      )}
    </aside>
  );
}
