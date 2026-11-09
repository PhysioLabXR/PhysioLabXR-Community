import {
  ArrowRight,
  Cpu,
  Devices,
  Function as FunctionIcon,
  Monitor,
} from "@phosphor-icons/react";
import type { Pipeline } from "../lib/api";
const icons = {
  device: Devices,
  processing: FunctionIcon,
  model: Cpu,
  application: Monitor,
};
const titles: Record<string, string> = {
  "simulated-emg": "Simulated EMG",
  bandpass: "Bandpass filter",
  "sliding-window": "Sliding window",
  "demo-emg-rms": "Gesture classifier",
  "web-preview": "Web application",
};
export function PipelineCanvas({
  pipeline,
  selected,
  onSelect,
  running,
}: {
  pipeline: Pipeline;
  selected: string;
  onSelect: (id: string) => void;
  running: boolean;
}) {
  return (
    <div className="canvas" aria-label="Pipeline graph">
      <div className="graph-caption">
        <span className="kicker">YOUR PIPELINE</span>
        <span>Signal → processing → inference → application</span>
      </div>
      <div className="nodes">
        {pipeline.nodes.map((node, i) => {
          const Icon = icons[node.kind];
          return (
            <div className="node-group" key={node.id}>
              <button
                className={`node ${selected === node.id ? "selected" : ""}`}
                aria-pressed={selected === node.id}
                onClick={() => onSelect(node.id)}
              >
                <span className="node-top">
                  <Icon size={22} />
                  <span className={`dot ${running ? "live" : ""}`} />
                </span>
                <span className="kicker">{node.kind}</span>
                <strong>
                  {titles[node.implementation] ?? node.implementation}
                </strong>
                <span className="node-meta">
                  {Object.entries(node.parameters ?? {})
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(" · ") ||
                    (node.kind === "device"
                      ? "EMG · 100 Hz · 1 channel"
                      : "Browser preview + export")}
                </span>
              </button>
              {i < pipeline.nodes.length - 1 && (
                <ArrowRight className="wire" size={28} />
              )}
            </div>
          );
        })}
      </div>
      <div className="canvas-note">
        Select a component to inspect its contract and parameters.
      </div>
    </div>
  );
}
