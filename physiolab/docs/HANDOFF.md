# Team handoff — 2026-11-08

Source: [Custom app development plan meeting](https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7)
and the supplied `PhysioLab Builder Import.zip`. The meeting transcript spells Haosen as
“Hausen”; the correct name used throughout this project is **Haosen**.

## Meeting direction

PhysioLab is a prompt-driven builder of custom physiological sensing apps. Users describe
their sensors and intended experience; the eventual agent assembles and deploys a pipeline.
Targets include phone, web, Quest XR and clinician dashboards. Device support starts with
Apple Watch and Oura, then phone motion and existing EEG/LSL connectors. Preprocessing
modules should expose constrained, tunable parameters. Open-source models stand in for
the not-yet-public physiological foundation model discussed in the meeting.

Validate near-term work with deterministic P300/EMG tasks. Focus/cognitive-load outcomes
need later self-report experiments and must not inherit the fixture's validation claim.
Pipeline evaluation asks: does it run, does it deploy, and does it satisfy the user's intent?
Compare generation providers against handcrafted golden pipelines, with a separate judge
and documented provider/model versions. Names mentioned in the meeting are candidates,
not verified model identifiers or committed integrations.

The meeting targets the YC batch on **2027-02-08** and a **WISC paper in April 2027**
(venue/deadline spelling to confirm). The next weekly sync is the first delivery checkpoint;
Linear starter issues use **2026-11-15** for the near-term inventory/review/prompt tasks.

## Ownership and first PRs

**Serena — device layer.** Implement capability inventory before connector promises: device,
data type, channel/unit definitions, nominal and observed rates, delivery mode, permissions,
platform limits, availability and timestamps. Apple Watch and Oura are required research
items; aim to demonstrate one real sensor stream by the next sync. Use `DeviceConnector` and
`SampleBatch`; preserve the simulation so the other layers work without hardware.

**Haosen — processing and models.** Extend the module registry with well-defined parameters,
units, state/reset behavior and sample-rate constraints. Review available open models and
licenses, document accepted modalities and latency, then integrate one behind
`PhysiologicalModel`. Start with simulated/replayed data. Keep deterministic evaluation and
physiological validity separate.

**Marvin — planner and application layer.** Turn ten example prompts into golden pipeline
fixtures and target app acceptance tests. Implement an agent behind `PipelinePlanner`,
validate outputs, and expose failures clearly. Deliver an end-to-end web target, then phone
and Quest paths, with deployment status and teardown. Add Render per-request workers only
with isolation and explicit cost/cleanup controls.

## Working agreement

1. Branch from `feat/physiolab-platform-skeleton` until it is merged. Create component PRs.
2. Coordinate schema changes first; regenerate API/TS artifacts in the same PR.
3. Every claimed connector/model/target must state whether it is simulated, replayed or live.
4. Include one reproducible input/output example and meaningful contract/integration tests.
5. Keep the existing desktop app working; migrate adapters incrementally through the bridge.
6. Move live work in Linear, including dependencies, evidence and remaining limitations.

## Completion evidence for a component

- Clean checkout setup works using the README commands.
- Data can cross its boundary with validated shape, units, clock and provenance.
- Failures/disconnects/cancellation are visible and clean up resources.
- A browser or hardware demo proves its user flow; a stub returning success is insufficient.
- Relevant tests pass and documentation includes setup/credentials without committed secrets.

The imported design covers more than this starter. Real wearable integrations, trained
physiological inference, free-form generation, phone/Quest publishing and Render worker
orchestration remain assigned implementation work; the scaffold exposes their seams.
