# Implementation backlog

Linear PHY is the live tracker. These twelve handoff tasks preserve the meeting scope.

All twelve issues are published as Todo in [Linear PHY](https://linear.app/physiolab/team/PHY/active), with four assigned to each owner. The links below open the live issues; key dependencies refer to this handoff. November 15 is the first checkpoint recorded in the near-term issue descriptions.

| Key | Owner | Task | Depends on | Initial target |
| --- | --- | --- | --- | --- |
| [PHY-5](https://linear.app/physiolab/issue/PHY-5/inventory-sensor-capabilities-and-confirm-the-first-live-data-path) (S1) | Serena | Inventory sensor capabilities and confirm the first live-data path | Ready | 2026-11-15 |
| [PHY-6](https://linear.app/physiolab/issue/PHY-6/connect-apple-watch-through-a-local-companion-bridge) (S2) | Serena | Connect Apple Watch through a local companion bridge | S1 | 2026-11-15 |
| [PHY-7](https://linear.app/physiolab/issue/PHY-7/implement-oura-v2-sync-with-consent-and-data-provenance) (S3) | Serena | Implement Oura V2 sync with consent and data provenance | S1 | After prerequisite review |
| [PHY-8](https://linear.app/physiolab/issue/PHY-8/bridge-a-legacy-physiolabxr-lsl-or-eeg-connector) (S4) | Serena | Bridge a legacy PhysioLabXR LSL or EEG connector | S1 | After prerequisite review |
| [PHY-9](https://linear.app/physiolab/issue/PHY-9/define-and-validate-the-preprocessing-module-registry) (H1) | Haosen | Define and validate the preprocessing module registry | Ready | 2026-11-15 |
| [PHY-10](https://linear.app/physiolab/issue/PHY-10/review-open-physiological-models-and-publish-model-cards) (H2) | Haosen | Review open physiological models and publish model cards | Ready | 2026-11-15 |
| [PHY-11](https://linear.app/physiolab/issue/PHY-11/integrate-the-first-open-physiological-model-adapter) (H3) | Haosen | Integrate the first open physiological model adapter | H1, H2 | After prerequisite review |
| [PHY-12](https://linear.app/physiolab/issue/PHY-12/create-deterministic-emg-and-p300-evaluation-fixtures) (H4) | Haosen | Create deterministic EMG and P300 evaluation fixtures | H1 | After prerequisite review |
| [PHY-13](https://linear.app/physiolab/issue/PHY-13/curate-ten-example-prompts-and-handcrafted-golden-pipelines) (M1) | Marvin | Curate ten example prompts and handcrafted golden pipelines | Ready | 2026-11-15 |
| [PHY-14](https://linear.app/physiolab/issue/PHY-14/implement-a-validated-pipeline-generation-agent) (M2) | Marvin | Implement a validated pipeline-generation agent | M1 | After prerequisite review |
| [PHY-15](https://linear.app/physiolab/issue/PHY-15/build-application-adapters-for-web-phone-and-quest) (M3) | Marvin | Build application adapters for web, phone and Quest | M1 | After prerequisite review |
| [PHY-16](https://linear.app/physiolab/issue/PHY-16/implement-render-per-request-workers-and-deployment-lifecycle) (M4) | Marvin | Implement Render per-request workers and deployment lifecycle | M2 | After prerequisite review |

## S1 — Inventory sensor capabilities and confirm the first live-data path

Linear: [PHY-5](https://linear.app/physiolab/issue/PHY-5/inventory-sensor-capabilities-and-confirm-the-first-live-data-path).

Owner: Serena

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Enumerate Apple Watch, Oura Ring, phone motion and existing EEG/LSL devices. Record data types, channels, units, nominal/observed rates, live vs synced delivery, permissions, platform limitations and clock source. Select one attainable live stream for the Nov 15 sync. Include evidence for availability rather than assuming every wearable streams continuously.

Start here: physiolab/backend/src/physiolab/devices/catalog.py.

Dependencies: Ready to start.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## S2 — Connect Apple Watch through a local companion bridge

Linear: [PHY-6](https://linear.app/physiolab/issue/PHY-6/connect-apple-watch-through-a-local-companion-bridge).

Owner: Serena

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Implement the chosen HealthKit/watchOS/iPhone bridge path behind DeviceConnector and normalize output to SampleBatch. Demonstrate real sensor data, permission denial, disconnect/reconnect and explicit Unix-time mapping. Document supported data types/rates and limitations. Preserve the simulated connector for hardware-free development.

Start here: physiolab/backend/src/physiolab/devices/base.py.

Dependencies: S1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## S3 — Implement Oura V2 sync with consent and data provenance

Linear: [PHY-7](https://linear.app/physiolab/issue/PHY-7/implement-oura-v2-sync-with-consent-and-data-provenance).

Owner: Serena

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Add OAuth V2 authorization and retrieval for the selected Oura data types. Persist source timestamps and last-sync time; distinguish synced/batch data from live streaming. Handle revoked/expired access and rate limits. Provide fixture-based tests and one authorized end-to-end retrieval; keep tokens server-side.

Start here: physiolab/backend/src/physiolab/devices/base.py.

Dependencies: S1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## S4 — Bridge a legacy PhysioLabXR LSL or EEG connector

Linear: [PHY-8](https://linear.app/physiolab/issue/PHY-8/bridge-a-legacy-physiolabxr-lsl-or-eeg-connector).

Owner: Serena

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Use LegacyDeviceAdapter to wrap one existing connector without importing Qt/native SDKs into the cloud API. Verify channel-major to sample-major conversion, units, channel count and clock mapping. Demonstrate start/read/stop and device loss with cleanup. Specify the authenticated local-to-runtime transport seam.

Start here: physiolab/backend/src/physiolab/devices/legacy.py.

Dependencies: S1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## H1 — Define and validate the preprocessing module registry

Linear: [PHY-9](https://linear.app/physiolab/issue/PHY-9/define-and-validate-the-preprocessing-module-registry).

Owner: Haosen

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Extend the bandpass/window starter into a documented registry with parameter bounds, sample-rate and modality compatibility, units, state/reset rules and error behavior. Test chunk continuity, window overlap, missing/out-of-order data and invalid Nyquist settings. Preserve the shared contracts and regenerate API types when they change.

Start here: physiolab/backend/src/physiolab/processing/modules.py.

Dependencies: Ready to start.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## H2 — Review open physiological models and publish model cards

Linear: [PHY-10](https://linear.app/physiolab/issue/PHY-10/review-open-physiological-models-and-publish-model-cards).

Owner: Haosen

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Review available open-source physiological/foundation models. Compare modalities, license, weights availability, preprocessing, sampling rate, output semantics, latency, hardware needs and promptability. Select a practical first model. Document why it fits and its validity limits; do not depend on the unavailable foundation model discussed in the meeting.

Start here: physiolab/backend/src/physiolab/models/base.py.

Dependencies: Ready to start.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## H3 — Integrate the first open physiological model adapter

Linear: [PHY-11](https://linear.app/physiolab/issue/PHY-11/integrate-the-first-open-physiological-model-adapter).

Owner: Haosen

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Implement PhysiologicalModel for the chosen model with fixed version/provenance and reproducible fixture inputs. Validate preprocessing compatibility and output schema. Run on simulated or replayed data initially; surface incompatible inputs and missing weights honestly. Supply a reproducible inference example and latency measurements.

Start here: physiolab/backend/src/physiolab/models/base.py.

Dependencies: H1, H2.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## H4 — Create deterministic EMG and P300 evaluation fixtures

Linear: [PHY-12](https://linear.app/physiolab/issue/PHY-12/create-deterministic-emg-and-p300-evaluation-fixtures).

Owner: Haosen

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Establish ground-truth EMG gesture and P300 tasks to validate end-to-end signal handling, inference and app responses. Check timing, dropped data and recovery; record expected labels/events and metrics. Separate pipeline correctness from physiological validity. Outline the later self-report study for focus/cognitive-load decoding.

Start here: physiolab/backend/tests/test_pipeline.py and examples/emg-pipeline.json.

Dependencies: H1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## M1 — Curate ten example prompts and handcrafted golden pipelines

Linear: [PHY-13](https://linear.app/physiolab/issue/PHY-13/curate-ten-example-prompts-and-handcrafted-golden-pipelines).

Owner: Marvin

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Start from examples/prompts.json and refine ten representative prompts spanning wearables, web/phone/Quest, clinician dashboard and research/BCI. For each, define required devices, processing, model, target, expected behavior, constraints and a golden pipeline or explicit unsupported result. Cover ambiguous prompts and unavailable capabilities.

Start here: physiolab/examples/prompts.json and examples/emg-pipeline.json.

Dependencies: Ready to start.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## M2 — Implement a validated pipeline-generation agent

Linear: [PHY-14](https://linear.app/physiolab/issue/PHY-14/implement-a-validated-pipeline-generation-agent).

Owner: Marvin

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Replace TemplatePlanner behind PipelinePlanner with a provider-configurable agent. Emit only schema-valid supported nodes/edges/parameters; validate before execution, explain missing capabilities and retain provenance. Compare candidate generation providers mentioned in the meeting against the golden fixtures. Score run success, deploy success and user intent with a separate judge and pinned model versions.

Start here: physiolab/backend/src/physiolab/orchestration/planner.py and validation.py.

Dependencies: M1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## M3 — Build application adapters for web, phone and Quest

Linear: [PHY-15](https://linear.app/physiolab/issue/PHY-15/build-application-adapters-for-web-phone-and-quest).

Owner: Marvin

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Extend the standalone HTML starter into working target-specific app adapters. Start with an end-to-end web app, then a phone path and Quest XR path. Consume typed inference events, implement connection/error states and document local/device setup. Verify each target on its real runtime before reporting deploy success; preserve a replayable demo.

Start here: physiolab/backend/src/physiolab/applications/export.py and web/src/components/SessionPanel.tsx.

Dependencies: M1.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7


## M4 — Implement Render per-request workers and deployment lifecycle

Linear: [PHY-16](https://linear.app/physiolab/issue/PHY-16/implement-render-per-request-workers-and-deployment-lifecycle).

Owner: Marvin

Context: PhysioLab team handoff, Nov 8, 2026.

Acceptance criteria:
Use render.yaml and DeploymentProvider as the starting seams. Add a control plane for per-request isolated runtime provisioning, job status, cancellation, teardown and failure cleanup. Define tenant authentication, secrets, persistent storage, quotas and cost limits. Test an isolated deployment with health/readiness checks and cleanup; the starter blueprint is only one simulation service.

Start here: physiolab/render.yaml, Dockerfile and backend/src/physiolab/applications/export.py.

Dependencies: M2.

Handoff: https://github.com/PhysioLabXR/PhysioLabXR-Community/tree/feat/physiolab-platform-skeleton/physiolab
Meeting: https://notes.granola.ai/d/bb8697db-17e8-4a97-8a2e-e5588fb289b7
