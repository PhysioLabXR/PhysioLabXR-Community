# Golden prompt fixtures (PHY-13, draft v0.2)

Each file in `examples/golden/` is the answer key for one prompt: what a correct planner
should return, and the handcrafted ideal pipeline for that prompt. `examples/golden/proposed/`
holds extra cases awaiting team approval. `examples/prompts.json` lists the refined ten.
`backend/tests/test_golden_fixtures.py` checks every file against `contracts.py` and
`validation.py`.

## Fields

| Field | Meaning |
| --- | --- |
| `fixture_version` | Format version (`0.2`) |
| `id` | Matches the filename and `prompts.json` |
| `original_prompt` | Wording before refinement (`null` for new prompts) |
| `prompt` | Refined request: names devices and target unless ambiguity is the point |
| `categories` | Any of `wearable`, `web`, `phone`, `quest`, `clinician-dashboard`, `research-bci` |
| `requested_target` | Target the user asked for, or `null` |
| `requirements` | Devices (delivery: simulated / batch / live / unverified), processing, model task + validation status, target |
| `expected_behavior` | What the finished app should visibly do |
| `constraints` | Limits the planner must respect |
| `ambiguities` | Open questions, why they matter, and the policy (ask / stated default) |
| `result` | The planner's correct answer (below) |
| `evaluation.intent_rubric` | Checks a separate judge model can score |

## Result statuses

| Status | Contains | Meaning |
| --- | --- | --- |
| `runnable` | `pipeline`, `provenance` | Runs today; passes `validate_pipeline` |
| `partial` | `available_layers`, `missing`, `target_pipeline` | Some layers exist |
| `unsupported` | `missing`, `target_pipeline` | Core capabilities missing |
| `requires_review` | `review_needed`, `missing`, `target_pipeline` | Clinical, consent or safety review first |
| `needs_clarification` | `questions` | Too vague to plan; ask first |
| `invalid_request` | `requested_pipeline`, `suggestion` | A requested setting is impossible (e.g. above Nyquist) |
| `refused` | `missing` | Crosses a safety boundary (arbitrary code) |

`missing` items have `layer`, `capability`, `blocked_by` (Linear issue, `unassigned`, or a named
decision) and `reason`. Every result has a plain-language `user_message`.

## Golden pipelines

- `pipeline` (runnable only) is executable today and must pass the validator.
- `target_pipeline` is the **handcrafted ideal** once missing parts exist. It is marked
  `executable: false`. The test checks it is a well-formed graph (devices in, application out,
  no cycles) **and** that today's validator rejects it. PHY-14 providers are scored against it.
- Future implementation names (`apple-watch`, `p300-classifier`, `quest-workspace`, ...) are
  proposals for Serena, Haosen and the app layer to confirm or rename.
- Multi-source prompts (focus, P300, multimodal) use fan-in graphs; the current runtime only
  runs one chain, which each fixture lists as a missing capability.

## Rules

1. Only `runnable` results carry an executable `pipeline`.
2. The planner never picks a device or target the user did not name.
3. Focus, stress or cognitive-load results never borrow the EMG fixture's validation claim.

## Changes from the original prompts

- All ten reworded to name devices, target and expected outcome.
- `breathing-feedback` moved to `proposed/` (overlapped `focus-wearables`); `stress-vague`
  added to the core ten so ambiguity is covered.

## Open questions for the team

- Target vocabulary: catalog lists only `web-preview`.
- Provenance: contracts have no `replayed` value and no planner provenance field.
- May a pipeline omit the model layer (recording-only prompts)?
- Should the proposed cases replace or extend the core ten?
