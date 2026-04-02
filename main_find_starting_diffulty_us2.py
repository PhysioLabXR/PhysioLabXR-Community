#!/usr/bin/env python3
"""
US2 prep helper (with FORCE assignment support):

1) For each participant (from PARTICIPANT_LOGS), load their US1 session-log JSON.
2) Extract SpaceShooter blocks only.
3) Compute a conservative per-block score from (targetShot, nontargetShot, mothershipHealth).
4) Recommend a US2 starting difficulty per participant.
5) Recommend a US2 condition per participant using stratified balancing by starting difficulty,
   while accommodating:
   - persisted assignments (stable across runs)
   - participant dropouts (not in PARTICIPANT_LOGS => not counted for balancing)
   - forced assignments (FORCE_CONDITIONS), e.g., today’s returnees must be O/C.

Key design goal:
- Re-running this script will NOT reshuffle existing assignments.
- Only assigns conditions to new participants (unless forced).
- Forced participants count toward bin/global counts so balancing adapts around them.

Requirements: numpy
"""

import json
import math
import os
import ntpath
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------
# USER CONFIG
# ---------------------------

LOG_ROOT = r"C:\UnityProjects\ReNaSuite\Assets\Logs"

# Key: participant id (int). Value: log file name (string) OR absolute path.
# NOTE: use int keys, not strings.
# NOTE: 3-16-2026
#   people who have not come back for US2 are commented out
#   Only add/uncomment people if they are here for US2
PARTICIPANT_LOGS: Dict[int, str] = {
    4: "ReNaSessionLog_02-28-2026-12-19-44.json",
    # 5:  "ReNaSessionLog_02-28-2026-14-13-20.json",
    6: "ReNaSessionLog_02-28-2026-15-32-56.json",
    8: "ReNaSessionLog_02-28-2026-19-24-30.json",
    9: "ReNaSessionLog_03-01-2026-13-59-23.json",
    # 10: "ReNaSessionLog_03-01-2026-15-36-19.json",
    # 11: "ReNaSessionLog_03-01-2026-17-59-03.json",
    12: "ReNaSessionLog_03-02-2026-10-06-47.json",
    13: "ReNaSessionLog_03-02-2026-15-52-08.json",
    # 14: "ReNaSessionLog_03-02-2026-18-20-30.json",
    15: "ReNaSessionLog_03-02-2026-18-20-30.json",
    16: "ReNaSessionLog_03-04-2026-18-10-44.json",
    # 17: "ReNaSessionLog_03-05-2026-21-57-42.json",
    18: "ReNaSessionLog_03-06-2026-22-21-04.json",
    19: "ReNaSessionLog_03-07-2026-18-03-04.json",
    20: "ReNaSessionLog_03-08-2026-11-28-28.json",
    # 21: "ReNaSessionLog_03-08-2026-16-19-52.json",
    # 22: "ReNaSessionLog_03-08-2026-20-33-04.json",
    23: "ReNaSessionLog_03-09-2026-15-14-48.json",
    # 24: "ReNaSessionLog_03-09-2026-19-09-30.json",
    25: "ReNaSessionLog_03-10-2026-10-56-42.json",
    27: "ReNaSessionLog_03-10-2026-21-07-56.json",
    28: "ReNaSessionLog_03-11-2026-10-23-14.json",
    29: "ReNaSessionLog_03-11-2026-19-46-10.json",

    33: "ReNaSessionLog_03-14-2026-16-42-51.json",
    34: "ReNaSessionLog_03-15-2026-15-40-18.json",
    35: r"C:\Users\Season\Downloads\wingman_35_us1_ReNaSessionLog_03-15-2026-21-27-33.json",
    36: r"C:\Users\Season\Downloads\wingman_36_us1_ReNaSessionLog_03-16-2026-14-10-14.json",
    37: r"C:\Users\Season\Downloads\wingman_37_us1_ReNaSessionLog_03-16-2026-16-24-52.json",
    38: r"C:\Users\Season\Downloads\wingman_38_us1_ReNaSessionLog_03-16-2026-20-39-51.json",
    39: r"C:\Users\Season\Downloads\wingman_39_us1_ReNaSessionLog_03-17-2026-21-28-27.json",
    40: r"C:\Users\Season\Downloads\wingman_40_us1_ReNaSessionLog_03-18-2026-09-37-54.json",
    41: r"C:\Users\Season\Downloads\wingman_41_us1_ReNaSessionLog_03-18-2026-11-31-24.json",
    43: r"C:\Users\Season\Downloads\wingman_43_us1_ReNaSessionLog_03-18-2026-16-47-54.json",
    45: r"C:\Users\Season\Downloads\wingman_45_us1_ReNaSessionLog_03-19-2026-19-39-08.json",
    46: r"C:\Users\Season\Downloads\wingman_46_us1_ReNaSessionLog_03-20-2026-12-48-48.json",
    48: "ReNaSessionLog_03-20-2026-18-33-47.json",
    49: r"C:\Users\Season\Downloads\wingman_49_us1_ReNaSessionLog_03-21-2026-12-08-25.json",
    50: "ReNaSessionLog_03-22-2026-13-52-56.json",
    51: r"C:\Users\Season\Downloads\wingman_51_us1_ReNaSessionLog_03-23-2026-10-54-11.json",
    53: r"C:\Users\Season\Downloads\wingman_53_us1_ReNaSessionLog_03-25-2026-20-59-50.json",
    54: r"ReNaSessionLog_03-26-2026-12-23-35.json",
    55: r"ReNaSessionLog_03-26-2026-18-54-40.json",
    56: r"C:\Users\Season\Downloads\wingman_56_us1_ReNaSessionLog_03-27-2026-19-58-19.json",
    57: r"ReNaSessionLog_03-28-2026-18-05-04.json",
    58: r"ReNaSessionLog_03-29-2026-12-41-56.json",
    59: r"ReNaSessionLog_03-29-2026-15-50-51.json",
    60: r"ReNaSessionLog_03-29-2026-18-52-48.json",
    62: r"C:\UnityProjects\ReNaSuite\Assets\Logs\ReNaSessionLog_03-31-2026-09-58-18.json"
}

# Force certain participants to specific conditions.
# Example: FORCE_CONDITIONS = {12: "O", 13: "C"}
# Forced assignments override auto-assignment (and will override any prior assignment).
FORCE_CONDITIONS: Dict[int, str] = {
    6: "C",
    8: "O",
    16: "O",
    9: "C",
    37: "C",
    36: "IE",
}

CONDITIONS = ["O", "C", "IE", "E"]

ASSIGNMENTS_FILE = os.path.join(LOG_ROOT, "us2_condition_assignments.json")

BIN_EDGES = [2.0, 3.0, 4.0]

DIFFICULTY_STEP = 0.5

K_LAST = 6
SCORE_THRESHOLD = 70.0
PRECISION_THRESHOLD = 0.70
HEALTH_THRESHOLD = 0.70
FALLBACK_STEPS = 1


# ---------------------------
# Core utilities
# ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def is_absolute_path(path_str: str) -> bool:
    """
    Returns True for:
    - native absolute paths on the current OS
    - Windows absolute paths like C:\\... even if parsing is awkward
    - UNC paths like \\\\server\\share\\...
    """
    s = os.path.expanduser(str(path_str))
    return os.path.isabs(s) or ntpath.isabs(s)


def resolve_log_path(log_entry: str, log_root: str) -> Path:
    """
    If log_entry is absolute, use it directly.
    Otherwise treat it as relative to LOG_ROOT.
    """
    s = os.path.expanduser(str(log_entry))
    if is_absolute_path(s):
        return Path(s)
    return Path(log_root) / s


def flatten_blocks(obj: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for _, v in obj.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        blocks.append(item)
            elif isinstance(v, dict):
                blocks.append(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                blocks.append(item)
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")
    return blocks


def is_spaceshooter_block(b: Dict[str, Any]) -> bool:
    return bool(b.get("isSpaceShooterBlock")) or (b.get("conditionType") == "SpaceShooter")


def extract_ss_rows(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for b in blocks:
        if not is_spaceshooter_block(b):
            continue

        d = b.get("difficulty", None)
        if d is None:
            continue
        try:
            d = float(d)
        except Exception:
            continue
        if d <= 0:
            continue

        rows.append(
            dict(
                blockID=int(b.get("blockID", -1)),
                difficulty=float(d),
                targetShot=float(b.get("targetShot", 0)),
                nontargetShot=float(b.get("nontargetShot", 0)),
                mothershipHealth=float(b.get("mothershipHealth", -1)),
                isPracticing=bool(b.get("isPracticing", False)),
            )
        )
    rows.sort(key=lambda r: r["blockID"])
    return rows


def health_to_norm(h: float) -> float:
    if h < 0:
        return 0.0
    if h > 1.5:
        return clamp(h / 100.0, 0.0, 1.0)
    return clamp(h, 0.0, 1.0)


def compute_block_score(
    target_shot: float,
    nontarget_shot: float,
    health_norm: float,
    target_ref: float,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> Tuple[float, float]:
    total = max(1.0, target_shot + nontarget_shot)
    precision = clamp(target_shot / total, 0.0, 1.0)

    denom = max(1e-6, math.log1p(max(1.0, target_ref)))
    throughput = clamp(math.log1p(max(0.0, target_shot)) / denom, 0.0, 1.0)

    s = 100.0 * (throughput ** alpha) * (precision ** beta) * (health_norm ** gamma)
    return (clamp(s, 0.0, 100.0), precision)


def recommend_start_difficulty(
    ss_rows: List[Dict[str, Any]],
    k_last: int,
    score_threshold: float,
    precision_threshold: float,
    health_threshold: float,
    step: float,
    conservative_fallback_steps: int,
) -> Dict[str, Any]:
    ss = [r for r in ss_rows if not r["isPracticing"]]
    if len(ss) == 0:
        return dict(error="No SpaceShooter blocks found (after filtering).")

    target_shots = np.array([r["targetShot"] for r in ss], dtype=float)
    target_ref = float(np.percentile(target_shots, 90)) if len(target_shots) >= 3 else float(np.max(target_shots))

    enriched = []
    for r in ss:
        hnorm = health_to_norm(r["mothershipHealth"])
        score, prec = compute_block_score(r["targetShot"], r["nontargetShot"], hnorm, target_ref=target_ref)
        enriched.append(dict(**r, healthNorm=hnorm, precision=prec, blockScore=score))

    last_k = enriched[-k_last:] if k_last > 0 else enriched[:]
    if len(last_k) == 0:
        last_k = enriched

    ok_blocks = [
        r for r in last_k
        if (r["blockScore"] >= score_threshold
            and r["precision"] >= precision_threshold
            and r["healthNorm"] >= health_threshold)
    ]

    diffs_last = np.array([r["difficulty"] for r in last_k], dtype=float)

    if len(ok_blocks) > 0:
        diffs_ok = np.array([r["difficulty"] for r in ok_blocks], dtype=float)
        weights = np.array([r["blockScore"] for r in ok_blocks], dtype=float)
        weights = np.clip(weights, 1e-6, None)
        d_hat = float(np.sum(diffs_ok * weights) / np.sum(weights))
        reason = f"Score-weighted mean difficulty over {len(ok_blocks)}/{len(last_k)} OK blocks in last {len(last_k)}."
    else:
        d_hat = float(np.median(diffs_last) - conservative_fallback_steps * step)
        reason = f"No OK blocks in last {len(last_k)}; used median(lastK) minus {conservative_fallback_steps} step(s)."

    d_min = float(min(r["difficulty"] for r in enriched))
    d_max = float(max(r["difficulty"] for r in enriched))
    d_rec = round_to_step(d_hat, step)
    d_rec = clamp(d_rec, d_min, d_max)

    return dict(
        recommendedDifficulty=float(d_rec),
        reason=reason,
        nSpaceShooterBlocks=len(enriched),
        targetRef90p=float(target_ref),
        lastBlockDifficulty=float(enriched[-1]["difficulty"]),
        lastBlockScore=float(enriched[-1]["blockScore"]),
        lastK=last_k,
        okBlocks=ok_blocks,
    )


# ---------------------------
# Condition assignment (stratified balancing with FORCE)
# ---------------------------

def bin_index(d: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if d < e:
            return i
    return len(edges)


def load_assignments(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"meta": {}, "assignments": {}}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "assignments" not in obj:
        return {"meta": {}, "assignments": obj}
    return obj


def save_assignments(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def validate_forces(force_map: Dict[int, str], conditions: List[str]) -> None:
    bad = {pid: c for pid, c in force_map.items() if c not in conditions}
    if bad:
        raise ValueError(f"FORCE_CONDITIONS contains invalid condition(s): {bad}. Valid: {conditions}")


def choose_condition_for_bin(
    b: int,
    counts_by_bin: List[Dict[str, int]],
    counts_global: Dict[str, int],
    conditions: List[str],
    participant_id: int,
) -> str:
    bin_counts = counts_by_bin[b]
    min_bin = min(bin_counts[c] for c in conditions)
    candidates = [c for c in conditions if bin_counts[c] == min_bin]

    if len(candidates) == 1:
        return candidates[0]

    min_global = min(counts_global[c] for c in candidates)
    candidates2 = [c for c in candidates if counts_global[c] == min_global]

    if len(candidates2) == 1:
        return candidates2[0]

    def h(cond: str) -> int:
        return hash((participant_id, cond)) & 0x7FFFFFFF

    candidates2.sort(key=h)
    return candidates2[0]


def assign_conditions_stratified(
    active_participants: List[int],
    start_difficulty: Dict[int, float],
    existing_assignments: Dict[str, str],
    conditions: List[str],
    bin_edges: List[float],
    force_conditions: Dict[int, str],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    validate_forces(force_conditions, conditions)

    active_set = set(active_participants)

    assigned_active: Dict[int, str] = {}
    for k, v in existing_assignments.items():
        try:
            pid = int(k)
        except Exception:
            continue
        if pid in active_set:
            assigned_active[pid] = v

    forced_active: Dict[int, str] = {pid: cond for pid, cond in force_conditions.items() if pid in active_set}

    fixed_active: Dict[int, str] = dict(assigned_active)
    for pid, cond in forced_active.items():
        fixed_active[pid] = cond

    n_bins = len(bin_edges) + 1
    counts_by_bin = [{c: 0 for c in conditions} for _ in range(n_bins)]
    counts_global = {c: 0 for c in conditions}

    for pid, cond in fixed_active.items():
        if pid not in start_difficulty:
            continue
        b = bin_index(start_difficulty[pid], bin_edges)
        counts_by_bin[b][cond] += 1
        counts_global[cond] += 1

    updated = dict(existing_assignments)
    newly_assigned: Dict[int, str] = {}
    forced_overrides: Dict[int, Tuple[str, str]] = {}

    for pid, cond in forced_active.items():
        old = updated.get(str(pid))
        if old is not None and old != cond:
            forced_overrides[pid] = (old, cond)
        updated[str(pid)] = cond

    for pid, cond in assigned_active.items():
        if pid in forced_active:
            continue
        updated[str(pid)] = cond

    for pid in sorted(active_participants):
        if pid not in start_difficulty:
            continue

        if pid in forced_active:
            continue
        if pid in assigned_active:
            continue

        b = bin_index(start_difficulty[pid], bin_edges)
        cond = choose_condition_for_bin(b, counts_by_bin, counts_global, conditions, pid)

        updated[str(pid)] = cond
        newly_assigned[pid] = cond

        counts_by_bin[b][cond] += 1
        counts_global[cond] += 1

    debug = {
        "counts_global_active": counts_global,
        "counts_by_bin_active": counts_by_bin,
        "newly_assigned": newly_assigned,
        "forced_active": forced_active,
        "forced_overrides": forced_overrides,
        "bin_edges": bin_edges,
        "conditions": conditions,
    }
    return updated, debug


# ---------------------------
# Main
# ---------------------------

def main():
    start_difficulty: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}
    errors: Dict[int, str] = {}

    for pid, fname in PARTICIPANT_LOGS.items():
        fpath = resolve_log_path(fname, LOG_ROOT)

        if not fpath.exists():
            errors[pid] = f"Missing log file: {fpath}"
            continue

        try:
            with fpath.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            blocks = flatten_blocks(obj)
            ss_rows = extract_ss_rows(blocks)
            rec = recommend_start_difficulty(
                ss_rows,
                k_last=K_LAST,
                score_threshold=SCORE_THRESHOLD,
                precision_threshold=PRECISION_THRESHOLD,
                health_threshold=HEALTH_THRESHOLD,
                step=DIFFICULTY_STEP,
                conservative_fallback_steps=FALLBACK_STEPS,
            )
            if "error" in rec:
                errors[pid] = rec["error"]
                continue
            start_difficulty[pid] = float(rec["recommendedDifficulty"])
            details[pid] = rec
        except Exception as e:
            errors[pid] = f"Exception while processing {fpath}: {e}"

    assign_obj = load_assignments(ASSIGNMENTS_FILE)
    existing_assignments = assign_obj.get("assignments", {})

    active_participants = sorted(PARTICIPANT_LOGS.keys())

    updated_assignments, debug = assign_conditions_stratified(
        active_participants=active_participants,
        start_difficulty=start_difficulty,
        existing_assignments=existing_assignments,
        conditions=CONDITIONS,
        bin_edges=BIN_EDGES,
        force_conditions=FORCE_CONDITIONS,
    )

    assign_obj["meta"] = {
        "note": "Auto-generated. Counts are balanced among ACTIVE participants only (keys in PARTICIPANT_LOGS). Forced assignments override existing and are included in balancing counts.",
        "log_root": LOG_ROOT,
        "k_last": K_LAST,
        "score_threshold": SCORE_THRESHOLD,
        "precision_threshold": PRECISION_THRESHOLD,
        "health_threshold": HEALTH_THRESHOLD,
        "difficulty_step": DIFFICULTY_STEP,
        "fallback_steps": FALLBACK_STEPS,
        "bin_edges": BIN_EDGES,
        "conditions": CONDITIONS,
        "force_conditions": {str(k): v for k, v in FORCE_CONDITIONS.items()},
    }
    assign_obj["assignments"] = updated_assignments
    save_assignments(ASSIGNMENTS_FILE, assign_obj)

    print("\n=== US2 Starting Difficulty + Condition Assignment ===")
    print(f"Active participants: {len(active_participants)}")
    print(f"Computed start difficulties: {len(start_difficulty)}")
    if errors:
        print(f"Errors/missing logs: {len(errors)}")

    if debug["forced_active"]:
        print("\nForced assignments (active):", debug["forced_active"])
    if debug["forced_overrides"]:
        print("Forced overrides (old -> new):", debug["forced_overrides"])

    header = f"{'PID':>5} {'d0':>5} {'cond':>4} {'note':<40}"
    print("\n" + header)
    print("-" * len(header))

    for pid in active_participants:
        cond = updated_assignments.get(str(pid), "NA")
        if pid in start_difficulty:
            d0 = start_difficulty[pid]
            if pid in debug["forced_active"]:
                note = "FORCED"
            elif pid in debug["newly_assigned"]:
                note = "NEW_ASSIGNED"
            else:
                note = "OK"
            print(f"{pid:>5} {d0:>5.1f} {cond:>4} {note:<40}")
        else:
            note = errors.get(pid, "No starting difficulty (skipped)")
            print(f"{pid:>5} {'NA':>5} {cond:>4} {note:<40}")

    print("\n=== Active counts (balanced within bins; forced included) ===")
    print("Global counts:", debug["counts_global_active"])
    print("By-bin counts:")
    for b, counts in enumerate(debug["counts_by_bin_active"]):
        if b == 0:
            rng = f"(-inf, {BIN_EDGES[0]})"
        elif b == len(BIN_EDGES):
            rng = f"[{BIN_EDGES[-1]}, inf)"
        else:
            rng = f"[{BIN_EDGES[b-1]}, {BIN_EDGES[b]})"
        print(f"  Bin {b} {rng}: {counts}")

    if errors:
        print("\n=== Errors / Missing logs ===")
        for pid, msg in errors.items():
            print(f"PID {pid}: {msg}")

    print(f"\nSaved/updated assignments: {ASSIGNMENTS_FILE}")


if __name__ == "__main__":
    main()