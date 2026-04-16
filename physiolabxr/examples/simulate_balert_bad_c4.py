#!/usr/bin/env python3
"""
simulate_balert_bad_c4.py

Streams a fake BAlert LSL outlet at 256 Hz.
All channels get realistic pink-noise EEG.
C4 is deliberately corrupted with one of several "bad channel" modes
so you can verify that RealtimeBadChannelInterp fixes it in physio_server.

Usage
-----
    python simulate_balert_bad_c4.py                  # default: saturated C4
    python simulate_balert_bad_c4.py --mode flatline   # C4 stuck at 0
    python simulate_balert_bad_c4.py --mode noise      # C4 = pure white noise (10x amplitude)
    python simulate_balert_bad_c4.py --mode spike      # C4 = periodic large spikes
    python simulate_balert_bad_c4.py --mode drift      # C4 = slow DC drift

Press Ctrl-C to stop.
"""

import argparse
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32

# ── stream config ────────────────────────────────────────────────────────────
STREAM_NAME   = "_BAlert"
STREAM_TYPE   = "EEG"
FS            = 256          # samples per second
CHUNK_SIZE    = 8            # samples pushed per iteration  (≈ 32 ms chunks)

# Matches BALERT_CHANNELS in physio_utils.py exactly
BALERT_CHANNELS = [
    "Fp1", "F7", "F8", "T4", "T6", "T5", "T3", "Fp2", "O1", "P3",
    "Pz",  "F3", "Fz", "F4", "C4", "P4", "POz", "C3", "Cz", "O2",
]
N_CHANNELS = len(BALERT_CHANNELS)           # 20
C4_IDX     = BALERT_CHANNELS.index("C4")   # 14

# ── realistic EEG simulation ─────────────────────────────────────────────────

def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simple 1/f pink noise via frequency-domain shaping.
    Returns array of length n, zero-mean, ~unit std before scaling.
    """
    white = rng.standard_normal(n)
    f     = np.fft.rfftfreq(n)
    f[0]  = 1.0                             # avoid div-by-zero at DC
    psd   = 1.0 / np.sqrt(f)
    pink  = np.fft.irfft(np.fft.rfft(white) * psd, n=n)
    pink -= pink.mean()
    std    = pink.std()
    return pink / std if std > 1e-9 else pink


class EEGSimulator:
    """
    Generates N_CHANNELS of pink-noise EEG at FS Hz.
    Typical scalp EEG amplitude: ~20–80 µV peak-to-peak.
    We use ±30 µV RMS per channel with mild spatial correlation.
    """
    AMP_UV = 30.0   # µV RMS per channel

    def __init__(self, seed: int = 42):
        self.rng    = np.random.default_rng(seed)
        self._t     = 0          # sample counter (for deterministic drift/spike)

    def next_chunk(self, n: int) -> np.ndarray:
        """Returns (N_CHANNELS, n) float32 array in µV."""
        # One pink-noise base signal for shared spatial correlation
        base = _pink_noise(n, self.rng) * self.AMP_UV

        chunk = np.empty((N_CHANNELS, n), dtype=np.float32)
        for ch in range(N_CHANNELS):
            indep      = _pink_noise(n, self.rng) * self.AMP_UV
            chunk[ch]  = (0.4 * base + 0.6 * indep).astype(np.float32)

        self._t += n
        return chunk


# ── bad-channel corruption modes ─────────────────────────────────────────────

def corrupt_c4(chunk: np.ndarray, mode: str, t_start: int, fs: int) -> np.ndarray:
    """
    Overwrites the C4 row of chunk (N_CHANNELS, T) in-place.

    Modes
    -----
    saturated : clipped rail signal (hard +/- saturation) — default
    flatline   : constant zero
    noise      : white noise at 10× normal amplitude
    spike      : brief high-amplitude spikes every ~0.5 s
    drift      : slow DC drift that grows over time
    """
    n   = chunk.shape[1]
    rng = np.random.default_rng(t_start)          # deterministic per chunk

    if mode == "flatline":
        chunk[C4_IDX, :] = 0.0

    elif mode == "noise":
        chunk[C4_IDX, :] = rng.standard_normal(n).astype(np.float32) * 300.0  # 300 µV RMS

    elif mode == "spike":
        spike_interval = int(fs * 0.5)             # spike every 0.5 s
        chunk[C4_IDX, :] = 0.0
        for k in range(n):
            if (t_start + k) % spike_interval < 3:  # 3-sample spike width
                chunk[C4_IDX, k] = 500.0 * (1 if (t_start + k) % 2 == 0 else -1)

    elif mode == "drift":
        t_vec = (t_start + np.arange(n, dtype=np.float32)) / float(fs)
        chunk[C4_IDX, :] = (t_vec * 15.0).astype(np.float32)  # 15 µV/s ramp

    else:  # "saturated" (default)
        RAIL = 200.0
        chunk[C4_IDX, :] = np.clip(chunk[C4_IDX, :] * 10.0, -RAIL, RAIL).astype(np.float32)

    return chunk


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fake BAlert LSL stream with bad C4")
    p.add_argument(
        "--mode",
        choices=["saturated", "flatline", "noise", "spike", "drift"],
        default="saturated",
        help="Corruption mode for C4 (default: saturated)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for EEG simulation",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    mode   = args.mode

    # ── create LSL outlet ────────────────────────────────────────────────────
    info = StreamInfo(
        name       = STREAM_NAME,
        type       = STREAM_TYPE,
        channel_count = N_CHANNELS,
        nominal_srate = FS,
        channel_format = cf_float32,
        source_id  = "simulate_balert_bad_c4",
    )

    # Label channels so any LSL viewer shows names correctly
    chans = info.desc().append_child("channels")
    for label in BALERT_CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit",  "microvolts")
        ch.append_child_value("type",  "EEG")

    outlet = StreamOutlet(info, chunk_size=CHUNK_SIZE, max_buffered=360)

    print(f"[sim] Stream '{STREAM_NAME}' open  |  {N_CHANNELS} ch @ {FS} Hz")
    print(f"[sim] Bad channel : C4 (index {C4_IDX})  |  mode = {mode}")
    print("[sim] Ctrl-C to stop.\n")

    sim            = EEGSimulator(seed=args.seed)
    t_sample       = 0                          # running sample counter
    interval_s     = CHUNK_SIZE / FS
    next_push_time = time.perf_counter()

    try:
        while True:
            chunk = sim.next_chunk(CHUNK_SIZE)      # (N_CH, CHUNK_SIZE)
            corrupt_c4(chunk, mode, t_sample, FS)
            t_sample += CHUNK_SIZE

            # LSL push_chunk expects list[list] or (samples, channels)
            # We have (channels, samples) → transpose
            outlet.push_chunk(chunk.T.tolist())

            next_push_time += interval_s
            sleep_for = next_push_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[sim] stopped.")


if __name__ == "__main__":
    main()
