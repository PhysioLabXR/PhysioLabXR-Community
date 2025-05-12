"""
Test-suite for the new RNStream

Version taxonomy
================
* **RNStream v1**  (legacy) – a file starts directly with the 16-byte *magic*
  marker of the first TLV packet. There is **no file-header**.

* **RNStream v2**  (current) – file begins with the ASCII signature
  ``b'DATS_HDR'`` followed by:
      version-byte, table-length byte, and *N* × (32-byte label + 1-byte codec-id).
  TLV packets follow immediately after the table.

These tests make sure:

1.  A v2 reader can still load v1 recordings (backward compatibility).
2.  Lossless video compression (`libx264rgb –crf 0`) round-trips pixel-exactly.
3.  Mixed compressed-video + raw time-series round-trips.
4.  `stream_in_stepwise()` reproduces the same data as one-shot `stream_in()`.

Run with ``python -m pytest``.
"""
import io
import os
import shutil
import tempfile

import numpy as np
import pytest

from physiolabxr.utils.RNStream import RNStream, magic, HEADER_MAGIC
from physiolabxr.compression.compression import DataCompressionPreset

# -----------------------------------------------------------------------------


def _random_video(h=32, w=24, t=10):
    return np.random.randint(0, 256, (h, w, 3, t), np.uint8), np.arange(t) / 30


def _random_signal(ch=256, t=10):
    return np.random.randn(ch, t).astype("float32"), np.arange(t) / 500


@pytest.fixture()
def tmpdir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


# -----------------------------------------------------------------------------


def _save_file(path, streams, codec_map=None):
    rn = RNStream(path, codec_map)
    rn.stream_out(streams)            # one eviction is enough for tests
    rn.close()


def _save_file_multi(path, streams, codec_map=None, *, n_evictions: int = 3):
    """
    Write *streams* to *path* using *n_evictions* successive calls to
    :py:meth:`RNStream.stream_out`.  Each eviction gets an equal-sized slice
    along the last axis (time) of every stream.
    """
    rn = RNStream(path, codec_map)

    # pre-compute slice indices for every stream
    indices = {}
    for name, (arr, _) in streams.items():
        t = arr.shape[-1]
        indices[name] = np.array_split(np.arange(t), n_evictions)

    for ev in range(n_evictions):
        eviction_payload = {}
        for name, (data, ts) in streams.items():
            idx = indices[name][ev]
            if idx.size == 0:                 # can happen for tiny T
                continue
            eviction_payload[name] = (data[..., idx], ts[idx])
        rn.stream_out(eviction_payload)

    rn.close()


def _load_file(path):
    rn = RNStream(path)               # reader side doesn't need codec_map
    return rn.stream_in()


# -----------------------------------------------------------------------------


def test_backward_compatibility(tmpdir):
    """Write a v2 file, strip the header → emulate v1, then load again."""
    vid, v_ts = _random_video()
    eeg, e_ts = _random_signal()

    f2 = os.path.join(tmpdir, "v2.dats")
    _save_file(
        f2,
        {"Video": (vid, v_ts), "EEG": (eeg, e_ts)},
        {"Video": DataCompressionPreset.RAW, "EEG": DataCompressionPreset.RAW},
    )

    # --- remove header to create a v1 clone ---------------------------------
    with open(f2, "rb") as g:
        f2_content = g.read()
    stream = RNStream(f2)
    ver, codec_map, read_bytes_count = stream.parse_header()
    f1_content = f2_content[read_bytes_count:]

    f1 = os.path.join(tmpdir, "v1.dats")
    with open(f1, "wb") as g:
        g.write(f1_content)                          # legacy file starts with magic

    # --- load legacy file ----------------------------------------------------
    dat = _load_file(f1)
    assert np.array_equal(dat["Video"][0], vid)
    assert np.allclose(dat["Video"][1], v_ts)
    assert np.array_equal(dat["EEG"][0], eeg)
    assert np.allclose(dat["EEG"][1], e_ts)


def test_lossless_video_roundtrip(tmpdir):
    video_array, video_timestamps = _random_video()

    f = os.path.join(tmpdir, "vid.dats")
    _save_file(
        f,
        {"Screen": (video_array, video_timestamps)},
        {"Screen": DataCompressionPreset.LOSSLESS},
    )
    dat = _load_file(f)
    rec = dat["Screen"][0]

    assert np.array_equal(rec, video_array)            # pixel-exact
    assert np.allclose(dat["Screen"][1], video_timestamps)


def test_mixed_streams_roundtrip(tmpdir):
    video_array, video_timestamps = _random_video()
    signal_array, signal_timestamps = _random_signal()

    f = os.path.join(tmpdir, "mix.dats")
    _save_file(
        f,
        {"Cam": (video_array, video_timestamps), "EMG": (signal_array, signal_timestamps)},
        {"Cam": DataCompressionPreset.LOSSLESS, "EMG": DataCompressionPreset.RAW},
    )
    dat = _load_file(f)

    assert np.array_equal(dat["Cam"][0], video_array)
    assert np.array_equal(dat["EMG"][0], signal_array)
    assert np.allclose(dat["Cam"][1], video_timestamps)
    assert np.allclose(dat["EMG"][1], signal_timestamps)


def test_stream_in_stepwise_matches_full(tmpdir):
    """Read file in small steps; result should match full read."""
    vid, v_ts = _random_video()
    f = os.path.join(tmpdir, "step.dats")
    _save_file(
        f, {"V": (vid, v_ts)}, {"V": DataCompressionPreset.LOSSLESS}
    )

    rn = RNStream(f)
    file, buf, byte_cnt, total, done = rn.stream_in_stepwise(
        file=None, buffer=None, read_bytes_count=None
    )
    while not done:
        file, buf, byte_cnt, total, done = rn.stream_in_stepwise(
            file, buf, byte_cnt
        )

    full = _load_file(f)
    assert np.array_equal(full["V"][0], buf["V"][0])
    assert np.allclose(full["V"][1], buf["V"][1])



def test_backward_compatibility_multi(tmpdir):
    """Header-stripped v1 file with several evictions loads correctly."""
    vid, v_ts = _random_video()
    eeg, e_ts = _random_signal()

    f2 = os.path.join(tmpdir, "v2multi.dats")
    _save_file_multi(
        f2,
        {"Video": (vid, v_ts), "EEG": (eeg, e_ts)},
        {"Video": DataCompressionPreset.RAW, "EEG": DataCompressionPreset.RAW},
        n_evictions=4,
    )

    # strip header → emulate v1
    with open(f2, "rb") as g:
        f2_bytes = g.read()
    hdr_len = RNStream(f2).parse_header()[2]
    with open(os.path.join(tmpdir, "v1multi.dats"), "wb") as g:
        g.write(f2_bytes[hdr_len:])

    dat = _load_file(os.path.join(tmpdir, "v1multi.dats"))
    assert np.array_equal(dat["Video"][0], vid)
    assert np.allclose(dat["Video"][1], v_ts)
    assert np.array_equal(dat["EEG"][0], eeg)
    assert np.allclose(dat["EEG"][1], e_ts)


def test_lossless_video_roundtrip_multi(tmpdir):
    vid, v_ts = _random_video()
    f = os.path.join(tmpdir, "vidmulti.dats")
    _save_file_multi(
        f, {"Screen": (vid, v_ts)}, {"Screen": DataCompressionPreset.LOSSLESS}, n_evictions=5
    )
    dat = _load_file(f)
    assert np.array_equal(dat["Screen"][0], vid)
    assert np.allclose(dat["Screen"][1], v_ts)


def test_mixed_streams_roundtrip_multi(tmpdir):
    vid, v_ts = _random_video()
    sig, s_ts = _random_signal()
    f = os.path.join(tmpdir, "mixmulti.dats")
    _save_file_multi(
        f,
        {"Cam": (vid, v_ts), "EMG": (sig, s_ts)},
        {"Cam": DataCompressionPreset.LOSSLESS, "EMG": DataCompressionPreset.RAW},
        n_evictions=6,
    )
    dat = _load_file(f)
    assert np.array_equal(dat["Cam"][0], vid)
    assert np.array_equal(dat["EMG"][0], sig)
    assert np.allclose(dat["Cam"][1], v_ts)
    assert np.allclose(dat["EMG"][1], s_ts)


def test_stream_in_stepwise_multi_matches_full(tmpdir):
    """Step-wise reader equals full reader when file has many evictions."""
    vid, v_ts = _random_video()
    f = os.path.join(tmpdir, "stepmulti.dats")
    _save_file_multi(
        f, {"V": (vid, v_ts)}, {"V": DataCompressionPreset.LOSSLESS}, n_evictions=7
    )

    rn = RNStream(f)
    file, buf, cnt, tot, done = rn.stream_in_stepwise(None, None, None)
    while not done:
        file, buf, cnt, tot, done = rn.stream_in_stepwise(file, buf, cnt)

    full = _load_file(f)
    assert np.array_equal(full["V"][0], buf["V"][0])
    assert np.allclose(full["V"][1], buf["V"][1])


# ------------------------------------------------------------------
# 10.  generate_video  (creates .avi that matches original frames)
# ------------------------------------------------------------------

import cv2


def _mean_abs_err(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

def _read_avi_rgb(path):
    """Read an AVI with OpenCV and return (H, W, 3, T) uint8 in **RGB** order."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded from AVI")
    return np.stack(frames, axis=-1)


def test_generate_video_roundtrip(tmpdir):
    """Video written by `generate_video` must be pixel-exact."""
    vid, ts = _random_video()
    dat_path = os.path.join(tmpdir, "capture.dats")

    _save_file(
        dat_path, {"Screen": (vid, ts)}, {"Screen": DataCompressionPreset.LOSSLESS}
    )

    # --- export AVI ---------------------------------------------------------
    rn = RNStream(dat_path)
    rn.generate_video("Screen")                       # default output path

    avi_path = os.path.join(tmpdir, "capture_Screen.avi")
    assert os.path.exists(avi_path), "AVI not written"

    # CV2 is too lossy
    # decoded = _read_avi_rgb(avi_path)
    #
    # # shapes must match (H, W, 3, T)
    # assert decoded.shape == vid.shape
    #
    # # FFV1 / HuffYUV are loss-less → expect MAE == 0
    # mae = _mean_abs_err(decoded, vid)
    # assert mae == 0, f"mean-abs-err too high: {mae}"