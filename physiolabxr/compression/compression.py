import io, warnings
from fractions import Fraction
from enum import Enum, unique
import numpy as np

try:
    import av  # TODO windows/macOs: ship with a static FFmpeg build. For Ubuntu, use system FFmpeg. If av can't be imported, let the user know and fall back to RAW.
except ImportError:
    av = None


# ──────────────────────────────────────────────────────────────────────────────
@unique
class DataCompressionPreset(Enum):
    RAW        = (0, None,            "")                       # id, codec, opts
    LOSSLESS   = (1, "libx264rgb",    "-crf 0 -preset veryfast -tune zerolatency -bf 0")

    def __init__(self, cid: int, codec: str | None, opts: str):
        self.cid        = cid          # 1-byte integer written to header
        self.codec      = codec
        self.ffmpeg_opts= opts

    def is_raw(self) -> bool:
        return self.codec is None

    def arglist(self):                 # convenience
        return self.ffmpeg_opts.split() if self.ffmpeg_opts else []

# reverse lookup used by stream_in
COMPRESSION_ID_TO_PRESET = {p.cid: p for p in DataCompressionPreset}


# ──────────────────────────────────────────────────────────────────────────────
class EncoderProxy:
    def __init__(self, *, preset: DataCompressionPreset,
                 width: int, height: int):
        self.disabled = preset.is_raw() or av is None
        self._buf = io.BytesIO()
        if self.disabled:
            return

        self.ctx = av.CodecContext.create(preset.codec, "w")
        self.ctx.width  = width
        self.ctx.height = height
        self.ctx.pix_fmt = "rgb24"
        self.ctx.time_base = Fraction(1, 1_000_000)   # μs PTS
        for k, v in zip(preset.arglist()[::2], preset.arglist()[1::2]):
            self.ctx.options[k.lstrip("-")] = v
        self._t0 = None

    # ------------------------------------------------------------------ #
    def push(self, frame: np.ndarray, ts_us: int):
        if self.disabled:
            self._buf.write(frame.tobytes())
            return

        if self._t0 is None:
            self._t0 = ts_us
        if frame.shape[-1] == 4:          # drop alpha
            frame = frame[..., :3]
        frm = av.VideoFrame.from_ndarray(frame, format="rgb24")
        frm.pts = ts_us - self._t0

        for pkt in self.ctx.encode(frm):
            self._buf.write(bytes(pkt))   # ← raw Annex-B packet

    # ------------------------------------------------------------------ #
    def pop(self, *, flush: bool = False) -> bytes:
        """
        Return bytes accumulated so far.
        If *flush* is True **once only**, also drain the encoder’s
        delayed frames.
        """
        if flush and not self.disabled and not getattr(self, "_flushed", False):
            try:
                for pkt in self.ctx.encode():           # flush
                    self._buf.write(bytes(pkt))
            except av.EOFError:                         # already flushed
                pass
            self._flushed = True

        data = self._buf.getvalue()
        self._buf.seek(0); self._buf.truncate(0)
        return data

    def close(self):
        return self.pop(flush=True)


def decode_h264(blob: bytes) -> np.ndarray:
    """
    Decode a raw Annex-B H-264 elementary stream and return
    an array shaped (H, W, 3, T) uint8 (RGB24).
    """
    frames: list[np.ndarray] = []

    with av.open(io.BytesIO(blob), "r", format="h264") as cont:
        # demuxer splits the bytestream into proper packets
        for packet in cont.demux(video=0):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise RuntimeError("no frames decoded from blob of size "
                           f"{len(blob)} bytes")

    return np.stack(frames, axis=-1)