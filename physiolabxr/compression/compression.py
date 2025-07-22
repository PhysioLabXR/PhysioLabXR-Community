import io, warnings
from fractions import Fraction
from enum import Enum, unique
import numpy as np

from physiolabxr.utils.bidict import Bidict

try:
    import av  # TODO windows/macOs: ship with a static FFmpeg build. For Ubuntu, use system FFmpeg. If av can't be imported, let the user know and fall back to RAW.
except ImportError:
    av = None


# ──────────────────────────────────────────────────────────────────────────────
from enum import Enum, unique


@unique
class DataCompressionPreset(Enum):
    _ignore_ = ("_codec", "_ffmpeg_opts", "_label")
    # value  = cid
    RAW      = "Raw (uncompressed)"
    LOSSLESS = "Lossless H-264 RGB"

    # ------------------------------------------------------------------
    #  convenience properties
    # ------------------------------------------------------------------
    @property
    def cid(self) -> int:                       # keep the old name
        return _CID[self]                       # 1-byte id written to file

    @property
    def codec(self) -> str | None:
        return _CODEC[self]

    @property
    def ffmpeg_opts(self) -> str:
        return _FFMPEG_OPTS[self]

    def is_raw(self) -> bool:
        return self.codec is None

    def arglist(self) -> list[str]:
        """Return ``ffmpeg`` option list."""
        return self.ffmpeg_opts.split() if self.ffmpeg_opts else []

    @classmethod
    def from_cid(cls, cid: int) -> "DataCompressionPreset":
        """
        Reverse-lookup from the 1-byte *cid* stored in the file header
        back to the enum member.

        >>> DataCompressionPreset.from_cid(1) is DataCompressionPreset.LOSSLESS
        True
        >>> DataCompressionPreset.from_cid(99)
        KeyError: 'unknown compression id: 99'
        """
        try:
            return _CID.inv[cid]                # Enum does the heavy lifting
        except ValueError:                 # but massage the error message
            raise KeyError(f"unknown compression id: {cid}") from None

_CODEC       = Bidict({
    DataCompressionPreset.RAW     : None,
    DataCompressionPreset.LOSSLESS: "libx264rgb",
})

_FFMPEG_OPTS = Bidict({
    DataCompressionPreset.RAW     : "",
    DataCompressionPreset.LOSSLESS: "-crf 0 -preset veryfast -tune zerolatency -bf 0",
})

_CID = Bidict({
    DataCompressionPreset.RAW: 0,
    DataCompressionPreset.LOSSLESS: 1,
})

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