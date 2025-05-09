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
    """
    TODO the data compression only supports video atm.
    """
    RAW               = (None, "")
    LOSSLESS          = ("libx264rgb", "-crf 0 -preset veryfast")  # TODO this is defaulted to for video streams in RecordinTab
    VISUALLY_LOSSLESS = ("libx264","-crf 16 -preset veryfast")  # TODO not tested

    def __init__(self, codec, opts):   # noqa: D401
        self.codec = codec
        self.ffmpeg_opts = opts
    def is_raw(self) -> bool:          # noqa: D401
        return self.codec is None
    def arglist(self) -> list[str]:
        return self.ffmpeg_opts.split() if self.ffmpeg_opts else []


# ──────────────────────────────────────────────────────────────────────────────
class EncoderProxy:
    """One long-lived encoder per stream   (frames→packets)."""

    def __init__(self, *, preset: DataCompressionPreset,
                 width: int, height: int):
        if av is None or preset.is_raw():
            self.disabled = True
            self._buf = io.BytesIO()
            return

        self.disabled = False
        self._buf = io.BytesIO()
        self._container = av.open(self._buf, mode="w", format="matroska")
        self._stream = self._container.add_stream(preset.codec, rate=30)  # 30 is the default rate, it will be overridden when we pass pts (micro-seconds since the first frame)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = 'rgb24'
        self._stream.time_base = Fraction(1, 1_000_000)   # μs PTS
        # handle extra options
        args = preset.arglist()
        for k, v in zip(args[::2], args[1::2]):
            self._stream.codec_context.options[k.lstrip("-")] = v
        self._t0 = None

    # ------------------------------------------------------------------ #
    def push(self, frame_array: np.ndarray, ts_us: int):
        """
        frame_array: (H, W, 3) or (H, W, 4) uint8
        ts_us    : timestamp in micro-seconds
        """
        if self.disabled:
            self._buf.write(frame_array.tobytes())  # RAW fallback
            return

        if self._t0 is None:
            self._t0 = ts_us
        # drop the alpha channel if present
        if frame_array.shape[-1] == 4:
            frm = av.VideoFrame.from_ndarray(frame_array[..., :3], format="rgb24")
        else:
            frm = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        frm.pts = ts_us - self._t0
        for pkt in self._stream.encode(frm):
            self._container.mux(pkt)

    # ------------------------------------------------------------------ #
    def pop(self) -> bytes:
        """
        Flush packets produced **so far** and return them.
        (Called once each eviction interval.)
        """
        if self.disabled:
            data = self._buf.getvalue()
            self._buf.seek(0); self._buf.truncate(0)
            return data

        data = self._buf.getvalue()
        self._buf.seek(0); self._buf.truncate(0)
        return data

    # ------------------------------------------------------------------ #
    def close(self) -> bytes:
        """Flush final packets when RNStream is destroyed."""
        if not self.disabled:
            self._container.mux(self._stream.encode())
            self._container.close()
        return self.pop()
