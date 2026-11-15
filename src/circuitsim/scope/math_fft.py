"""Qt-free oscilloscope math traces, periodic-Hann FFT, XY mode, and cursors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np


class MathOp(str, Enum):
    """Supported two-channel math operations."""

    SUB = "CH1-CH2"
    ADD = "CH1+CH2"
    MUL = "CH1*CH2"


@dataclass(frozen=True)
class MathConfig:
    """Configuration for derived oscilloscope math trace."""

    enabled: bool = False
    op: MathOp = MathOp.SUB
    source_a: str = "CH1"
    source_b: str = "CH2"

    def __post_init__(self) -> None:
        if not isinstance(self.op, MathOp):
            object.__setattr__(self, "op", MathOp(self.op))


def compute_math_trace(va: np.ndarray, vb: np.ndarray, op: MathOp | str) -> np.ndarray:
    """Compute point-wise math trace ``CH1-CH2``, ``CH1+CH2``, or ``CH1*CH2``."""
    a = np.asarray(va, dtype=np.float64)
    b = np.asarray(vb, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Math inputs must be 1D arrays of matching shape")
    op_enum = op if isinstance(op, MathOp) else MathOp(op)
    if op_enum == MathOp.SUB:
        out = a - b
    elif op_enum == MathOp.ADD:
        out = a + b
    else:
        out = a * b
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class FFTResult:
    """Single-sided FFT magnitude spectrum (in V peak and dBV) and peak readout."""

    freqs_hz: np.ndarray
    magnitude_v: np.ndarray
    magnitude_dbv: np.ndarray
    fundamental_freq_hz: float | None
    fundamental_mag_v: float | None
    fundamental_dbv: float | None
    rbw_hz: float
    window_name: str = "hann"

    def __post_init__(self) -> None:
        self._lock_arrays()

    def _lock_arrays(self) -> None:
        for arr in (self.freqs_hz, self.magnitude_v, self.magnitude_dbv):
            arr.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock_arrays()


def _periodic_window(n: int, name: str) -> tuple[np.ndarray, float]:
    norm_name = name.strip().lower()
    if norm_name in ("hann", "hanning"):
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)
        # Periodic Hann normalized energy factor across 3 mainlobe bins: sum(w^2)/n / cg^2 = 0.375 / 0.25 = 1.5
        return w, 1.5
    if norm_name in ("rect", "rectangular", "boxcar"):
        w = np.ones(n, dtype=np.float64)
        return w, 1.0
    raise ValueError(f"Unsupported FFT window {name!r}; expected 'hann' or 'rect'")


def compute_fft(
    t: np.ndarray, v: np.ndarray, window: str = "hann"
) -> FFTResult:
    """Compute single-sided FFT with periodic Hann window and coherent-gain correction.

    Amplitude scaling sets a coherent sine ``A * sin(2*pi*f*t)`` to peak ``A`` V
    (``20*log10(A)`` dBV). ``fundamental_mag_v`` integrates mainlobe energy around the
    dominant AC peak so off-bin tones remain accurate within < 0.1 dB.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if t_arr.ndim != 1 or v_arr.ndim != 1 or len(t_arr) != len(v_arr):
        raise ValueError("t and v must be 1D arrays of identical length")

    n = len(t_arr)
    if n < 4 or float(t_arr[-1] - t_arr[0]) <= 0.0:
        empty = np.empty(0, dtype=np.float64)
        return FFTResult(
            freqs_hz=empty,
            magnitude_v=empty.copy(),
            magnitude_dbv=empty.copy(),
            fundamental_freq_hz=None,
            fundamental_mag_v=None,
            fundamental_dbv=None,
            rbw_hz=0.0,
            window_name=window,
        )

    dt_span = float(t_arr[-1] - t_arr[0]) / (n - 1)
    diffs = np.diff(t_arr)
    if float(np.max(np.abs(diffs - dt_span))) > 1e-6 * dt_span:
        t_u = t_arr[0] + np.arange(n, dtype=np.float64) * dt_span
        y_u = np.interp(t_u, t_arr, v_arr)
    else:
        y_u = v_arr

    w, mainlobe_factor = _periodic_window(n, window)
    coherent_gain = float(np.mean(w))
    spec = np.fft.rfft(y_u * w)
    mag = (2.0 / (n * coherent_gain)) * np.abs(spec)
    mag[0] *= 0.5
    if n % 2 == 0 and len(mag) > 1:
        mag[-1] *= 0.5

    freqs = np.fft.rfftfreq(n, d=dt_span)
    mag_dbv = 20.0 * np.log10(np.maximum(mag, 1e-12))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    # Equivalent Noise Bandwidth / 3dB mainlobe RBW: ~1.5 * df for Hann, 1.0 * df for rect
    rbw_hz = (1.44 * df) if mainlobe_factor > 1.0 else df

    fund_freq: float | None = None
    fund_mag: float | None = None
    fund_dbv: float | None = None
    if len(mag) > 2:
        k_ac = 1 + int(np.argmax(mag[1:]))
        if mag[k_ac] > 1e-12:
            lo = max(1, k_ac - 1)
            hi = min(len(mag), k_ac + 2)
            # Sub-bin parabolic frequency interpolation on log magnitude
            if 1 <= k_ac < len(mag) - 1:
                y0 = float(mag_dbv[k_ac - 1])
                y1 = float(mag_dbv[k_ac])
                y2 = float(mag_dbv[k_ac + 1])
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                delta = max(-0.5, min(0.5, delta))
                fund_freq = float((k_ac + delta) * df)
            else:
                fund_freq = float(freqs[k_ac])

            energy_bins = float(np.sum(mag[lo:hi] ** 2))
            fund_mag = math.sqrt(energy_bins / mainlobe_factor)
            fund_dbv = 20.0 * math.log10(max(fund_mag, 1e-12))

    return FFTResult(
        freqs_hz=freqs,
        magnitude_v=mag,
        magnitude_dbv=mag_dbv,
        fundamental_freq_hz=fund_freq,
        fundamental_mag_v=fund_mag,
        fundamental_dbv=fund_dbv,
        rbw_hz=rbw_hz,
        window_name=window,
    )


@dataclass(frozen=True)
class XYTrace:
    """Two-channel Lissajous / X-Y trace."""

    x: np.ndarray
    y: np.ndarray
    x_channel: str = "CH1"
    y_channel: str = "CH2"

    def __post_init__(self) -> None:
        x_arr = np.asarray(self.x, dtype=np.float64)
        y_arr = np.asarray(self.y, dtype=np.float64)
        if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
            raise ValueError("XYTrace x and y must be 1D arrays of matching shape")
        if x_arr is not self.x:
            object.__setattr__(self, "x", x_arr)
        if y_arr is not self.y:
            object.__setattr__(self, "y", y_arr)
        x_arr.setflags(write=False)
        y_arr.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.x.setflags(write=False)
        self.y.setflags(write=False)


def compute_xy(
    x: np.ndarray,
    y: np.ndarray,
    x_channel: str = "CH1",
    y_channel: str = "CH2",
) -> XYTrace:
    """Create immutable ``XYTrace`` from matched channel voltage vectors."""
    return XYTrace(x=np.array(x, dtype=np.float64), y=np.array(y, dtype=np.float64), x_channel=x_channel, y_channel=y_channel)


@dataclass(frozen=True)
class CursorReadout:
    """Dual vertical/horizontal cursor difference readout."""

    t1: float
    t2: float
    dt: float
    inv_dt_hz: float | None
    v1: float
    v2: float
    dv: float


def compute_cursors(
    t: np.ndarray,
    v: np.ndarray,
    t1: float,
    t2: float,
    *,
    v1: float | None = None,
    v2: float | None = None,
) -> CursorReadout:
    """Compute time/frequency delta and voltage delta between two cursor positions."""
    t_arr = np.asarray(t, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if len(t_arr) > 0 and len(t_arr) == len(v_arr):
        val1 = float(np.interp(t1, t_arr, v_arr)) if v1 is None else float(v1)
        val2 = float(np.interp(t2, t_arr, v_arr)) if v2 is None else float(v2)
    else:
        val1 = 0.0 if v1 is None else float(v1)
        val2 = 0.0 if v2 is None else float(v2)

    dt = float(t2 - t1)
    inv_dt = (1.0 / abs(dt)) if abs(dt) > 1e-18 else None
    return CursorReadout(
        t1=float(t1),
        t2=float(t2),
        dt=dt,
        inv_dt_hz=inv_dt,
        v1=val1,
        v2=val2,
        dv=float(val2 - val1),
    )
