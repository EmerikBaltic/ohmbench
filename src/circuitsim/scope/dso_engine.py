"""Qt-free Voltcraft DSO-1062D signal processing, trigger, math, FFT, and cursor engine."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any

import numpy as np

from circuitsim.scope.measurements import measure_channel
from circuitsim.scope.trigger import find_rising_midpoint_crossings

DSO_HORIZONTAL_DIVS: int = 12
DSO_VERTICAL_DIVS: int = 8

DSO_VOLTS_PER_DIV: tuple[float, ...] = (
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
)

DSO_SEC_PER_DIV: tuple[float, ...] = (
    2e-9,
    5e-9,
    1e-8,
    2e-8,
    5e-8,
    1e-7,
    2e-7,
    5e-7,
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
)

DSO_AVG_COUNTS: tuple[int, ...] = (4, 16, 64, 128)
DSO_MEM_DEPTHS: tuple[int, ...] = (4000, 40000, 512000, 1000000)


@dataclass
class DSOChannelState:
    """Vertical channel settings matching DSO-1062D CH1/CH2 menus."""

    enabled: bool = True
    volts_per_div: float = 2.0
    position_div: float = 0.0
    coupling: str = "DC"  # "DC", "AC", "GND"
    probe_atten: float = 1.0  # 1.0 or 10.0
    invert: bool = False
    bw_limit: bool = False


@dataclass
class DSOMathState:
    """MATH & FFT subsystem state."""

    enabled: bool = False
    operation: str = "CH1-CH2"  # "CH1+CH2", "CH1-CH2", "CH1*CH2", "CH1/CH2", "FFT"
    fft_source: str = "CH1"  # "CH1", "CH2"
    scale: float = 2.0
    position_div: float = 0.0
    fft_window: str = "HANNING"


@dataclass
class DSOHorizontalState:
    """Horizontal timebase state."""

    sec_per_div: float = 5e-3
    position_s: float = 0.0
    format_mode: str = "YT"  # "YT", "XY"
    zoom_enabled: bool = False
    zoom_sec_per_div: float = 200e-6


@dataclass
class DSOTriggerState:
    """Edge/auto/normal/single trigger configuration."""

    source: str = "CH1"  # "CH1", "CH2", "EXT"
    slope: str = "RISING"  # "RISING", "FALLING", "EITHER"
    level_v: float = 0.0
    mode: str = "AUTO"  # "AUTO", "NORMAL", "SINGLE"
    coupling: str = "DC"  # "DC", "AC"
    holdoff_s: float = 0.0
    single_armed: bool = False
    forced: bool = False


@dataclass
class DSOAcquireState:
    """Acquisition mode & depth settings."""

    acq_type: str = "REAL_TIME"  # "REAL_TIME", "EQU_TIME"
    acq_mode: str = "NORMAL"  # "NORMAL", "PEAK", "AVERAGE"
    avg_count: int = 4
    mem_depth: int = 4000


@dataclass
class DSODisplayState:
    """LCD display rendering options."""

    render_style: str = "VECTORS"  # "VECTORS", "DOTS"
    persistence: bool = False
    grid_brightness: float = 0.8


@dataclass
class DSOCursorState:
    """Manual measurement cursor state."""

    cursor_mode: str = "OFF"  # "OFF", "VOLTAGE", "TIME"
    cursor_source: str = "CH1"  # "CH1", "CH2"
    v1: float = -2.0
    v2: float = 2.0
    t1: float = -1e-3
    t2: float = 1e-3


@dataclass
class DSOState:
    """Complete mutable state of one DSO-1062D oscilloscope instance."""

    ch1: DSOChannelState = field(default_factory=DSOChannelState)
    ch2: DSOChannelState = field(default_factory=DSOChannelState)
    math: DSOMathState = field(default_factory=DSOMathState)
    horizontal: DSOHorizontalState = field(default_factory=DSOHorizontalState)
    trigger: DSOTriggerState = field(default_factory=DSOTriggerState)
    acquire: DSOAcquireState = field(default_factory=DSOAcquireState)
    display: DSODisplayState = field(default_factory=DSODisplayState)
    cursors: DSOCursorState = field(default_factory=DSOCursorState)
    running: bool = True
    auto_measure_enabled: bool = True

    @property
    def sec_per_div(self) -> float:
        return self.horizontal.sec_per_div

    @sec_per_div.setter
    def sec_per_div(self, value: float) -> None:
        self.horizontal.sec_per_div = float(value)


@dataclass(frozen=True)
class DSOCursorReadout:
    """Computed delta values for manual voltage/time cursors."""

    mode: str
    source: str
    v1: float
    v2: float
    t1: float
    t2: float
    delta_v: float
    delta_t: float
    inv_delta_t_hz: float | None


@dataclass(frozen=True)
class DSOChannelMeasurements:
    """Single-channel IEEE-style waveform measurements."""

    v_pp: float
    v_max: float
    v_min: float
    v_avg: float
    v_rms: float
    freq_hz: float | None
    period_s: float | None
    duty_cycle_pct: float | None


@dataclass(frozen=True)
class DSOFrame:
    """Processed DSO-1062D frame ready for screen rendering & CSV export."""

    times_rel_s: np.ndarray
    ch1_v: np.ndarray
    ch2_v: np.ndarray
    math_v: np.ndarray
    fft_freqs_hz: np.ndarray
    fft_mag_db: np.ndarray
    fft_peak_hz: float | None
    triggered: bool
    trigger_status: str  # "Trig'd", "Auto", "Wait", "Stop"
    trigger_time_s: float | None
    trig_freq_hz: float | None
    captured_single: bool
    ch1_meas: DSOChannelMeasurements
    ch2_meas: DSOChannelMeasurements
    cursor_readout: DSOCursorReadout
    persistence_traces: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    t_left_s: float = 0.0
    t_right_s: float = 0.06
    span_s: float = 0.06
    x_norm: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    roll_mode: bool = False

    @property
    def ch1_measurements(self) -> DSOChannelMeasurements:
        return self.ch1_meas

    @property
    def ch2_measurements(self) -> DSOChannelMeasurements:
        return self.ch2_meas

    @property
    def hw_freq_counter_hz(self) -> float | None:
        return self.trig_freq_hz


class DSORingBuffer:
    """Pre-allocated circular NumPy buffer storing up to `capacity` multi-channel samples."""

    def __init__(self, capacity: int = 65536) -> None:
        if capacity < 1:
            raise ValueError(f"DSORingBuffer capacity must be >= 1, got {capacity}.")
        self.capacity: int = int(capacity)
        self._times = np.zeros(self.capacity, dtype=np.float64)
        self._ch1 = np.zeros(self.capacity, dtype=np.float64)
        self._ch2 = np.zeros(self.capacity, dtype=np.float64)
        self._ext = np.zeros(self.capacity, dtype=np.float64)
        self._out_times = np.zeros(self.capacity, dtype=np.float64)
        self._out_ch1 = np.zeros(self.capacity, dtype=np.float64)
        self._out_ch2 = np.zeros(self.capacity, dtype=np.float64)
        self._out_ext = np.zeros(self.capacity, dtype=np.float64)
        self._head: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = 0
        self._size = 0

    def last_time(self) -> float | None:
        if self._size == 0:
            return None
        return float(self._times[(self._head - 1) % self.capacity])

    def last_sample(self) -> tuple[float, float, float, float] | None:
        if self._size == 0:
            return None
        idx = (self._head - 1) % self.capacity
        return (
            float(self._times[idx]),
            float(self._ch1[idx]),
            float(self._ch2[idx]),
            float(self._ext[idx]),
        )

    def append_sample(self, t: float, ch1: float, ch2: float, ext: float) -> None:
        idx = self._head
        self._times[idx] = float(t)
        self._ch1[idx] = float(ch1)
        self._ch2[idx] = float(ch2)
        self._ext[idx] = float(ext)
        self._head = (idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def append_chunk(
        self,
        t: np.ndarray,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ext: np.ndarray,
    ) -> None:
        t_arr = np.asarray(t, dtype=np.float64)
        c1_arr = np.asarray(ch1, dtype=np.float64)
        c2_arr = np.asarray(ch2, dtype=np.float64)
        e_arr = np.asarray(ext, dtype=np.float64)
        n = min(len(t_arr), len(c1_arr), len(c2_arr), len(e_arr))
        if n <= 0:
            return
        if n >= self.capacity:
            tail = slice(n - self.capacity, n)
            self._times[:] = t_arr[tail]
            self._ch1[:] = c1_arr[tail]
            self._ch2[:] = c2_arr[tail]
            self._ext[:] = e_arr[tail]
            self._head = 0
            self._size = self.capacity
            return
        first = min(n, self.capacity - self._head)
        h = self._head
        self._times[h : h + first] = t_arr[:first]
        self._ch1[h : h + first] = c1_arr[:first]
        self._ch2[h : h + first] = c2_arr[:first]
        self._ext[h : h + first] = e_arr[:first]
        rem = n - first
        if rem > 0:
            self._times[:rem] = t_arr[first:n]
            self._ch1[:rem] = c1_arr[first:n]
            self._ch2[:rem] = c2_arr[first:n]
            self._ext[:rem] = e_arr[first:n]
        self._head = (h + n) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def set_arrays(
        self,
        t: np.ndarray,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ext: np.ndarray,
    ) -> None:
        self.clear()
        self.append_chunk(t, ch1, ch2, ext)

    def ordered_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._size == 0:
            return (
                self._out_times[:0],
                self._out_ch1[:0],
                self._out_ch2[:0],
                self._out_ext[:0],
            )
        start = (self._head - self._size) % self.capacity
        if start + self._size <= self.capacity:
            end = start + self._size
            return (
                self._times[start:end],
                self._ch1[start:end],
                self._ch2[start:end],
                self._ext[start:end],
            )
        first = self.capacity - start
        second = self._size - first
        for src, dst in (
            (self._times, self._out_times),
            (self._ch1, self._out_ch1),
            (self._ch2, self._out_ch2),
            (self._ext, self._out_ext),
        ):
            dst[:first] = src[start:]
            dst[first : self._size] = src[:second]
        return (
            self._out_times[: self._size],
            self._out_ch1[: self._size],
            self._out_ch2[: self._size],
            self._out_ext[: self._size],
        )


def decimate_xy_minmax(
    x_norm: np.ndarray,
    y: np.ndarray,
    *,
    max_columns: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """Strip non-finite points and compute pixel-column min/max extrema above 2000 samples."""
    n = min(len(x_norm), len(y))
    if n <= 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    xv = np.asarray(x_norm[:n], dtype=np.float64)
    yv = np.asarray(y[:n], dtype=np.float64)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if not bool(np.all(valid)):
        xv = xv[valid]
        yv = yv[valid]
    m = len(xv)
    if m <= 2000:
        return xv, yv
    n_cols = min(max(int(max_columns), 64), m // 2, 800)
    edges = np.linspace(0, m, n_cols + 1, dtype=np.int64)
    starts = edges[:-1]
    ends = np.maximum(edges[1:] - 1, starts)
    y_mins = np.minimum.reduceat(yv, starts)
    y_maxs = np.maximum.reduceat(yv, starts)
    x_out = np.empty(2 * n_cols, dtype=np.float64)
    y_out = np.empty(2 * n_cols, dtype=np.float64)
    x_out[0::2] = xv[starts]
    x_out[1::2] = xv[ends]
    y_out[0::2] = y_mins
    y_out[1::2] = y_maxs
    return x_out, y_out


def snap_dso_step(value: float, steps: tuple[float, ...]) -> float:
    """Snap a positive value to the nearest 1-2-5 table step."""
    val = max(float(value), steps[0])
    return min(steps, key=lambda s: abs(math.log10(s) - math.log10(val)))


def _estimate_dc_offset(times: np.ndarray, sig: np.ndarray) -> float:
    """Estimate accurate DC offset using integer periods when >= 2 crossings exist."""
    if len(sig) == 0:
        return 0.0
    if len(times) == len(sig) and len(times) >= 4:
        rises, vpp = find_rising_midpoint_crossings(times, sig)
        if vpp > 1e-12 and len(rises) >= 2:
            t0, t1 = float(rises[0]), float(rises[-1])
            dur = t1 - t0
            if dur > 1e-15:
                v0 = float(np.interp(t0, times, sig))
                v1 = float(np.interp(t1, times, sig))
                i0 = int(np.searchsorted(times, t0, side="right"))
                i1 = int(np.searchsorted(times, t1, side="left"))
                t_seg = np.concatenate(([t0], times[i0:i1], [t1]))
                v_seg = np.concatenate(([v0], sig[i0:i1], [v1]))
                integ = (
                    float(np.trapezoid(v_seg, t_seg))
                    if hasattr(np, "trapezoid")
                    else float(np.trapz(v_seg, t_seg))
                )
                return integ / dur
    return float(np.mean(sig))


def _condition_channel(
    times: np.ndarray, raw: np.ndarray, cfg: DSOChannelState
) -> np.ndarray:
    """Apply coupling (`DC`, `AC`, `GND`), invert, and 20 MHz BW smoothing."""
    arr = np.asarray(raw, dtype=np.float64)
    coupling = cfg.coupling.upper()
    if coupling == "GND":
        out = np.zeros_like(arr)
    elif coupling == "AC":
        out = arr - _estimate_dc_offset(times, arr)
    else:
        out = arr.copy()
    if cfg.invert:
        out = -out
    if cfg.bw_limit and len(out) >= 5:
        padded = np.pad(out, 2, mode="edge")
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
        out = np.convolve(padded, kernel, mode="valid")
    return out


def _latest_contiguous_slice(times: np.ndarray) -> slice:
    """Isolate the most recent monotonically increasing run after a simulation restart."""
    n = len(times)
    if n < 2:
        return slice(0, n)
    diffs = np.diff(times)
    reset_idxs = np.flatnonzero(diffs <= 0.0)
    if len(reset_idxs) == 0:
        return slice(0, n)
    return slice(int(reset_idxs[-1]) + 1, n)


def _find_edge_crossings(
    times: np.ndarray,
    trig_sig: np.ndarray,
    level_v: float,
    slope: str,
    holdoff_s: float = 0.0,
) -> np.ndarray:
    """Return linearly interpolated crossing timestamps with noise hysteresis."""
    if len(times) < 2 or len(times) != len(trig_sig):
        return np.empty(0, dtype=np.float64)
    v_min = float(np.min(trig_sig))
    v_max = float(np.max(trig_sig))
    v_pp = v_max - v_min
    v_mid = 0.5 * (v_max + v_min)
    if v_pp < max(1e-4, 1e-5 * abs(v_mid)):
        return np.empty(0, dtype=np.float64)

    hyst = min(max(0.05 * v_pp, 1e-4), 0.25 * v_pp)
    low_arm = max(level_v - hyst, 0.5 * (level_v + v_min)) if level_v > v_min else level_v
    high_arm = min(level_v + hyst, 0.5 * (level_v + v_max)) if level_v < v_max else level_v

    slope_u = slope.upper()
    want_rise = slope_u in ("RISING", "EITHER")
    want_fall = slope_u in ("FALLING", "EITHER")

    a = trig_sig[:-1]
    b = trig_sig[1:]
    cand_parts: list[np.ndarray] = []
    if want_rise:
        cand_parts.append(np.flatnonzero((a < level_v) & (b >= level_v)))
    if want_fall:
        cand_parts.append(np.flatnonzero((a > level_v) & (b <= level_v)))
    if not cand_parts:
        return np.empty(0, dtype=np.float64)
    cands = cand_parts[0] if len(cand_parts) == 1 else np.sort(np.concatenate(cand_parts))
    if len(cands) == 0:
        return np.empty(0, dtype=np.float64)

    v0 = float(trig_sig[0])
    crossings: list[float] = []
    last_t = -math.inf
    last_rise_idx: int | None = None
    last_fall_idx: int | None = None
    for idx_np in cands:
        i = int(idx_np)
        av = float(trig_sig[i])
        bv = float(trig_sig[i + 1])
        ta = float(times[i])
        tb = float(times[i + 1])
        if want_rise and av < level_v <= bv:
            if last_rise_idx is None:
                armed = (v0 <= low_arm or v0 < level_v - 0.1 * hyst) or (
                    float(np.min(trig_sig[: i + 1])) <= low_arm
                )
            else:
                armed = (i > last_rise_idx) and (
                    float(np.min(trig_sig[last_rise_idx + 1 : i + 1])) <= low_arm
                )
            if armed:
                dv = bv - av
                frac = max(0.0, min(1.0, ((level_v - av) / dv) if abs(dv) > 1e-18 else 0.0))
                tc = ta + frac * (tb - ta)
                if holdoff_s <= 0.0 or (tc - last_t) >= holdoff_s:
                    crossings.append(tc)
                    last_t = tc
                last_rise_idx = i
        if want_fall and av > level_v >= bv:
            if last_fall_idx is None:
                armed = (v0 >= high_arm or v0 > level_v + 0.1 * hyst) or (
                    float(np.max(trig_sig[: i + 1])) >= high_arm
                )
            else:
                armed = (i > last_fall_idx) and (
                    float(np.max(trig_sig[last_fall_idx + 1 : i + 1])) >= high_arm
                )
            if armed:
                dv = bv - av
                frac = max(0.0, min(1.0, ((level_v - av) / dv) if abs(dv) > 1e-18 else 0.0))
                tc = ta + frac * (tb - ta)
                if holdoff_s <= 0.0 or (tc - last_t) >= holdoff_s:
                    crossings.append(tc)
                    last_t = tc
                last_fall_idx = i

    return np.array(crossings, dtype=np.float64) if crossings else np.empty(0, dtype=np.float64)


def _measure_dso_channel(times: np.ndarray, sig: np.ndarray) -> DSOChannelMeasurements:
    if len(times) < 2 or len(sig) < 2:
        val = float(sig[0]) if len(sig) > 0 else 0.0
        return DSOChannelMeasurements(0.0, val, val, val, abs(val), None, None, None)
    n = min(len(times), len(sig))
    if n > 4096:
        step = n // 4096
        t_m = times[:n:step]
        s_m = sig[:n:step]
        v_max_full = float(np.max(sig[:n]))
        v_min_full = float(np.min(sig[:n]))
    else:
        t_m = times[:n]
        s_m = sig[:n]
        v_max_full = None
        v_min_full = None
    m = measure_channel(t_m, s_m)
    v_max = v_max_full if v_max_full is not None else float(m.vmax)
    v_min = v_min_full if v_min_full is not None else float(m.vmin)
    vpp = max(0.0, v_max - v_min) if v_max_full is not None else float(m.vpp)
    vavg = float(m.vavg)
    freq = None if m.frequency_hz is None else float(m.frequency_hz)
    per = None if m.period_s is None else float(m.period_s)
    if vpp < max(1e-4, 1e-4 * abs(vavg)) or (freq is not None and freq < 1.0):
        freq = None
        per = None
    return DSOChannelMeasurements(
        v_pp=vpp,
        v_max=v_max,
        v_min=v_min,
        v_avg=vavg,
        v_rms=float(m.vrms),
        freq_hz=freq,
        period_s=per,
        duty_cycle_pct=None if m.duty_cycle_pct is None else float(m.duty_cycle_pct),
    )


def _compute_fft_trace(
    times: np.ndarray, sig: np.ndarray, window_name: str = "HANNING"
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Compute AC-coupled windowed magnitude spectrum in dBV and parabolic peak frequency."""
    n = len(sig)
    if n < 8 or len(times) != n:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, None
    dt = float(times[-1] - times[0]) / max(1, n - 1)
    if dt <= 0.0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, None

    ac = sig - float(np.mean(sig))
    w_u = window_name.upper()
    if w_u == "HAMMING":
        win = np.hamming(n)
    elif w_u == "BLACKMAN":
        win = np.blackman(n)
    elif w_u == "RECTANGLE":
        win = np.ones(n, dtype=np.float64)
    else:
        win = np.hanning(n)

    w_sum = max(float(np.sum(win)), 1e-12)
    spec = np.fft.rfft(ac * win)
    mag = np.abs(spec) * (2.0 / w_sum)
    if len(mag) > 0:
        mag[0] *= 0.5
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))
    freqs = np.fft.rfftfreq(n, d=dt)

    if len(mag) <= 2:
        return freqs, mag_db, None
    peak_idx = int(np.argmax(mag[1:]) + 1)
    if mag[peak_idx] <= 1e-9:
        return freqs, mag_db, None

    # Parabolic peak interpolation around `peak_idx` for sub-bin frequency accuracy
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    delta = 0.0
    if 1 <= peak_idx < len(mag) - 1:
        y0 = float(mag[peak_idx - 1])
        y1 = float(mag[peak_idx])
        y2 = float(mag[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-18:
            delta = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))
    peak_hz = max(0.0, float(freqs[peak_idx]) + delta * df)
    return freqs, mag_db, peak_hz


def _build_cursor_readout(cursors: DSOCursorState) -> DSOCursorReadout:
    dv = float(cursors.v2 - cursors.v1)
    dt = float(cursors.t2 - cursors.t1)
    inv_dt = (1.0 / abs(dt)) if abs(dt) > 1e-15 else None
    return DSOCursorReadout(
        mode=cursors.cursor_mode.upper(),
        source=cursors.cursor_source.upper(),
        v1=float(cursors.v1),
        v2=float(cursors.v2),
        t1=float(cursors.t1),
        t2=float(cursors.t2),
        delta_v=dv,
        delta_t=dt,
        inv_delta_t_hz=inv_dt,
    )


def compute_set_to_50_percent(signal: np.ndarray) -> float:
    """Return 50% midpoint `(max + min) / 2.0` of `signal`."""
    arr = np.asarray(signal, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return 0.5 * (float(np.max(arr)) + float(np.min(arr)))


def reset_default_setup() -> DSOState:
    """Restore factory default DSO-1062D configuration."""
    return DSOState()


def save_setup_json(state: DSOState) -> str:
    """Serialize `DSOState` cleanly to JSON for setup Save / `.circ` persistence."""
    return json.dumps(asdict(state), sort_keys=True, indent=2)


def recall_setup_json(text: str) -> DSOState:
    """Deserialize JSON produced by `save_setup_json` back into a `DSOState`."""
    if not text or not str(text).strip():
        return DSOState()
    raw: dict[str, Any] = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("DSO setup JSON must decode to an object.")

    def _merge(cls: type[Any], payload: Any) -> Any:
        inst = cls()
        if isinstance(payload, dict):
            for k, v in payload.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)
        return inst

    return DSOState(
        ch1=_merge(DSOChannelState, raw.get("ch1")),
        ch2=_merge(DSOChannelState, raw.get("ch2")),
        math=_merge(DSOMathState, raw.get("math")),
        horizontal=_merge(DSOHorizontalState, raw.get("horizontal")),
        trigger=_merge(DSOTriggerState, raw.get("trigger")),
        acquire=_merge(DSOAcquireState, raw.get("acquire")),
        display=_merge(DSODisplayState, raw.get("display")),
        cursors=_merge(DSOCursorState, raw.get("cursors")),
        running=bool(raw.get("running", True)),
        auto_measure_enabled=bool(raw.get("auto_measure_enabled", True)),
    )


def _auto_scale_channel(raw: np.ndarray, cfg: DSOChannelState) -> DSOChannelState:
    updated = copy.deepcopy(cfg)
    arr = np.asarray(raw, dtype=np.float64)
    if len(arr) == 0:
        return updated
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    v_pp = v_max - v_min
    v_mid = 0.5 * (v_max + v_min)
    if v_pp <= 1e-6:
        v_div = 2.0
        for cand in DSO_VOLTS_PER_DIV:
            if abs(v_mid) / cand <= 3.5:
                v_div = cand
                break
        updated.volts_per_div = v_div
        updated.position_div = max(-4.0, min(4.0, -v_mid / v_div))
        return updated

    chosen = DSO_VOLTS_PER_DIV[-1]
    for cand in DSO_VOLTS_PER_DIV:
        span_ok = (v_pp / cand) <= 6.05
        center_ok = abs(v_mid / cand) <= 3.8
        if span_ok and center_ok:
            chosen = cand
            break
    updated.volts_per_div = chosen
    if abs(v_mid) <= 0.1 * max(v_pp, 1e-6):
        updated.position_div = 0.0
    else:
        pos = -v_mid / chosen
        updated.position_div = max(-4.0, min(4.0, round(pos * 20.0) / 20.0))
    return updated


def compute_auto_set(
    times: np.ndarray,
    ch1_raw: np.ndarray,
    ch2_raw: np.ndarray,
    state: DSOState,
) -> DSOState:
    """Compute ideal V/div, vertical centering, sec/div (~2-3 cycles), and 50% trigger."""
    t_arr = np.asarray(times, dtype=np.float64)
    c1 = np.asarray(ch1_raw, dtype=np.float64)
    c2 = np.asarray(ch2_raw, dtype=np.float64)
    new_state = copy.deepcopy(state)
    new_state.ch1 = _auto_scale_channel(c1, new_state.ch1)
    new_state.ch2 = _auto_scale_channel(c2, new_state.ch2)

    active = c1 if (new_state.ch1.enabled and (len(c1) > 0 and np.ptp(c1) > 1e-6)) else c2
    active_name = "CH1" if active is c1 else "CH2"
    if len(active) == 0:
        active = c1
        active_name = "CH1"

    new_state.trigger.source = active_name
    new_state.trigger.slope = "RISING"
    new_state.trigger.mode = "AUTO"
    new_state.trigger.level_v = compute_set_to_50_percent(active)

    if len(t_arr) >= 4 and len(active) == len(t_arr) and float(np.ptp(active)) > 1e-6:
        meas = _measure_dso_channel(t_arr, active)
        if meas.freq_hz is not None and meas.freq_hz > 0.0:
            period_s = 1.0 / meas.freq_hz
            # Target ~3.0 full cycles across the 12 horizontal divisions
            target_sec_div = (3.0 * period_s) / float(DSO_HORIZONTAL_DIVS)
            new_state.horizontal.sec_per_div = snap_dso_step(target_sec_div, DSO_SEC_PER_DIV)
        else:
            span = max(float(t_arr[-1] - t_arr[0]), 1e-9)
            new_state.horizontal.sec_per_div = snap_dso_step(
                span / float(DSO_HORIZONTAL_DIVS), DSO_SEC_PER_DIV
            )
    new_state.horizontal.position_s = 0.0
    new_state.running = True
    return new_state


def process_dso_capture(
    times: np.ndarray,
    ch1_raw: np.ndarray,
    ch2_raw: np.ndarray,
    ext_raw: np.ndarray | None,
    state: DSOState,
    *,
    prev_frame: DSOFrame | None = None,
    live_streaming: bool = False,
) -> DSOFrame:
    """Process raw captured voltages into a full `DSOFrame` per current `DSOState`."""
    t_arr = np.asarray(times, dtype=np.float64)
    c1_in = np.asarray(ch1_raw, dtype=np.float64)
    c2_in = np.asarray(ch2_raw, dtype=np.float64)
    ext_in = (
        np.asarray(ext_raw, dtype=np.float64)
        if ext_raw is not None
        else np.zeros_like(c1_in)
    )
    cursor_ro = _build_cursor_readout(state.cursors)

    # Hold existing trace when stopped and not manually forced
    forced = bool(state.trigger.forced)
    if forced:
        state.trigger.forced = False

    if not state.running and not forced and prev_frame is not None:
        if not state.trigger.single_armed:
            return DSOFrame(
                times_rel_s=prev_frame.times_rel_s,
                ch1_v=prev_frame.ch1_v,
                ch2_v=prev_frame.ch2_v,
                math_v=prev_frame.math_v,
                fft_freqs_hz=prev_frame.fft_freqs_hz,
                fft_mag_db=prev_frame.fft_mag_db,
                fft_peak_hz=prev_frame.fft_peak_hz,
                triggered=False,
                trigger_status="Stop",
                trigger_time_s=prev_frame.trigger_time_s,
                trig_freq_hz=prev_frame.trig_freq_hz,
                captured_single=False,
                ch1_meas=prev_frame.ch1_meas,
                ch2_meas=prev_frame.ch2_meas,
                cursor_readout=cursor_ro,
                persistence_traces=prev_frame.persistence_traces,
                t_left_s=prev_frame.t_left_s,
                t_right_s=prev_frame.t_right_s,
                span_s=prev_frame.span_s,
                x_norm=prev_frame.x_norm,
                roll_mode=prev_frame.roll_mode,
            )

    # Guard against short/empty live buffers overwriting an existing valid waveform during rapid toggles
    if prev_frame is not None and len(prev_frame.ch1_v) >= 16 and len(t_arr) < 16:
        return DSOFrame(
            times_rel_s=prev_frame.times_rel_s,
            ch1_v=prev_frame.ch1_v,
            ch2_v=prev_frame.ch2_v,
            math_v=prev_frame.math_v,
            fft_freqs_hz=prev_frame.fft_freqs_hz,
            fft_mag_db=prev_frame.fft_mag_db,
            fft_peak_hz=prev_frame.fft_peak_hz,
            triggered=prev_frame.triggered,
            trigger_status="Stop" if not state.running else prev_frame.trigger_status,
            trigger_time_s=prev_frame.trigger_time_s,
            trig_freq_hz=prev_frame.trig_freq_hz,
            captured_single=False,
            ch1_meas=prev_frame.ch1_meas,
            ch2_meas=prev_frame.ch2_meas,
            cursor_readout=cursor_ro,
            persistence_traces=prev_frame.persistence_traces,
            t_left_s=prev_frame.t_left_s,
            t_right_s=prev_frame.t_right_s,
            span_s=prev_frame.span_s,
            x_norm=prev_frame.x_norm,
            roll_mode=prev_frame.roll_mode,
        )

    contig = _latest_contiguous_slice(t_arr)
    t_arr = t_arr[contig]
    c1_in = np.clip(np.nan_to_num(c1_in[contig], nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    c2_in = np.clip(np.nan_to_num(c2_in[contig], nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    ext_in = np.clip(np.nan_to_num(ext_in[contig], nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)

    ch1_cond = _condition_channel(t_arr, c1_in, state.ch1)
    ch2_cond = _condition_channel(t_arr, c2_in, state.ch2)

    trig_src_name = state.trigger.source.upper()
    if trig_src_name == "CH2":
        trig_base = ch2_cond
    elif trig_src_name == "EXT":
        trig_base = ext_in.copy()
    else:
        trig_base = ch1_cond

    if state.trigger.coupling.upper() == "AC":
        trig_sig = trig_base - _estimate_dc_offset(t_arr, trig_base)
    else:
        trig_sig = trig_base

    crossings = _find_edge_crossings(
        t_arr,
        trig_sig,
        state.trigger.level_v,
        state.trigger.slope,
        state.trigger.holdoff_s,
    )

    trig_freq_hz: float | None = None
    if len(crossings) >= 2:
        diffs = np.diff(crossings)
        pos_diffs = diffs[diffs > 1e-15]
        if len(pos_diffs) > 0:
            med_p = float(np.median(pos_diffs))
            valid_p = pos_diffs[pos_diffs <= 2.5 * med_p]
            mean_p = float(np.mean(valid_p)) if len(valid_p) > 0 else med_p
            if mean_p > 1e-15 and (1.0 / mean_p) >= 1.0:
                trig_freq_hz = 1.0 / mean_p

    eff_sec_div = (
        state.horizontal.zoom_sec_per_div
        if state.horizontal.zoom_enabled
        else state.horizontal.sec_per_div
    )
    span_s = max(float(eff_sec_div) * float(DSO_HORIZONTAL_DIVS), 1e-12)
    half_span = 0.5 * span_s
    t0 = float(t_arr[0]) if len(t_arr) > 0 else 0.0
    t_end = float(t_arr[-1]) if len(t_arr) > 0 else 0.0
    t_mid = 0.5 * (t0 + t_end)
    pos_s = float(state.horizontal.position_s)

    trig_mode = state.trigger.mode.upper()
    has_crossing = len(crossings) > 0
    captured_single = False
    single_anchor_t: float | None = None

    if forced:
        triggered = True
        trigger_status = "Trig'd"
    elif trig_mode == "SINGLE":
        if not state.trigger.single_armed:
            if prev_frame is not None:
                return DSOFrame(
                    times_rel_s=prev_frame.times_rel_s,
                    ch1_v=prev_frame.ch1_v,
                    ch2_v=prev_frame.ch2_v,
                    math_v=prev_frame.math_v,
                    fft_freqs_hz=prev_frame.fft_freqs_hz,
                    fft_mag_db=prev_frame.fft_mag_db,
                    fft_peak_hz=prev_frame.fft_peak_hz,
                    triggered=False,
                    trigger_status="Stop",
                    trigger_time_s=prev_frame.trigger_time_s,
                    trig_freq_hz=prev_frame.trig_freq_hz,
                    captured_single=False,
                    ch1_meas=prev_frame.ch1_meas,
                    ch2_meas=prev_frame.ch2_meas,
                    cursor_readout=cursor_ro,
                    persistence_traces=prev_frame.persistence_traces,
                    t_left_s=prev_frame.t_left_s,
                    t_right_s=prev_frame.t_right_s,
                    span_s=prev_frame.span_s,
                    x_norm=prev_frame.x_norm,
                    roll_mode=prev_frame.roll_mode,
                )
            triggered = False
            trigger_status = "Stop"
        elif has_crossing:
            single_anchor_t = float(crossings[0])
            cand_right = single_anchor_t - pos_s + half_span
            triggered = True
            trigger_status = "Trig'd"
            if (not live_streaming) or (t_end >= cand_right - 1e-9):
                captured_single = True
                state.trigger.single_armed = False
                state.running = False
        else:
            triggered = False
            trigger_status = "Wait"
    elif trig_mode == "NORMAL":
        if has_crossing:
            triggered = True
            trigger_status = "Trig'd"
        else:
            if prev_frame is not None:
                return DSOFrame(
                    times_rel_s=prev_frame.times_rel_s,
                    ch1_v=prev_frame.ch1_v,
                    ch2_v=prev_frame.ch2_v,
                    math_v=prev_frame.math_v,
                    fft_freqs_hz=prev_frame.fft_freqs_hz,
                    fft_mag_db=prev_frame.fft_mag_db,
                    fft_peak_hz=prev_frame.fft_peak_hz,
                    triggered=prev_frame.triggered,
                    trigger_status="Trig'd" if prev_frame.triggered else "Wait",
                    trigger_time_s=prev_frame.trigger_time_s,
                    trig_freq_hz=prev_frame.trig_freq_hz if prev_frame.triggered else trig_freq_hz,
                    captured_single=False,
                    ch1_meas=prev_frame.ch1_meas,
                    ch2_meas=prev_frame.ch2_meas,
                    cursor_readout=cursor_ro,
                    persistence_traces=prev_frame.persistence_traces,
                    t_left_s=prev_frame.t_left_s,
                    t_right_s=prev_frame.t_right_s,
                    span_s=prev_frame.span_s,
                    x_norm=prev_frame.x_norm,
                    roll_mode=prev_frame.roll_mode,
                )
            triggered = False
            trigger_status = "Wait"
    else:  # AUTO
        if has_crossing:
            triggered = True
            trigger_status = "Trig'd"
        else:
            triggered = False
            trigger_status = "Auto"

    # Roll mode applies ONLY in AUTO mode (slow >= 100 ms/div or free-run untriggered AUTO)
    is_roll_mode = (trig_mode == "AUTO") and (
        (eff_sec_div >= 0.1 and not forced) or (not has_crossing and not forced)
    )

    if is_roll_mode:
        trigger_time_s: float | None = float(crossings[-1]) if has_crossing else None
        t_right = t_end
        t_left = t_right - span_s
        t_ref = 0.5 * (t_left + t_right)
    else:
        if single_anchor_t is not None:
            trigger_time_s = single_anchor_t
        elif has_crossing:
            latest_ok: float | None = None
            for cand_t in reversed(crossings):
                c_val = float(cand_t)
                c_ref = c_val - pos_s
                if (c_ref - half_span) >= (t0 - 1e-12) and (c_ref + half_span) <= (t_end + 1e-12):
                    latest_ok = c_val
                    break
            if latest_ok is None:
                for cand_t in reversed(crossings):
                    c_val = float(cand_t)
                    if (c_val - pos_s + half_span) <= (t_end + 1e-12):
                        latest_ok = c_val
                        break
            if latest_ok is None:
                latest_ok = float(crossings[-1])
            trigger_time_s = latest_ok
        elif forced:
            trigger_time_s = (
                max(t0 + half_span, t_end - half_span)
                if (t_end - t0) >= span_s
                else t_mid
            )
        else:
            trigger_time_s = None

        if (
            trig_mode == "NORMAL"
            and trigger_time_s is not None
            and prev_frame is not None
            and prev_frame.triggered
            and prev_frame.trigger_time_s is not None
            and trigger_time_s <= prev_frame.trigger_time_s + 1e-9
            and len(prev_frame.x_norm) > 0
            and float(prev_frame.x_norm[-1]) >= 1.0 - 1e-3
            and t0 > prev_frame.t_left_s + 1e-9
        ):
            return DSOFrame(
                times_rel_s=prev_frame.times_rel_s,
                ch1_v=prev_frame.ch1_v,
                ch2_v=prev_frame.ch2_v,
                math_v=prev_frame.math_v,
                fft_freqs_hz=prev_frame.fft_freqs_hz,
                fft_mag_db=prev_frame.fft_mag_db,
                fft_peak_hz=prev_frame.fft_peak_hz,
                triggered=True,
                trigger_status="Trig'd",
                trigger_time_s=prev_frame.trigger_time_s,
                trig_freq_hz=prev_frame.trig_freq_hz,
                captured_single=False,
                ch1_meas=prev_frame.ch1_meas,
                ch2_meas=prev_frame.ch2_meas,
                cursor_readout=cursor_ro,
                persistence_traces=prev_frame.persistence_traces,
                t_left_s=prev_frame.t_left_s,
                t_right_s=prev_frame.t_right_s,
                span_s=prev_frame.span_s,
                x_norm=prev_frame.x_norm,
                roll_mode=prev_frame.roll_mode,
            )

        if trigger_time_s is not None:
            t_ref = trigger_time_s - pos_s
            t_left = t_ref - half_span
            t_right = t_left + span_s
        else:
            # Waiting for NORMAL/SINGLE trigger -> show trailing edge without stretching
            t_right = t_end
            t_left = t_right - span_s
            t_ref = 0.5 * (t_left + t_right)

    if len(t_arr) > 0:
        mask_w = (t_arr >= (t_left - 1e-12)) & (t_arr <= (t_right + 1e-12))
        if int(np.sum(mask_w)) >= 1:
            t_win = t_arr[mask_w]
            ch1_win = ch1_cond[mask_w]
            ch2_win = ch2_cond[mask_w]
        else:
            t_win = t_arr
            ch1_win = ch1_cond
            ch2_win = ch2_cond
    else:
        t_win = t_arr
        ch1_win = ch1_cond
        ch2_win = ch2_cond

    # Exponential / cumulative running average in AVERAGE acquisition mode
    if (
        state.acquire.acq_mode.upper() == "AVERAGE"
        and prev_frame is not None
        and len(prev_frame.ch1_v) == len(ch1_win)
        and len(ch1_win) > 0
    ):
        alpha = 1.0 / max(1, int(state.acquire.avg_count))
        ch1_win = (1.0 - alpha) * prev_frame.ch1_v + alpha * ch1_win
        ch2_win = (1.0 - alpha) * prev_frame.ch2_v + alpha * ch2_win

    times_rel = t_win - t_ref
    x_norm = (
        np.clip((t_win - t_left) / span_s, 0.0, 1.0)
        if len(t_win) > 0
        else np.empty(0, dtype=np.float64)
    )

    # Math trace
    op = state.math.operation.upper()
    if op == "CH1+CH2":
        math_v = ch1_win + ch2_win
    elif op == "CH1*CH2":
        math_v = ch1_win * ch2_win
    elif op == "CH1/CH2":
        denom = np.where(np.abs(ch2_win) > 1e-6, ch2_win, np.where(ch2_win >= 0.0, 1e-6, -1e-6))
        math_v = ch1_win / denom
    else:
        math_v = ch1_win - ch2_win

    if state.math.enabled:
        fft_sig = ch2_cond if state.math.fft_source.upper() == "CH2" else ch1_cond
        if len(t_arr) > 8192:
            fft_t = t_arr[-8192:]
            fft_sig = fft_sig[-8192:]
        else:
            fft_t = t_arr
        fft_freqs, fft_db, fft_peak = _compute_fft_trace(
            fft_t, fft_sig, window_name=state.math.fft_window
        )
    else:
        empty_fft = np.empty(0, dtype=np.float64)
        fft_freqs, fft_db, fft_peak = empty_fft, empty_fft, None

    ch1_meas = _measure_dso_channel(t_win, ch1_win)
    ch2_meas = _measure_dso_channel(t_win, ch2_win)
    if ch1_meas.freq_hz is None and len(t_arr) >= 8:
        full1 = _measure_dso_channel(t_arr, ch1_cond)
        if full1.freq_hz is not None:
            ch1_meas = DSOChannelMeasurements(
                v_pp=ch1_meas.v_pp,
                v_max=ch1_meas.v_max,
                v_min=ch1_meas.v_min,
                v_avg=ch1_meas.v_avg,
                v_rms=ch1_meas.v_rms,
                freq_hz=full1.freq_hz,
                period_s=full1.period_s,
                duty_cycle_pct=ch1_meas.duty_cycle_pct,
            )
    if ch2_meas.freq_hz is None and len(t_arr) >= 8:
        full2 = _measure_dso_channel(t_arr, ch2_cond)
        if full2.freq_hz is not None:
            ch2_meas = DSOChannelMeasurements(
                v_pp=ch2_meas.v_pp,
                v_max=ch2_meas.v_max,
                v_min=ch2_meas.v_min,
                v_avg=ch2_meas.v_avg,
                v_rms=ch2_meas.v_rms,
                freq_hz=full2.freq_hz,
                period_s=full2.period_s,
                duty_cycle_pct=ch2_meas.duty_cycle_pct,
            )

    win_crossings = (
        crossings[(crossings >= t_left - 1e-12) & (crossings <= t_right + 1e-12)]
        if len(crossings) >= 2
        else np.empty(0, dtype=np.float64)
    )
    if len(win_crossings) >= 2:
        w_diffs = np.diff(win_crossings)
        w_pos = w_diffs[w_diffs > 1e-15]
        if len(w_pos) > 0:
            mean_wp = float(np.mean(w_pos))
            if mean_wp > 1e-15 and (1.0 / mean_wp) >= 1.0:
                trig_freq_hz = 1.0 / mean_wp

    if trig_src_name == "CH1" and ch1_meas.freq_hz is not None and (trig_freq_hz is None or len(win_crossings) < 3):
        trig_freq_hz = ch1_meas.freq_hz
    elif trig_src_name == "CH2" and ch2_meas.freq_hz is not None and (trig_freq_hz is None or len(win_crossings) < 3):
        trig_freq_hz = ch2_meas.freq_hz
    elif trig_freq_hz is None:
        trig_freq_hz = (
            ch2_meas.freq_hz if trig_src_name == "CH2" else ch1_meas.freq_hz
        )

    pers_traces: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    if state.display.persistence:
        hist = list(prev_frame.persistence_traces) if prev_frame is not None else []
        hist.append((ch1_win.copy(), ch2_win.copy()))
        pers_traces = tuple(hist[-5:])

    return DSOFrame(
        times_rel_s=times_rel,
        ch1_v=ch1_win,
        ch2_v=ch2_win,
        math_v=math_v,
        fft_freqs_hz=fft_freqs,
        fft_mag_db=fft_db,
        fft_peak_hz=fft_peak,
        triggered=triggered,
        trigger_status=trigger_status,
        trigger_time_s=trigger_time_s,
        trig_freq_hz=trig_freq_hz,
        captured_single=captured_single,
        ch1_meas=ch1_meas,
        ch2_meas=ch2_meas,
        cursor_readout=cursor_ro,
        persistence_traces=pers_traces,
        t_left_s=t_left,
        t_right_s=t_right,
        span_s=span_s,
        x_norm=x_norm,
        roll_mode=is_roll_mode,
    )
