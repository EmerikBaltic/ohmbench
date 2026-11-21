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


def _find_edge_crossings(
    times: np.ndarray,
    trig_sig: np.ndarray,
    level_v: float,
    slope: str,
    holdoff_s: float = 0.0,
) -> np.ndarray:
    """Return linearly interpolated crossing timestamps for the requested slope."""
    if len(times) < 2 or len(times) != len(trig_sig):
        return np.empty(0, dtype=np.float64)
    a = trig_sig[:-1]
    b = trig_sig[1:]
    slope_u = slope.upper()
    rising_mask = (a < level_v) & (b >= level_v)
    falling_mask = (a > level_v) & (b <= level_v)
    if slope_u == "RISING":
        mask = rising_mask
    elif slope_u == "FALLING":
        mask = falling_mask
    else:
        mask = rising_mask | falling_mask

    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.empty(0, dtype=np.float64)
    dv = b[idx] - a[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(dv) > 1e-18, (level_v - a[idx]) / dv, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    crossings = times[idx] + frac * (times[idx + 1] - times[idx])
    if holdoff_s > 0.0 and len(crossings) > 1:
        kept = [float(crossings[0])]
        last_t = kept[0]
        for tc in crossings[1:]:
            if float(tc) - last_t >= holdoff_s:
                kept.append(float(tc))
                last_t = float(tc)
        return np.array(kept, dtype=np.float64)
    return crossings


def _measure_dso_channel(times: np.ndarray, sig: np.ndarray) -> DSOChannelMeasurements:
    if len(times) < 2 or len(sig) < 2:
        val = float(sig[0]) if len(sig) > 0 else 0.0
        return DSOChannelMeasurements(0.0, val, val, val, abs(val), None, None, None)
    m = measure_channel(times, sig)
    return DSOChannelMeasurements(
        v_pp=float(m.vpp),
        v_max=float(m.vmax),
        v_min=float(m.vmin),
        v_avg=float(m.vavg),
        v_rms=float(m.vrms),
        freq_hz=None if m.frequency_hz is None else float(m.frequency_hz),
        period_s=None if m.period_s is None else float(m.period_s),
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
        span_ok = (v_pp / cand) <= 5.5
        center_ok = abs(v_mid / cand) <= 3.8
        if span_ok and center_ok:
            chosen = cand
            break
    updated.volts_per_div = chosen
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
            # Target ~2.4 to ~3.5 full cycles across the 12 horizontal divisions
            target_sec_div = (2.5 * period_s) / float(DSO_HORIZONTAL_DIVS)
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
        total_span = float(crossings[-1] - crossings[0])
        if total_span > 1e-15:
            trig_freq_hz = (len(crossings) - 1) / total_span

    trig_mode = state.trigger.mode.upper()
    has_crossing = len(crossings) > 0
    captured_single = False

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
            triggered = True
            captured_single = True
            state.trigger.single_armed = False
            state.running = False
            trigger_status = "Trig'd"
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
                    triggered=False,
                    trigger_status="Wait",
                    trigger_time_s=prev_frame.trigger_time_s,
                    trig_freq_hz=trig_freq_hz,
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

    # Choose trigger reference & exact 12-division horizontal window [t_left, t_right]
    eff_sec_div = (
        state.horizontal.zoom_sec_per_div
        if state.horizontal.zoom_enabled
        else state.horizontal.sec_per_div
    )
    span_s = max(float(eff_sec_div) * float(DSO_HORIZONTAL_DIVS), 1e-12)
    t0 = float(t_arr[0]) if len(t_arr) > 0 else 0.0
    t_end = float(t_arr[-1]) if len(t_arr) > 0 else 0.0
    t_mid = 0.5 * (t0 + t_end)
    pos_s = float(state.horizontal.position_s)

    is_roll_mode = (eff_sec_div >= 0.1) or (
        trig_mode == "AUTO" and not has_crossing and not forced
    )

    if is_roll_mode:
        trigger_time_s: float | None = float(crossings[-1]) if has_crossing else None
        t_right = t_end
        t_left = t_right - span_s
        t_ref = 0.5 * (t_left + t_right)
    else:
        if has_crossing:
            half_span = 0.5 * span_s
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
                mid_cross_idx = int(np.argmin(np.abs(crossings - t_mid)))
                latest_ok = float(crossings[mid_cross_idx])
            trigger_time_s = latest_ok
        elif forced:
            trigger_time_s = (
                max(t0 + 0.5 * span_s, t_end - 0.5 * span_s)
                if (t_end - t0) >= span_s
                else t_mid
            )
        else:
            trigger_time_s = None

        base_t = trigger_time_s if trigger_time_s is not None else t_mid
        t_ref = base_t - pos_s
        t_left = t_ref - 0.5 * span_s
        t_right = t_left + span_s

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

    fft_sig = ch2_cond if state.math.fft_source.upper() == "CH2" else ch1_cond
    fft_freqs, fft_db, fft_peak = _compute_fft_trace(
        t_arr, fft_sig, window_name=state.math.fft_window
    )

    ch1_meas = _measure_dso_channel(t_win, ch1_win)
    ch2_meas = _measure_dso_channel(t_win, ch2_win)
    if trig_freq_hz is None:
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
