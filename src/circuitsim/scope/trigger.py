"""Qt-free vectorized oscilloscope trigger engine and AC/integer-period DC removal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

VALID_TRIGGER_SOURCES: tuple[str, ...] = ("CH1", "CH2", "CH3", "CH4")


class TriggerEdge(str, Enum):
    """Trigger slope / edge selection."""

    RISING = "rising"
    FALLING = "falling"
    EITHER = "either"


class TriggerMode(str, Enum):
    """Trigger acquisition mode."""

    AUTO = "auto"
    NORMAL = "normal"
    SINGLE = "single"


class TriggerCoupling(str, Enum):
    """Trigger path coupling."""

    DC = "dc"
    AC = "ac"


@dataclass(frozen=True)
class TriggerConfig:
    """Immutable oscilloscope trigger configuration."""

    source: str = "CH1"
    level_v: float = 0.0
    edge: TriggerEdge = TriggerEdge.RISING
    mode: TriggerMode = TriggerMode.AUTO
    coupling: TriggerCoupling = TriggerCoupling.DC
    hysteresis_div: float = 0.05
    holdoff_s: float = 0.0
    auto_timeout_s: float = 0.1
    pre_trigger_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.source not in VALID_TRIGGER_SOURCES:
            raise ValueError(
                f"trigger source must be one of {VALID_TRIGGER_SOURCES}, got {self.source!r}"
            )
        if not math.isfinite(self.level_v):
            raise ValueError(f"level_v must be finite, got {self.level_v!r}")
        if not isinstance(self.edge, TriggerEdge):
            object.__setattr__(self, "edge", TriggerEdge(self.edge))
        if not isinstance(self.mode, TriggerMode):
            object.__setattr__(self, "mode", TriggerMode(self.mode))
        if not isinstance(self.coupling, TriggerCoupling):
            object.__setattr__(self, "coupling", TriggerCoupling(self.coupling))
        if not math.isfinite(self.hysteresis_div) or self.hysteresis_div < 0.0:
            raise ValueError(f"hysteresis_div must be >= 0, got {self.hysteresis_div!r}")
        if not math.isfinite(self.holdoff_s) or self.holdoff_s < 0.0:
            raise ValueError(f"holdoff_s must be >= 0, got {self.holdoff_s!r}")
        if not math.isfinite(self.auto_timeout_s) or self.auto_timeout_s <= 0.0:
            raise ValueError(f"auto_timeout_s must be > 0, got {self.auto_timeout_s!r}")
        if (
            not math.isfinite(self.pre_trigger_fraction)
            or not 0.0 <= self.pre_trigger_fraction <= 1.0
        ):
            raise ValueError(
                f"pre_trigger_fraction must be in [0, 1], got {self.pre_trigger_fraction!r}"
            )


@dataclass(frozen=True)
class TriggerEvent:
    """Detected trigger crossing with linear sub-sample interpolated timestamp."""

    index: int
    time_s: float
    level_v: float
    edge: TriggerEdge


def find_rising_midpoint_crossings(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """Return sub-sample interpolated rising midpoint crossing times and estimated Vpp."""
    if len(t) < 3 or len(t) != len(v):
        return np.empty(0, dtype=np.float64), 0.0

    v_max = float(np.max(v))
    v_min = float(np.min(v))
    vpp = v_max - v_min
    if vpp <= 1e-12:
        return np.empty(0, dtype=np.float64), vpp

    mid = 0.5 * (v_max + v_min)
    h = 0.1 * vpp
    cross_times: list[float] = []

    arm_low = np.flatnonzero(v <= mid - h)
    raw_rise = np.flatnonzero((v[:-1] < mid) & (v[1:] >= mid))
    if len(arm_low) == 0 or len(raw_rise) == 0:
        return np.empty(0, dtype=np.float64), vpp

    next_min_idx = int(arm_low[0])
    ri = int(np.searchsorted(raw_rise, next_min_idx, side="left"))
    while ri < len(raw_rise):
        idx = int(raw_rise[ri])
        dv = float(v[idx + 1] - v[idx])
        frac = 0.0 if abs(dv) < 1e-18 else max(0.0, min(1.0, (mid - float(v[idx])) / dv))
        t_cross = float(t[idx] + frac * (t[idx + 1] - t[idx]))
        cross_times.append(t_cross)

        pos_arm = int(np.searchsorted(arm_low, idx + 1, side="left"))
        if pos_arm >= len(arm_low):
            break
        next_arm = int(arm_low[pos_arm])
        ri = int(np.searchsorted(raw_rise, next_arm, side="left"))

    return np.asarray(cross_times, dtype=np.float64), vpp


def estimate_dc_offset_integer_periods(t: np.ndarray, v: np.ndarray) -> float:
    """Estimate DC mean over the largest integer number of periods, or full-record mean."""
    if len(v) == 0:
        return 0.0
    cross_times, _vpp = find_rising_midpoint_crossings(t, v)
    if len(cross_times) < 2:
        return float(np.mean(v))

    t0 = float(cross_times[0])
    t1 = float(cross_times[-1])
    duration = t1 - t0
    if duration <= 1e-18:
        return float(np.mean(v))

    v_mid = float(np.interp(t0, t, v))
    i0 = int(np.searchsorted(t, t0, side="right"))
    i1 = int(np.searchsorted(t, t1, side="left"))
    t_seg = np.concatenate(([t0], t[i0:i1], [t1]))
    v_seg = np.concatenate(([v_mid], v[i0:i1], [v_mid]))
    area = float(np.trapezoid(v_seg, t_seg)) if hasattr(np, "trapezoid") else float(np.trapz(v_seg, t_seg))
    return area / duration


def ac_couple_signal(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove DC component using integer-period mean with full-record fallback."""
    arr = np.asarray(v, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    t_arr = np.asarray(t, dtype=np.float64)
    dc = estimate_dc_offset_integer_periods(t_arr, arr)
    return arr - dc


def _interpolate_crossing_time(t: np.ndarray, sig: np.ndarray, idx: int, level: float) -> float:
    dv = float(sig[idx + 1] - sig[idx])
    frac = 0.0 if abs(dv) < 1e-18 else (level - float(sig[idx])) / dv
    frac = max(0.0, min(1.0, frac))
    return float(t[idx] + frac * (t[idx + 1] - t[idx]))


def _find_armed_single_slope(
    t: np.ndarray,
    sig: np.ndarray,
    level: float,
    hyst_v: float,
    holdoff_s: float,
    rising: bool,
) -> list[TriggerEvent]:
    if rising:
        raw = np.flatnonzero((sig[:-1] < level) & (sig[1:] >= level))
        arm = np.flatnonzero(sig <= level - hyst_v) if hyst_v > 0.0 else None
        edge = TriggerEdge.RISING
    else:
        raw = np.flatnonzero((sig[:-1] > level) & (sig[1:] <= level))
        arm = np.flatnonzero(sig >= level + hyst_v) if hyst_v > 0.0 else None
        edge = TriggerEdge.FALLING

    if len(raw) == 0:
        return []
    if arm is not None and len(arm) == 0:
        return []

    events: list[TriggerEvent] = []
    next_min_idx = int(arm[0]) if arm is not None else 0
    ri = int(np.searchsorted(raw, next_min_idx, side="left"))
    last_trig_t = -math.inf

    while ri < len(raw):
        idx = int(raw[ri])
        t_cross = _interpolate_crossing_time(t, sig, idx, level)
        if t_cross - last_trig_t >= holdoff_s:
            events.append(TriggerEvent(index=idx, time_s=t_cross, level_v=level, edge=edge))
            last_trig_t = t_cross
        if arm is not None:
            pos_arm = int(np.searchsorted(arm, idx + 1, side="left"))
            if pos_arm >= len(arm):
                break
            ri = int(np.searchsorted(raw, int(arm[pos_arm]), side="left"))
        else:
            ri += 1

    return events


def find_trigger_crossings(
    t: np.ndarray,
    v: np.ndarray,
    config: TriggerConfig,
    *,
    volts_per_div: float = 1.0,
) -> list[TriggerEvent]:
    """Find valid trigger crossings enforcing edge, hysteresis, holdoff, and AC coupling."""
    t_arr = np.asarray(t, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if len(t_arr) < 2 or len(t_arr) != len(v_arr):
        return []

    sig = ac_couple_signal(t_arr, v_arr) if config.coupling == TriggerCoupling.AC else v_arr
    hyst_v = max(0.0, config.hysteresis_div * volts_per_div)

    if config.edge == TriggerEdge.RISING:
        return _find_armed_single_slope(
            t_arr, sig, config.level_v, hyst_v, config.holdoff_s, rising=True
        )
    if config.edge == TriggerEdge.FALLING:
        return _find_armed_single_slope(
            t_arr, sig, config.level_v, hyst_v, config.holdoff_s, rising=False
        )

    # TriggerEdge.EITHER: find armed rising & falling independently (before shared holdoff),
    # then merge in chronological order to enforce mutual holdoff across either edge.
    rise_ev = _find_armed_single_slope(
        t_arr, sig, config.level_v, hyst_v, 0.0, rising=True
    )
    fall_ev = _find_armed_single_slope(
        t_arr, sig, config.level_v, hyst_v, 0.0, rising=False
    )
    merged = sorted(rise_ev + fall_ev, key=lambda e: (e.time_s, e.index))
    if config.holdoff_s <= 0.0:
        return merged

    filtered: list[TriggerEvent] = []
    last_trig_t = -math.inf
    for ev in merged:
        if ev.time_s - last_trig_t >= config.holdoff_s:
            filtered.append(ev)
            last_trig_t = ev.time_s
    return filtered


def select_window_trigger(
    events: list[TriggerEvent],
    t_min: float,
    t_max: float,
    window_duration_s: float,
    pre_trigger_fraction: float = 0.5,
) -> TriggerEvent | None:
    """Pick the latest trigger event for display.

    Prefers an event whose full pre/post window fits in ``[t_min, t_max]``; falls
    back to an event whose post-trigger tail ``t_trig + (1-pre)*W <= t_max`` fits
    so single-shot startup steps near ``t = 0`` still trigger cleanly.
    """
    if not events or window_duration_s <= 0.0:
        return None
    pre = pre_trigger_fraction * window_duration_s
    post = (1.0 - pre_trigger_fraction) * window_duration_s
    eps = 1e-12 * max(1.0, window_duration_s)

    best_full: TriggerEvent | None = None
    best_post_only: TriggerEvent | None = None
    for ev in reversed(events):
        if ev.time_s <= t_max - post + eps:
            if best_post_only is None:
                best_post_only = ev
            if ev.time_s >= t_min + pre - eps:
                best_full = ev
                break
    return best_full if best_full is not None else best_post_only
