"""Qt-free automated oscilloscope waveform measurements and phase lag calculation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from circuitsim.scope.trigger import find_rising_midpoint_crossings


@dataclass(frozen=True)
class ChannelMeasurements:
    """Automated IEEE-style single-channel waveform measurements."""

    vpp: float
    vmax: float
    vmin: float
    vavg: float
    vrms: float
    frequency_hz: float | None = None
    period_s: float | None = None
    duty_cycle_pct: float | None = None
    rise_time_s: float | None = None
    fall_time_s: float | None = None
    reason: str | None = None


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _integrate_window_stats(
    t: np.ndarray, v: np.ndarray, t0: float, t1: float
) -> tuple[float, float, float]:
    """Return ``(vavg, vrms, duty_cycle_pct_above_mid)`` over continuous ``[t0, t1]``."""
    v0 = float(np.interp(t0, t, v))
    v1 = float(np.interp(t1, t, v))
    i0 = int(np.searchsorted(t, t0, side="right"))
    i1 = int(np.searchsorted(t, t1, side="left"))
    t_seg = np.concatenate(([t0], t[i0:i1], [t1]))
    v_seg = np.concatenate(([v0], v[i0:i1], [v1]))
    dur = float(t1 - t0)
    if dur <= 1e-18:
        val = float(np.mean(v_seg))
        return val, abs(val), 0.0

    vavg = _trapezoid(v_seg, t_seg) / dur
    vrms = math.sqrt(max(0.0, _trapezoid(v_seg * v_seg, t_seg) / dur))

    mid = 0.5 * (float(np.max(v_seg)) + float(np.min(v_seg)))
    # Piecewise linear exact time above `mid`
    dt_seg = t_seg[1:] - t_seg[:-1]
    va = v_seg[:-1] - mid
    vb = v_seg[1:] - mid
    both_hi = (va >= 0.0) & (vb >= 0.0)
    cross_up = (va < 0.0) & (vb >= 0.0)
    cross_dn = (va >= 0.0) & (vb < 0.0)
    high_time = float(np.sum(dt_seg[both_hi]))
    if np.any(cross_up):
        frac_dn = (-va[cross_up]) / (vb[cross_up] - va[cross_up])
        high_time += float(np.sum(dt_seg[cross_up] * (1.0 - frac_dn)))
    if np.any(cross_dn):
        frac_hi = va[cross_dn] / (va[cross_dn] - vb[cross_dn])
        high_time += float(np.sum(dt_seg[cross_dn] * frac_hi))
    duty = max(0.0, min(100.0, 100.0 * high_time / dur))
    return vavg, vrms, duty


def _interpolated_threshold_times(
    t: np.ndarray, v: np.ndarray, thresh: float, *, rising: bool
) -> tuple[np.ndarray, np.ndarray]:
    if rising:
        idx = np.flatnonzero((v[:-1] < thresh) & (v[1:] >= thresh))
    else:
        idx = np.flatnonzero((v[:-1] > thresh) & (v[1:] <= thresh))
    if len(idx) == 0:
        return idx, np.empty(0, dtype=np.float64)
    dv = v[idx + 1] - v[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(dv) > 1e-18, (thresh - v[idx]) / dv, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    times = t[idx] + frac * (t[idx + 1] - t[idx])
    return idx, times


def _measure_edge_time(
    t: np.ndarray, v: np.ndarray, v_low: float, v_high: float, *, rising: bool
) -> float | None:
    amp = v_high - v_low
    if amp <= 1e-12 or len(t) < 2:
        return None
    v10 = v_low + 0.1 * amp
    v90 = v_low + 0.9 * amp

    if rising:
        idx_start, t_start = _interpolated_threshold_times(t, v, v10, rising=True)
        idx_end, t_end = _interpolated_threshold_times(t, v, v90, rising=True)
        reset_mask = v < v10
    else:
        idx_start, t_start = _interpolated_threshold_times(t, v, v90, rising=False)
        idx_end, t_end = _interpolated_threshold_times(t, v, v10, rising=False)
        reset_mask = v > v90

    if len(t_start) == 0 or len(t_end) == 0:
        return None

    durations: list[float] = []
    ei = 0
    for si in range(len(idx_start)):
        i_s = int(idx_start[si])
        ts = float(t_start[si])
        while ei < len(idx_end) and (int(idx_end[ei]) < i_s or float(t_end[ei]) <= ts):
            ei += 1
        if ei >= len(idx_end):
            break
        i_e = int(idx_end[ei])
        te = float(t_end[ei])
        # Verify no reset occurred between `i_s + 1` and `i_e`
        if i_e > i_s + 1 and np.any(reset_mask[i_s + 1 : i_e + 1]):
            continue
        if te > ts:
            durations.append(te - ts)

    if not durations:
        return None
    return float(np.median(durations))


def measure_channel(t: np.ndarray, v: np.ndarray) -> ChannelMeasurements:
    """Compute automated measurements over ``(t, v)``."""
    t_arr = np.asarray(t, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if len(t_arr) == 0 or len(t_arr) != len(v_arr):
        return ChannelMeasurements(
            vpp=0.0,
            vmax=0.0,
            vmin=0.0,
            vavg=0.0,
            vrms=0.0,
            reason="empty signal",
        )

    vmax = float(np.max(v_arr))
    vmin = float(np.min(v_arr))
    vpp = vmax - vmin

    if vpp <= 1e-12 or len(t_arr) < 2:
        v_const = float(np.mean(v_arr))
        return ChannelMeasurements(
            vpp=vpp,
            vmax=vmax,
            vmin=vmin,
            vavg=v_const,
            vrms=abs(v_const),
            reason="no complete period",
        )

    rise_times_mid, _ = find_rising_midpoint_crossings(t_arr, v_arr)
    rise_time_s = _measure_edge_time(t_arr, v_arr, vmin, vmax, rising=True)
    fall_time_s = _measure_edge_time(t_arr, v_arr, vmin, vmax, rising=False)

    if len(rise_times_mid) >= 2:
        period_s = float(rise_times_mid[-1] - rise_times_mid[0]) / (len(rise_times_mid) - 1)
        freq_hz = 1.0 / period_s if period_s > 0.0 else None
        vavg, vrms, duty_pct = _integrate_window_stats(
            t_arr, v_arr, float(rise_times_mid[0]), float(rise_times_mid[-1])
        )
        return ChannelMeasurements(
            vpp=vpp,
            vmax=vmax,
            vmin=vmin,
            vavg=vavg,
            vrms=vrms,
            frequency_hz=freq_hz,
            period_s=period_s,
            duty_cycle_pct=duty_pct,
            rise_time_s=rise_time_s,
            fall_time_s=fall_time_s,
            reason=None,
        )

    # Fewer than 1 complete period in record
    vavg, vrms, _ = _integrate_window_stats(t_arr, v_arr, float(t_arr[0]), float(t_arr[-1]))
    return ChannelMeasurements(
        vpp=vpp,
        vmax=vmax,
        vmin=vmin,
        vavg=vavg,
        vrms=vrms,
        frequency_hz=None,
        period_s=None,
        duty_cycle_pct=None,
        rise_time_s=rise_time_s,
        fall_time_s=fall_time_s,
        reason="no complete period",
    )


def measure_phase_lag_deg(
    t: np.ndarray, v_ref: np.ndarray, v_target: np.ndarray
) -> tuple[float | None, str | None]:
    """Measure phase lag of ``v_target`` behind ``v_ref`` in degrees within ``(-180, 180]``.

    Positive return values indicate ``v_target`` lags ``v_ref`` in time
    (e.g., an RC low-pass output lags its input by ``+45 deg`` at ``f = f_c``).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    ref_arr = np.asarray(v_ref, dtype=np.float64)
    tgt_arr = np.asarray(v_target, dtype=np.float64)
    if len(t_arr) < 3 or len(ref_arr) != len(t_arr) or len(tgt_arr) != len(t_arr):
        return None, "empty signal"

    r_ref, vpp_ref = find_rising_midpoint_crossings(t_arr, ref_arr)
    r_tgt, vpp_tgt = find_rising_midpoint_crossings(t_arr, tgt_arr)
    if vpp_ref <= 1e-12 or vpp_tgt <= 1e-12:
        return None, "insufficient amplitude"
    if len(r_ref) < 2 or len(r_tgt) < 1:
        return None, "no complete period"

    period = float(r_ref[-1] - r_ref[0]) / (len(r_ref) - 1)
    if period <= 1e-18:
        return None, "no complete period"

    # Match each reference crossing to nearest target crossing within +-0.5 period
    lags: list[float] = []
    for tr in r_ref:
        diffs = r_tgt - tr
        nearest_idx = int(np.argmin(np.abs(diffs)))
        dt_lag = float(diffs[nearest_idx])
        if abs(dt_lag) <= 0.6 * period:
            deg = ((dt_lag / period) * 360.0 + 180.0) % 360.0 - 180.0
            lags.append(deg)

    if not lags:
        dt_lag = float(r_tgt[0] - r_ref[0])
        deg = ((dt_lag / period) * 360.0 + 180.0) % 360.0 - 180.0
        return float(deg), None

    return float(np.median(lags)), None
