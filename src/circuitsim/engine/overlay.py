"""Qt-free overlay value-to-color, dot velocity, and power probe helpers."""

from __future__ import annotations

import math

_NEUTRAL_GRAY: tuple[int, int, int] = (128, 128, 128)
_POSITIVE_GREEN: tuple[int, int, int] = (32, 224, 96)
_NEGATIVE_RED: tuple[int, int, int] = (236, 64, 64)


def _lerp_channel(start: int, target: int, t: float) -> int:
    return max(0, min(255, int(round(start + (target - start) * t))))


def map_voltage_to_rgb(voltage: float, v_ref: float = 12.0) -> tuple[int, int, int]:
    """Map node voltage (V) to RGB: 0 V = neutral gray, +V -> green, -V -> red."""
    if not math.isfinite(voltage) or not math.isfinite(v_ref):
        return _NEUTRAL_GRAY
    ref = max(abs(float(v_ref)), 1e-12)
    v = float(voltage)
    if abs(v) <= 1e-12:
        return _NEUTRAL_GRAY
    t = min(1.0, abs(v) / ref)
    target = _POSITIVE_GREEN if v > 0.0 else _NEGATIVE_RED
    return (
        _lerp_channel(_NEUTRAL_GRAY[0], target[0], t),
        _lerp_channel(_NEUTRAL_GRAY[1], target[1], t),
        _lerp_channel(_NEUTRAL_GRAY[2], target[2], t),
    )


def compute_current_dot_velocity(
    current_amps: float, i_scale: float = 0.05
) -> float:
    """Return signed normalized dot speed in [-1.0, 1.0] (0.0 when |I| < 1e-9 A)."""
    if not math.isfinite(current_amps) or not math.isfinite(i_scale):
        return 0.0
    i_val = float(current_amps)
    if abs(i_val) < 1e-9:
        return 0.0
    scale = max(abs(float(i_scale)), 1e-12)
    return max(-1.0, min(1.0, i_val / scale))


def format_watts_si(power_w: float, *, sig_figs: int = 4) -> str:
    """Format power in W / mW / uW / kW cleanly without Qt dependencies."""
    if not math.isfinite(power_w):
        return f"{power_w} W"
    if abs(power_w) < 1e-15:
        return "0 W"
    for scale, prefix in (
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "u"),
        (1e-9, "n"),
        (1e-12, "p"),
    ):
        if abs(power_w) >= scale * 0.999999:
            scaled = power_w / scale
            return f"{scaled:.{sig_figs}g} {prefix}W"
    scaled = power_w / 1e-12
    return f"{scaled:.{sig_figs}g} pW"


def format_power_probe_text(
    comp_name: str,
    power_w: float,
    *,
    voltage_v: float | None = None,
    current_a: float | None = None,
    is_running: bool = True,
    is_source: bool = False,
) -> str:
    """Format Alt-hover instantaneous power probe reading in W or mW."""
    if not is_running:
        return f"{comp_name}: P = -- W (start simulation)"
    p_val = float(power_w)
    if is_source and p_val < -1e-15:
        p_str = format_watts_si(abs(p_val))
        header = f"{comp_name}: Supplied P = {p_str} ({format_watts_si(p_val)})"
    else:
        p_str = format_watts_si(p_val)
        header = f"{comp_name}: P = {p_str}"
    if voltage_v is not None and current_a is not None:
        return f"{header}\nV = {voltage_v:.4g} V | I = {current_a * 1000.0:.4g} mA"
    return header
