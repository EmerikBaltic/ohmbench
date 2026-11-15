"""Qt-free 4-channel oscilloscope vertical scaling, probe attenuation, and coupling.

Probe convention:
    ``SampleBlock`` columns store circuit-referred node voltages ``V_node`` (in
    volts at the circuit node). In a physical oscilloscope, a 10x passive probe
    attenuates the BNC input by 10x (``V_bnc = V_node / 10``) and the channel
    front-end multiplies by ``probe_attenuation`` so circuit-referred voltage
    matches for both ``1x`` and ``10x`` settings (``(V_node / P) * P == V_node``)
    when the channel setting matches the probe. Passing
    ``raw_is_attenuated_tip=True`` scales a raw BNC voltage up by
    ``config.probe_attenuation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from circuitsim.scope.trigger import ac_couple_signal

CHANNEL_NAMES: tuple[str, ...] = ("CH1", "CH2", "CH3", "CH4")
DEFAULT_VERTICAL_DIVISIONS = 8

CHANNEL_COLOR_HEX: dict[str, str] = {
    "CH1": "#FFD600",
    "CH2": "#00E5FF",
    "CH3": "#FF4081",
    "CH4": "#69F0AE",
}
CHANNEL_COLORS: dict[str, str] = CHANNEL_COLOR_HEX

VOLTS_PER_DIV_STEPS: tuple[float, ...] = (
    1e-3,
    2e-3,
    5e-3,
    10e-3,
    20e-3,
    50e-3,
    100e-3,
    200e-3,
    500e-3,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)
MIN_VOLTS_PER_DIV = VOLTS_PER_DIV_STEPS[0]
MAX_VOLTS_PER_DIV = VOLTS_PER_DIV_STEPS[-1]
PROBE_ATTENUATIONS: tuple[float, ...] = (1.0, 10.0)


class ChannelCoupling(str, Enum):
    """Vertical input coupling modes."""

    DC = "dc"
    AC = "ac"
    GND = "gnd"


def snap_volts_per_div(volts_per_div: float) -> float:
    """Return the nearest standard 1-2-5 vertical scale within ``[1 mV/div, 1 kV/div]``."""
    val = float(volts_per_div)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"volts_per_div must be positive and finite, got {volts_per_div!r}")
    clamped = max(MIN_VOLTS_PER_DIV, min(MAX_VOLTS_PER_DIV, val))
    return min(
        VOLTS_PER_DIV_STEPS,
        key=lambda step: abs(math.log10(step) - math.log10(clamped)),
    )


@dataclass(frozen=True)
class ChannelConfig:
    """Immutable configuration for one oscilloscope channel (CH1..CH4)."""

    name: str
    enabled: bool = False
    node_name: str | None = None
    volts_per_div: float = 1.0
    offset_div: float = 0.0
    coupling: ChannelCoupling = ChannelCoupling.DC
    probe_attenuation: float = 1.0
    invert: bool = False
    color_hex: str | None = None

    def __post_init__(self) -> None:
        if self.name not in CHANNEL_NAMES:
            raise ValueError(f"Channel name must be one of {CHANNEL_NAMES}, got {self.name!r}")
        if (
            not math.isfinite(self.volts_per_div)
            or self.volts_per_div < MIN_VOLTS_PER_DIV
            or self.volts_per_div > MAX_VOLTS_PER_DIV
        ):
            raise ValueError(
                f"volts_per_div must be in [{MIN_VOLTS_PER_DIV}, {MAX_VOLTS_PER_DIV}], got {self.volts_per_div!r}"
            )
        if not math.isfinite(self.offset_div):
            raise ValueError(f"offset_div must be finite, got {self.offset_div!r}")
        if not isinstance(self.coupling, ChannelCoupling):
            object.__setattr__(self, "coupling", ChannelCoupling(self.coupling))
        probe = float(self.probe_attenuation)
        if probe not in PROBE_ATTENUATIONS:
            raise ValueError(
                f"probe_attenuation must be in {PROBE_ATTENUATIONS}, got {self.probe_attenuation!r}"
            )
        object.__setattr__(self, "probe_attenuation", probe)
        if self.color_hex is None:
            object.__setattr__(self, "color_hex", CHANNEL_COLOR_HEX[self.name])


def default_channels() -> dict[str, ChannelConfig]:
    """Return default 4-channel mapping with CH1 enabled and CH2..CH4 disabled."""
    return {
        name: ChannelConfig(name=name, enabled=(name == "CH1"))
        for name in CHANNEL_NAMES
    }


def process_channel_signal(
    raw_v: np.ndarray,
    t: np.ndarray,
    config: ChannelConfig,
    *,
    raw_is_attenuated_tip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply probe scaling, DC/AC/GND coupling, inversion, and vertical scaling.

    Returns ``(volts, display_div_y)`` where both 1D float64 arrays are read-only.
    """
    arr = np.asarray(raw_v, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    if arr.shape != t_arr.shape or arr.ndim != 1:
        raise ValueError("raw_v and t must be 1D arrays of matching shape")

    if len(arr) == 0:
        empty = np.empty(0, dtype=np.float64)
        empty2 = np.empty(0, dtype=np.float64)
        empty.setflags(write=False)
        empty2.setflags(write=False)
        return empty, empty2

    if config.coupling == ChannelCoupling.GND:
        volts = np.zeros_like(arr)
    else:
        scaled = arr * config.probe_attenuation if raw_is_attenuated_tip else arr.copy()
        if config.coupling == ChannelCoupling.AC:
            volts = ac_couple_signal(t_arr, scaled)
        else:
            volts = scaled
        if config.invert:
            volts = -volts

    display_div_y = (volts / config.volts_per_div) + config.offset_div
    volts.setflags(write=False)
    display_div_y.setflags(write=False)
    return volts, display_div_y
