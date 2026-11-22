"""Placeable benchtop Function Generator component (Qt-free)."""

from __future__ import annotations

import math

from circuitsim.components.sources import ACVoltageSource, WaveformType
from circuitsim.dialogs.value_parser import (
    format_engineering_value,
    parse_engineering_value,
)


def _coerce_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    txt = str(val).strip().lower()
    if txt in ("1", "true", "on", "yes"):
        return True
    if txt in ("0", "false", "off", "no", ""):
        return False
    return bool(val)


class FunctionGenerator(ACVoltageSource):
    """Programmable benchtop Function Generator (`FG1`, terminals `out`, `gnd`).

    Subclasses `ACVoltageSource` so MNA DC, transient, and AC frequency-domain
    solvers treat its output branch (and optional 50 Ω output impedance via
    `r_internal`) uniformly.
    """

    PIN_NAMES: tuple[str, ...] = ("out", "gnd")

    def __init__(
        self,
        name: str,
        out_node: str | None = None,
        gnd_node: str | None = None,
        *,
        waveform: WaveformType | str = WaveformType.SINE,
        frequency: float | str = 100.0,
        amplitude: float | str = 5.0,
        dc_offset: float | str = 0.0,
        duty_cycle: float = 0.5,
        output_impedance: float | str = 0.0,
        output_enabled: bool = True,
        phase_deg: float | str = 0.0,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        n_out = out_node or f"{name}_out"
        n_gnd = gnd_node or f"{name}_gnd"
        parsed_freq = max(0.1, min(1e7, parse_engineering_value(frequency)))
        parsed_duty = max(0.05, min(0.95, float(duty_cycle)))
        parsed_zout = max(0.0, parse_engineering_value(output_impedance))
        wf = WaveformType(str(waveform).lower()) if not isinstance(waveform, WaveformType) else waveform
        super().__init__(
            name=name,
            node_pos=n_out,
            node_neg=n_gnd,
            waveform=wf,
            amplitude=amplitude,
            frequency=parsed_freq,
            phase_deg=phase_deg,
            dc_offset=dc_offset,
            duty_cycle=parsed_duty,
            r_internal=parsed_zout,
            label=label,
            show_value=show_value,
            color=color,
        )
        self.output_enabled: bool = bool(output_enabled)

    @property
    def component_type(self) -> str:
        return "function_generator"

    @property
    def output_impedance(self) -> float:
        """Internal source impedance in Ohms (`0.0` for Hi-Z, `50.0` for 50 Ω)."""
        return self.r_internal

    @output_impedance.setter
    def output_impedance(self, value: float | str) -> None:
        parsed = max(0.0, parse_engineering_value(value))
        self.r_internal = parsed

    @property
    def display_value(self) -> str:
        if not self.output_enabled:
            return "OFF"
        amp_s = format_engineering_value(self.amplitude, "V", sig_figs=3)
        freq_s = format_engineering_value(self.frequency, "Hz", sig_figs=3)
        tag_map = {
            WaveformType.SINE: "SIN",
            WaveformType.SQUARE: "SQ",
            WaveformType.TRIANGLE: "TRI",
            WaveformType.SAWTOOTH: "SAW",
            WaveformType.PULSE: "PLS",
        }
        tag = tag_map.get(self.waveform, self.waveform.value[:3].upper())
        return f"{amp_s} {freq_s} {tag}"

    def voltage_at(self, sim_time: float) -> float:
        """Evaluate smooth instantaneous output voltage at `sim_time` (seconds)."""
        if not self.output_enabled:
            return 0.0

        phase_cycles = self.phase_deg / 360.0
        amp = self.amplitude
        off = self.dc_offset

        if self.waveform == WaveformType.SINE:
            return off + amp * math.sin(2.0 * math.pi * (self.frequency * sim_time + phase_cycles))

        phi = (self.frequency * sim_time + phase_cycles) % 1.0

        if self.waveform == WaveformType.SQUARE:
            d = max(0.05, min(0.95, float(self.duty_cycle)))
            edge = min(0.005, 0.25 * d, 0.25 * (1.0 - d))
            if phi < d:
                if edge > 0.0 and phi < edge:
                    val = -amp + (2.0 * amp) * (phi / edge)
                else:
                    val = amp
            else:
                dt_fall = phi - d
                if edge > 0.0 and dt_fall < edge:
                    val = amp - (2.0 * amp) * (dt_fall / edge)
                else:
                    val = -amp
            return off + val

        if self.waveform == WaveformType.TRIANGLE:
            d = max(0.05, min(0.95, float(self.duty_cycle)))
            if phi < d:
                val = -amp + (2.0 * amp) * (phi / d)
            else:
                val = amp - (2.0 * amp) * ((phi - d) / (1.0 - d))
            return off + val

        if self.waveform == WaveformType.SAWTOOTH:
            flyback = 0.02
            rise_end = 1.0 - flyback
            if phi < rise_end:
                val = -amp + (2.0 * amp) * (phi / rise_end)
            else:
                val = amp - (2.0 * amp) * ((phi - rise_end) / flyback)
            return off + val

        # PULSE fallback
        d = max(0.05, min(0.95, float(self.duty_cycle)))
        edge = min(0.005, 0.25 * d, 0.25 * (1.0 - d))
        if phi < d:
            val = amp * (phi / edge) if (edge > 0.0 and phi < edge) else amp
        else:
            dt_fall = phi - d
            val = amp * (1.0 - dt_fall / edge) if (edge > 0.0 and dt_fall < edge) else 0.0
        return off + val

    def get_parameter_value(self, key: str) -> float:
        if key == "output_impedance":
            return float(self.r_internal)
        if key == "output_enabled":
            return 1.0 if self.output_enabled else 0.0
        return super().get_parameter_value(key)

    def set_parameter_value(self, key: str, raw_value: float | str | bool) -> float:  # type: ignore[override]
        if key == "waveform":
            wf = (
                raw_value
                if isinstance(raw_value, WaveformType)
                else WaveformType(str(raw_value).strip().lower())
            )
            self.waveform = wf
            return 0.0
        if key == "output_enabled":
            self.output_enabled = _coerce_bool(raw_value)
            return 1.0 if self.output_enabled else 0.0
        if key == "duty_cycle":
            parsed = float(raw_value)
            if parsed > 1.0 and parsed <= 100.0:
                parsed = parsed / 100.0
            clamped = max(0.05, min(0.95, parsed))
            self.duty_cycle = clamped
            return clamped
        if key in ("output_impedance", "r_internal"):
            parsed = max(0.0, parse_engineering_value(raw_value))
            self.r_internal = parsed
            return parsed
        if key == "frequency":
            parsed = max(0.1, min(1e7, parse_engineering_value(raw_value)))
            self.frequency = parsed
            return parsed
        if key == "phase_deg":
            parsed = parse_engineering_value(raw_value)
            self.phase_deg = parsed
            return parsed
        return super().set_parameter_value(key, raw_value)
