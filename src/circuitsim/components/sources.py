"""Ideal and non-ideal DC and AC voltage/current source models."""

from __future__ import annotations

from enum import Enum
import math

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value


class WaveformType(str, Enum):
    """Supported AC source waveform shapes (Section 3.1 and Section 4)."""

    SINE = "sine"
    SQUARE = "square"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    PULSE = "pulse"


class DCVoltageSource(Component):
    """DC voltage source maintaining V(node_pos) - V(node_neg) = voltage - I_load * r_internal.

    Sign conventions:
        - `node_pos` is the positive (+) terminal.
        - `node_neg` is the negative (-) terminal.
        - `self.voltage` is the terminal voltage V(node_pos) - V(node_neg).
        - `self.current` follows the passive convention used by MNA branch variables
          (positive when current flows into `node_pos` and out of `node_neg`, or
          negative when delivering power out of `node_pos`). `delivered_current`
          and `delivered_power` provide the source-convention values.
    """

    def __init__(
        self,
        name: str,
        node_pos: str,
        node_neg: str,
        voltage: float | str = 5.0,
        *,
        r_internal: float | str = 0.0,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_pos, node_neg),
            label=label,
            show_value=show_value,
            color=color,
        )
        parsed_v = parse_engineering_value(voltage)
        parsed_r = parse_engineering_value(r_internal)
        if parsed_r < 0.0:
            raise ValueError(
                f"DCVoltageSource {name!r}: internal resistance cannot be negative ({parsed_r})."
            )
        self.dc_voltage: float = parsed_v
        self.r_internal: float = parsed_r

    @property
    def component_type(self) -> str:
        return "dc_voltage"

    def voltage_at(self, sim_time: float = 0.0) -> float:
        """Return open-circuit source voltage at `sim_time`."""
        _ = sim_time
        return self.dc_voltage

    @property
    def delivered_current(self) -> float:
        """Current flowing OUT of the positive terminal into the circuit (Amperes)."""
        return -self.current

    @property
    def delivered_power(self) -> float:
        """Power supplied by the source to the circuit (Watts)."""
        return -self.power


class ACVoltageSource(Component):
    """AC voltage source supporting sine, square, triangle, sawtooth, and pulse waveforms.

    Attributes:
        waveform: One of `WaveformType` ('sine', 'square', 'triangle', 'sawtooth', 'pulse').
        amplitude: Peak amplitude A in Volts.
            - For SINE, SQUARE, TRIANGLE, SAWTOOTH: swings between `dc_offset - A`
              and `dc_offset + A`.
            - For PULSE: switches between `dc_offset` (low) and `dc_offset + A` (high).
        frequency: Signal frequency f in Hz (must be > 0).
        phase_deg: Initial phase angle in degrees at t = 0.
        dc_offset: DC offset voltage in Volts added to the waveform.
        duty_cycle: Fraction of each period spent in the high state for SQUARE/PULSE
            waveforms, in (0.0, 1.0) (default 0.5).
        r_internal: Optional series internal resistance in Ohms (default 0.0).
    """

    def __init__(
        self,
        name: str,
        node_pos: str,
        node_neg: str,
        *,
        waveform: WaveformType | str = WaveformType.SINE,
        amplitude: float | str = 5.0,
        frequency: float | str = 1000.0,
        phase_deg: float | str = 0.0,
        dc_offset: float | str = 0.0,
        duty_cycle: float = 0.5,
        r_internal: float | str = 0.0,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_pos, node_neg),
            label=label,
            show_value=show_value,
            color=color,
        )
        parsed_amp = parse_engineering_value(amplitude)
        parsed_freq = parse_engineering_value(frequency)
        parsed_phase = parse_engineering_value(phase_deg)
        parsed_offset = parse_engineering_value(dc_offset)
        parsed_r = parse_engineering_value(r_internal)

        if parsed_amp < 0.0:
            raise ValueError(f"ACVoltageSource {name!r}: amplitude cannot be negative.")
        if parsed_freq <= 0.0:
            raise ValueError(f"ACVoltageSource {name!r}: frequency must be > 0 Hz.")
        if not (0.0 < duty_cycle < 1.0):
            raise ValueError(f"ACVoltageSource {name!r}: duty_cycle must be in (0, 1).")
        if parsed_r < 0.0:
            raise ValueError(f"ACVoltageSource {name!r}: r_internal cannot be negative.")

        self.waveform: WaveformType = (
            WaveformType(waveform) if not isinstance(waveform, WaveformType) else waveform
        )
        self.amplitude: float = parsed_amp
        self.frequency: float = parsed_freq
        self.phase_deg: float = parsed_phase
        self.dc_offset: float = parsed_offset
        self.duty_cycle: float = float(duty_cycle)
        self.r_internal: float = parsed_r

    @property
    def component_type(self) -> str:
        return "ac_voltage"

    @property
    def period(self) -> float:
        """Signal period T = 1 / f in seconds."""
        return 1.0 / self.frequency

    def voltage_at(self, sim_time: float) -> float:
        """Evaluate the instantaneous source voltage at `sim_time` (seconds)."""
        phase_cycles = self.phase_deg / 360.0
        if self.waveform == WaveformType.SINE:
            angle_rad = 2.0 * math.pi * (self.frequency * sim_time + phase_cycles)
            return self.dc_offset + self.amplitude * math.sin(angle_rad)

        # Normalized cycle position phi in [0.0, 1.0)
        phi = (self.frequency * sim_time + phase_cycles) % 1.0

        if self.waveform == WaveformType.SQUARE:
            val = self.amplitude if phi < self.duty_cycle else -self.amplitude
            return self.dc_offset + val

        if self.waveform == WaveformType.PULSE:
            val = self.amplitude if phi < self.duty_cycle else 0.0
            return self.dc_offset + val

        if self.waveform == WaveformType.TRIANGLE:
            # Starts at 0 at phi=0, rises to +A at phi=0.25, falls to -A at phi=0.75,
            # returns to 0 at phi=1.0 (matches phase of sine wave).
            if phi < 0.25:
                tri = 4.0 * phi
            elif phi < 0.75:
                tri = 2.0 - 4.0 * phi
            else:
                tri = -4.0 + 4.0 * phi
            return self.dc_offset + self.amplitude * tri

        if self.waveform == WaveformType.SAWTOOTH:
            # Ramps linearly from -A at phi=0 to +A as phi -> 1.0
            return self.dc_offset + self.amplitude * (2.0 * phi - 1.0)

        raise ValueError(f"Unsupported waveform {self.waveform!r}.")

    @property
    def delivered_current(self) -> float:
        """Current flowing OUT of the positive terminal into the circuit (Amperes)."""
        return -self.current

    @property
    def delivered_power(self) -> float:
        """Power supplied by the source to the circuit (Watts)."""
        return -self.power


class WavVoltageSource(ACVoltageSource):
    """Arbitrary PCM `.wav` audio signal voltage source (`V(t)` via linear interpolation).

    Inherits from `ACVoltageSource` so MNA DC, Transient, and AC solvers handle it
    through the standard time-varying voltage-source stamps without special cases.
    """

    def __init__(
        self,
        name: str,
        node_pos: str,
        node_neg: str,
        *,
        wav_path: str | None = None,
        samples: list[float] | tuple[float, ...] | object | None = None,
        sample_rate_hz: int = 44100,
        gain_v: float | str = 1.0,
        dc_offset: float | str = 0.0,
        loop: bool = False,
        r_internal: float | str = 0.0,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        import numpy as np
        from circuitsim.io.wav_signal import load_wav_samples

        parsed_gain = parse_engineering_value(gain_v)
        parsed_offset = parse_engineering_value(dc_offset)
        self.gain_v: float = parsed_gain
        self.loop: bool = bool(loop)
        self.wav_path: str = str(wav_path) if wav_path else ""

        if wav_path:
            wav_data = load_wav_samples(wav_path, gain_v=1.0, dc_offset_v=0.0)
            self.sample_rate_hz: int = wav_data.sample_rate_hz
            base = wav_data.samples_v
        else:
            if int(sample_rate_hz) <= 0:
                raise ValueError(f"WavVoltageSource {name!r}: sample_rate_hz must be > 0.")
            self.sample_rate_hz = int(sample_rate_hz)
            if samples is None:
                base = np.zeros(1, dtype=np.float64)
            else:
                base = np.asarray(samples, dtype=np.float64).reshape(-1)
                if len(base) == 0:
                    base = np.zeros(1, dtype=np.float64)
                base.setflags(write=False)

        self._base_samples_v = base
        peak = float(np.max(np.abs(base))) * abs(parsed_gain) if len(base) > 0 else 0.0
        eff_freq = float(self.sample_rate_hz) / max(len(base), 1)
        super().__init__(
            name=name,
            node_pos=node_pos,
            node_neg=node_neg,
            amplitude=peak,
            frequency=max(eff_freq, 1e-3),
            dc_offset=parsed_offset,
            r_internal=r_internal,
            label=label,
            show_value=show_value,
            color=color,
        )

    @property
    def component_type(self) -> str:
        return "wav_voltage"

    @property
    def duration_s(self) -> float:
        if self.sample_rate_hz <= 0 or len(self._base_samples_v) <= 1:
            return 0.0
        return float(len(self._base_samples_v) - 1) / float(self.sample_rate_hz)

    @property
    def samples_v(self):
        return self._base_samples_v * self.gain_v + self.dc_offset

    def load_wav(self, path: str) -> None:
        from circuitsim.io.wav_signal import load_wav_samples

        wav_data = load_wav_samples(path, gain_v=1.0, dc_offset_v=0.0)
        self.wav_path = str(path)
        self.sample_rate_hz = wav_data.sample_rate_hz
        self._base_samples_v = wav_data.samples_v

    def voltage_at(self, sim_time: float) -> float:
        n = len(self._base_samples_v)
        if n == 0 or self.sample_rate_hz <= 0:
            return self.dc_offset
        if sim_time <= 0.0:
            return self.dc_offset + self.gain_v * float(self._base_samples_v[0])
        dur = float(n) / float(self.sample_rate_hz)
        if self.loop and dur > 0.0:
            t_eff = sim_time % dur
        else:
            t_eff = sim_time
        pos = t_eff * float(self.sample_rate_hz)
        if pos >= n - 1:
            idx_end = 0 if (self.loop and pos >= n) else n - 1
            return self.dc_offset + self.gain_v * float(self._base_samples_v[idx_end])
        idx = int(math.floor(pos))
        frac = pos - idx
        v0 = float(self._base_samples_v[idx])
        v1 = float(self._base_samples_v[idx + 1])
        return self.dc_offset + self.gain_v * (v0 + frac * (v1 - v0))


class DCCurrentSource(Component):
    """Ideal DC current source injecting `dc_current` from `node_neg` into `node_pos`.

    External circuit current leaves `node_pos` and returns to `node_neg`.
    """

    def __init__(
        self,
        name: str,
        node_pos: str,
        node_neg: str,
        current: float | str = 0.01,
        *,
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        super().__init__(
            name=name,
            nodes=(node_pos, node_neg),
            label=label,
            show_value=show_value,
            color=color,
        )
        self.dc_current: float = parse_engineering_value(current)

    @property
    def component_type(self) -> str:
        return "dc_current"
