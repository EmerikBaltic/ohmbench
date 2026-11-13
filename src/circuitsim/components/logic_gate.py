"""Behavioral digital logic gate component model (pure Python, no Qt imports)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value

GATE_OPERATIONS: tuple[str, ...] = (
    "not",
    "and",
    "or",
    "nand",
    "nor",
    "xor",
    "xnor",
)

GATE_COMPONENT_TYPES: tuple[str, ...] = tuple(f"gate_{op}" for op in GATE_OPERATIONS)

# Standard logic family presets (voltages in V).
# Values represent conservative datasheet thresholds and unloaded/light-load levels.
LOGIC_FAMILY_PRESETS: dict[str, dict[str, float]] = {
    "5V_CMOS": {
        "v_il": 1.5,
        "v_ih": 3.5,
        "v_ol": 0.05,
        "v_oh": 4.95,
    },
    "5V_TTL": {
        "v_il": 0.8,
        "v_ih": 2.0,
        "v_ol": 0.4,
        "v_oh": 3.4,
    },
    "3V3_LVCMOS": {
        "v_il": 0.8,
        "v_ih": 2.0,
        "v_ol": 0.1,
        "v_oh": 3.2,
    },
    "Custom": {
        "v_il": 1.5,
        "v_ih": 3.5,
        "v_ol": 0.05,
        "v_oh": 4.95,
    },
}

_FAMILY_ALIASES: dict[str, str] = {
    "5v_cmos": "5V_CMOS",
    "5v cmos": "5V_CMOS",
    "cmos": "5V_CMOS",
    "5v_ttl": "5V_TTL",
    "5v ttl": "5V_TTL",
    "ttl": "5V_TTL",
    "3v3_lvcmos": "3V3_LVCMOS",
    "3.3v_lvcmos": "3V3_LVCMOS",
    "3.3v lvcmos": "3V3_LVCMOS",
    "3v3": "3V3_LVCMOS",
    "3.3v": "3V3_LVCMOS",
    "custom": "Custom",
}

_SUPPLY_MODES: tuple[str, ...] = ("implicit", "explicit")


def normalize_logic_family(family: str) -> str:
    """Resolve user-facing or alias logic family names to a canonical preset key."""
    key = family.strip()
    if key in LOGIC_FAMILY_PRESETS:
        return key
    mapped = _FAMILY_ALIASES.get(key.lower())
    if mapped is not None:
        return mapped
    valid = ", ".join(LOGIC_FAMILY_PRESETS)
    raise ValueError(f"Unknown logic family {family!r}. Valid options: {valid}.")


def evaluate_gate_boolean(operation: str, inputs: tuple[bool, ...] | list[bool]) -> bool:
    """Evaluate Boolean truth function for `operation` over digital input tuple `inputs`."""
    op = operation.strip().lower()
    if op == "not":
        if len(inputs) != 1:
            raise ValueError(f"Gate 'not' requires 1 input, got {len(inputs)}.")
        return not bool(inputs[0])
    if len(inputs) < 2:
        raise ValueError(f"Gate {op!r} requires at least 2 inputs, got {len(inputs)}.")
    if op == "and":
        return all(inputs)
    if op == "or":
        return any(inputs)
    if op == "nand":
        return not all(inputs)
    if op == "nor":
        return not any(inputs)
    if op == "xor":
        acc = False
        for bit in inputs:
            acc ^= bool(bit)
        return acc
    if op == "xnor":
        acc = False
        for bit in inputs:
            acc ^= bool(bit)
        return not acc
    raise ValueError(f"Unsupported gate operation {operation!r}.")


class LogicGate(Component):
    """Behavioral digital logic gate compatible with fixed-matrix MNA transient stepping.

    Electrical model:
        - Each input pin connects to implicit ground `"0"` through a large fixed
          input resistance ``r_in`` (default 10 MΩ), drawing negligible load current
          while avoiding isolated high-Z singularities when wired cleanly.
        - The output pin connects to implicit ground `"0"` through a fixed
          Thévenin/Norton output resistance ``r_out`` (default 100 Ω) driven by
          an internal voltage source ``v_drive(t)`` that switches smoothly between
          ``v_ol`` (Low) and ``v_oh`` (High).
        - Because ``r_in`` and ``r_out`` remain constant across logic states,
          the global MNA conductance matrix is assembled and LU-factorized once,
          and every gate transition updates only the RHS current vector.
        - Causality and smoothing:
          State transitions sensed at step ``t`` are scheduled to begin after
          propagation delay ``t_pd`` and ramp linearly from ``v_start`` to
          ``v_target`` over ``effective_edge = max(t_edge, 2 * dt)`` so
          Trapezoidal integration stays free of step discontinuities.
    """

    DEFAULT_R_IN: float = 1e7    # 10 MΩ input resistance to ground
    DEFAULT_R_OUT: float = 100.0  # 100 Ω output drive resistance
    DEFAULT_T_PD: float = 10e-9   # 10 ns propagation delay
    DEFAULT_T_EDGE: float = 2e-9  # 2 ns output rise/fall transition time

    def __init__(
        self,
        name: str,
        operation: str,
        input_nodes: tuple[str, ...] | list[str] | str,
        output_node: str,
        *,
        family: str = "5V_CMOS",
        t_pd: float | str = DEFAULT_T_PD,
        t_edge: float | str = DEFAULT_T_EDGE,
        v_il: float | str | None = None,
        v_ih: float | str | None = None,
        v_ol: float | str | None = None,
        v_oh: float | str | None = None,
        r_in: float | str = DEFAULT_R_IN,
        r_out: float | str = DEFAULT_R_OUT,
        initial_state: bool = False,
        supply_mode: str = "implicit",
        label: str | None = None,
        show_value: bool = True,
        color: str = "#FFFFFF",
    ) -> None:
        op = operation.strip().lower()
        if op not in GATE_OPERATIONS:
            valid_ops = ", ".join(GATE_OPERATIONS)
            raise ValueError(f"Unsupported gate operation {operation!r}. Valid: {valid_ops}.")

        if isinstance(input_nodes, str):
            in_nodes: tuple[str, ...] = (input_nodes,)
        else:
            in_nodes = tuple(input_nodes)

        expected_inputs = 1 if op == "not" else 2
        if len(in_nodes) != expected_inputs:
            raise ValueError(
                f"Gate {name!r} ({op}): expected {expected_inputs} input node(s), got {len(in_nodes)}."
            )

        out_clean = str(output_node).strip()
        if not out_clean:
            raise ValueError(f"Gate {name!r}: output_node must be non-empty.")

        super().__init__(
            name,
            (*in_nodes, out_clean),
            label=label,
            show_value=show_value,
            color=color,
        )

        canon_family = normalize_logic_family(family)
        preset = LOGIC_FAMILY_PRESETS[canon_family]

        parsed_t_pd = parse_engineering_value(t_pd)
        parsed_t_edge = parse_engineering_value(t_edge)
        parsed_v_il = parse_engineering_value(v_il) if v_il is not None else preset["v_il"]
        parsed_v_ih = parse_engineering_value(v_ih) if v_ih is not None else preset["v_ih"]
        parsed_v_ol = parse_engineering_value(v_ol) if v_ol is not None else preset["v_ol"]
        parsed_v_oh = parse_engineering_value(v_oh) if v_oh is not None else preset["v_oh"]
        parsed_r_in = parse_engineering_value(r_in)
        parsed_r_out = parse_engineering_value(r_out)

        if parsed_t_pd < 0.0:
            raise ValueError(f"Gate {name!r}: t_pd cannot be negative (got {t_pd!r}).")
        if parsed_t_edge < 0.0:
            raise ValueError(f"Gate {name!r}: t_edge cannot be negative (got {t_edge!r}).")
        if parsed_r_in <= 0.0:
            raise ValueError(f"Gate {name!r}: r_in must be > 0 Ω (got {r_in!r}).")
        if parsed_r_out <= 0.0:
            raise ValueError(f"Gate {name!r}: r_out must be > 0 Ω (got {r_out!r}).")
        if parsed_v_il >= parsed_v_ih:
            raise ValueError(
                f"Gate {name!r}: v_il ({parsed_v_il} V) must be strictly less than v_ih ({parsed_v_ih} V)."
            )
        if parsed_v_ol >= parsed_v_oh:
            raise ValueError(
                f"Gate {name!r}: v_ol ({parsed_v_ol} V) must be strictly less than v_oh ({parsed_v_oh} V)."
            )
        if supply_mode not in _SUPPLY_MODES:
            raise ValueError(
                f"Gate {name!r}: unsupported supply_mode {supply_mode!r}; expected one of {_SUPPLY_MODES}."
            )

        if canon_family != "Custom" and (
            abs(parsed_v_il - preset["v_il"]) > 1e-12
            or abs(parsed_v_ih - preset["v_ih"]) > 1e-12
            or abs(parsed_v_ol - preset["v_ol"]) > 1e-12
            or abs(parsed_v_oh - preset["v_oh"]) > 1e-12
        ):
            canon_family = "Custom"

        self.operation: str = op
        self.family: str = canon_family
        self.t_pd: float = parsed_t_pd
        self.t_edge: float = parsed_t_edge
        self.v_il: float = parsed_v_il
        self.v_ih: float = parsed_v_ih
        self.v_ol: float = parsed_v_ol
        self.v_oh: float = parsed_v_oh
        self.r_in: float = parsed_r_in
        self.r_out: float = parsed_r_out
        self.initial_state: bool = bool(initial_state)
        self.supply_mode: str = supply_mode

        self.input_pin_names: tuple[str, ...] = ("in",) if op == "not" else ("a", "b")
        self.output_pin_name: str = "out"

        # Internal digital/transient state variables.
        self._output_level: bool = self.initial_state
        self._last_scheduled_level: bool = self.initial_state
        self._drive_voltage: float = self.level_voltage(self.initial_state)
        self._pin_levels: list[bool] = [False] * len(self.input_nodes)
        self._midband_dwell_s: list[float] = [0.0] * len(self.input_nodes)
        self._pending_events: deque[tuple[float, bool]] = deque()
        # Active output ramp: (v_start, v_end, t_start, t_end, target_level)
        self._active_ramp: tuple[float, float, float, float, bool] | None = None

    @property
    def input_nodes(self) -> tuple[str, ...]:
        """Input terminal node identifiers, tracking dynamic `.nodes` rebinding."""
        return tuple(self.nodes[:-1])

    @property
    def output_node(self) -> str:
        """Output terminal node identifier, tracking dynamic `.nodes` rebinding."""
        return self.nodes[-1]

    @property
    def component_type(self) -> str:
        return f"gate_{self.operation}"

    @property
    def g_in(self) -> float:
        """Input conductance to ground per input pin (S)."""
        return 1.0 / self.r_in

    @property
    def g_out(self) -> float:
        """Output conductance to ground at output node (S)."""
        return 1.0 / self.r_out

    @property
    def output_level(self) -> bool:
        """Latest digital output state (True = High, False = Low)."""
        return self._output_level

    @property
    def drive_voltage(self) -> float:
        """Instantaneous open-circuit internal drive voltage behind `r_out` (V)."""
        return self._drive_voltage

    @property
    def pin_levels(self) -> tuple[bool, ...]:
        """Latest Schmitt-filtered Boolean state sensed on each input pin."""
        return tuple(self._pin_levels)

    def level_voltage(self, level: bool) -> float:
        """Return `v_oh` when `level` is True or `v_ol` when False."""
        return self.v_oh if level else self.v_ol

    def set_family_preset(self, family: str) -> None:
        """Apply threshold and rail preset values from `LOGIC_FAMILY_PRESETS`."""
        canon = normalize_logic_family(family)
        self.family = canon
        if canon != "Custom":
            preset = LOGIC_FAMILY_PRESETS[canon]
            self.v_il = preset["v_il"]
            self.v_ih = preset["v_ih"]
            self.v_ol = preset["v_ol"]
            self.v_oh = preset["v_oh"]
            self._drive_voltage = self.level_voltage(self._output_level)

    def set_parameter_value(self, key: str, raw_value: float | str) -> float:
        """Validate parameter update and keep logic thresholds/rails consistent."""
        prev = self.get_parameter_value(key)
        updated = super().set_parameter_value(key, raw_value)
        if self.v_il >= self.v_ih:
            setattr(self, key, prev)
            raise ValueError(
                f"Gate {self.name!r}: v_il ({self.v_il} V) must remain strictly less than v_ih ({self.v_ih} V)."
            )
        if self.v_ol >= self.v_oh:
            setattr(self, key, prev)
            raise ValueError(
                f"Gate {self.name!r}: v_ol ({self.v_ol} V) must remain strictly less than v_oh ({self.v_oh} V)."
            )
        if key in ("v_il", "v_ih", "v_ol", "v_oh") and self.family != "Custom":
            preset = LOGIC_FAMILY_PRESETS[self.family]
            if any(abs(getattr(self, k) - preset[k]) > 1e-12 for k in ("v_il", "v_ih", "v_ol", "v_oh")):
                self.family = "Custom"
        if key in ("v_ol", "v_oh") and self._active_ramp is None:
            self._drive_voltage = self.level_voltage(self._output_level)
        return updated

    def set_parameters_batch(
        self, params: Mapping[str, float | str | bool]
    ) -> dict[str, object]:
        """Atomically validate and apply multiple LogicGate parameters & settings."""
        snapshot = (
            self.family,
            self.t_pd,
            self.t_edge,
            self.v_il,
            self.v_ih,
            self.v_ol,
            self.v_oh,
            self.r_in,
            self.r_out,
            self.initial_state,
            self.supply_mode,
            self._drive_voltage,
        )
        parsed: dict[str, object] = {}
        param_keys = {p.key for p in self.get_editable_parameters()}
        try:
            for key, raw_val in params.items():
                if key == "family":
                    canon = normalize_logic_family(str(raw_val))
                    self.family = canon
                    parsed[key] = canon
                elif key == "initial_state":
                    lvl = bool(raw_val)
                    self.initial_state = lvl
                    parsed[key] = lvl
                elif key == "supply_mode":
                    self.supply_mode = str(raw_val)
                    parsed[key] = self.supply_mode
                elif key in param_keys:
                    parsed[key] = super().set_parameter_value(key, raw_val)  # type: ignore[arg-type]

            if self.v_il >= self.v_ih:
                raise ValueError(
                    f"Gate {self.name!r}: v_il ({self.v_il} V) must remain strictly less than v_ih ({self.v_ih} V)."
                )
            if self.v_ol >= self.v_oh:
                raise ValueError(
                    f"Gate {self.name!r}: v_ol ({self.v_ol} V) must remain strictly less than v_oh ({self.v_oh} V)."
                )

            thresh_keys = ("v_il", "v_ih", "v_ol", "v_oh")
            if "family" in parsed and parsed["family"] != "Custom":
                preset = LOGIC_FAMILY_PRESETS[str(parsed["family"])]
                if any(k not in parsed for k in thresh_keys):
                    for k in thresh_keys:
                        if k not in parsed:
                            setattr(self, k, preset[k])
                elif any(
                    abs(getattr(self, k) - preset[k]) > 1e-12 for k in thresh_keys
                ):
                    self.family = "Custom"
            elif "family" not in parsed and any(k in parsed for k in thresh_keys):
                if self.family != "Custom":
                    preset = LOGIC_FAMILY_PRESETS[self.family]
                    if any(
                        abs(getattr(self, k) - preset[k]) > 1e-12 for k in thresh_keys
                    ):
                        self.family = "Custom"

            if self._active_ramp is None:
                self._drive_voltage = self.level_voltage(self._output_level)
        except Exception:
            (
                self.family,
                self.t_pd,
                self.t_edge,
                self.v_il,
                self.v_ih,
                self.v_ol,
                self.v_oh,
                self.r_in,
                self.r_out,
                self.initial_state,
                self.supply_mode,
                self._drive_voltage,
            ) = snapshot
            raise
        return parsed

    def reset_state(self) -> None:
        """Reset all transient queues, input history, and operating telemetry to initial state."""
        super().reset_state()
        self._output_level = self.initial_state
        self._last_scheduled_level = self.initial_state
        self._drive_voltage = self.level_voltage(self.initial_state)
        self._pin_levels = [False] * len(self.input_nodes)
        self._midband_dwell_s = [0.0] * len(self.input_nodes)
        self._pending_events.clear()
        self._active_ramp = None

    def set_steady_state_level(self, level: bool) -> None:
        """Immediately snap internal drive and output state to `level` (for DC / t=0 init)."""
        lvl = bool(level)
        self._output_level = lvl
        self._last_scheduled_level = lvl
        self._drive_voltage = self.level_voltage(lvl)
        self._pending_events.clear()
        self._active_ramp = None

    def _sense_pin(self, pin_idx: int, v_pin: float, dt: float) -> bool:
        """Update Schmitt-hysteresis Boolean state and mid-band dwell counter for `pin_idx`."""
        if v_pin >= self.v_ih:
            self._pin_levels[pin_idx] = True
            self._midband_dwell_s[pin_idx] = 0.0
        elif v_pin <= self.v_il:
            self._pin_levels[pin_idx] = False
            self._midband_dwell_s[pin_idx] = 0.0
        else:
            # Strictly inside (v_il, v_ih): retain prior latched logic level.
            if dt > 0.0:
                self._midband_dwell_s[pin_idx] += dt
        return self._pin_levels[pin_idx]

    def evaluate_inputs(self, node_voltages: dict[str, float], dt: float = 0.0) -> bool:
        """Sense all input pins against `(v_il, v_ih)` hysteresis and return Boolean target."""
        sensed: list[bool] = []
        for idx, node in enumerate(self.input_nodes):
            v_pin = node_voltages.get(node, 0.0)
            sensed.append(self._sense_pin(idx, v_pin, dt))
        return evaluate_gate_boolean(self.operation, sensed)

    def current_output_voltage(self, sim_time: float) -> float:
        """Evaluate the open-circuit Thévenin drive voltage `v_drive(sim_time)`."""
        ramp = self._active_ramp
        if ramp is None:
            return self._drive_voltage
        v_start, v_end, t_start, t_end, _ = ramp
        if t_end <= t_start or sim_time >= t_end:
            return v_end
        if sim_time <= t_start:
            return v_start
        frac = (sim_time - t_start) / (t_end - t_start)
        return v_start + (v_end - v_start) * frac

    def prepare_step(self, t_next: float, dt: float) -> None:
        """Activate any due propagation-delay events and update `drive_voltage` for `t_next`."""
        effective_edge = max(self.t_edge, 2.0 * dt) if dt > 0.0 else self.t_edge
        while self._pending_events and self._pending_events[0][0] <= t_next + 1e-15:
            t_trig, target_level = self._pending_events.popleft()
            v_start = self.current_output_voltage(t_trig)
            v_end = self.level_voltage(target_level)
            if abs(v_end - v_start) < 1e-15 or effective_edge <= 0.0:
                self._active_ramp = None
                self._drive_voltage = v_end
                self._output_level = target_level
            else:
                self._active_ramp = (
                    v_start,
                    v_end,
                    t_trig,
                    t_trig + effective_edge,
                    target_level,
                )

        if self._active_ramp is not None:
            v_start, v_end, t_start, t_end, target_level = self._active_ramp
            v_now = self.current_output_voltage(t_next)
            self._drive_voltage = v_now
            midpoint = 0.5 * (self.v_ol + self.v_oh)
            self._output_level = v_now >= midpoint
            if t_next >= t_end - 1e-15:
                self._drive_voltage = v_end
                self._output_level = target_level
                self._active_ramp = None

    def sample_inputs_and_schedule(
        self,
        node_voltages: dict[str, float],
        sim_time: float,
        dt: float,
    ) -> None:
        """Sample solved inputs at `sim_time` and enqueue any future output transition."""
        desired_level = self.evaluate_inputs(node_voltages, dt=dt)
        if desired_level != self._last_scheduled_level:
            # Clamp causal delay >= dt in transient so same-step algebraic loops cannot occur.
            effective_t_pd = max(self.t_pd, dt) if dt > 0.0 else self.t_pd
            self._pending_events.append((sim_time + effective_t_pd, desired_level))
            self._last_scheduled_level = desired_level

    def indeterminate_pins(self, node_voltages: dict[str, float], dt: float = 0.0) -> list[tuple[str, float]]:
        """Return `(pin_name, v_pin)` entries currently stuck in the indeterminate band `(v_il, v_ih)`.

        During DC analysis (`dt <= 0.0`), any pin strictly between `v_il` and `v_ih`
        is reported immediately. During transient analysis (`dt > 0.0`), a pin is
        reported only once it lingers strictly in `(v_il, v_ih)` longer than
        `2.0 * max(t_edge, dt)` so fast valid transitions do not raise false alarms.
        """
        dwell_limit = 2.0 * max(self.t_edge, dt) if dt > 0.0 else 0.0
        bad: list[tuple[str, float]] = []
        for idx, (pin_name, node) in enumerate(zip(self.input_pin_names, self.input_nodes)):
            v_pin = node_voltages.get(node, 0.0)
            if self.v_il < v_pin < self.v_ih:
                if dt <= 0.0 or self._midband_dwell_s[idx] > dwell_limit:
                    bad.append((pin_name, v_pin))
        return bad
