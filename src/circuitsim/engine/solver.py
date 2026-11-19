"""Modified Nodal Analysis (MNA) DC and Transient solver using NumPy and SciPy LU."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from circuitsim.components.bjt import BJTBase
from circuitsim.components.capacitor import Capacitor
from circuitsim.components.diode import Diode
from circuitsim.components.ground import Ground
from circuitsim.components.inductor import Inductor
from circuitsim.components.led import LED
from circuitsim.components.lightbulb import Lightbulb
from circuitsim.components.logic_gate import LogicGate
from circuitsim.components.mcu import MCUBlock
from circuitsim.components.meters import Ammeter, Voltmeter
from circuitsim.components.mosfet import MOSFETBase
from circuitsim.components.ne555 import NE555
from circuitsim.components.opamp import OpAmp
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.power import Fuse, SPDTRelay, Transformer
from circuitsim.components.resistor import Resistor
from circuitsim.components.sequential import DigitalSequentialBlock
from circuitsim.components.sources import (
    ACVoltageSource,
    DCCurrentSource,
    DCVoltageSource,
)
from circuitsim.components.base import Component
from circuitsim.components.switch import Switch
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.engine.errors import CircuitWarning, SingularMatrixError
from circuitsim.engine.netlist import Circuit

_TAG_RESISTOR = 0
_TAG_LIGHTBULB = 1
_TAG_VOLTMETER = 2
_TAG_AMMETER = 3
_TAG_LED = 4
_TAG_SWITCH = 5
_TAG_CAPACITOR = 6
_TAG_INDUCTOR = 7
_TAG_LOGIC_GATE = 8
_TAG_DC_VS = 9
_TAG_AC_VS = 10
_TAG_DC_IS = 11
_TAG_GROUND = 12
_TAG_OTHER = 13
_TAG_POTENTIOMETER = 14
_TAG_DIODE = 15
_TAG_MCU = 16
_TAG_BJT = 17
_TAG_MOSFET = 18
_TAG_OPAMP = 19
_TAG_NE555 = 20
_TAG_FUSE = 21
_TAG_TRANSFORMER = 22
_TAG_RELAY = 23
_TAG_SEQUENTIAL = 24


class IntegrationMethod(str, Enum):
    """Numerical integration method for reactive companion models."""

    TRAPEZOIDAL = "trapezoidal"
    BACKWARD_EULER = "backward_euler"


@dataclass
class SolveResult:
    """Result of a single DC operating point or single transient step solve.

    Attributes:
        node_voltages: Mapping from every node name (including grounded nodes)
            to its solved electric potential in Volts.
        branch_currents: Mapping from component name to its solved branch current
            in Amperes (passive sign convention: positive from node[0] to node[1]).
        sim_time: Simulation timestamp in seconds (0.0 for DC operating point).
        dt: Step size used for this solve in seconds (0.0 for DC).
        integration_method: Integration method used for this step ('dc',
            'backward_euler', or 'trapezoidal').
        condition_number: 2-norm condition number of the MNA matrix A, if computed.
        warnings: Non-fatal circuit warnings detected during this solve.
        rebuilt_matrix: True if the MNA matrix A was assembled and LU-factorized
            on this call; False if a cached LU factorization was reused.
    """

    node_voltages: dict[str, float]
    branch_currents: dict[str, float]
    sim_time: float = 0.0
    dt: float = 0.0
    integration_method: str = "dc"
    condition_number: float | None = None
    warnings: list[CircuitWarning] = field(default_factory=list)
    rebuilt_matrix: bool = True


@dataclass
class TransientResult:
    """Time-series results from a multi-step transient simulation run.

    Attributes:
        times: 1D NumPy array of simulation timestamps in seconds.
        node_voltages: Mapping from node name to 1D NumPy array of voltages (V).
        branch_currents: Mapping from component name to 1D NumPy array of currents (A).
        component_voltages: Mapping from component name to 1D NumPy array of terminal
            voltages V(node_a) - V(node_b) (V).
        warnings: Deduplicated list of all `CircuitWarning` objects raised during the run.
        matrix_rebuild_count: Number of times `lu_factor` was called across the run.
        step_count: Total number of transient steps executed.
    """

    times: np.ndarray
    node_voltages: dict[str, np.ndarray]
    branch_currents: dict[str, np.ndarray]
    component_voltages: dict[str, np.ndarray]
    warnings: list[CircuitWarning] = field(default_factory=list)
    matrix_rebuild_count: int = 0
    step_count: int = 0


class MNASolver:
    """Modified Nodal Analysis (MNA) solver for DC and Transient circuit analysis.

    Numerical design highlights:
        1. `GMIN` (default 1e-12 S) is placed from every non-ground node to ground
           to prevent singular matrices when nodes are isolated behind open switches
           or series capacitors.
        2. LU factorization caching (`scipy.linalg.lu_factor` / `lu_solve`):
           During fixed-step transient simulation, the MNA matrix A depends only on
           component conductances, `dt`, and the active integration method. Once
           assembled and factorized, `(lu, piv)` is reused across hundreds or
           thousands of time steps until a switch toggles, `dt` or the integration
           method changes, or a thermal filament updates.
        3. Backward Euler startup and switch-event damping:
           Trapezoidal integration is second-order accurate (O(dt^2)) and energy-
           conserving (no artificial damping on lossless LC tanks), but it is not
           L-stable: step discontinuities (such as t = 0 startup or a switch
           toggling) cause step-to-step alternating ringing in the companion history
           currents. Using Backward Euler on the first step and for 2 damping steps
           upon any switch toggle settles discontinuous jumps cleanly before
           resuming Trapezoidal integration.
    """

    DEFAULT_GMIN: float = 1e-12
    KICKBACK_VOLTAGE_WARNING_MIN: float = 100.0

    def __init__(
        self,
        circuit: Circuit,
        *,
        gmin: float = DEFAULT_GMIN,
        regularize_ideal_loops: bool = False,
        startup_perturbation_uv: float = 1.0,
    ) -> None:
        if gmin < 0.0:
            raise ValueError("gmin cannot be negative.")
        self.circuit: Circuit = circuit
        self.gmin: float = float(gmin)
        self.regularize_ideal_loops: bool = bool(regularize_ideal_loops)
        self.startup_perturbation_uv: float = float(startup_perturbation_uv)
        self.last_convergence_method: str = "direct"

        # Transient clock and damping counter
        self.sim_time: float = 0.0
        self._transient_step_index: int = 0
        # Number of upcoming transient steps that must use Backward Euler
        # (initialized to 1 for the very first transient step, and set to 2
        # whenever a switch toggles to settle the impulse derivative).
        self._be_steps_remaining: int = 1
        self._startup_perturbation_applied: bool = False
        self._auto_regularized_warnings: list[CircuitWarning] = []

        # Cached MNA system state
        self._cached_mode: str = ""
        self._cached_circuit_version: int = -1
        self._cached_conductance_signature: tuple[float, ...] = ()
        self._node_to_idx: dict[str, int] = {}
        self._vs_to_idx: dict[str, int] = {}
        self._matrix_a: np.ndarray | None = None
        self._lu_piv: tuple[np.ndarray, np.ndarray] | None = None
        self._topology_warnings: list[CircuitWarning] = []
        self._logic_t0_initialized: bool = False
        self._t0_node_voltages: dict[str, float] = {}

        # Per-circuit-version component type buckets (avoids repeated ABC isinstance checks in inner transient loops)
        self._buckets_version: int = -1
        self._buckets_len: int = -1
        self._bucket_grounded_nodes: tuple[str, ...] = ()
        self._bucket_tagged_comps: tuple[tuple[Component, int], ...] = ()
        self._bucket_rhs_comps: tuple[tuple[Component, int], ...] = ()
        self._bucket_lightbulbs: tuple[Lightbulb, ...] = ()
        self._bucket_voltmeters: tuple[Voltmeter, ...] = ()
        self._bucket_leds: tuple[LED, ...] = ()
        self._bucket_diodes: tuple[Diode, ...] = ()
        self._bucket_bjts: tuple[BJTBase, ...] = ()
        self._bucket_mosfets: tuple[MOSFETBase, ...] = ()
        self._bucket_opamps: tuple[OpAmp, ...] = ()
        self._bucket_switches: tuple[Switch, ...] = ()
        self._bucket_inductors: tuple[Inductor, ...] = ()
        self._bucket_logic_gates: tuple[LogicGate, ...] = ()
        self._bucket_mcus: tuple[MCUBlock, ...] = ()
        self._bucket_ne555: tuple[NE555, ...] = ()
        self._bucket_fuses: tuple[Fuse, ...] = ()
        self._bucket_transformers: tuple[Transformer, ...] = ()
        self._bucket_relays: tuple[SPDTRelay, ...] = ()
        self._bucket_sequential: tuple[DigitalSequentialBlock, ...] = ()

    def invalidate_cache(self) -> None:
        """Force the solver to rebuild and re-factorize the MNA matrix on next solve."""
        self._lu_piv = None
        self._cached_circuit_version = -1
        self._cached_mode = ""
        self._buckets_version = -1

    def _ensure_component_buckets(self) -> None:
        """Partition circuit components once per topology version to keep inner loops free of ABC checks."""
        circ_ver = self.circuit.version
        raw_comps = self.circuit.flat_components()
        if self._buckets_version == circ_ver and self._buckets_len == len(raw_comps):
            return

        tagged: list[tuple[Component, int]] = []
        rhs_tagged: list[tuple[Component, int]] = []
        grounded_nodes: list[str] = []
        lightbulbs: list[Lightbulb] = []
        voltmeters: list[Voltmeter] = []
        leds: list[LED] = []
        diodes: list[Diode] = []
        bjts: list[BJTBase] = []
        mosfets: list[MOSFETBase] = []
        opamps: list[OpAmp] = []
        switches: list[Switch] = []
        inductors: list[Inductor] = []
        logic_gates: list[LogicGate] = []
        mcus: list[MCUBlock] = []
        ne555_ics: list[NE555] = []
        fuses: list[Fuse] = []
        transformers: list[Transformer] = []
        relays: list[SPDTRelay] = []
        sequential_blocks: list[DigitalSequentialBlock] = []

        for comp in raw_comps:
            if isinstance(comp, Ground):
                tagged.append((comp, _TAG_GROUND))
                grounded_nodes.append(comp.node)
            elif isinstance(comp, Resistor):
                tagged.append((comp, _TAG_RESISTOR))
            elif isinstance(comp, Lightbulb):
                tagged.append((comp, _TAG_LIGHTBULB))
                lightbulbs.append(comp)
            elif isinstance(comp, Voltmeter):
                tagged.append((comp, _TAG_VOLTMETER))
                voltmeters.append(comp)
            elif isinstance(comp, Ammeter):
                tagged.append((comp, _TAG_AMMETER))
                rhs_tagged.append((comp, _TAG_AMMETER))
            elif isinstance(comp, LED):
                tagged.append((comp, _TAG_LED))
                rhs_tagged.append((comp, _TAG_LED))
                leds.append(comp)
            elif isinstance(comp, Diode):
                tagged.append((comp, _TAG_DIODE))
                rhs_tagged.append((comp, _TAG_DIODE))
                diodes.append(comp)
            elif isinstance(comp, BJTBase):
                tagged.append((comp, _TAG_BJT))
                rhs_tagged.append((comp, _TAG_BJT))
                bjts.append(comp)
            elif isinstance(comp, MOSFETBase):
                tagged.append((comp, _TAG_MOSFET))
                rhs_tagged.append((comp, _TAG_MOSFET))
                mosfets.append(comp)
            elif isinstance(comp, OpAmp):
                tagged.append((comp, _TAG_OPAMP))
                rhs_tagged.append((comp, _TAG_OPAMP))
                opamps.append(comp)
            elif isinstance(comp, Switch):
                tagged.append((comp, _TAG_SWITCH))
                rhs_tagged.append((comp, _TAG_SWITCH))
                switches.append(comp)
            elif isinstance(comp, Capacitor):
                tagged.append((comp, _TAG_CAPACITOR))
                rhs_tagged.append((comp, _TAG_CAPACITOR))
            elif isinstance(comp, Inductor):
                tagged.append((comp, _TAG_INDUCTOR))
                rhs_tagged.append((comp, _TAG_INDUCTOR))
                inductors.append(comp)
            elif isinstance(comp, LogicGate):
                tagged.append((comp, _TAG_LOGIC_GATE))
                rhs_tagged.append((comp, _TAG_LOGIC_GATE))
                logic_gates.append(comp)
            elif isinstance(comp, MCUBlock):
                tagged.append((comp, _TAG_MCU))
                rhs_tagged.append((comp, _TAG_MCU))
                mcus.append(comp)
            elif isinstance(comp, NE555):
                tagged.append((comp, _TAG_NE555))
                ne555_ics.append(comp)
            elif isinstance(comp, Fuse):
                tagged.append((comp, _TAG_FUSE))
                fuses.append(comp)
            elif isinstance(comp, Transformer):
                tagged.append((comp, _TAG_TRANSFORMER))
                transformers.append(comp)
            elif isinstance(comp, SPDTRelay):
                tagged.append((comp, _TAG_RELAY))
                relays.append(comp)
            elif isinstance(comp, DigitalSequentialBlock):
                tagged.append((comp, _TAG_SEQUENTIAL))
                rhs_tagged.append((comp, _TAG_SEQUENTIAL))
                sequential_blocks.append(comp)
            elif isinstance(comp, Potentiometer):
                tagged.append((comp, _TAG_POTENTIOMETER))
            elif isinstance(comp, DCVoltageSource):
                tagged.append((comp, _TAG_DC_VS))
                rhs_tagged.append((comp, _TAG_DC_VS))
            elif isinstance(comp, ACVoltageSource):
                tagged.append((comp, _TAG_AC_VS))
                rhs_tagged.append((comp, _TAG_AC_VS))
            elif isinstance(comp, DCCurrentSource):
                tagged.append((comp, _TAG_DC_IS))
                rhs_tagged.append((comp, _TAG_DC_IS))
            else:
                tagged.append((comp, _TAG_OTHER))

        self._bucket_grounded_nodes = tuple(grounded_nodes)
        self._bucket_tagged_comps = tuple(tagged)
        self._bucket_rhs_comps = tuple(rhs_tagged)
        self._bucket_lightbulbs = tuple(lightbulbs)
        self._bucket_voltmeters = tuple(voltmeters)
        self._bucket_leds = tuple(leds)
        self._bucket_diodes = tuple(diodes)
        self._bucket_bjts = tuple(bjts)
        self._bucket_mosfets = tuple(mosfets)
        self._bucket_opamps = tuple(opamps)
        self._bucket_switches = tuple(switches)
        self._bucket_inductors = tuple(inductors)
        self._bucket_logic_gates = tuple(logic_gates)
        self._bucket_mcus = tuple(mcus)
        self._bucket_ne555 = tuple(ne555_ics)
        self._bucket_fuses = tuple(fuses)
        self._bucket_transformers = tuple(transformers)
        self._bucket_relays = tuple(relays)
        self._bucket_sequential = tuple(sequential_blocks)
        self._buckets_len = len(tagged)
        self._buckets_version = circ_ver

    def reset_transient(self) -> None:
        """Reset simulation time, component states, and solver cache."""
        self.sim_time = 0.0
        self._transient_step_index = 0
        self._be_steps_remaining = 1
        self._logic_t0_initialized = False
        self._t0_node_voltages = {}
        self._startup_perturbation_applied = False
        self._auto_regularized_warnings = []
        self.last_convergence_method = "direct"
        self.circuit.reset_state()
        self.invalidate_cache()

    @property
    def matrix_a(self) -> np.ndarray | None:
        """Return a copy of the most recently assembled MNA matrix A."""
        return None if self._matrix_a is None else self._matrix_a.copy()

    def compute_condition_number(self) -> float:
        """Return the 2-norm condition number of the active MNA matrix A."""
        if self._matrix_a is None:
            raise RuntimeError("MNA matrix has not been assembled yet.")
        return float(np.linalg.cond(self._matrix_a))

    def choose_default_dt(self, duration: float | None = None) -> float:
        """Choose a safe default transient time step `dt` based on circuit timescales.

        Default Time-Step Rule:
            1. AC sources: at least 100 steps per period of the highest frequency
               AC source (`dt <= 1 / (100 * f_max)`), matching Section 7.2.
            2. RL time constants: at least 50 steps per RL time constant
               (`dt <= tau_L / 50`) for active inductive branches.
            3. RC time constants: at least 50 steps per RC time constant
               (`dt <= R_th * C / 50`) when R_th * C > 0.
            4. LC resonant tanks: at least 100 steps per resonant period
               (`dt <= 2 * pi * sqrt(L * C) / 100`).
            5. Total duration: if `duration` is specified, at least 200 steps across
               the run (`dt <= duration / 200`).
            6. Fallback default for purely resistive DC circuits: `dt = 1e-4 s` (100 us).
        """
        candidates: list[float] = []
        if duration is not None and duration > 0.0:
            candidates.append(duration / 200.0)
        else:
            candidates.append(1e-4)

        inductances: list[float] = []
        capacitances: list[float] = []

        for comp in self.circuit.flat_components():
            if isinstance(comp, ACVoltageSource) and comp.frequency > 0.0:
                candidates.append(1.0 / (100.0 * comp.frequency))
            elif isinstance(comp, Inductor):
                inductances.append(comp.inductance)
                tau_l = comp.time_constant(self.circuit)
                if math.isfinite(tau_l) and tau_l > 1e-9:
                    candidates.append(tau_l / 50.0)
            elif isinstance(comp, Capacitor):
                capacitances.append(comp.capacitance)
                r_th = self.circuit.thevenin_resistance(
                    comp.nodes[0], comp.nodes[1], exclude_component=comp.name
                )
                tau_c = r_th * comp.capacitance
                if math.isfinite(tau_c) and tau_c > 1e-9:
                    candidates.append(tau_c / 50.0)
            elif isinstance(comp, LogicGate):
                if comp.t_pd > 0.0:
                    candidates.append(comp.t_pd / 10.0)
                if comp.t_edge > 0.0:
                    candidates.append(comp.t_edge / 2.0)

        if inductances and capacitances:
            min_l = min(inductances)
            min_c = min(capacitances)
            t_lc = 2.0 * math.pi * math.sqrt(min_l * min_c)
            if t_lc > 0.0:
                candidates.append(t_lc / 100.0)

        chosen = min(candidates)
        return max(1e-12, chosen)

    def _stamp_conductance(
        self,
        matrix_a: np.ndarray,
        node_a: str,
        node_b: str,
        g: float,
    ) -> None:
        """Stamp a two-terminal conductance `g` between `node_a` and `node_b`."""
        idx_a = self._node_to_idx.get(node_a)
        idx_b = self._node_to_idx.get(node_b)
        if idx_a is not None:
            matrix_a[idx_a, idx_a] += g
        if idx_b is not None:
            matrix_a[idx_b, idx_b] += g
        if idx_a is not None and idx_b is not None:
            matrix_a[idx_a, idx_b] -= g
            matrix_a[idx_b, idx_a] -= g

    @staticmethod
    def _stamp_small_matrix(
        matrix_a: np.ndarray,
        node_to_idx: dict[str, int],
        nodes: tuple[str, ...],
        g_mat: tuple[tuple[float, ...], ...],
    ) -> None:
        """Stamp an NxN small-signal Jacobian matrix for a multi-terminal device."""
        idxs = [node_to_idx.get(n) for n in nodes]
        for r_i, row_idx in enumerate(idxs):
            if row_idx is None:
                continue
            row_vals = g_mat[r_i]
            for c_j, col_idx in enumerate(idxs):
                if col_idx is not None:
                    matrix_a[row_idx, col_idx] += row_vals[c_j]

    @staticmethod
    def _stamp_small_rhs(
        rhs_b: np.ndarray,
        node_to_idx: dict[str, int],
        nodes: tuple[str, ...],
        rhs_vec: tuple[float, ...],
    ) -> None:
        """Stamp an N-element Norton RHS vector for a multi-terminal device."""
        for node, val in zip(nodes, rhs_vec):
            idx = node_to_idx.get(node)
            if idx is not None:
                rhs_b[idx] += val

    def _voltage_sources(self) -> list[DCVoltageSource | ACVoltageSource | Ammeter]:
        return [
            comp
            for comp in self.circuit.flat_components()
            if isinstance(comp, (DCVoltageSource, ACVoltageSource))
            or (isinstance(comp, Ammeter) and comp.is_ideal)
        ]

    def _dc_conductance_signature(
        self, *, extra_gmin: float = 0.0, loop_reg_r: float = 0.0
    ) -> tuple[float, ...]:
        sig: list[float] = [max(self.gmin, extra_gmin), loop_reg_r]
        for comp in self.circuit.flat_components():
            if isinstance(comp, (Resistor, Switch, Fuse, Lightbulb, Voltmeter, Ammeter, LED)):
                if isinstance(comp, (Voltmeter, Ammeter)) and comp.is_ideal:
                    sig.append(0.0)
                else:
                    sig.append(comp.conductance)
            elif isinstance(comp, Diode):
                sig.append(comp.companion_model(0.0, "dc")[0])
            elif isinstance(comp, BJTBase):
                sig.extend((comp.operating_vbe, comp.operating_vbc))
            elif isinstance(comp, MOSFETBase):
                sig.extend((comp.operating_vg, comp.operating_vd, comp.operating_vs))
            elif isinstance(comp, OpAmp):
                sig.append(comp.operating_vd)
            elif isinstance(comp, Inductor):
                sig.append(comp.dc_conductance)
            elif isinstance(comp, LogicGate):
                sig.extend((comp.g_in, comp.g_out))
            elif isinstance(comp, (MCUBlock, DigitalSequentialBlock)):
                sig.extend(comp.pin_conductance(p) for p in comp.pin_names)
            elif isinstance(comp, NE555):
                sig.extend(
                    (
                        1.0 if comp.output_high else 0.0,
                        comp.r_div,
                        comp.r_out,
                        comp.r_dis_on,
                    )
                )
            elif isinstance(comp, Transformer):
                sig.extend((comp.turns_ratio, comp.r_winding_ohm, comp.l_primary_h, comp.coupling_k))
            elif isinstance(comp, SPDTRelay):
                sig.extend((1.0 if comp.energized else 0.0, comp.coil_resistance_ohm))
            elif isinstance(comp, Potentiometer):
                sig.extend((comp.g_aw, comp.g_wb))
            elif isinstance(comp, (DCVoltageSource, ACVoltageSource)):
                sig.append(comp.r_internal)
        return tuple(sig)

    def _transient_conductance_signature(
        self,
        dt: float,
        method_str: str,
        *,
        extra_gmin: float = 0.0,
        loop_reg_r: float = 0.0,
    ) -> tuple[float, ...]:
        self._ensure_component_buckets()
        method_code = 1.0 if method_str == "backward_euler" else 2.0
        sig: list[float] = [max(self.gmin, extra_gmin), loop_reg_r, dt, method_code]
        for comp, tag in self._bucket_tagged_comps:
            if tag <= _TAG_LIGHTBULB:
                sig.append(comp.conductance)
            elif tag <= _TAG_AMMETER:
                sig.append(0.0 if comp.is_ideal else comp.conductance)
            elif tag <= _TAG_INDUCTOR:
                g_eq, _ = comp.companion_model(dt, method_str)
                sig.append(g_eq)
            elif tag == _TAG_DIODE:
                g_eq, _ = comp.companion_model(dt, method_str)
                sig.append(g_eq)
            elif tag == _TAG_BJT:
                sig.extend((comp.operating_vbe, comp.operating_vbc))
            elif tag == _TAG_MOSFET:
                sig.extend((comp.operating_vg, comp.operating_vd, comp.operating_vs))
            elif tag == _TAG_OPAMP:
                sig.append(comp.operating_vd)
            elif tag == _TAG_LOGIC_GATE:
                sig.extend((comp.g_in, comp.g_out))
            elif tag in (_TAG_MCU, _TAG_SEQUENTIAL):
                sig.extend(comp.pin_conductance(p) for p in comp.pin_names)
            elif tag == _TAG_NE555:
                sig.extend(
                    (
                        1.0 if comp.output_high else 0.0,
                        comp.r_div,
                        comp.r_out,
                        comp.r_dis_on,
                    )
                )
            elif tag == _TAG_FUSE:
                sig.append(comp.conductance)
            elif tag == _TAG_TRANSFORMER:
                sig.extend((comp.turns_ratio, comp.r_winding_ohm, comp.l_primary_h, comp.coupling_k))
            elif tag == _TAG_RELAY:
                sig.extend((1.0 if comp.energized else 0.0, comp.coil_resistance_ohm))
            elif tag == _TAG_POTENTIOMETER:
                sig.extend((comp.g_aw, comp.g_wb))
            elif tag <= _TAG_AC_VS:
                sig.append(comp.r_internal)
        return tuple(sig)

    def _build_mna_matrix(
        self,
        *,
        mode: str,
        dt: float,
        method_str: str,
        eff_gmin: float,
        effective_loop_reg_r: float,
    ) -> np.ndarray:
        non_gnd_nodes = self.circuit.non_ground_nodes()
        self._node_to_idx = {node: idx for idx, node in enumerate(non_gnd_nodes)}

        voltage_sources = self._voltage_sources()
        n_nodes = len(non_gnd_nodes)
        n_vs = len(voltage_sources)
        self._vs_to_idx = {
            comp.name: n_nodes + k for k, comp in enumerate(voltage_sources)
        }

        dim = n_nodes + n_vs
        matrix_a = np.zeros((dim, dim), dtype=np.float64)

        # 1. Stamp GMIN from every non-ground node to ground.
        for idx in range(n_nodes):
            matrix_a[idx, idx] += eff_gmin

        # 2. Stamp conductive and companion two-terminal components.
        for comp in self.circuit.flat_components():
            if isinstance(comp, Ground):
                continue
            if isinstance(comp, (Resistor, Fuse, Lightbulb, Voltmeter, Ammeter)):
                if isinstance(comp, (Voltmeter, Ammeter)) and comp.is_ideal:
                    continue
                self._stamp_conductance(
                    matrix_a, comp.nodes[0], comp.nodes[1], comp.conductance
                )
            elif isinstance(comp, LED):
                g_led, _ = comp.companion_model(
                    dt if mode != "dc" else 0.0,
                    method_str if mode != "dc" else "dc",
                )
                self._stamp_conductance(matrix_a, comp.nodes[0], comp.nodes[1], g_led)
            elif isinstance(comp, Diode):
                g_d, _ = comp.companion_model(
                    dt if mode != "dc" else 0.0,
                    method_str if mode != "dc" else "dc",
                )
                self._stamp_conductance(matrix_a, comp.nodes[0], comp.nodes[1], g_d)
            elif isinstance(comp, (BJTBase, MOSFETBase, OpAmp, NE555, Transformer, SPDTRelay)):
                g_mat, _ = comp.companion_stamp()
                self._stamp_small_matrix(matrix_a, self._node_to_idx, comp.nodes, g_mat)
            elif isinstance(comp, Switch):
                if mode == "dc":
                    g_sw = comp.conductance
                else:
                    g_sw, _ = comp.companion_model(dt, method_str)
                self._stamp_conductance(matrix_a, comp.nodes[0], comp.nodes[1], g_sw)
            elif isinstance(comp, Capacitor):
                if mode != "dc":
                    g_eq, _ = comp.companion_model(dt, method_str)
                    self._stamp_conductance(
                        matrix_a, comp.nodes[0], comp.nodes[1], g_eq
                    )
            elif isinstance(comp, Potentiometer):
                self._stamp_conductance(matrix_a, comp.node_a, comp.node_w, comp.g_aw)
                self._stamp_conductance(matrix_a, comp.node_w, comp.node_b, comp.g_wb)
            elif isinstance(comp, LogicGate):
                for in_node in comp.input_nodes:
                    idx_in = self._node_to_idx.get(in_node)
                    if idx_in is not None:
                        matrix_a[idx_in, idx_in] += comp.g_in
                idx_out = self._node_to_idx.get(comp.output_node)
                if idx_out is not None:
                    matrix_a[idx_out, idx_out] += comp.g_out
            elif isinstance(comp, (MCUBlock, DigitalSequentialBlock)):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    idx_p = self._node_to_idx.get(p_node)
                    if idx_p is not None:
                        matrix_a[idx_p, idx_p] += comp.pin_conductance(pin)
            elif isinstance(comp, Inductor):
                if mode == "dc":
                    self._stamp_conductance(
                        matrix_a, comp.nodes[0], comp.nodes[1], comp.dc_conductance
                    )
                else:
                    g_eq, _ = comp.companion_model(dt, method_str)
                    self._stamp_conductance(
                        matrix_a, comp.nodes[0], comp.nodes[1], g_eq
                    )

        # 3. Stamp voltage sources (KCL incidence + branch KVL equation).
        for comp in voltage_sources:
            row_k = self._vs_to_idx[comp.name]
            idx_pos = self._node_to_idx.get(comp.nodes[0])
            idx_neg = self._node_to_idx.get(comp.nodes[1])

            if idx_pos is not None:
                matrix_a[idx_pos, row_k] += 1.0
                matrix_a[row_k, idx_pos] += 1.0
            if idx_neg is not None:
                matrix_a[idx_neg, row_k] -= 1.0
                matrix_a[row_k, idx_neg] -= 1.0

            if isinstance(comp, Ammeter) and getattr(comp, "is_ideal", False):
                r_int = comp.r_internal if self.circuit.is_ammeter_in_parallel(comp) else 0.0
            else:
                r_int = getattr(comp, "r_internal", 0.0)
            if r_int > 0.0:
                matrix_a[row_k, row_k] += r_int
            elif effective_loop_reg_r > 0.0:
                matrix_a[row_k, row_k] += effective_loop_reg_r

        return matrix_a

    def _assemble_and_factorize(
        self,
        *,
        mode: str,
        dt: float = 0.0,
        method_str: str = "trapezoidal",
        extra_gmin: float = 0.0,
        loop_reg_r: float = 0.0,
    ) -> None:
        """Assemble the MNA matrix A (for DC or transient) and compute its LU factors."""
        eff_gmin = max(self.gmin, extra_gmin)
        eff_loop_reg = loop_reg_r if loop_reg_r > 0.0 else (1e-6 if self.regularize_ideal_loops else 0.0)

        matrix_a = self._build_mna_matrix(
            mode=mode,
            dt=dt,
            method_str=method_str,
            eff_gmin=eff_gmin,
            effective_loop_reg_r=eff_loop_reg,
        )
        dim = matrix_a.shape[0]
        if dim > 0:
            needs_auto_reg = False
            try:
                lu_piv = lu_factor(matrix_a, check_finite=True)
                u_diag = np.diag(lu_piv[0])
                if not np.all(np.isfinite(u_diag)) or float(np.min(np.abs(u_diag))) <= 1e-14:
                    needs_auto_reg = True
            except (ValueError, np.linalg.LinAlgError) as exc:
                lu_piv = None
                needs_auto_reg = True
                orig_exc = exc
            else:
                orig_exc = None

            if needs_auto_reg and eff_loop_reg <= 0.0:
                reg_gmin = max(eff_gmin, 1e-9)
                reg_r = 1e-6
                matrix_reg = self._build_mna_matrix(
                    mode=mode,
                    dt=dt,
                    method_str=method_str,
                    eff_gmin=reg_gmin,
                    effective_loop_reg_r=reg_r,
                )
                try:
                    lu_piv = lu_factor(matrix_reg, check_finite=True)
                    u_diag2 = np.diag(lu_piv[0])
                    if not np.all(np.isfinite(u_diag2)) or float(np.min(np.abs(u_diag2))) <= 1e-14:
                        raise SingularMatrixError(
                            "MNA matrix remains singular after ideal loop regularization."
                        )
                    matrix_a = matrix_reg
                    eff_gmin = reg_gmin
                    eff_loop_reg = reg_r
                    self.last_convergence_method = "regularized"
                    if not any(w.code == "ideal_loop_regularized" for w in self._auto_regularized_warnings):
                        self._auto_regularized_warnings.append(
                            CircuitWarning(
                                code="ideal_loop_regularized",
                                message="Singular or ideal loop branch was regularized with 1 uOhm / 1 nS stabilizing elements.",
                                component_names=(),
                            )
                        )
                except (ValueError, np.linalg.LinAlgError) as exc2:
                    raise SingularMatrixError(
                        f"MNA matrix is singular and could not be factorized: {exc2}"
                    ) from exc2
            elif needs_auto_reg:
                if orig_exc is not None:
                    raise SingularMatrixError(
                        f"MNA matrix is singular and could not be factorized: {orig_exc}"
                    ) from orig_exc
                raise SingularMatrixError("MNA matrix has a zero pivot after regularization.")

            assert lu_piv is not None
            self._matrix_a = matrix_a
            self._lu_piv = lu_piv
        else:
            self._matrix_a = matrix_a
            self._lu_piv = (
                np.zeros((0, 0), dtype=np.float64),
                np.zeros(0, dtype=np.int32),
            )

        self._cached_mode = mode
        self._cached_circuit_version = self.circuit.version
        if mode == "dc":
            self._cached_conductance_signature = self._dc_conductance_signature(
                extra_gmin=eff_gmin, loop_reg_r=eff_loop_reg
            )
        else:
            self._cached_conductance_signature = self._transient_conductance_signature(
                dt, method_str, extra_gmin=eff_gmin, loop_reg_r=eff_loop_reg
            )

    def _build_rhs_dc(self, *, source_scale: float = 1.0) -> np.ndarray:
        """Assemble the right-hand side excitation vector b for DC operating point."""
        dim = len(self._node_to_idx) + len(self._vs_to_idx)
        rhs_b = np.zeros(dim, dtype=np.float64)

        for comp in self.circuit.flat_components():
            if isinstance(comp, DCCurrentSource):
                idx_pos = self._node_to_idx.get(comp.nodes[0])
                idx_neg = self._node_to_idx.get(comp.nodes[1])
                i_dc = comp.dc_current * source_scale
                if idx_pos is not None:
                    rhs_b[idx_pos] += i_dc
                if idx_neg is not None:
                    rhs_b[idx_neg] -= i_dc
            elif isinstance(comp, DCVoltageSource):
                row_k = self._vs_to_idx[comp.name]
                rhs_b[row_k] = comp.dc_voltage * source_scale
            elif isinstance(comp, ACVoltageSource):
                row_k = self._vs_to_idx[comp.name]
                rhs_b[row_k] = comp.voltage_at(0.0) * source_scale
            elif isinstance(comp, Ammeter) and comp.is_ideal:
                row_k = self._vs_to_idx[comp.name]
                rhs_b[row_k] = 0.0
            elif isinstance(comp, LED):
                _, i_eq = comp.companion_model(0.0, "dc")
                idx_a = self._node_to_idx.get(comp.nodes[0])
                idx_k = self._node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] += i_eq
                if idx_k is not None:
                    rhs_b[idx_k] -= i_eq
            elif isinstance(comp, Diode):
                _, i_eq = comp.companion_model(0.0, "dc")
                idx_a = self._node_to_idx.get(comp.nodes[0])
                idx_k = self._node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] += i_eq
                if idx_k is not None:
                    rhs_b[idx_k] -= i_eq
            elif isinstance(comp, (BJTBase, MOSFETBase, OpAmp)):
                _, rhs_3 = comp.companion_stamp()
                self._stamp_small_rhs(rhs_b, self._node_to_idx, comp.nodes, rhs_3)
            elif isinstance(comp, LogicGate):
                idx_out = self._node_to_idx.get(comp.output_node)
                if idx_out is not None:
                    rhs_b[idx_out] += comp.g_out * comp.drive_voltage * source_scale
            elif isinstance(comp, (MCUBlock, DigitalSequentialBlock)):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    if comp.is_output_pin(pin):
                        idx_p = self._node_to_idx.get(p_node)
                        if idx_p is not None:
                            rhs_b[idx_p] += comp.pin_conductance(pin) * comp.drive_voltage(pin) * source_scale

        return rhs_b

    def _build_rhs_transient(
        self, sim_time: float, dt: float, method_str: str, *, source_scale: float = 1.0
    ) -> np.ndarray:
        """Assemble the right-hand side vector b for a transient step at `sim_time`."""
        self._ensure_component_buckets()
        dim = len(self._node_to_idx) + len(self._vs_to_idx)
        rhs_b = np.zeros(dim, dtype=np.float64)
        node_to_idx = self._node_to_idx

        for comp, tag in self._bucket_rhs_comps:
            if tag == _TAG_DC_IS:
                idx_pos = node_to_idx.get(comp.nodes[0])
                idx_neg = node_to_idx.get(comp.nodes[1])
                i_dc = comp.dc_current * source_scale
                if idx_pos is not None:
                    rhs_b[idx_pos] += i_dc
                if idx_neg is not None:
                    rhs_b[idx_neg] -= i_dc
            elif tag == _TAG_DC_VS or tag == _TAG_AC_VS:
                row_k = self._vs_to_idx[comp.name]
                rhs_b[row_k] = comp.voltage_at(sim_time) * source_scale
            elif tag == _TAG_AMMETER:
                if comp.is_ideal:
                    row_k = self._vs_to_idx[comp.name]
                    rhs_b[row_k] = 0.0
            elif tag == _TAG_LED:
                _, i_eq = comp.companion_model(dt, method_str)
                idx_a = node_to_idx.get(comp.nodes[0])
                idx_k = node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] += i_eq
                if idx_k is not None:
                    rhs_b[idx_k] -= i_eq
            elif tag == _TAG_DIODE:
                _, i_eq = comp.companion_model(dt, method_str)
                idx_a = node_to_idx.get(comp.nodes[0])
                idx_k = node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] += i_eq
                if idx_k is not None:
                    rhs_b[idx_k] -= i_eq
            elif tag in (_TAG_BJT, _TAG_MOSFET, _TAG_OPAMP):
                _, rhs_3 = comp.companion_stamp()
                self._stamp_small_rhs(rhs_b, node_to_idx, comp.nodes, rhs_3)
            elif tag == _TAG_SWITCH:
                if comp.c_parasitic > 0.0 or comp.arcing:
                    _, i_sw_eq = comp.companion_model(dt, method_str)
                    idx_a = node_to_idx.get(comp.nodes[0])
                    idx_b = node_to_idx.get(comp.nodes[1])
                    if idx_a is not None:
                        rhs_b[idx_a] += i_sw_eq
                    if idx_b is not None:
                        rhs_b[idx_b] -= i_sw_eq
            elif tag == _TAG_CAPACITOR:
                # i_ab(t_n) = G_eq * v_ab(t_n) - I_eq
                # KCL at node_a: +G_eq * v_ab - I_eq => move +I_eq to RHS at node_a
                _, i_eq = comp.companion_model(dt, method_str)
                idx_a = node_to_idx.get(comp.nodes[0])
                idx_b = node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] += i_eq
                if idx_b is not None:
                    rhs_b[idx_b] -= i_eq
            elif tag == _TAG_INDUCTOR:
                # i_ab(t_n) = G_eq * v_ab(t_n) + I_hist
                # KCL at node_a: +G_eq * v_ab + I_hist => move -I_hist to RHS at node_a
                _, i_hist = comp.companion_model(dt, method_str)
                idx_a = node_to_idx.get(comp.nodes[0])
                idx_b = node_to_idx.get(comp.nodes[1])
                if idx_a is not None:
                    rhs_b[idx_a] -= i_hist
                if idx_b is not None:
                    rhs_b[idx_b] += i_hist
            elif tag == _TAG_LOGIC_GATE:
                idx_out = node_to_idx.get(comp.output_node)
                if idx_out is not None:
                    rhs_b[idx_out] += comp.g_out * comp.drive_voltage * source_scale
            elif tag in (_TAG_MCU, _TAG_SEQUENTIAL):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    if comp.is_output_pin(pin):
                        idx_p = node_to_idx.get(p_node)
                        if idx_p is not None:
                            rhs_b[idx_p] += comp.pin_conductance(pin) * comp.drive_voltage(pin) * source_scale

        return rhs_b

    def _logic_gates(self) -> list[LogicGate]:
        self._ensure_component_buckets()
        return list(self._bucket_logic_gates)

    def _collect_gate_runtime_warnings(
        self,
        node_voltages: dict[str, float],
        *,
        dt: float,
        base_warnings: list[CircuitWarning],
    ) -> list[CircuitWarning]:
        """Augment/refine logic-gate topology warnings with solved voltages and currents."""
        gates = self._logic_gates()
        if not gates:
            return base_warnings

        warnings = list(base_warnings)
        gates_by_name = {g.name: g for g in gates}

        # Refine output contention and supply-short warnings with live current values.
        refined: list[CircuitWarning] = []
        for w in warnings:
            if w.code == "output_contention" and w.component_names:
                drivers = [gates_by_name[n] for n in w.component_names if n in gates_by_name]
                if len(drivers) >= 2:
                    v_out = node_voltages.get(drivers[0].output_node, 0.0)
                    max_drive = max(g.drive_voltage for g in drivers)
                    min_drive = min(g.drive_voltage for g in drivers)
                    i_peak = max(
                        abs((g.drive_voltage - v_out) * g.g_out) for g in drivers
                    )
                    if max_drive - min_drive > 0.5:
                        refined.append(
                            CircuitWarning(
                                code="output_contention",
                                message=(
                                    f"Logic output contention on net {drivers[0].output_node}: "
                                    f"gates {', '.join(w.component_names)} drive opposing levels "
                                    f"(measured contention current {i_peak * 1e3:.1f} mA)"
                                ),
                                component_names=w.component_names,
                            )
                        )
                        continue
            elif w.code == "output_shorted_to_supply" and w.component_names:
                g0 = gates_by_name.get(w.component_names[0])
                if g0 is not None:
                    v_out = node_voltages.get(g0.output_node, 0.0)
                    i_short = abs((g0.drive_voltage - v_out) * g0.g_out)
                    target = (
                        f"voltage source ({', '.join(w.component_names[1:])})"
                        if len(w.component_names) > 1
                        else "Ground"
                    )
                    refined.append(
                        CircuitWarning(
                            code="output_shorted_to_supply",
                            message=(
                                f"Gate {g0.name} output is shorted to {target}: "
                                f"clamped at {v_out:.2f} V ({i_short * 1e3:.1f} mA short current)"
                            ),
                            component_names=w.component_names,
                        )
                    )
                    continue
            refined.append(w)

        # Detect inputs lingering strictly inside (V_IL, V_IH).
        for gate in gates:
            bad_pins = gate.indeterminate_pins(node_voltages, dt=dt)
            for pin_name, v_pin in bad_pins:
                refined.append(
                    CircuitWarning(
                        code="indeterminate_logic_input",
                        message=(
                            f"Indeterminate logic input on {gate.name}.{pin_name}: "
                            f"measured {v_pin:.2f} V is strictly between V_IL ({gate.v_il:g} V) "
                            f"and V_IH ({gate.v_ih:g} V) for {gate.family}"
                        ),
                        component_names=(gate.name,),
                    )
                )

        return refined

    def _relax_logic_gates_dc(
        self, *, source_scale: float = 1.0
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run Gauss-Seidel relaxation over `LogicGate`s using the cached DC LU matrix.

        Combinational DAGs and bistable latches converge monotonically to an exact fixed
        point in at most `len(gates) + 1` sweeps. Astable ring oscillators trigger
        cycle detection once a previously seen logic-state signature recurs and exit
        cleanly in a symmetry-broken single-wave state.
        """
        assert self._lu_piv is not None
        assert self._matrix_a is not None
        gates = self._logic_gates()
        dim = self._matrix_a.shape[0]

        def solve_once() -> tuple[np.ndarray, dict[str, float]]:
            if dim > 0:
                rhs_b = self._build_rhs_dc(source_scale=source_scale)
                sol_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                if not np.all(np.isfinite(sol_x)):
                    raise SingularMatrixError("MNA solve produced non-finite values.")
            else:
                sol_x = np.zeros(0, dtype=np.float64)
            nv: dict[str, float] = {g: 0.0 for g in self.circuit.grounded_nodes()}
            for node, idx in self._node_to_idx.items():
                nv[node] = float(sol_x[idx])
            return sol_x, nv

        if not gates and not self._bucket_mcus:
            return solve_once()

        seen_states: set[tuple[object, ...]] = {
            (
                tuple(g.output_level for g in gates),
                tuple(
                    round(m.drive_voltage(p), 6)
                    for m in self._bucket_mcus
                    for p in m.pin_names
                    if m.is_output_pin(p)
                ),
            )
        }
        max_sweeps = max(2 * (len(gates) + len(self._bucket_mcus)) + 4, 8)
        for _ in range(max_sweeps):
            any_changed = False
            for gate in gates:
                _, nv = solve_once()
                target = gate.evaluate_inputs(nv, dt=0.0)
                if target != gate.output_level:
                    gate.set_steady_state_level(target)
                    any_changed = True
            for mcu in self._bucket_mcus:
                _, nv = solve_once()
                if mcu.evaluate_firmware(nv, 0.0, 0.0):
                    any_changed = True
            if not any_changed:
                break
            sig = (
                tuple(g.output_level for g in gates),
                tuple(
                    round(m.drive_voltage(p), 6)
                    for m in self._bucket_mcus
                    for p in m.pin_names
                    if m.is_output_pin(p)
                ),
            )
            if sig in seen_states:
                break
            seen_states.add(sig)

        return solve_once()

    def _initialize_logic_gates_t0(self, step_dt: float) -> None:
        """Settle initial t = 0+ logic gate & MCU states and queue any initial ring edges."""
        self._logic_t0_initialized = True
        self._ensure_component_buckets()
        gates = self._logic_gates()
        mcus = self._bucket_mcus
        if not gates and not mcus:
            self._t0_node_voltages = {}
            return

        for gate in gates:
            gate.reset_state()
        for mcu in mcus:
            mcu.reset_state()

        capacitors = [c for c in self.circuit.flat_components() if isinstance(c, Capacitor)]
        inductors = [c for c in self.circuit.flat_components() if isinstance(c, Inductor)]
        non_gnd_nodes = self.circuit.non_ground_nodes()
        node_to_idx = {node: idx for idx, node in enumerate(non_gnd_nodes)}
        voltage_sources = self._voltage_sources()

        n_nodes = len(non_gnd_nodes)
        n_vs = len(voltage_sources)
        n_caps = len(capacitors)
        dim = n_nodes + n_vs + n_caps

        if dim == 0:
            self._t0_node_voltages = {g: 0.0 for g in self.circuit.grounded_nodes()}
            for gate in gates:
                gate.sample_inputs_and_schedule(self._t0_node_voltages, 0.0, step_dt)
            for mcu in mcus:
                mcu.evaluate_firmware(self._t0_node_voltages, 0.0, step_dt)
            return

        mat = np.zeros((dim, dim), dtype=np.float64)
        base_rhs = np.zeros(dim, dtype=np.float64)

        for idx in range(n_nodes):
            mat[idx, idx] += max(self.gmin, 1e-15)

        for comp in self.circuit.flat_components():
            if isinstance(comp, (Resistor, Switch, Lightbulb, Voltmeter, Ammeter)):
                if isinstance(comp, (Voltmeter, Ammeter)) and getattr(comp, "is_ideal", False):
                    continue
                g = comp.conductance
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    mat[ia, ia] += g
                if ib is not None:
                    mat[ib, ib] += g
                if ia is not None and ib is not None:
                    mat[ia, ib] -= g
                    mat[ib, ia] -= g
            elif isinstance(comp, LED):
                g_led, i_eq = comp.companion_model(0.0, "dc")
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    mat[ia, ia] += g_led
                    base_rhs[ia] += i_eq
                if ib is not None:
                    mat[ib, ib] += g_led
                    base_rhs[ib] -= i_eq
                if ia is not None and ib is not None:
                    mat[ia, ib] -= g_led
                    mat[ib, ia] -= g_led
            elif isinstance(comp, Diode):
                g_d, i_eq = comp.companion_model(0.0, "dc")
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    mat[ia, ia] += g_d
                    base_rhs[ia] += i_eq
                if ib is not None:
                    mat[ib, ib] += g_d
                    base_rhs[ib] -= i_eq
                if ia is not None and ib is not None:
                    mat[ia, ib] -= g_d
                    mat[ib, ia] -= g_d
            elif isinstance(comp, (BJTBase, MOSFETBase, OpAmp)):
                g_mat, rhs_3 = comp.companion_stamp()
                self._stamp_small_matrix(mat, node_to_idx, comp.nodes, g_mat)
                self._stamp_small_rhs(base_rhs, node_to_idx, comp.nodes, rhs_3)
            elif isinstance(comp, Potentiometer):
                for na, nb, g in (
                    (comp.node_a, comp.node_w, comp.g_aw),
                    (comp.node_w, comp.node_b, comp.g_wb),
                ):
                    ia = node_to_idx.get(na)
                    ib = node_to_idx.get(nb)
                    if ia is not None:
                        mat[ia, ia] += g
                    if ib is not None:
                        mat[ib, ib] += g
                    if ia is not None and ib is not None:
                        mat[ia, ib] -= g
                        mat[ib, ia] -= g
            elif isinstance(comp, LogicGate):
                for in_node in comp.input_nodes:
                    ii = node_to_idx.get(in_node)
                    if ii is not None:
                        mat[ii, ii] += comp.g_in
                io = node_to_idx.get(comp.output_node)
                if io is not None:
                    mat[io, io] += comp.g_out
            elif isinstance(comp, MCUBlock):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    ip = node_to_idx.get(p_node)
                    if ip is not None:
                        mat[ip, ip] += comp.pin_conductance(pin)
            elif isinstance(comp, DCCurrentSource):
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    base_rhs[ia] += comp.dc_current
                if ib is not None:
                    base_rhs[ib] -= comp.dc_current
            elif isinstance(comp, Inductor):
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    base_rhs[ia] -= comp.ind_current
                if ib is not None:
                    base_rhs[ib] += comp.ind_current

        for k, vs in enumerate(voltage_sources):
            row_k = n_nodes + k
            ia = node_to_idx.get(vs.nodes[0])
            ib = node_to_idx.get(vs.nodes[1])
            if ia is not None:
                mat[ia, row_k] += 1.0
                mat[row_k, ia] += 1.0
            if ib is not None:
                mat[ib, row_k] -= 1.0
                mat[row_k, ib] -= 1.0
            if isinstance(vs, Ammeter) and getattr(vs, "is_ideal", False):
                r_int = vs.r_internal if self.circuit.is_ammeter_in_parallel(vs) else 0.0
            else:
                r_int = getattr(vs, "r_internal", 0.0)
            if r_int > 0.0:
                mat[row_k, row_k] += r_int
            base_rhs[row_k] = vs.voltage_at(0.0) if hasattr(vs, "voltage_at") else 0.0

        for k, cap in enumerate(capacitors):
            row_k = n_nodes + n_vs + k
            ia = node_to_idx.get(cap.nodes[0])
            ib = node_to_idx.get(cap.nodes[1])
            if ia is not None:
                mat[ia, row_k] += 1.0
                mat[row_k, ia] += 1.0
            if ib is not None:
                mat[ib, row_k] -= 1.0
                mat[row_k, ib] -= 1.0
            if cap.esr > 0.0:
                mat[row_k, row_k] -= cap.esr
            base_rhs[row_k] = cap.cap_voltage

        try:
            t0_lu = lu_factor(mat, check_finite=True)
        except (ValueError, np.linalg.LinAlgError):
            return

        def solve_t0() -> dict[str, float]:
            rhs = base_rhs.copy()
            for gate in gates:
                io = node_to_idx.get(gate.output_node)
                if io is not None:
                    rhs[io] += gate.g_out * gate.drive_voltage
            for mcu in mcus:
                for pin, p_node in zip(mcu.pin_names, mcu.nodes):
                    if mcu.is_output_pin(pin):
                        ip = node_to_idx.get(p_node)
                        if ip is not None:
                            rhs[ip] += mcu.pin_conductance(pin) * mcu.drive_voltage(pin)
            sol = lu_solve(t0_lu, rhs, check_finite=True)
            nv = {g: 0.0 for g in self.circuit.grounded_nodes()}
            for node, idx in node_to_idx.items():
                nv[node] = float(sol[idx])
            return nv

        seen_states: set[tuple[object, ...]] = {
            (
                tuple(g.output_level for g in gates),
                tuple(
                    round(m.drive_voltage(p), 6)
                    for m in mcus
                    for p in m.pin_names
                    if m.is_output_pin(p)
                ),
            )
        }
        max_sweeps = max(2 * (len(gates) + len(mcus)) + 4, 8)
        for _ in range(max_sweeps):
            any_changed = False
            for gate in gates:
                nv = solve_t0()
                target = gate.evaluate_inputs(nv, dt=0.0)
                if target != gate.output_level:
                    gate.set_steady_state_level(target)
                    any_changed = True
            for mcu in mcus:
                nv = solve_t0()
                if mcu.evaluate_firmware(nv, 0.0, 0.0):
                    any_changed = True
            if not any_changed:
                break
            sig = (
                tuple(g.output_level for g in gates),
                tuple(
                    round(m.drive_voltage(p), 6)
                    for m in mcus
                    for p in m.pin_names
                    if m.is_output_pin(p)
                ),
            )
            if sig in seen_states:
                break
            seen_states.add(sig)

        node_v0 = solve_t0()
        self._t0_node_voltages = node_v0
        for gate in gates:
            v_out = node_v0.get(gate.output_node, 0.0)
            i_out = (gate.drive_voltage - v_out) * gate.g_out
            p_out = (gate.drive_voltage - v_out) * i_out
            p_in = sum((node_v0.get(n, 0.0) ** 2) * gate.g_in for n in gate.input_nodes)
            gate.voltage = v_out
            gate.current = i_out
            gate.power = p_out + p_in
            gate.sample_inputs_and_schedule(node_v0, 0.0, step_dt)
        for mcu in mcus:
            self._update_mcu_telemetry(mcu, node_v0)
            mcu.evaluate_firmware(node_v0, 0.0, step_dt)

    def _maybe_apply_startup_perturbation(self) -> None:
        """Inject a deterministic +1 uV perturbation once at t = 0+ for symmetric oscillators."""
        if self._startup_perturbation_applied or self.startup_perturbation_uv == 0.0:
            return
        caps = sorted(
            (c for c in self.circuit.flat_components() if isinstance(c, Capacitor)),
            key=lambda c: c.name,
        )
        if not caps:
            return
        inds = [c for c in self.circuit.flat_components() if isinstance(c, Inductor)]
        gates = [c for c in self.circuit.flat_components() if isinstance(c, LogicGate)]
        has_ac = any(isinstance(c, ACVoltageSource) for c in self.circuit.flat_components())
        has_diode = any(isinstance(c, Diode) for c in self.circuit.flat_components())
        has_active_nl = any(
            isinstance(c, (BJTBase, MOSFETBase, OpAmp)) for c in self.circuit.flat_components()
        )
        all_zero_ic = all(abs(c.initial_voltage) == 0.0 for c in caps) and all(
            abs(i.initial_current) == 0.0 for i in inds
        )
        if all_zero_ic and (
            len(gates) >= 1
            or (not has_ac and (len(caps) >= 2 or len(inds) >= 1 or has_diode or has_active_nl))
        ):
            caps[0].cap_voltage += self.startup_perturbation_uv * 1e-6
            self._startup_perturbation_applied = True

    @staticmethod
    def _update_mcu_telemetry(mcu: MCUBlock, node_voltages: dict[str, float]) -> float:
        """Update MCU voltage/current/power telemetry and return primary digital drive current."""
        mcu.sense_nodes(node_voltages)
        total_power = 0.0
        primary_i = 0.0
        for pin, p_node in zip(mcu.pin_names, mcu.nodes):
            v_p = node_voltages.get(p_node, 0.0)
            g_p = mcu.pin_conductance(pin)
            if mcu.is_output_pin(pin):
                i_out = (mcu.drive_voltage(pin) - v_p) * g_p
                total_power += abs((mcu.drive_voltage(pin) - v_p) * i_out)
                if pin == "D0":
                    primary_i = i_out
            else:
                total_power += (v_p**2) * g_p
        mcu.voltage = node_voltages.get(mcu.nodes[2], 0.0)  # D0 output node voltage
        mcu.current = primary_i
        mcu.power = total_power
        return primary_i

    @staticmethod
    def _update_sequential_telemetry(seq: DigitalSequentialBlock, node_voltages: dict[str, float]) -> float:
        """Update sequential digital / 7-seg telemetry and return primary output current."""
        total_power = 0.0
        primary_i = 0.0
        primary_v = node_voltages.get(seq.nodes[0], 0.0) if seq.nodes else 0.0
        first_out_seen = False
        for pin, p_node in zip(seq.pin_names, seq.nodes):
            v_p = node_voltages.get(p_node, 0.0)
            g_p = seq.pin_conductance(pin)
            if seq.is_output_pin(pin):
                i_out = (seq.drive_voltage(pin) - v_p) * g_p
                total_power += abs((seq.drive_voltage(pin) - v_p) * i_out)
                if not first_out_seen:
                    first_out_seen = True
                    primary_i = i_out
                    primary_v = v_p
            else:
                total_power += (v_p**2) * g_p
        seq.voltage = primary_v
        seq.current = primary_i
        seq.power = total_power
        return primary_i

    @staticmethod
    def _update_bjt_telemetry(bjt: BJTBase, node_voltages: dict[str, float]) -> float:
        """Update BJT terminal voltages, currents `(i_b, i_c, i_e)`, and total power."""
        vb = node_voltages.get(bjt.node_b, 0.0)
        vc = node_voltages.get(bjt.node_c, 0.0)
        ve = node_voltages.get(bjt.node_e, 0.0)
        s = float(bjt._POLARITY)
        bjt.operating_vbe = s * (vb - ve)
        bjt.operating_vbc = s * (vb - vc)
        ib, ic, ie = bjt.terminal_currents_at(vb, vc, ve)
        bjt.i_b = ib
        bjt.i_c = ic
        bjt.i_e = ie
        bjt.voltage = vc - ve
        bjt.current = ic
        bjt.power = abs((vc - ve) * ic + (vb - ve) * ib)
        return ic

    @staticmethod
    def _update_mosfet_telemetry(mosfet: MOSFETBase, node_voltages: dict[str, float]) -> float:
        """Update MOSFET terminal operating voltages, drain current `i_d`, and power."""
        vg = node_voltages.get(mosfet.node_g, 0.0)
        vd = node_voltages.get(mosfet.node_d, 0.0)
        vs = node_voltages.get(mosfet.node_s, 0.0)
        mosfet.operating_vg = vg
        mosfet.operating_vd = vd
        mosfet.operating_vs = vs
        i_d = mosfet.drain_current_at(vg, vd, vs)
        mosfet.i_d = i_d
        mosfet.voltage = vd - vs
        mosfet.current = i_d
        mosfet.power = abs((vd - vs) * i_d)
        return i_d

    @staticmethod
    def _update_opamp_telemetry(opamp: OpAmp, node_voltages: dict[str, float]) -> float:
        """Update Op-Amp differential input state, output current, and power."""
        vp = node_voltages.get(opamp.node_in_pos, 0.0)
        vn = node_voltages.get(opamp.node_in_neg, 0.0)
        vo = node_voltages.get(opamp.node_out, 0.0)
        vd = vp - vn
        opamp.operating_vd = vd
        v_target, _ = opamp.eval_transfer(vd)
        i_out = (v_target - vo) / opamp.r_out
        i_in = vd / opamp.r_in
        opamp.voltage = vo
        opamp.current = i_out
        opamp.power = abs((v_target - vo) * i_out) + abs(vd * i_in)
        return i_out

    @staticmethod
    def _update_ne555_telemetry(ic: NE555, node_voltages: dict[str, float]) -> float:
        """Update NE555 timer terminal telemetry and return output pin current."""
        vcc = node_voltages.get(ic.node_vcc, 0.0)
        vgnd = node_voltages.get(ic.node_gnd, 0.0)
        vout = node_voltages.get(ic.node_out, 0.0)
        vdis = node_voltages.get(ic.node_dis, 0.0)
        v_target = vcc if ic.output_high else vgnd
        i_out = (v_target - vout) / ic.r_out
        r_dis = ic.r_dis_off if ic.output_high else ic.r_dis_on
        i_dis = (vdis - vgnd) / r_dis
        p_div = ((vcc - vgnd) ** 2) / (3.0 * ic.r_div)
        ic.voltage = vout - vgnd
        ic.current = i_out
        ic.power = abs(i_out * (v_target - vout)) + abs(i_dis * (vdis - vgnd)) + p_div
        return i_out

    @staticmethod
    def _update_transformer_telemetry(xfmr: Transformer, node_voltages: dict[str, float]) -> float:
        """Update coupled transformer primary/secondary voltages, secondary current, and loss."""
        vp = node_voltages.get(xfmr.node_p1, 0.0) - node_voltages.get(xfmr.node_p2, 0.0)
        vs = node_voltages.get(xfmr.node_s1, 0.0) - node_voltages.get(xfmr.node_s2, 0.0)
        n = xfmr.turns_ratio * xfmr.coupling_k
        g_w = 1.0 / max(xfmr.r_winding_ohm, 1e-4)
        i_sec = (n * vp - vs) * g_w
        xfmr.voltage = vs
        xfmr.current = i_sec
        xfmr.power = abs((n * vp - vs) * i_sec)
        return i_sec

    @staticmethod
    def _update_relay_telemetry(relay: SPDTRelay, node_voltages: dict[str, float]) -> float:
        """Update SPDT relay coil voltage, contact current, and coil dissipation."""
        v_coil = node_voltages.get(relay.node_coil_pos, 0.0) - node_voltages.get(relay.node_coil_neg, 0.0)
        i_coil = v_coil / relay.coil_resistance_ohm
        v_com = node_voltages.get(relay.node_com, 0.0)
        v_active = node_voltages.get(relay.node_no if relay.energized else relay.node_nc, 0.0)
        i_contact = (v_com - v_active) / relay.r_on
        relay.voltage = v_coil
        relay.current = i_contact
        relay.power = abs(v_coil * i_coil) + abs((v_com - v_active) * i_contact)
        return i_contact

    def _update_led_discrete_states(
        self, node_voltages: dict[str, float]
    ) -> bool:
        """Advance piecewise LED conduction states; return True if any changed."""
        state_changed = False
        for comp in self._bucket_leds:
            if not comp.burned_out:
                v_ak = node_voltages.get(comp.nodes[0], 0.0) - node_voltages.get(comp.nodes[1], 0.0)
                if comp.is_conducting:
                    i_ak = (v_ak - comp.forward_voltage) / comp.r_on
                    if i_ak < -1e-9 or v_ak < comp.forward_voltage - 1e-4:
                        comp.is_conducting = False
                        state_changed = True
                else:
                    if v_ak > comp.forward_voltage + 1e-4:
                        comp.is_conducting = True
                        state_changed = True
        return state_changed

    def _solve_nonlinear_core(
        self,
        *,
        mode: str,
        sim_time: float,
        dt: float,
        method_str: str,
        source_scale: float = 1.0,
        extra_gmin: float = 0.0,
        max_iters: int = 80,
        relax_gates: bool = False,
    ) -> tuple[np.ndarray, dict[str, float], bool, bool, bool]:
        """Solve one DC or transient step using damped Newton-Raphson."""
        self._ensure_component_buckets()
        diodes = self._bucket_diodes
        bjts = self._bucket_bjts
        mosfets = self._bucket_mosfets
        opamps = self._bucket_opamps
        leds = self._bucket_leds
        gates = self._bucket_logic_gates

        if not diodes and not bjts and not mosfets and not opamps:
            n_outer = 10 if leds else 1
        else:
            n_outer = max(1, max_iters)

        rebuilt_any = False
        damping_used = False
        prev_node_voltages: dict[str, float] | None = None
        grounded = self._bucket_grounded_nodes
        eff_gmin = max(self.gmin, extra_gmin)
        eff_loop_reg = 1e-6 if self.regularize_ideal_loops else 0.0

        for _it in range(n_outer):
            if mode == "dc":
                current_sig = self._dc_conductance_signature(
                    extra_gmin=eff_gmin, loop_reg_r=eff_loop_reg
                )
            else:
                current_sig = self._transient_conductance_signature(
                    dt, method_str, extra_gmin=eff_gmin, loop_reg_r=eff_loop_reg
                )

            needs_rebuild = (
                self._lu_piv is None
                or self._cached_mode != mode
                or self._cached_circuit_version != self.circuit.version
                or self._cached_conductance_signature != current_sig
            )
            if needs_rebuild:
                try:
                    self._assemble_and_factorize(
                        mode=mode,
                        dt=dt,
                        method_str=method_str,
                        extra_gmin=extra_gmin,
                    )
                    rebuilt_any = True
                except SingularMatrixError:
                    return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any, damping_used)

            assert self._lu_piv is not None
            assert self._matrix_a is not None
            dim = self._matrix_a.shape[0]

            try:
                if relax_gates and (gates or self._bucket_mcus):
                    solution_x, node_voltages = self._relax_logic_gates_dc(
                        source_scale=source_scale
                    )
                else:
                    if dim > 0:
                        if mode == "dc":
                            rhs_b = self._build_rhs_dc(source_scale=source_scale)
                        else:
                            rhs_b = self._build_rhs_transient(
                                sim_time, dt, method_str, source_scale=source_scale
                            )
                        solution_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                        if not np.all(np.isfinite(solution_x)):
                            return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any, damping_used)
                    else:
                        solution_x = np.zeros(0, dtype=np.float64)
                    node_voltages = {g: 0.0 for g in grounded}
                    for node, idx in self._node_to_idx.items():
                        node_voltages[node] = float(solution_x[idx])
            except (ValueError, np.linalg.LinAlgError, SingularMatrixError):
                return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any, damping_used)

            led_changed = self._update_led_discrete_states(node_voltages) if leds else False

            if not diodes and not bjts and not mosfets and not opamps:
                if not led_changed:
                    return (solution_x, node_voltages, True, rebuilt_any, damping_used)
                continue

            max_vd_err = 0.0
            any_limited = False
            for comp in diodes:
                v_ak = node_voltages.get(comp.nodes[0], 0.0) - node_voltages.get(
                    comp.nodes[1], 0.0
                )
                vd_sol, _, _ = comp.terminal_model(v_ak, vd_guess=comp.operating_vd)
                vd_old = comp.operating_vd
                max_vd_err = max(max_vd_err, abs(vd_sol - vd_old))
                vd_next, was_lim = comp.limit_junction_voltage(vd_sol, vd_old)
                if was_lim:
                    any_limited = True
                    damping_used = True
                comp.operating_vd = vd_next
                i_j = comp.junction_current(vd_next)
                comp.operating_vak = vd_next + comp.rs * i_j

            for bjt in bjts:
                s = float(bjt._POLARITY)
                vbe_sol = s * (
                    node_voltages.get(bjt.node_b, 0.0) - node_voltages.get(bjt.node_e, 0.0)
                )
                vbc_sol = s * (
                    node_voltages.get(bjt.node_b, 0.0) - node_voltages.get(bjt.node_c, 0.0)
                )
                max_vd_err = max(
                    max_vd_err,
                    abs(vbe_sol - bjt.operating_vbe),
                    abs(vbc_sol - bjt.operating_vbc),
                )
                vbe_next, lim1 = bjt.limit_junction_voltage(vbe_sol, bjt.operating_vbe)
                vbc_next, lim2 = bjt.limit_junction_voltage(vbc_sol, bjt.operating_vbc)
                if lim1 or lim2:
                    any_limited = True
                    damping_used = True
                bjt.operating_vbe = vbe_next
                bjt.operating_vbc = vbc_next

            for mosfet in mosfets:
                vg_sol = node_voltages.get(mosfet.node_g, 0.0)
                vd_sol = node_voltages.get(mosfet.node_d, 0.0)
                vs_sol = node_voltages.get(mosfet.node_s, 0.0)
                max_vd_err = max(
                    max_vd_err,
                    abs(vg_sol - mosfet.operating_vg),
                    abs(vd_sol - mosfet.operating_vd),
                    abs(vs_sol - mosfet.operating_vs),
                )
                vg_next, vd_next, vs_next, m_lim = mosfet.limit_terminal_voltages(
                    vg_sol, vd_sol, vs_sol
                )
                if m_lim:
                    any_limited = True
                    damping_used = True
                mosfet.operating_vg = vg_next
                mosfet.operating_vd = vd_next
                mosfet.operating_vs = vs_next

            for opamp in opamps:
                vd_sol = node_voltages.get(opamp.node_in_pos, 0.0) - node_voltages.get(
                    opamp.node_in_neg, 0.0
                )
                max_vd_err = max(max_vd_err, abs(vd_sol - opamp.operating_vd))
                vd_next, op_lim = opamp.limit_diff_voltage(vd_sol, opamp.operating_vd)
                if op_lim:
                    any_limited = True
                    damping_used = True
                opamp.operating_vd = vd_next

            if prev_node_voltages is not None:
                max_dv = max(
                    (abs(node_voltages[n] - prev_node_voltages[n]) for n in node_voltages),
                    default=0.0,
                )
                v_scale = max((abs(v) for v in node_voltages.values()), default=0.0)
                v_tol = 1e-8 + 1e-6 * v_scale
                if (
                    max_dv <= v_tol
                    and max_vd_err <= 1e-7
                    and not any_limited
                    and not led_changed
                ):
                    return (solution_x, node_voltages, True, rebuilt_any, damping_used)

            prev_node_voltages = node_voltages

        return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any, damping_used)

    def _snapshot_nonlinear_states(self) -> tuple[Any, ...]:
        diodes = self._bucket_diodes
        leds = self._bucket_leds
        bjts = self._bucket_bjts
        mosfets = self._bucket_mosfets
        opamps = self._bucket_opamps
        return (
            [d.operating_vd for d in diodes],
            [d.operating_vak for d in diodes],
            [bool(l.is_conducting) for l in leds],
            [(b.operating_vbe, b.operating_vbc) for b in bjts],
            [(m.operating_vg, m.operating_vd, m.operating_vs) for m in mosfets],
            [o.operating_vd for o in opamps],
        )

    def _restore_nonlinear_states(self, snap: tuple[Any, ...]) -> None:
        vds, vaks, led_states = snap[0], snap[1], snap[2]
        bjt_states = snap[3] if len(snap) > 3 else []
        mos_states = snap[4] if len(snap) > 4 else []
        op_states = snap[5] if len(snap) > 5 else []
        for d, vd, vak in zip(self._bucket_diodes, vds, vaks):
            d.operating_vd = vd
            d.operating_vak = vak
        for led, st in zip(self._bucket_leds, led_states):
            led.is_conducting = st
        for bjt, (vbe, vbc) in zip(self._bucket_bjts, bjt_states):
            bjt.operating_vbe = vbe
            bjt.operating_vbc = vbc
        for mos, (vg, vd, vs) in zip(self._bucket_mosfets, mos_states):
            mos.operating_vg = vg
            mos.operating_vd = vd
            mos.operating_vs = vs
        for op, vd in zip(self._bucket_opamps, op_states):
            op.operating_vd = vd

    def _reset_nonlinear_guess(self) -> None:
        for d in self._bucket_diodes:
            d.operating_vd = 0.0
            d.operating_vak = 0.0
        for led in self._bucket_leds:
            if not led.burned_out:
                led.is_conducting = False
        for bjt in self._bucket_bjts:
            bjt.operating_vbe = 0.0
            bjt.operating_vbc = 0.0
        for mos in self._bucket_mosfets:
            mos.operating_vg = 0.0
            mos.operating_vd = 0.0
            mos.operating_vs = 0.0
        for op in self._bucket_opamps:
            op.operating_vd = 0.0

    def _recover_via_gmin_stepping(
        self,
        *,
        mode: str,
        sim_time: float,
        dt: float,
        method_str: str,
        relax_gates: bool,
    ) -> tuple[np.ndarray, dict[str, float], bool, bool]:
        """Attempt Gmin-stepping homotopy from large shunt conductance down to self.gmin."""
        self._reset_nonlinear_guess()
        gmin_schedule = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 0.0)
        rebuilt_any = False
        sol_x = np.zeros(0, dtype=np.float64)
        nv: dict[str, float] = {}
        ok = False
        for g_extra in gmin_schedule:
            sol_x, nv, ok, reb, _ = self._solve_nonlinear_core(
                mode=mode,
                sim_time=sim_time,
                dt=dt,
                method_str=method_str,
                source_scale=1.0,
                extra_gmin=g_extra,
                max_iters=60,
                relax_gates=relax_gates,
            )
            rebuilt_any = rebuilt_any or reb
            if not ok:
                return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any)
        return (sol_x, nv, ok, rebuilt_any)

    def _recover_via_source_stepping(
        self,
        *,
        mode: str,
        sim_time: float,
        dt: float,
        method_str: str,
        relax_gates: bool,
    ) -> tuple[np.ndarray, dict[str, float], bool, bool]:
        """Attempt adaptive source-stepping continuation from alpha = 0 to 1."""
        self._reset_nonlinear_guess()
        rebuilt_any = False
        # Anchor at alpha = 0.0 first.
        sol_x, nv, ok0, reb0, _ = self._solve_nonlinear_core(
            mode=mode,
            sim_time=sim_time,
            dt=dt,
            method_str=method_str,
            source_scale=0.0,
            extra_gmin=0.0,
            max_iters=40,
            relax_gates=relax_gates,
        )
        rebuilt_any = rebuilt_any or reb0
        if not ok0:
            return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any)

        alpha = 0.0
        d_alpha = 0.1
        while alpha < 1.0 - 1e-12:
            snap = self._snapshot_nonlinear_states()
            alpha_next = min(1.0, alpha + d_alpha)
            sol_x, nv, ok, reb, _ = self._solve_nonlinear_core(
                mode=mode,
                sim_time=sim_time,
                dt=dt,
                method_str=method_str,
                source_scale=alpha_next,
                extra_gmin=0.0,
                max_iters=60,
                relax_gates=relax_gates,
            )
            rebuilt_any = rebuilt_any or reb
            if ok:
                alpha = alpha_next
                d_alpha = min(0.25, d_alpha * 1.5)
            else:
                self._restore_nonlinear_states(snap)
                d_alpha *= 0.5
                if d_alpha < 1e-4:
                    return (np.zeros(0, dtype=np.float64), {}, False, rebuilt_any)

        return (sol_x, nv, True, rebuilt_any)

    def _solve_step_with_recovery(
        self,
        *,
        mode: str,
        sim_time: float = 0.0,
        dt: float = 0.0,
        method_str: str = "dc",
        direct_max_iters: int = 80,
        recovery_mode: str = "auto",
        relax_gates: bool = False,
    ) -> tuple[np.ndarray, dict[str, float], bool]:
        """Solve nonlinear MNA step with Gmin and Source stepping recovery fallbacks."""
        self._auto_regularized_warnings = []
        if recovery_mode == "gmin":
            sol_x, nv, ok, reb = self._recover_via_gmin_stepping(
                mode=mode, sim_time=sim_time, dt=dt, method_str=method_str, relax_gates=relax_gates
            )
            if ok:
                self.last_convergence_method = "gmin_stepping"
                return (sol_x, nv, reb)
            raise SingularMatrixError("Gmin stepping failed to converge.")

        if recovery_mode == "source":
            sol_x, nv, ok, reb = self._recover_via_source_stepping(
                mode=mode, sim_time=sim_time, dt=dt, method_str=method_str, relax_gates=relax_gates
            )
            if ok:
                self.last_convergence_method = "source_stepping"
                return (sol_x, nv, reb)
            raise SingularMatrixError("Source stepping failed to converge.")

        initial_snap = self._snapshot_nonlinear_states()
        sol_x, nv, ok, reb1, damped = self._solve_nonlinear_core(
            mode=mode,
            sim_time=sim_time,
            dt=dt,
            method_str=method_str,
            source_scale=1.0,
            extra_gmin=0.0,
            max_iters=direct_max_iters,
            relax_gates=relax_gates,
        )
        if ok:
            if self._auto_regularized_warnings:
                self.last_convergence_method = "regularized"
            elif damped:
                self.last_convergence_method = "newton_damped"
            else:
                self.last_convergence_method = "direct"
            return (sol_x, nv, reb1)

        if recovery_mode == "direct":
            self._restore_nonlinear_states(initial_snap)
            raise SingularMatrixError("Direct Newton-Raphson did not converge.")

        sol_x, nv, ok_g, reb2 = self._recover_via_gmin_stepping(
            mode=mode, sim_time=sim_time, dt=dt, method_str=method_str, relax_gates=relax_gates
        )
        if ok_g:
            self.last_convergence_method = "gmin_stepping"
            return (sol_x, nv, reb1 or reb2)

        sol_x, nv, ok_s, reb3 = self._recover_via_source_stepping(
            mode=mode, sim_time=sim_time, dt=dt, method_str=method_str, relax_gates=relax_gates
        )
        if ok_s:
            self.last_convergence_method = "source_stepping"
            return (sol_x, nv, reb1 or reb2 or reb3)

        self._restore_nonlinear_states(initial_snap)
        raise SingularMatrixError(
            f"Solver failed to converge after Newton-Raphson, Gmin stepping, and Source stepping (mode={mode})."
        )

    def solve_dc(
        self,
        *,
        compute_cond: bool = False,
        direct_max_iters: int = 80,
        recovery_mode: str = "auto",
    ) -> SolveResult:
        """Solve the DC operating point of the circuit.

        Capacitors are treated as open circuits (keeping any configured initial
        voltage if isolated) and inductors are modeled using their winding
        resistance (`max(dcr, 1 mΩ)`).
        """
        warnings = self.circuit.validate_topology(
            regularize_ideal_loops=self.regularize_ideal_loops
        )
        self._topology_warnings = list(warnings)
        self._ensure_component_buckets()

        gates = self._logic_gates()
        for gate in gates:
            gate.reset_state()
        for mcu in self._bucket_mcus:
            mcu.reset_state()

        ne555_ics = self._bucket_ne555
        relays = self._bucket_relays
        fuses = self._bucket_fuses
        seq_blocks = self._bucket_sequential
        has_stateful_ics = bool(ne555_ics or relays or fuses or seq_blocks)
        rebuilt_any = False
        solution_x = np.zeros(0, dtype=np.float64)
        node_voltages: dict[str, float] = {}
        dim = 0

        for _ic_pass in range(6 if has_stateful_ics else 1):
            if (
                not self._bucket_diodes
                and not self._bucket_leds
                and not self._bucket_bjts
                and not self._bucket_mosfets
                and not self._bucket_opamps
                and recovery_mode == "auto"
            ):
                self._auto_regularized_warnings = []
                current_sig = self._dc_conductance_signature(
                    extra_gmin=0.0,
                    loop_reg_r=(1e-6 if self.regularize_ideal_loops else 0.0),
                )
                needs_rebuild = (
                    self._lu_piv is None
                    or self._cached_mode != "dc"
                    or self._cached_circuit_version != self.circuit.version
                    or self._cached_conductance_signature != current_sig
                )
                if needs_rebuild:
                    self._assemble_and_factorize(mode="dc")
                    rebuilt_any = True
                assert self._lu_piv is not None
                assert self._matrix_a is not None
                dim = self._matrix_a.shape[0]
                if gates or self._bucket_mcus:
                    solution_x, node_voltages = self._relax_logic_gates_dc()
                else:
                    if dim > 0:
                        rhs_b = self._build_rhs_dc()
                        solution_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                        if not np.all(np.isfinite(solution_x)):
                            raise SingularMatrixError("MNA solve produced non-finite values.")
                    else:
                        solution_x = np.zeros(0, dtype=np.float64)
                    node_voltages = {g: 0.0 for g in self.circuit.grounded_nodes()}
                    for node, idx in self._node_to_idx.items():
                        node_voltages[node] = float(solution_x[idx])
                self.last_convergence_method = (
                    "regularized"
                    if (self._auto_regularized_warnings or self.regularize_ideal_loops)
                    else "direct"
                )
            else:
                solution_x, node_voltages, pass_rebuilt = self._solve_step_with_recovery(
                    mode="dc",
                    sim_time=0.0,
                    dt=0.0,
                    method_str="dc",
                    direct_max_iters=direct_max_iters,
                    recovery_mode=recovery_mode,
                    relax_gates=True,
                )
                rebuilt_any = rebuilt_any or pass_rebuilt
                assert self._matrix_a is not None
                dim = self._matrix_a.shape[0]

            if not has_stateful_ics:
                break
            state_changed = False
            for ic in ne555_ics:
                if ic.evaluate_latch(node_voltages):
                    state_changed = True
            for relay in relays:
                if relay.evaluate_coil(node_voltages):
                    state_changed = True
            for fuse in fuses:
                if not fuse.is_blown:
                    va_f = node_voltages.get(fuse.nodes[0], 0.0)
                    vb_f = node_voltages.get(fuse.nodes[1], 0.0)
                    if fuse.check_blow((va_f - vb_f) * fuse.conductance):
                        state_changed = True
            for seq in seq_blocks:
                if seq.step_sequential(node_voltages):
                    state_changed = True
            if not state_changed:
                break

        for w in self._auto_regularized_warnings:
            if w.code not in {ew.code for ew in warnings}:
                warnings.append(w)

        branch_currents: dict[str, float] = {}
        for comp in self.circuit.flat_components():
            if isinstance(comp, Ground):
                comp.voltage = 0.0
                comp.current = 0.0
                comp.power = 0.0
                branch_currents[comp.name] = 0.0
                continue

            if isinstance(comp, Potentiometer):
                va = node_voltages.get(comp.node_a, 0.0)
                vw = node_voltages.get(comp.node_w, 0.0)
                vb = node_voltages.get(comp.node_b, 0.0)
                iaw = (va - vw) * comp.g_aw
                iwb = (vw - vb) * comp.g_wb
                comp.voltage = va - vb
                comp.current = iaw
                comp.power = (va - vw) * iaw + (vw - vb) * iwb
                branch_currents[comp.name] = iaw
                continue

            if isinstance(comp, LogicGate):
                v_out = node_voltages.get(comp.output_node, 0.0)
                i_out = (comp.drive_voltage - v_out) * comp.g_out
                p_out = (comp.drive_voltage - v_out) * i_out
                p_in = sum((node_voltages.get(n, 0.0) ** 2) * comp.g_in for n in comp.input_nodes)
                comp.voltage = v_out
                comp.current = i_out
                comp.power = p_out + p_in
                branch_currents[comp.name] = i_out
                continue

            if isinstance(comp, MCUBlock):
                branch_currents[comp.name] = self._update_mcu_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, DigitalSequentialBlock):
                branch_currents[comp.name] = self._update_sequential_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, BJTBase):
                branch_currents[comp.name] = self._update_bjt_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, MOSFETBase):
                branch_currents[comp.name] = self._update_mosfet_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, OpAmp):
                branch_currents[comp.name] = self._update_opamp_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, NE555):
                branch_currents[comp.name] = self._update_ne555_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, Transformer):
                branch_currents[comp.name] = self._update_transformer_telemetry(comp, node_voltages)
                continue

            if isinstance(comp, SPDTRelay):
                branch_currents[comp.name] = self._update_relay_telemetry(comp, node_voltages)
                continue

            v_a = node_voltages.get(comp.nodes[0], 0.0)
            v_b = node_voltages.get(comp.nodes[1], 0.0)
            v_ab = v_a - v_b
            comp.voltage = v_ab

            if isinstance(comp, (Resistor, Switch, Fuse, Lightbulb, Voltmeter, Ammeter)):
                if isinstance(comp, Voltmeter):
                    if comp.is_ideal:
                        i_ab = 0.0
                        comp.current = 0.0
                        comp.power = 0.0
                        comp.record_sample(0.0, comp.voltage, 0.0)
                    else:
                        i_ab = v_ab * comp.conductance
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(0.0, comp.voltage, 0.0)
                elif isinstance(comp, Ammeter):
                    if comp.is_ideal:
                        row_k = self._vs_to_idx[comp.name]
                        i_ab = float(solution_x[row_k])
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(0.0, comp.current, 0.0)
                    else:
                        i_ab = v_ab * comp.conductance
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(0.0, comp.current, 0.0)
                else:
                    i_ab = v_ab * comp.conductance
                    comp.current = i_ab
                    comp.power = v_ab * i_ab
            elif isinstance(comp, LED):
                if comp.burned_out:
                    i_ab = v_ab / comp.r_off
                    comp.is_conducting = False
                    comp.brightness = 0.0
                    comp.is_over_current = False
                elif comp.is_conducting:
                    i_ab = (v_ab - comp.forward_voltage) / comp.r_on
                else:
                    i_ab = v_ab / comp.r_off
                comp.current = i_ab
                comp.power = v_ab * i_ab
                if comp.burned_out or i_ab < 1e-5:
                    comp.brightness = 0.0
                else:
                    comp.brightness = math.sqrt(max(0.0, i_ab) / comp.rated_current)
                comp.is_over_current = (i_ab > comp.max_current and not comp.burned_out)
                if comp.burnout_enabled and i_ab >= comp.burnout_current:
                    comp.burned_out = True
                    comp.is_conducting = False
                    comp.brightness = 0.0
                    comp.is_over_current = False
            elif isinstance(comp, Diode):
                vd_fin, i_ab, _ = comp.terminal_model(
                    v_ab, vd_guess=comp.operating_vd
                )
                comp.operating_vd = vd_fin
                comp.operating_vak = v_ab
                comp.current = i_ab
                comp.power = v_ab * i_ab
            elif isinstance(comp, Capacitor):
                comp.current = 0.0
                comp.power = 0.0
                comp.cap_voltage = v_ab
                comp.cap_current = 0.0
                i_ab = 0.0
            elif isinstance(comp, Inductor):
                i_ab = v_ab * comp.dc_conductance
                comp.current = i_ab
                comp.power = v_ab * i_ab
                comp.ind_current = i_ab
                comp.ind_voltage = 0.0
            elif isinstance(comp, (DCVoltageSource, ACVoltageSource)):
                row_k = self._vs_to_idx[comp.name]
                i_ab = float(solution_x[row_k])
                comp.current = i_ab
                comp.power = v_ab * i_ab
            elif isinstance(comp, DCCurrentSource):
                i_ab = -comp.dc_current
                comp.current = i_ab
                comp.power = v_ab * i_ab
            else:
                i_ab = 0.0

            branch_currents[comp.name] = i_ab

        for comp in self.circuit.flat_components():
            if isinstance(comp, Voltmeter):
                self._update_voltmeter_loading(comp)

        warnings = self._collect_gate_runtime_warnings(
            node_voltages, dt=0.0, base_warnings=warnings
        )
        cond_num = self.compute_condition_number() if (compute_cond and dim > 0) else None
        return SolveResult(
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            sim_time=0.0,
            dt=0.0,
            integration_method="dc",
            condition_number=cond_num,
            warnings=warnings,
            rebuilt_matrix=rebuilt_any,
        )

    def _initialize_t0_derivatives(self) -> None:
        """Compute exact t = 0+ voltages/currents given initial capacitor V0 and inductor I0.

        Why this matters:
            At t = 0+, each capacitor acts as an exact voltage source `cap_voltage`
            (with series resistance `esr`) and each inductor acts as an exact current
            source `ind_current`. Solving the exact t = 0+ circuit gives `v_L(0+)` and
            `i_C(0+)` to full 64-bit machine precision without any finite-difference
            probe error, preserving exact energy conservation and 2nd-order convergence
            for Trapezoidal integration from step 0.
        """
        capacitors = [c for c in self.circuit.flat_components() if isinstance(c, Capacitor)]
        inductors = [c for c in self.circuit.flat_components() if isinstance(c, Inductor)]
        if not capacitors and not inductors:
            return

        non_gnd_nodes = self.circuit.non_ground_nodes()
        node_to_idx = {node: idx for idx, node in enumerate(non_gnd_nodes)}
        voltage_sources = self._voltage_sources()

        n_nodes = len(non_gnd_nodes)
        n_vs = len(voltage_sources)
        n_caps = len(capacitors)
        dim = n_nodes + n_vs + n_caps
        if dim == 0:
            return

        mat = np.zeros((dim, dim), dtype=np.float64)
        rhs = np.zeros(dim, dtype=np.float64)

        for idx in range(n_nodes):
            mat[idx, idx] += max(self.gmin, 1e-15)

        for comp in self.circuit.flat_components():
            if isinstance(comp, (Resistor, Switch, Fuse, Lightbulb, Voltmeter, Ammeter)):
                g = comp.conductance
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    mat[ia, ia] += g
                if ib is not None:
                    mat[ib, ib] += g
                if ia is not None and ib is not None:
                    mat[ia, ib] -= g
                    mat[ib, ia] -= g
            elif isinstance(comp, DCCurrentSource):
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    rhs[ia] += comp.dc_current
                if ib is not None:
                    rhs[ib] -= comp.dc_current
            elif isinstance(comp, Inductor):
                # At t = 0+, an inductor carries current `ind_current` from node_a to node_b
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    rhs[ia] -= comp.ind_current
                if ib is not None:
                    rhs[ib] += comp.ind_current
            elif isinstance(comp, Potentiometer):
                for na, nb, g in (
                    (comp.node_a, comp.node_w, comp.g_aw),
                    (comp.node_w, comp.node_b, comp.g_wb),
                ):
                    ia = node_to_idx.get(na)
                    ib = node_to_idx.get(nb)
                    if ia is not None:
                        mat[ia, ia] += g
                    if ib is not None:
                        mat[ib, ib] += g
                    if ia is not None and ib is not None:
                        mat[ia, ib] -= g
                        mat[ib, ia] -= g
            elif isinstance(comp, LogicGate):
                for in_node in comp.input_nodes:
                    ii = node_to_idx.get(in_node)
                    if ii is not None:
                        mat[ii, ii] += comp.g_in
                io = node_to_idx.get(comp.output_node)
                if io is not None:
                    mat[io, io] += comp.g_out
                    rhs[io] += comp.g_out * comp.drive_voltage
            elif isinstance(comp, (MCUBlock, DigitalSequentialBlock)):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    ip = node_to_idx.get(p_node)
                    if ip is not None:
                        gp = comp.pin_conductance(pin)
                        mat[ip, ip] += gp
                        if comp.is_output_pin(pin):
                            rhs[ip] += gp * comp.drive_voltage(pin)
            elif isinstance(comp, Diode):
                g_d, i_eq = comp.companion_model(0.0, "dc")
                ia = node_to_idx.get(comp.nodes[0])
                ib = node_to_idx.get(comp.nodes[1])
                if ia is not None:
                    mat[ia, ia] += g_d
                    rhs[ia] += i_eq
                if ib is not None:
                    mat[ib, ib] += g_d
                    rhs[ib] -= i_eq
                if ia is not None and ib is not None:
                    mat[ia, ib] -= g_d
                    mat[ib, ia] -= g_d
            elif isinstance(comp, (BJTBase, MOSFETBase, OpAmp, NE555, Transformer, SPDTRelay)):
                g_mat, rhs_3 = comp.companion_stamp()
                self._stamp_small_matrix(mat, node_to_idx, comp.nodes, g_mat)
                self._stamp_small_rhs(rhs, node_to_idx, comp.nodes, rhs_3)

        for k, vs in enumerate(voltage_sources):
            row_k = n_nodes + k
            ia = node_to_idx.get(vs.nodes[0])
            ib = node_to_idx.get(vs.nodes[1])
            if ia is not None:
                mat[ia, row_k] += 1.0
                mat[row_k, ia] += 1.0
            if ib is not None:
                mat[ib, row_k] -= 1.0
                mat[row_k, ib] -= 1.0
            if vs.r_internal > 0.0:
                mat[row_k, row_k] += vs.r_internal
            elif self.regularize_ideal_loops:
                mat[row_k, row_k] += 1e-6
            rhs[row_k] = vs.voltage_at(0.0)

        for k, cap in enumerate(capacitors):
            row_k = n_nodes + n_vs + k
            ia = node_to_idx.get(cap.nodes[0])
            ib = node_to_idx.get(cap.nodes[1])
            # Branch variable is i_C(0+) flowing from node_a to node_b:
            # V(node_a) - V(node_b) - esr * i_C(0+) = cap_voltage
            if ia is not None:
                mat[ia, row_k] += 1.0
                mat[row_k, ia] += 1.0
            if ib is not None:
                mat[ib, row_k] -= 1.0
                mat[row_k, ib] -= 1.0
            if cap.esr > 0.0:
                mat[row_k, row_k] -= cap.esr
            rhs[row_k] = cap.cap_voltage

        try:
            sol_x = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            # Regularize ideal capacitor / voltage-source loops with tiny 1 uOhm impedance.
            mat_reg = mat.copy()
            for r_idx in range(n_nodes):
                mat_reg[r_idx, r_idx] += 1e-9
            for r_idx in range(n_nodes, n_nodes + n_vs):
                if abs(mat_reg[r_idx, r_idx]) < 1e-12:
                    mat_reg[r_idx, r_idx] += 1e-6
            for r_idx in range(n_nodes + n_vs, dim):
                if abs(mat_reg[r_idx, r_idx]) < 1e-12:
                    mat_reg[r_idx, r_idx] -= 1e-6
            try:
                sol_x = np.linalg.solve(mat_reg, rhs)
            except np.linalg.LinAlgError:
                return

        node_v: dict[str, float] = {g: 0.0 for g in self.circuit.grounded_nodes()}
        for node, idx in node_to_idx.items():
            node_v[node] = float(sol_x[idx])

        for k, cap in enumerate(capacitors):
            row_k = n_nodes + n_vs + k
            i_c0 = float(sol_x[row_k])
            v_ab = node_v.get(cap.nodes[0], 0.0) - node_v.get(cap.nodes[1], 0.0)
            cap.cap_current = i_c0
            cap.voltage = v_ab
            cap.current = i_c0
            cap.power = v_ab * i_c0

        for ind in inductors:
            v_ab = node_v.get(ind.nodes[0], 0.0) - node_v.get(ind.nodes[1], 0.0)
            ind.ind_voltage = v_ab - ind.dcr * ind.ind_current
            ind.voltage = v_ab
            ind.current = ind.ind_current
            ind.power = v_ab * ind.ind_current

        self.invalidate_cache()

    def step_transient(
        self,
        dt: float,
        *,
        method: IntegrationMethod | str = IntegrationMethod.TRAPEZOIDAL,
        use_be_on_first_step: bool = True,
        use_be_on_switch: bool = True,
        compute_cond: bool = False,
        _substep_depth: int = 0,
    ) -> SolveResult:
        """Advance the transient simulation by one time step `dt` (in seconds).

        Args:
            dt: Time step size in seconds (must be > 0).
            method: Base integration method (`TRAPEZOIDAL` by default, or
                `BACKWARD_EULER`).
            use_be_on_first_step: When True (default), uses Backward Euler on the
                very first step (`step_index == 0`), as specified in Milestone 1.
            use_be_on_switch: When True (default), automatically switches to
                Backward Euler when any `Switch` toggles so discontinuous jumps do
                not trigger Trapezoidal numerical ringing.
            compute_cond: Whether to compute the 2-norm condition number on this step.
        """
        if dt <= 0.0:
            raise ValueError(f"Transient step dt must be > 0, got {dt}.")

        self._ensure_component_buckets()
        if self._transient_step_index == 0:
            self._topology_warnings = self.circuit.validate_topology(
                regularize_ideal_loops=self.regularize_ideal_loops
            )
            self._maybe_apply_startup_perturbation()
            if not self._logic_t0_initialized:
                self._initialize_logic_gates_t0(dt)
            if not use_be_on_first_step:
                self._initialize_t0_derivatives()
                self._be_steps_remaining = 0

        # Snapshot lightweight pre-step state only if nonlinear devices/switches exist and bisection is still allowed.
        has_smooth_nl = bool(
            self._bucket_diodes
            or self._bucket_bjts
            or self._bucket_mosfets
            or self._bucket_opamps
        )
        can_bisect = bool(
            _substep_depth < 6 and (has_smooth_nl or self._bucket_leds or self._bucket_switches)
        )
        if can_bisect:
            snap_sim_time = self.sim_time
            snap_step_idx = self._transient_step_index
            snap_be_rem = self._be_steps_remaining
            snap_nl = self._snapshot_nonlinear_states()
            snap_sw = [(s.closed, s.arcing) for s in self._bucket_switches]
        else:
            snap_sim_time = self.sim_time
            snap_step_idx = self._transient_step_index
            snap_be_rem = self._be_steps_remaining
            snap_nl = ([], [], [])
            snap_sw = []

        t_next = self.sim_time + dt

        # 1. Check timed switches, thermal lightbulb states, and due logic-gate transitions.
        switch_toggled = False
        switch_opened = False
        for sw in self._bucket_switches:
            was_closed = sw.closed
            if sw.check_timed_event(t_next):
                switch_toggled = True
                if was_closed and not sw.closed:
                    switch_opened = True
        if self._transient_step_index > 0:
            for bulb in self._bucket_lightbulbs:
                bulb.advance_thermal_state(dt)
        for gate in self._bucket_logic_gates:
            gate.prepare_step(t_next, dt)

        if switch_toggled and use_be_on_switch:
            # Use Backward Euler on the switching step and two settling steps after it
            # so stiff impulse currents decay before Trapezoidal integration resumes.
            self._be_steps_remaining = max(self._be_steps_remaining, 3)

        base_method = (
            IntegrationMethod(method)
            if not isinstance(method, IntegrationMethod)
            else method
        )
        if base_method == IntegrationMethod.BACKWARD_EULER:
            active_method = IntegrationMethod.BACKWARD_EULER
        elif self._be_steps_remaining > 0:
            active_method = IntegrationMethod.BACKWARD_EULER
            self._be_steps_remaining -= 1
        else:
            active_method = IntegrationMethod.TRAPEZOIDAL

        method_str = active_method.value
        needs_rebuild = False

        try:
            if has_smooth_nl:
                solution_x, node_voltages, needs_rebuild = self._solve_step_with_recovery(
                    mode="transient",
                    sim_time=t_next,
                    dt=dt,
                    method_str=method_str,
                    direct_max_iters=80,
                    relax_gates=False,
                )
                assert self._matrix_a is not None
                dim = self._matrix_a.shape[0]
            else:
                # 2. Check if we can reuse the cached LU factorization or must rebuild A.
                eff_loop_reg = 1e-6 if self.regularize_ideal_loops else 0.0
                current_sig = self._transient_conductance_signature(
                    dt, method_str, extra_gmin=0.0, loop_reg_r=eff_loop_reg
                )
                needs_rebuild = (
                    self._lu_piv is None
                    or self._cached_mode != "transient"
                    or self._cached_circuit_version != self.circuit.version
                    or self._cached_conductance_signature != current_sig
                )

                if needs_rebuild:
                    self._assemble_and_factorize(
                        mode="transient", dt=dt, method_str=method_str
                    )

                assert self._lu_piv is not None
                assert self._matrix_a is not None

                # 3. Assemble RHS vector b(t_next) and solve using cached LU factors.
                dim = self._matrix_a.shape[0]
                if dim > 0:
                    rhs_b = self._build_rhs_transient(t_next, dt, method_str)
                    solution_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                    if not np.all(np.isfinite(solution_x)):
                        raise SingularMatrixError(
                            f"Transient solve did not converge at t = {t_next:.6g} s."
                        )
                else:
                    solution_x = np.zeros(0, dtype=np.float64)

                node_voltages = {}
                for gnd_node in self._bucket_grounded_nodes:
                    node_voltages[gnd_node] = 0.0
                for node, idx in self._node_to_idx.items():
                    node_voltages[node] = float(solution_x[idx])

            # 3b. Check for switch contact breakdown arc ignition or extinction.
            arc_state_changed = False
            arc_ignited_switches: list[str] = []
            for comp in self._bucket_switches:
                if not comp.closed:
                    v_sw = node_voltages.get(comp.nodes[0], 0.0) - node_voltages.get(
                        comp.nodes[1], 0.0
                    )
                    if not comp.arcing and comp.check_arc_ignition(v_sw):
                        arc_state_changed = True
                        arc_ignited_switches.append(comp.name)
                    elif comp.arcing and comp.check_arc_extinction(v_sw):
                        arc_state_changed = True

            if arc_state_changed:
                if use_be_on_switch:
                    active_method = IntegrationMethod.BACKWARD_EULER
                    method_str = active_method.value
                    self._be_steps_remaining = max(self._be_steps_remaining, 2)
                if has_smooth_nl:
                    solution_x, node_voltages, reb_arc = self._solve_step_with_recovery(
                        mode="transient",
                        sim_time=t_next,
                        dt=dt,
                        method_str=method_str,
                        direct_max_iters=80,
                        relax_gates=False,
                    )
                    needs_rebuild = needs_rebuild or reb_arc
                else:
                    self._assemble_and_factorize(
                        mode="transient", dt=dt, method_str=method_str
                    )
                    needs_rebuild = True
                    assert self._lu_piv is not None
                    if dim > 0:
                        rhs_b = self._build_rhs_transient(t_next, dt, method_str)
                        solution_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                        for node, idx in self._node_to_idx.items():
                            node_voltages[node] = float(solution_x[idx])

            # 3c. Check for LED conduction state transitions (when no smooth nonlinear loop ran).
            if self._bucket_leds and not has_smooth_nl:
                for _diode_iter in range(5):
                    led_state_changed = self._update_led_discrete_states(node_voltages)
                    if not led_state_changed:
                        break
                    self._assemble_and_factorize(
                        mode="transient", dt=dt, method_str=method_str
                    )
                    needs_rebuild = True
                    assert self._lu_piv is not None
                    if dim > 0:
                        rhs_b = self._build_rhs_transient(t_next, dt, method_str)
                        solution_x = lu_solve(self._lu_piv, rhs_b, check_finite=True)
                        for node, idx in self._node_to_idx.items():
                            node_voltages[node] = float(solution_x[idx])
        except (SingularMatrixError, ValueError, np.linalg.LinAlgError):
            if not can_bisect:
                raise
            # Restore pre-step discrete state and adaptively bisect the timestep cleanly.
            self.sim_time = snap_sim_time
            self._transient_step_index = snap_step_idx
            self._be_steps_remaining = snap_be_rem
            self._restore_nonlinear_states(snap_nl)
            for sw, (was_cl, was_arc) in zip(self._bucket_switches, snap_sw):
                sw.closed = was_cl
                sw.arcing = was_arc
            half_dt = 0.5 * dt
            self.step_transient(
                half_dt,
                method=method,
                use_be_on_first_step=use_be_on_first_step,
                use_be_on_switch=use_be_on_switch,
                compute_cond=False,
                _substep_depth=_substep_depth + 1,
            )
            res_second = self.step_transient(
                half_dt,
                method=method,
                use_be_on_first_step=use_be_on_first_step,
                use_be_on_switch=use_be_on_switch,
                compute_cond=compute_cond,
                _substep_depth=_substep_depth + 1,
            )
            self.last_convergence_method = "substep_recovery"
            return res_second

        # 3d. Sample logic-gate inputs & execute offline MCU callbacks after step convergence.
        for comp in self._bucket_logic_gates:
            comp.sample_inputs_and_schedule(node_voltages, t_next, dt)
        for mcu in self._bucket_mcus:
            mcu.evaluate_firmware(node_voltages, t_next, dt)
        for ic in self._bucket_ne555:
            if ic.evaluate_latch(node_voltages):
                self._be_steps_remaining = max(self._be_steps_remaining, 2)
        for relay in self._bucket_relays:
            if relay.evaluate_coil(node_voltages):
                self._be_steps_remaining = max(self._be_steps_remaining, 2)
        for fuse in self._bucket_fuses:
            if not fuse.is_blown:
                va_f = node_voltages.get(fuse.nodes[0], 0.0)
                vb_f = node_voltages.get(fuse.nodes[1], 0.0)
                if fuse.check_blow((va_f - vb_f) * fuse.conductance):
                    self._be_steps_remaining = max(self._be_steps_remaining, 2)
        for seq in self._bucket_sequential:
            seq.step_sequential(node_voltages)

        # 4. Update component voltages, currents, companion states, and meter buffers.
        step_warnings = self._collect_gate_runtime_warnings(
            node_voltages, dt=dt, base_warnings=self._topology_warnings
        )
        branch_currents: dict[str, float] = {}

        # Nominal source voltage scale for inductive kickback comparison
        if self._bucket_inductors:
            max_source_v = max(
                (
                    abs(c.dc_voltage)
                    if tag == _TAG_DC_VS
                    else (abs(c.amplitude) + abs(c.dc_offset))
                    if tag == _TAG_AC_VS
                    else 0.0
                    for c, tag in self._bucket_tagged_comps
                ),
                default=10.0,
            )
            kickback_threshold = max(
                self.KICKBACK_VOLTAGE_WARNING_MIN, 5.0 * max_source_v
            )
        else:
            max_source_v = 0.0
            kickback_threshold = self.KICKBACK_VOLTAGE_WARNING_MIN

        for comp, tag in self._bucket_tagged_comps:
            if tag == _TAG_GROUND:
                comp.voltage = 0.0
                comp.current = 0.0
                comp.power = 0.0
                branch_currents[comp.name] = 0.0
                continue

            if tag == _TAG_POTENTIOMETER:
                va = node_voltages.get(comp.node_a, 0.0)
                vw = node_voltages.get(comp.node_w, 0.0)
                vb = node_voltages.get(comp.node_b, 0.0)
                iaw = (va - vw) * comp.g_aw
                iwb = (vw - vb) * comp.g_wb
                comp.voltage = va - vb
                comp.current = iaw
                comp.power = (va - vw) * iaw + (vw - vb) * iwb
                branch_currents[comp.name] = iaw
                continue

            if tag == _TAG_LOGIC_GATE:
                v_out = node_voltages.get(comp.output_node, 0.0)
                i_out = (comp.drive_voltage - v_out) * comp.g_out
                p_out = (comp.drive_voltage - v_out) * i_out
                p_in = sum((node_voltages.get(n, 0.0) ** 2) * comp.g_in for n in comp.input_nodes)
                comp.voltage = v_out
                comp.current = i_out
                comp.power = p_out + p_in
                branch_currents[comp.name] = i_out
                continue

            if tag == _TAG_MCU:
                branch_currents[comp.name] = self._update_mcu_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_SEQUENTIAL:
                branch_currents[comp.name] = self._update_sequential_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_BJT:
                branch_currents[comp.name] = self._update_bjt_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_MOSFET:
                branch_currents[comp.name] = self._update_mosfet_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_OPAMP:
                branch_currents[comp.name] = self._update_opamp_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_NE555:
                branch_currents[comp.name] = self._update_ne555_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_TRANSFORMER:
                branch_currents[comp.name] = self._update_transformer_telemetry(comp, node_voltages)
                continue

            if tag == _TAG_RELAY:
                branch_currents[comp.name] = self._update_relay_telemetry(comp, node_voltages)
                continue

            v_a = node_voltages.get(comp.nodes[0], 0.0)
            v_b = node_voltages.get(comp.nodes[1], 0.0)
            v_ab = v_a - v_b

            if tag == _TAG_FUSE:
                i_ab = v_ab * comp.conductance
                comp.voltage = v_ab
                comp.current = i_ab
                comp.power = v_ab * i_ab
                branch_currents[comp.name] = i_ab
                continue

            if tag <= _TAG_AMMETER:
                if tag == _TAG_VOLTMETER:
                    if comp.is_ideal:
                        i_ab = 0.0
                        comp.voltage = v_ab
                        comp.current = 0.0
                        comp.power = 0.0
                        comp.record_sample(t_next, comp.voltage, dt)
                    else:
                        i_ab = v_ab * comp.conductance
                        comp.voltage = v_ab
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(t_next, comp.voltage, dt)
                elif tag == _TAG_AMMETER:
                    if comp.is_ideal:
                        row_k = self._vs_to_idx[comp.name]
                        i_ab = float(solution_x[row_k])
                        comp.voltage = v_ab
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(t_next, comp.current, dt)
                    else:
                        i_ab = v_ab * comp.conductance
                        comp.voltage = v_ab
                        comp.current = i_ab
                        comp.power = v_ab * i_ab
                        comp.record_sample(t_next, comp.current, dt)
                else:
                    i_ab = v_ab * comp.conductance
                    comp.voltage = v_ab
                    comp.current = i_ab
                    comp.power = v_ab * i_ab
            elif tag == _TAG_LED:
                comp.voltage = v_ab
                if comp.burned_out:
                    i_ab = v_ab / comp.r_off
                    comp.is_conducting = False
                    comp.brightness = 0.0
                    comp.is_over_current = False
                elif comp.is_conducting:
                    i_ab = (v_ab - comp.forward_voltage) / comp.r_on
                else:
                    i_ab = v_ab / comp.r_off
                comp.current = i_ab
                comp.power = v_ab * i_ab
                if comp.burned_out or i_ab < 1e-5:
                    comp.brightness = 0.0
                else:
                    comp.brightness = math.sqrt(max(0.0, i_ab) / comp.rated_current)
                comp.is_over_current = (i_ab > comp.max_current and not comp.burned_out)
                if comp.burnout_enabled and i_ab >= comp.burnout_current:
                    comp.burned_out = True
                    comp.is_conducting = False
                    comp.brightness = 0.0
                    comp.is_over_current = False
            elif tag == _TAG_DIODE:
                vd_fin, i_ab, _ = comp.terminal_model(
                    v_ab, vd_guess=comp.operating_vd
                )
                comp.operating_vd = vd_fin
                comp.operating_vak = v_ab
                comp.voltage = v_ab
                comp.current = i_ab
                comp.power = v_ab * i_ab
            elif tag == _TAG_SWITCH:
                comp.update_companion_state(v_ab, dt, method_str)
                i_ab = comp.current
            elif tag == _TAG_CAPACITOR:
                g_eq, i_eq = comp.companion_model(dt, method_str)
                i_ab = g_eq * v_ab - i_eq
                comp.update_companion_state(v_ab, i_ab)
            elif tag == _TAG_INDUCTOR:
                g_eq, i_hist = comp.companion_model(dt, method_str)
                i_ab = g_eq * v_ab + i_hist
                comp.update_companion_state(v_ab, i_ab)
                if (
                    abs(comp.ind_voltage) >= kickback_threshold
                    or (switch_opened and abs(comp.ind_voltage) > 2.0 * max_source_v)
                    or bool(arc_ignited_switches)
                ):
                    step_warnings.append(
                        CircuitWarning(
                            code="inductive_kickback",
                            message="Inductive kickback, consider a flyback diode",
                            component_names=(comp.name,),
                        )
                    )
            elif tag == _TAG_DC_VS or tag == _TAG_AC_VS:
                row_k = self._vs_to_idx[comp.name]
                i_ab = float(solution_x[row_k])
                comp.voltage = v_ab
                comp.current = i_ab
                comp.power = v_ab * i_ab
            elif tag == _TAG_DC_IS:
                i_ab = -comp.dc_current
                comp.voltage = v_ab
                comp.current = i_ab
                comp.power = v_ab * i_ab
            else:
                i_ab = 0.0

            branch_currents[comp.name] = i_ab

        for comp in self._bucket_voltmeters:
            self._update_voltmeter_loading(comp)

        self.sim_time = t_next
        self._transient_step_index += 1
        cond_num = self.compute_condition_number() if (compute_cond and dim > 0) else None

        return SolveResult(
            node_voltages=node_voltages,
            branch_currents=branch_currents,
            sim_time=t_next,
            dt=dt,
            integration_method=method_str,
            condition_number=cond_num,
            warnings=step_warnings,
            rebuilt_matrix=needs_rebuild,
        )

    def solve_transient(
        self,
        duration: float,
        dt: float | None = None,
        *,
        method: IntegrationMethod | str = IntegrationMethod.TRAPEZOIDAL,
        use_be_on_first_step: bool = True,
        use_be_on_switch: bool = True,
        reset_before_run: bool = True,
    ) -> TransientResult:
        """Run a transient simulation from `t = 0` (or current `sim_time`) for `duration` s.

        Returns a `TransientResult` with NumPy waveform arrays for every node and
        component, including the initial state at index 0 (`t = 0`).
        """
        if duration <= 0.0:
            raise ValueError(f"Transient duration must be > 0 s, got {duration}.")

        if reset_before_run:
            self.reset_transient()

        step_dt = dt if (dt is not None and dt > 0.0) else self.choose_default_dt(duration)
        n_steps = max(1, int(round(duration / step_dt)))

        # Validate topology and initialize t = 0+ states for index 0 recording.
        all_warnings: list[CircuitWarning] = list(
            self.circuit.validate_topology(
                regularize_ideal_loops=self.regularize_ideal_loops
            )
        )
        self._topology_warnings = list(all_warnings)
        self._maybe_apply_startup_perturbation()
        self._initialize_logic_gates_t0(step_dt)
        self._initialize_t0_derivatives()
        if use_be_on_first_step:
            self._be_steps_remaining = 1
        else:
            self._be_steps_remaining = 0

        all_nodes = sorted(self.circuit.all_nodes())
        comp_names = [c.name for c in self.circuit.flat_components()]

        times = np.zeros(n_steps + 1, dtype=np.float64)
        node_history: dict[str, np.ndarray] = {
            n: np.zeros(n_steps + 1, dtype=np.float64) for n in all_nodes
        }
        curr_history: dict[str, np.ndarray] = {
            name: np.zeros(n_steps + 1, dtype=np.float64) for name in comp_names
        }
        volt_history: dict[str, np.ndarray] = {
            name: np.zeros(n_steps + 1, dtype=np.float64) for name in comp_names
        }

        # Record t = 0 state
        times[0] = self.sim_time
        gnd_nodes = self.circuit.grounded_nodes()
        if self._t0_node_voltages:
            for n, v in self._t0_node_voltages.items():
                if n in node_history:
                    node_history[n][0] = v
        for comp in self.circuit.flat_components():
            curr_history[comp.name][0] = comp.current
            volt_history[comp.name][0] = comp.voltage
            if not self._t0_node_voltages and not isinstance(comp, Ground) and len(comp.nodes) >= 2:
                if comp.nodes[1] in gnd_nodes and comp.nodes[0] not in gnd_nodes:
                    node_history[comp.nodes[0]][0] = comp.voltage

        rebuild_count = 0
        all_warnings = (
            self._collect_gate_runtime_warnings(
                self._t0_node_voltages, dt=step_dt, base_warnings=all_warnings
            )
            if self._t0_node_voltages
            else all_warnings
        )
        seen_warning_idx: dict[tuple[str, tuple[str, ...]], int] = {
            (w.code, w.component_names): idx for idx, w in enumerate(all_warnings)
        }

        for k in range(1, n_steps + 1):
            res = self.step_transient(
                step_dt,
                method=method,
                use_be_on_first_step=use_be_on_first_step,
                use_be_on_switch=use_be_on_switch,
            )
            if res.rebuilt_matrix:
                rebuild_count += 1
            times[k] = res.sim_time
            for n, v in res.node_voltages.items():
                node_history[n][k] = v
            for comp in self.circuit.flat_components():
                curr_history[comp.name][k] = comp.current
                volt_history[comp.name][k] = comp.voltage
            for w in res.warnings:
                key = (w.code, w.component_names)
                existing_idx = seen_warning_idx.get(key)
                if existing_idx is None:
                    seen_warning_idx[key] = len(all_warnings)
                    all_warnings.append(w)
                else:
                    all_warnings[existing_idx] = w

        return TransientResult(
            times=times,
            node_voltages=node_history,
            branch_currents=curr_history,
            component_voltages=volt_history,
            warnings=all_warnings,
            matrix_rebuild_count=rebuild_count,
            step_count=n_steps,
        )

    def _update_voltmeter_loading(self, comp: Voltmeter) -> None:
        """Evaluate voltmeter loading effect against circuit Thévenin resistance.

        If the loading changes the circuit by >= 0.1% (loading error), format
        the user-facing diagnostic string:
        '{VM} draws {I_vm} from the {R_th} resistor; the reading is {pct}% lower than the unloaded voltage.'
        """
        if comp.is_ideal:
            comp.loading_error_message = None
            comp.loading_error_percent = 0.0
            return

        try:
            r_th = self.circuit.thevenin_resistance(
                comp.nodes[0], comp.nodes[1], exclude_component=comp.name
            )
        except Exception:
            r_th = 0.0

        if r_th > 0.0 and math.isfinite(r_th):
            loading_frac = r_th / (r_th + comp.r_internal)
            loading_pct = loading_frac * 100.0
            if round(loading_pct, 1) >= 0.1:
                comp.loading_error_percent = loading_pct
                i_vm = comp.voltage / comp.r_internal
                comp.loading_error_message = (
                    f"{comp.name} draws {format_engineering_value(abs(i_vm), 'A', sig_figs=2)} "
                    f"from the {format_engineering_value(r_th, 'ohm', sig_figs=3)} resistor; "
                    f"the reading is {loading_pct:.1f}% lower than the unloaded voltage."
                )
            else:
                comp.loading_error_message = None
                comp.loading_error_percent = 0.0
        else:
            comp.loading_error_message = None
            comp.loading_error_percent = 0.0
