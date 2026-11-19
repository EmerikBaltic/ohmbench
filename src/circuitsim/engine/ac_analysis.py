"""Qt-free small-signal AC phasor, Bode, input impedance (Z_in), and reflection (Gamma) solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np

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
from circuitsim.components.sources import (
    ACVoltageSource,
    DCCurrentSource,
    DCVoltageSource,
)
from circuitsim.components.switch import Switch
from circuitsim.engine.errors import CircuitError
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.solver import MNASolver, SolveResult


@dataclass(frozen=True)
class ACFrequencySample:
    """Single-frequency AC response at one stimulus frequency."""

    frequency_hz: float
    v_phasor: complex
    mag_v: float
    db: float
    phase_deg: float
    z_in: complex
    gamma: complex


@dataclass(frozen=True)
class ACAnalysisResult:
    """Complete small-signal AC frequency-sweep result."""

    frequencies_hz: np.ndarray
    omega_rad_s: np.ndarray
    source_name: str
    stimulus_amplitude: float
    z0_ohm: float
    node_phasors: dict[str, np.ndarray]
    node_mag_v: dict[str, np.ndarray]
    node_db: dict[str, np.ndarray]
    node_phase_deg: dict[str, np.ndarray]
    z_in: np.ndarray
    gamma: np.ndarray
    dc_operating_point: SolveResult

    @property
    def z_in_mag_ohm(self) -> np.ndarray:
        return np.abs(self.z_in)

    @property
    def z_in_phase_deg(self) -> np.ndarray:
        return np.angle(self.z_in, deg=True)

    @property
    def gamma_mag(self) -> np.ndarray:
        return np.abs(self.gamma)

    def bode(
        self, node: str, *, unwrap_phase: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(frequencies_hz, magnitude_db, phase_deg)` for `node`."""
        if node not in self.node_phasors:
            raise KeyError(f"Unknown node {node!r} in ACAnalysisResult.")
        phase = np.angle(self.node_phasors[node], deg=True)
        if unwrap_phase:
            phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))
        return self.frequencies_hz, self.node_db[node], phase

    def sample_at(self, node: str, index: int) -> ACFrequencySample:
        """Return a scalar summary sample for `node` at sweep index `index`."""
        return ACFrequencySample(
            frequency_hz=float(self.frequencies_hz[index]),
            v_phasor=complex(self.node_phasors[node][index]),
            mag_v=float(self.node_mag_v[node][index]),
            db=float(self.node_db[node][index]),
            phase_deg=float(self.node_phase_deg[node][index]),
            z_in=complex(self.z_in[index]),
            gamma=complex(self.gamma[index]),
        )


def _generate_frequencies(
    *,
    f_start: float,
    f_stop: float,
    points_per_decade: int,
    num_points: int | None,
    sweep_scale: str,
    frequencies_hz: Iterable[float] | None,
) -> np.ndarray:
    if frequencies_hz is not None:
        freqs = np.asarray([float(f) for f in frequencies_hz], dtype=np.float64)
        if freqs.ndim != 1 or len(freqs) == 0:
            raise ValueError("frequencies_hz must contain at least one positive frequency.")
        if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0.0):
            raise ValueError("All AC sweep frequencies must be finite and > 0 Hz.")
        return freqs

    if f_start <= 0.0 or f_stop < f_start:
        raise ValueError(
            f"AC sweep bounds must satisfy 0 < f_start <= f_stop (got {f_start}, {f_stop})."
        )
    if f_start == f_stop:
        return np.array([float(f_start)], dtype=np.float64)

    if num_points is None:
        decades = math.log10(f_stop / f_start)
        n_pts = max(2, int(round(decades * max(1, points_per_decade))) + 1)
    else:
        n_pts = max(2, int(num_points))

    if sweep_scale.lower() == "linear":
        return np.linspace(f_start, f_stop, n_pts, dtype=np.float64)
    return np.geomspace(f_start, f_stop, n_pts, dtype=np.float64)


def _select_ac_source(
    circuit: Circuit, source_name: str | None
) -> DCVoltageSource | ACVoltageSource:
    voltage_sources = [
        c for c in circuit.components if isinstance(c, (ACVoltageSource, DCVoltageSource))
    ]
    if not voltage_sources:
        raise CircuitError(
            "AC analysis requires at least one ACVoltageSource or DCVoltageSource stimulus."
        )
    if source_name is not None:
        for vs in voltage_sources:
            if vs.name == source_name:
                return vs
        raise KeyError(f"AC stimulus source {source_name!r} not found in circuit.")

    for vs in voltage_sources:
        if isinstance(vs, ACVoltageSource):
            return vs
    return voltage_sources[0]


def _stamp_complex_admittance(
    mat: np.ndarray,
    node_to_idx: dict[str, int],
    node_a: str,
    node_b: str,
    y_val: complex,
) -> None:
    ia = node_to_idx.get(node_a)
    ib = node_to_idx.get(node_b)
    if ia is not None:
        mat[ia, ia] += y_val
    if ib is not None:
        mat[ib, ib] += y_val
    if ia is not None and ib is not None:
        mat[ia, ib] -= y_val
        mat[ib, ia] -= y_val


def run_ac_sweep(
    circuit: Circuit,
    *,
    f_start: float = 1.0,
    f_stop: float = 1e6,
    points_per_decade: int = 20,
    num_points: int | None = None,
    sweep_scale: str = "log",
    frequencies_hz: Iterable[float] | None = None,
    source_name: str | None = None,
    stimulus_amplitude: float = 1.0,
    stimulus_phase_deg: float = 0.0,
    z0_ohm: float = 50.0,
    gmin: float | None = None,
    regularize_ideal_loops: bool = False,
) -> ACAnalysisResult:
    """Linearize `circuit` around its DC operating point and solve small-signal AC MNA.

    Computes complex node voltages ``V(jw)``, magnitude in dB referenced to
    ``stimulus_amplitude``, phase in degrees, driving-point input impedance
    ``Z_in(jw)``, and reflection coefficient ``Gamma(jw) = (Z_in - Z0) / (Z_in + Z0)``.
    """
    eff_gmin = MNASolver.DEFAULT_GMIN if gmin is None else float(gmin)
    if z0_ohm <= 0.0:
        raise ValueError(f"z0_ohm must be > 0, got {z0_ohm}.")
    if stimulus_amplitude <= 0.0:
        raise ValueError(f"stimulus_amplitude must be > 0, got {stimulus_amplitude}.")

    freqs = _generate_frequencies(
        f_start=f_start,
        f_stop=f_stop,
        points_per_decade=points_per_decade,
        num_points=num_points,
        sweep_scale=sweep_scale,
        frequencies_hz=frequencies_hz,
    )
    omegas = 2.0 * math.pi * freqs
    n_f = len(freqs)

    # 1. Linearize nonlinear / thermal / diode components at DC operating point.
    dc_solver = MNASolver(
        circuit, gmin=eff_gmin, regularize_ideal_loops=regularize_ideal_loops
    )
    dc_res = dc_solver.solve_dc()
    stim_src = _select_ac_source(circuit, source_name)

    non_gnd_nodes = circuit.non_ground_nodes()
    grounded_nodes = circuit.grounded_nodes()
    all_nodes = sorted(circuit.all_nodes())
    node_to_idx = {node: idx for idx, node in enumerate(non_gnd_nodes)}

    vs_list: list[DCVoltageSource | ACVoltageSource | Ammeter] = [
        c
        for c in circuit.components
        if isinstance(c, (DCVoltageSource, ACVoltageSource))
        or (isinstance(c, Ammeter) and c.is_ideal)
    ]
    n_nodes = len(non_gnd_nodes)
    n_vs = len(vs_list)
    dim = n_nodes + n_vs
    vs_to_idx = {c.name: n_nodes + k for k, c in enumerate(vs_list)}

    phasor_stim = complex(
        stimulus_amplitude * math.cos(math.radians(stimulus_phase_deg)),
        stimulus_amplitude * math.sin(math.radians(stimulus_phase_deg)),
    )

    node_phasors: dict[str, np.ndarray] = {
        node: np.zeros(n_f, dtype=np.complex128) for node in all_nodes
    }
    z_in_arr = np.zeros(n_f, dtype=np.complex128)
    gamma_arr = np.zeros(n_f, dtype=np.complex128)

    stim_row = vs_to_idx[stim_src.name]
    stim_pos_node = stim_src.nodes[0]
    stim_neg_node = stim_src.nodes[1]

    for k_freq, omega in enumerate(omegas):
        mat_y = np.zeros((dim, dim), dtype=np.complex128)
        rhs_c = np.zeros(dim, dtype=np.complex128)

        for idx in range(n_nodes):
            mat_y[idx, idx] += eff_gmin

        for comp in circuit.components:
            if isinstance(comp, Ground) or isinstance(comp, DCCurrentSource):
                continue
            if isinstance(comp, (Resistor, Fuse, Lightbulb, Voltmeter, Ammeter)):
                if isinstance(comp, (Voltmeter, Ammeter)) and comp.is_ideal:
                    continue
                _stamp_complex_admittance(
                    mat_y, node_to_idx, comp.nodes[0], comp.nodes[1], complex(comp.conductance)
                )
            elif isinstance(comp, (LED, Diode)):
                g_lin = comp.companion_model(0.0, "dc")[0]
                _stamp_complex_admittance(
                    mat_y, node_to_idx, comp.nodes[0], comp.nodes[1], complex(g_lin)
                )
            elif isinstance(comp, (BJTBase, MOSFETBase, OpAmp, NE555, Transformer, SPDTRelay)):
                g_mat, _ = comp.companion_stamp()
                idxs = [node_to_idx.get(n) for n in comp.nodes]
                for r_i, row_idx in enumerate(idxs):
                    if row_idx is None:
                        continue
                    for c_j, col_idx in enumerate(idxs):
                        if col_idx is not None:
                            mat_y[row_idx, col_idx] += g_mat[r_i][c_j]
            elif isinstance(comp, Switch):
                y_sw = complex(comp.conductance)
                if getattr(comp, "c_parasitic", 0.0) > 0.0:
                    y_sw += 1j * omega * comp.c_parasitic
                _stamp_complex_admittance(mat_y, node_to_idx, comp.nodes[0], comp.nodes[1], y_sw)
            elif isinstance(comp, Capacitor):
                esr = getattr(comp, "esr", 0.0)
                if esr > 0.0:
                    denom = 1.0 + 1j * omega * comp.capacitance * esr
                    y_c = (1j * omega * comp.capacitance) / denom
                else:
                    y_c = 1j * omega * comp.capacitance
                _stamp_complex_admittance(mat_y, node_to_idx, comp.nodes[0], comp.nodes[1], y_c)
            elif isinstance(comp, Inductor):
                r_dc = getattr(comp, "dcr", 0.0)
                z_l = complex(r_dc, omega * comp.inductance)
                y_l = 1.0 / z_l
                _stamp_complex_admittance(mat_y, node_to_idx, comp.nodes[0], comp.nodes[1], y_l)
            elif isinstance(comp, Potentiometer):
                _stamp_complex_admittance(mat_y, node_to_idx, comp.node_a, comp.node_w, complex(comp.g_aw))
                _stamp_complex_admittance(mat_y, node_to_idx, comp.node_w, comp.node_b, complex(comp.g_wb))
            elif isinstance(comp, LogicGate):
                for in_node in comp.input_nodes:
                    ii = node_to_idx.get(in_node)
                    if ii is not None:
                        mat_y[ii, ii] += comp.g_in
                io = node_to_idx.get(comp.output_node)
                if io is not None:
                    mat_y[io, io] += comp.g_out
            elif isinstance(comp, MCUBlock):
                for pin, p_node in zip(comp.pin_names, comp.nodes):
                    ip = node_to_idx.get(p_node)
                    if ip is not None:
                        mat_y[ip, ip] += comp.pin_conductance(pin)

        for comp in vs_list:
            row_k = vs_to_idx[comp.name]
            ia = node_to_idx.get(comp.nodes[0])
            ib = node_to_idx.get(comp.nodes[1])
            if ia is not None:
                mat_y[ia, row_k] += 1.0
                mat_y[row_k, ia] += 1.0
            if ib is not None:
                mat_y[ib, row_k] -= 1.0
                mat_y[row_k, ib] -= 1.0

            if isinstance(comp, Ammeter) and getattr(comp, "is_ideal", False):
                r_int = comp.r_internal if circuit.is_ammeter_in_parallel(comp) else 0.0
            else:
                r_int = getattr(comp, "r_internal", 0.0)
            if r_int > 0.0:
                mat_y[row_k, row_k] += r_int
            elif regularize_ideal_loops:
                mat_y[row_k, row_k] += 1e-6

            if comp.name == stim_src.name:
                rhs_c[row_k] = phasor_stim

        sol_c = np.linalg.solve(mat_y, rhs_c)

        for gnd in grounded_nodes:
            node_phasors[gnd][k_freq] = 0.0j
        for node, idx in node_to_idx.items():
            node_phasors[node][k_freq] = complex(sol_c[idx])

        v_port = node_phasors[stim_pos_node][k_freq] - node_phasors[stim_neg_node][k_freq]
        i_in = -complex(sol_c[stim_row])
        if abs(i_in) <= 1e-18:
            z_val = complex(math.inf, 0.0)
            g_val = complex(1.0, 0.0)
        else:
            z_val = v_port / i_in
            g_val = (z_val - z0_ohm) / (z_val + z0_ohm)
        z_in_arr[k_freq] = z_val
        gamma_arr[k_freq] = g_val

    node_mag_v: dict[str, np.ndarray] = {}
    node_db: dict[str, np.ndarray] = {}
    node_phase_deg: dict[str, np.ndarray] = {}
    for node, arr in node_phasors.items():
        mags = np.abs(arr)
        node_mag_v[node] = mags
        ref = np.maximum(mags / stimulus_amplitude, 1e-15)
        node_db[node] = 20.0 * np.log10(ref)
        node_phase_deg[node] = np.angle(arr, deg=True)

    return ACAnalysisResult(
        frequencies_hz=freqs,
        omega_rad_s=omegas,
        source_name=stim_src.name,
        stimulus_amplitude=float(stimulus_amplitude),
        z0_ohm=float(z0_ohm),
        node_phasors=node_phasors,
        node_mag_v=node_mag_v,
        node_db=node_db,
        node_phase_deg=node_phase_deg,
        z_in=z_in_arr,
        gamma=gamma_arr,
        dc_operating_point=dc_res,
    )
