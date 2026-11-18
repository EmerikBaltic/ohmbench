"""6 built-in, solver-ready WireGraph Test Jigs for benchtop verification (Qt-free)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Diode,
    Ground,
    Inductor,
    LogicGate,
    Potentiometer,
    Resistor,
    Voltmeter,
)
from circuitsim.io.circ_serializer import PlacedComponentSpec
from circuitsim.wiring.model import JunctionRef, TerminalRef, WireGraph
from circuitsim.wiring.serializer import CircuitDocumentV2


@dataclass(frozen=True)
class TestJigInfo:
    """Descriptor for one built-in benchtop test jig."""

    jig_id: str
    title: str
    description: str


BUILTIN_TEST_JIGS: tuple[TestJigInfo, ...] = (
    TestJigInfo(
        "rc_lowpass",
        "1 kHz RC Low-Pass Filter",
        "1 V AC source + 1 kΩ + 159.15 nF (-3 dB at 1 kHz)",
    ),
    TestJigInfo(
        "rlc_resonance",
        "Series RLC 31.8 kHz Resonator",
        "50 Ω + 1 mH + 25 nF series tank matched to 50 Ω at resonance",
    ),
    TestJigInfo(
        "diode_clipper",
        "Symmetric Shockley Diode Clipper",
        "5 Vp 1 kHz sine through 1 kΩ clipped by anti-parallel 1N4148 diodes (~±0.65 V)",
    ),
    TestJigInfo(
        "half_wave_rectifier",
        "Half-Wave Rectifier & Filter",
        "5 Vp AC source -> 1N4148 diode -> 1 kΩ // 10 uF smoothed DC output",
    ),
    TestJigInfo(
        "ring_oscillator",
        "3-Stage Inverter Ring Oscillator",
        "Three 5V CMOS NOT gates in closed ring (f ≈ 1 / (6 * t_pd))",
    ),
    TestJigInfo(
        "potentiometer_divider",
        "Potentiometer Voltage Divider + Meter",
        "5 V DC source -> 10 kΩ potentiometer (50% wiper) monitored by Voltmeter",
    ),
)


def list_test_jigs() -> tuple[TestJigInfo, ...]:
    """Return metadata for all 6 built-in test jigs."""
    return BUILTIN_TEST_JIGS


def _make_rc_lowpass() -> CircuitDocumentV2:
    c_val = 1.0 / (2.0 * math.pi * 1000.0 * 1000.0)
    placed = [
        PlacedComponentSpec(
            component=ACVoltageSource("V1", "n_vin", "0", amplitude=1.0, frequency=1000.0),
            center_px=(100.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Resistor("R1", "n_vin", "n_vout", resistance=1000.0),
            center_px=(240.0, 140.0),
        ),
        PlacedComponentSpec(
            component=Capacitor("C1", "n_vout", "0", capacitance=c_val, esr=0.0),
            center_px=(360.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(240.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    g = g.add_junction((360.0, 140.0), "J1")
    g = g.add_wire(TerminalRef("R1", "b"), JunctionRef("J1"))
    g = g.add_wire(JunctionRef("J1"), TerminalRef("C1", "a"))
    g = g.add_junction((240.0, 260.0), "J2")
    g = g.add_wire(TerminalRef("V1", "neg"), JunctionRef("J2"))
    g = g.add_wire(TerminalRef("C1", "b"), JunctionRef("J2"))
    g = g.add_wire(TerminalRef("GND1", "gnd"), JunctionRef("J2"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_rlc_resonance() -> CircuitDocumentV2:
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(1e-3 * 25e-9))
    placed = [
        PlacedComponentSpec(
            component=ACVoltageSource("V1", "n_vin", "0", amplitude=1.0, frequency=f0),
            center_px=(80.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Resistor("R1", "n_vin", "n1", resistance=50.0),
            center_px=(200.0, 140.0),
        ),
        PlacedComponentSpec(
            component=Inductor("L1", "n1", "n2", inductance=1e-3, dcr=1e-3),
            center_px=(320.0, 140.0),
        ),
        PlacedComponentSpec(
            component=Capacitor("C1", "n2", "0", capacitance=25e-9, esr=0.0),
            center_px=(420.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(240.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    g = g.add_wire(TerminalRef("R1", "b"), TerminalRef("L1", "a"))
    g = g.add_wire(TerminalRef("L1", "b"), TerminalRef("C1", "a"))
    g = g.add_junction((240.0, 260.0), "J1")
    g = g.add_wire(TerminalRef("V1", "neg"), JunctionRef("J1"))
    g = g.add_wire(TerminalRef("C1", "b"), JunctionRef("J1"))
    g = g.add_wire(TerminalRef("GND1", "gnd"), JunctionRef("J1"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_diode_clipper() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(
            component=ACVoltageSource("V1", "n_vin", "0", amplitude=5.0, frequency=1000.0),
            center_px=(80.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Resistor("R1", "n_vin", "n_vout", resistance=1000.0),
            center_px=(200.0, 140.0),
        ),
        PlacedComponentSpec(
            component=Diode("D1", "n_vout", "0"),
            center_px=(320.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Diode("D2", "0", "n_vout"),
            center_px=(420.0, 200.0),
            rotation_deg=270,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(240.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    g = g.add_junction((320.0, 140.0), "J_top")
    g = g.add_wire(TerminalRef("R1", "b"), JunctionRef("J_top"))
    g = g.add_wire(JunctionRef("J_top"), TerminalRef("D1", "anode"))
    g = g.add_wire(JunctionRef("J_top"), TerminalRef("D2", "cathode"))
    g = g.add_junction((240.0, 260.0), "J_gnd")
    g = g.add_wire(TerminalRef("V1", "neg"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("D1", "cathode"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("D2", "anode"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("GND1", "gnd"), JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_half_wave_rectifier() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(
            component=ACVoltageSource("V1", "n_vin", "0", amplitude=5.0, frequency=1000.0),
            center_px=(80.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Diode("D1", "n_vin", "n_dc"),
            center_px=(200.0, 140.0),
        ),
        PlacedComponentSpec(
            component=Resistor("R1", "n_dc", "0", resistance=1000.0),
            center_px=(320.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Capacitor("C1", "n_dc", "0", capacitance=10e-6),
            center_px=(420.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(240.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("D1", "anode"))
    g = g.add_junction((320.0, 140.0), "J_dc")
    g = g.add_wire(TerminalRef("D1", "cathode"), JunctionRef("J_dc"))
    g = g.add_wire(JunctionRef("J_dc"), TerminalRef("R1", "a"))
    g = g.add_wire(JunctionRef("J_dc"), TerminalRef("C1", "a"))
    g = g.add_junction((240.0, 260.0), "J_gnd")
    g = g.add_wire(TerminalRef("V1", "neg"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("R1", "b"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("C1", "b"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("GND1", "gnd"), JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_ring_oscillator() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(
            component=LogicGate("U1", "not", "n3", "n1", t_pd=10e-9, t_edge=2e-9),
            center_px=(140.0, 180.0),
        ),
        PlacedComponentSpec(
            component=LogicGate("U2", "not", "n1", "n2", t_pd=10e-9, t_edge=2e-9),
            center_px=(280.0, 180.0),
        ),
        PlacedComponentSpec(
            component=LogicGate("U3", "not", "n2", "n3", t_pd=10e-9, t_edge=2e-9),
            center_px=(420.0, 180.0),
        ),
        PlacedComponentSpec(
            component=Resistor("R_load", "n3", "0", resistance=100_000.0),
            center_px=(520.0, 220.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(520.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("U1", "out"), TerminalRef("U2", "in"))
    g = g.add_wire(TerminalRef("U2", "out"), TerminalRef("U3", "in"))
    g = g.add_junction((480.0, 180.0), "J_fb")
    g = g.add_wire(TerminalRef("U3", "out"), JunctionRef("J_fb"))
    g = g.add_wire(JunctionRef("J_fb"), TerminalRef("U1", "in"), hints=[(480.0, 100.0), (80.0, 100.0)])
    g = g.add_wire(JunctionRef("J_fb"), TerminalRef("R_load", "a"))
    g = g.add_wire(TerminalRef("R_load", "b"), TerminalRef("GND1", "gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_potentiometer_divider() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(
            component=DCVoltageSource("V1", "n_in", "0", voltage=5.0),
            center_px=(100.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Potentiometer("P1", "n_in", "n_w", "0", resistance=10000.0, alpha=0.5),
            center_px=(240.0, 180.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Voltmeter("VM1", "n_w", "0"),
            center_px=(380.0, 200.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(
            component=Ground("GND1", "0"),
            center_px=(240.0, 300.0),
        ),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("P1", "a"))
    g = g.add_wire(TerminalRef("P1", "w"), TerminalRef("VM1", "pos"))
    g = g.add_junction((240.0, 260.0), "J_gnd")
    g = g.add_wire(TerminalRef("V1", "neg"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("P1", "b"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("VM1", "neg"), JunctionRef("J_gnd"))
    g = g.add_wire(TerminalRef("GND1", "gnd"), JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


_JIG_BUILDERS = {
    "rc_lowpass": _make_rc_lowpass,
    "rlc_resonance": _make_rlc_resonance,
    "diode_clipper": _make_diode_clipper,
    "half_wave_rectifier": _make_half_wave_rectifier,
    "ring_oscillator": _make_ring_oscillator,
    "potentiometer_divider": _make_potentiometer_divider,
}


def build_test_jig(jig_id: str) -> CircuitDocumentV2:
    """Construct a solver-ready `CircuitDocumentV2` for one of the 6 built-in Test Jigs."""
    builder = _JIG_BUILDERS.get(jig_id)
    if builder is None:
        valid = ", ".join(sorted(_JIG_BUILDERS))
        raise KeyError(f"Unknown test jig {jig_id!r}; expected one of: {valid}")
    return builder()
