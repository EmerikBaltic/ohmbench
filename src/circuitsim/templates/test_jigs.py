"""18 built-in, solver-ready WireGraph Test Jigs for benchtop verification (Qt-free)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    Comparator,
    Counter4Bit,
    DCVoltageSource,
    Diode,
    Fuse,
    Ground,
    Inductor,
    LED,
    LogicGate,
    MCUBlock,
    NE555,
    NMOS,
    NPNBJT,
    OpAmp,
    Potentiometer,
    Resistor,
    SPDTRelay,
    SevenSegDisplay,
    Transformer,
    Voltmeter,
    WaveformType,
    ZenerDiode,
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
    TestJigInfo(
        "common_emitter_npn_amp",
        "Common-Emitter NPN BJT Amplifier",
        "12 V voltage-divider biased 2N3904 NPN CE stage with bypass & coupling caps",
    ),
    TestJigInfo(
        "zener_5v1_regulator",
        "5.1 V Zener Shunt Voltage Regulator",
        "12 V DC -> 500 Ω series resistor -> 5.1 V Zener + 1 kΩ load",
    ),
    TestJigInfo(
        "nmos_low_side_driver",
        "NMOS Low-Side LED Switch Driver",
        "5 V gate drive on enhancement NMOS sinking 150 Ω + red LED load",
    ),
    TestJigInfo(
        "opamp_non_inverting_amp",
        "Op-Amp Non-Inverting Amplifier (4x)",
        "2 V input amplified 4.00x (Rf=30 kΩ, Rg=10 kΩ) to +8.00 V",
    ),
    TestJigInfo(
        "opamp_relaxation_oscillator",
        "Op-Amp RC Relaxation Oscillator",
        "±5 V Schmitt-trigger Op-Amp hysteresis divider + RC feedback timing loop",
    ),
    TestJigInfo(
        "mcu_analog_threshold_controller",
        "MCU Analog Threshold LED Controller",
        "MCUBlock sampling 2.5 V A0 divider (ADC ~512) to drive D0 LED indicator",
    ),
    TestJigInfo(
        "555_astable_oscillator",
        "NE555 Timer Astable Oscillator",
        "9 V 555 astable oscillator (R_A=1k, R_B=2k, C=1uF) swinging 3 V..6 V",
    ),
    TestJigInfo(
        "transformer_ac_supply",
        "2:1 Transformer AC Step-Down Supply",
        "20 Vp 100 Hz primary stepped down to 10 Vp across 1 kΩ secondary load",
    ),
    TestJigInfo(
        "spdt_relay_driver",
        "SPDT Relay Dual-Load Switch",
        "6 V coil energizes SPDT relay switching 12 V bus from NC load to NO load",
    ),
    TestJigInfo(
        "overcurrent_fuse_demo",
        "Overcurrent Fuse Protection Demo",
        "12 V supply through 0.5 A resettable fuse into 100 Ω normal load",
    ),
    TestJigInfo(
        "counter_4bit_seven_seg",
        "4-Bit Counter + 7-Segment Hex Display",
        "5 V 1 kHz clock driving 4-bit binary counter directly into 7-segment display",
    ),
    TestJigInfo(
        "comparator_threshold_detector",
        "Voltage Comparator Threshold Detector",
        "LM393-style comparator comparing 5 Vp AC sine against 2.5 V reference",
    ),
)


def list_test_jigs() -> tuple[TestJigInfo, ...]:
    """Return metadata for all 18 built-in test jigs."""
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


def _make_common_emitter_npn_amp() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("VCC", "n_vcc", "0", voltage=12.0), (80.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(ACVoltageSource("VIN", "n_vin", "0", amplitude=0.02, frequency=1000.0), (80.0, 340.0), rotation_deg=90),
        PlacedComponentSpec(Capacitor("CIN", "n_vin", "n_b", capacitance=1e-6), (180.0, 280.0)),
        PlacedComponentSpec(Resistor("R1", "n_vcc", "n_b", resistance=47_000.0), (260.0, 140.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R2", "n_b", "0", resistance=10_000.0), (260.0, 320.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RC", "n_vcc", "n_c", resistance=2_200.0), (380.0, 140.0), rotation_deg=90),
        PlacedComponentSpec(NPNBJT("Q1", "n_b", "n_c", "n_e", beta_f=100.0), (380.0, 240.0)),
        PlacedComponentSpec(Resistor("RE", "n_e", "0", resistance=1_000.0), (380.0, 340.0), rotation_deg=90),
        PlacedComponentSpec(Capacitor("CE", "n_e", "0", capacitance=47e-6), (480.0, 340.0), rotation_deg=90),
        PlacedComponentSpec(Capacitor("COUT", "n_c", "n_out", capacitance=1e-6), (500.0, 200.0)),
        PlacedComponentSpec(Resistor("RL", "n_out", "0", resistance=10_000.0), (600.0, 280.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (300.0, 440.0)),
    ]
    g = WireGraph()
    g = g.add_junction((260.0, 80.0), "J_vcc")
    g = g.add_wire(TerminalRef("VCC", "pos"), JunctionRef("J_vcc"))
    g = g.add_wire(JunctionRef("J_vcc"), TerminalRef("R1", "a"))
    g = g.add_wire(JunctionRef("J_vcc"), TerminalRef("RC", "a"))

    g = g.add_junction((260.0, 240.0), "J_b")
    g = g.add_wire(TerminalRef("R1", "b"), JunctionRef("J_b"))
    g = g.add_wire(JunctionRef("J_b"), TerminalRef("R2", "a"))
    g = g.add_wire(JunctionRef("J_b"), TerminalRef("CIN", "b"))
    g = g.add_wire(JunctionRef("J_b"), TerminalRef("Q1", "B"))
    g = g.add_wire(TerminalRef("VIN", "pos"), TerminalRef("CIN", "a"))

    g = g.add_junction((420.0, 200.0), "J_c")
    g = g.add_wire(TerminalRef("RC", "b"), JunctionRef("J_c"))
    g = g.add_wire(JunctionRef("J_c"), TerminalRef("Q1", "C"))
    g = g.add_wire(JunctionRef("J_c"), TerminalRef("COUT", "a"))
    g = g.add_wire(TerminalRef("COUT", "b"), TerminalRef("RL", "a"))

    g = g.add_junction((420.0, 280.0), "J_e")
    g = g.add_wire(TerminalRef("Q1", "E"), JunctionRef("J_e"))
    g = g.add_wire(JunctionRef("J_e"), TerminalRef("RE", "a"))
    g = g.add_wire(JunctionRef("J_e"), TerminalRef("CE", "a"))

    g = g.add_junction((300.0, 400.0), "J_gnd")
    for t in (
        TerminalRef("VCC", "neg"),
        TerminalRef("VIN", "neg"),
        TerminalRef("R2", "b"),
        TerminalRef("RE", "b"),
        TerminalRef("CE", "b"),
        TerminalRef("RL", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_zener_5v1_regulator() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("V1", "n_in", "0", voltage=12.0), (100.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R1", "n_in", "n_out", resistance=500.0), (220.0, 140.0)),
        PlacedComponentSpec(ZenerDiode("DZ1", "0", "n_out", v_z=5.1), (340.0, 200.0), rotation_deg=270),
        PlacedComponentSpec(Resistor("RL", "n_out", "0", resistance=1_000.0), (460.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (260.0, 300.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("R1", "a"))
    g = g.add_junction((340.0, 140.0), "J_out")
    g = g.add_wire(TerminalRef("R1", "b"), JunctionRef("J_out"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("DZ1", "cathode"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("RL", "a"))
    g = g.add_junction((260.0, 260.0), "J_gnd")
    for t in (
        TerminalRef("V1", "neg"),
        TerminalRef("DZ1", "anode"),
        TerminalRef("RL", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_nmos_low_side_driver() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("VDD", "n_vdd", "0", voltage=5.0), (80.0, 180.0), rotation_deg=90),
        PlacedComponentSpec(DCVoltageSource("VG", "n_g", "0", voltage=5.0), (160.0, 240.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RD", "n_vdd", "n_a", resistance=150.0), (260.0, 100.0)),
        PlacedComponentSpec(LED("D1", "n_a", "n_d", forward_voltage=2.0), (380.0, 140.0), rotation_deg=90),
        PlacedComponentSpec(NMOS("M1", "n_g", "n_d", "0", v_th=1.5, k_p=0.05), (380.0, 240.0)),
        PlacedComponentSpec(Ground("GND1", "0"), (240.0, 340.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("VDD", "pos"), TerminalRef("RD", "a"))
    g = g.add_wire(TerminalRef("RD", "b"), TerminalRef("D1", "anode"))
    g = g.add_wire(TerminalRef("D1", "cathode"), TerminalRef("M1", "D"))
    g = g.add_wire(TerminalRef("VG", "pos"), TerminalRef("M1", "G"))
    g = g.add_junction((240.0, 300.0), "J_gnd")
    for t in (
        TerminalRef("VDD", "neg"),
        TerminalRef("VG", "neg"),
        TerminalRef("M1", "S"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_opamp_non_inverting_amp() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("VIN", "n_in", "0", voltage=2.0), (80.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(OpAmp("U1", "n_in", "n_inv", "n_out"), (260.0, 200.0)),
        PlacedComponentSpec(Resistor("RF", "n_inv", "n_out", resistance=30_000.0), (260.0, 100.0)),
        PlacedComponentSpec(Resistor("RG", "n_inv", "0", resistance=10_000.0), (180.0, 300.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RL", "n_out", "0", resistance=10_000.0), (420.0, 260.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (240.0, 380.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("VIN", "pos"), TerminalRef("U1", "in_pos"))
    g = g.add_junction((180.0, 220.0), "J_inv")
    g = g.add_wire(TerminalRef("U1", "in_neg"), JunctionRef("J_inv"))
    g = g.add_wire(JunctionRef("J_inv"), TerminalRef("RG", "a"))
    g = g.add_wire(JunctionRef("J_inv"), TerminalRef("RF", "a"))
    g = g.add_junction((360.0, 200.0), "J_out")
    g = g.add_wire(TerminalRef("U1", "out"), JunctionRef("J_out"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("RF", "b"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("RL", "a"))
    g = g.add_junction((240.0, 340.0), "J_gnd")
    for t in (
        TerminalRef("VIN", "neg"),
        TerminalRef("RG", "b"),
        TerminalRef("RL", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_opamp_relaxation_oscillator() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(OpAmp("U1", "n_pos", "n_inv", "n_out", v_rail_pos=5.0, v_rail_neg=-5.0), (260.0, 200.0)),
        PlacedComponentSpec(Resistor("RF", "n_out", "n_inv", resistance=2_000.0), (260.0, 100.0)),
        PlacedComponentSpec(Capacitor("C1", "n_inv", "0", capacitance=22e-9, initial_voltage=0.0), (140.0, 280.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R1", "n_out", "n_pos", resistance=10_000.0), (260.0, 300.0)),
        PlacedComponentSpec(Resistor("R2", "n_pos", "0", resistance=10_000.0), (140.0, 360.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RL", "n_out", "0", resistance=10_000.0), (420.0, 280.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (260.0, 420.0)),
    ]
    g = WireGraph()
    g = g.add_junction((360.0, 200.0), "J_out")
    g = g.add_wire(TerminalRef("U1", "out"), JunctionRef("J_out"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("RF", "a"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("R1", "a"))
    g = g.add_wire(JunctionRef("J_out"), TerminalRef("RL", "a"))

    g = g.add_junction((160.0, 180.0), "J_inv")
    g = g.add_wire(TerminalRef("U1", "in_neg"), JunctionRef("J_inv"))
    g = g.add_wire(JunctionRef("J_inv"), TerminalRef("RF", "b"))
    g = g.add_wire(JunctionRef("J_inv"), TerminalRef("C1", "a"))

    g = g.add_junction((160.0, 240.0), "J_pos")
    g = g.add_wire(TerminalRef("U1", "in_pos"), JunctionRef("J_pos"))
    g = g.add_wire(JunctionRef("J_pos"), TerminalRef("R1", "b"))
    g = g.add_wire(JunctionRef("J_pos"), TerminalRef("R2", "a"))

    g = g.add_junction((260.0, 380.0), "J_gnd")
    for t in (
        TerminalRef("C1", "b"),
        TerminalRef("R2", "b"),
        TerminalRef("RL", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_mcu_analog_threshold_controller() -> CircuitDocumentV2:
    code_src = (
        "val = mcu.adc_read('A0')\n"
        "d0_on = val > 400\n"
        "mcu.digital_write('D0', d0_on)\n"
        "mcu.log(f'A0={val} D0={d0_on}')\n"
    )
    placed = [
        PlacedComponentSpec(DCVoltageSource("V1", "n_vcc", "0", voltage=5.0), (80.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R_top", "n_vcc", "n_adc", resistance=10_000.0), (180.0, 140.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R_bot", "n_adc", "0", resistance=10_000.0), (180.0, 260.0), rotation_deg=90),
        PlacedComponentSpec(MCUBlock("U1", "n_adc", "0", "n_d0", "0", "0", "0", code=code_src), (340.0, 200.0)),
        PlacedComponentSpec(Resistor("R_led", "n_d0", "n_a", resistance=220.0), (500.0, 160.0)),
        PlacedComponentSpec(LED("D1", "n_a", "0", forward_voltage=2.0), (580.0, 240.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (300.0, 360.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("R_top", "a"))
    g = g.add_junction((180.0, 200.0), "J_adc")
    g = g.add_wire(TerminalRef("R_top", "b"), JunctionRef("J_adc"))
    g = g.add_wire(JunctionRef("J_adc"), TerminalRef("R_bot", "a"))
    g = g.add_wire(JunctionRef("J_adc"), TerminalRef("U1", "A0"))

    g = g.add_wire(TerminalRef("U1", "D0"), TerminalRef("R_led", "a"))
    g = g.add_wire(TerminalRef("R_led", "b"), TerminalRef("D1", "anode"))

    g = g.add_junction((300.0, 320.0), "J_gnd")
    for t in (
        TerminalRef("V1", "neg"),
        TerminalRef("R_bot", "b"),
        TerminalRef("D1", "cathode"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_555_astable_oscillator() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("V1", "n_vcc", "0", voltage=9.0), (80.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RA", "n_vcc", "n_dis", resistance=1_000.0), (180.0, 80.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("RB", "n_dis", "n_cap", resistance=2_000.0), (180.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Capacitor("C1", "n_cap", "0", capacitance=1e-6), (180.0, 320.0), rotation_deg=90),
        PlacedComponentSpec(NE555("U1", "n_vcc", "0", "n_cap", "n_cap", "n_dis", "n_out", "n_vcc"), (340.0, 200.0)),
        PlacedComponentSpec(Resistor("R_OUT", "n_out", "0", resistance=1_000.0), (500.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (260.0, 420.0)),
    ]
    g = WireGraph()
    g = g.add_junction((200.0, 20.0), "J_vcc")
    for t in (
        TerminalRef("V1", "pos"),
        TerminalRef("RA", "a"),
        TerminalRef("U1", "vcc"),
        TerminalRef("U1", "reset"),
    ):
        g = g.add_wire(t, JunctionRef("J_vcc"))
    g = g.add_junction((240.0, 140.0), "J_dis")
    for t in (TerminalRef("RA", "b"), TerminalRef("RB", "a"), TerminalRef("U1", "dis")):
        g = g.add_wire(t, JunctionRef("J_dis"))
    g = g.add_junction((240.0, 260.0), "J_cap")
    for t in (
        TerminalRef("RB", "b"),
        TerminalRef("C1", "a"),
        TerminalRef("U1", "trig"),
        TerminalRef("U1", "thr"),
    ):
        g = g.add_wire(t, JunctionRef("J_cap"))
    g = g.add_wire(TerminalRef("U1", "out"), TerminalRef("R_OUT", "a"))
    g = g.add_junction((260.0, 380.0), "J_gnd")
    for t in (
        TerminalRef("V1", "neg"),
        TerminalRef("C1", "b"),
        TerminalRef("U1", "gnd"),
        TerminalRef("R_OUT", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_transformer_ac_supply() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(ACVoltageSource("V1", "n_p", "0", amplitude=20.0, frequency=100.0), (100.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Transformer("T1", "n_p", "0", "n_s", "0", turns_ratio=0.5), (260.0, 200.0)),
        PlacedComponentSpec(Resistor("R_LOAD", "n_s", "0", resistance=1_000.0), (420.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (260.0, 320.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("T1", "p1"))
    g = g.add_wire(TerminalRef("T1", "s1"), TerminalRef("R_LOAD", "a"))
    g = g.add_junction((260.0, 280.0), "J_gnd")
    for t in (
        TerminalRef("V1", "neg"),
        TerminalRef("T1", "p2"),
        TerminalRef("T1", "s2"),
        TerminalRef("R_LOAD", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_spdt_relay_driver() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("V_COIL", "n_coil", "0", voltage=6.0), (80.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(SPDTRelay("K1", "n_coil", "0", "n_com", "n_no", "n_nc"), (260.0, 200.0)),
        PlacedComponentSpec(DCVoltageSource("V_BUS", "n_com", "0", voltage=12.0), (440.0, 80.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R_NO", "n_no", "0", resistance=100.0), (440.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Resistor("R_NC", "n_nc", "0", resistance=100.0), (440.0, 320.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (260.0, 380.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V_COIL", "pos"), TerminalRef("K1", "coil_pos"))
    g = g.add_wire(TerminalRef("K1", "com"), TerminalRef("V_BUS", "pos"))
    g = g.add_wire(TerminalRef("K1", "no"), TerminalRef("R_NO", "a"))
    g = g.add_wire(TerminalRef("K1", "nc"), TerminalRef("R_NC", "a"))
    g = g.add_junction((260.0, 340.0), "J_gnd")
    for t in (
        TerminalRef("V_COIL", "neg"),
        TerminalRef("K1", "coil_neg"),
        TerminalRef("V_BUS", "neg"),
        TerminalRef("R_NO", "b"),
        TerminalRef("R_NC", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_overcurrent_fuse_demo() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(DCVoltageSource("V1", "n_in", "0", voltage=12.0), (100.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Fuse("F1", "n_in", "n_load", rating_amps=0.5), (240.0, 140.0)),
        PlacedComponentSpec(Resistor("R_LOAD", "n_load", "0", resistance=100.0), (380.0, 200.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (240.0, 320.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V1", "pos"), TerminalRef("F1", "a"))
    g = g.add_wire(TerminalRef("F1", "b"), TerminalRef("R_LOAD", "a"))
    g = g.add_junction((240.0, 280.0), "J_gnd")
    for t in (
        TerminalRef("V1", "neg"),
        TerminalRef("R_LOAD", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_counter_4bit_seven_seg() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(
            ACVoltageSource(
                "V_CLK",
                "clk",
                "0",
                amplitude=2.5,
                frequency=1000.0,
                dc_offset=2.5,
                waveform=WaveformType.SQUARE,
            ),
            (80.0, 180.0),
            rotation_deg=90,
        ),
        PlacedComponentSpec(DCVoltageSource("V_RST", "rst", "0", voltage=0.0), (80.0, 300.0), rotation_deg=90),
        PlacedComponentSpec(Counter4Bit("U1", "clk", "rst", "q0", "q1", "q2", "q3"), (260.0, 220.0)),
        PlacedComponentSpec(SevenSegDisplay("DS1", "q0", "q1", "q2", "q3"), (460.0, 220.0)),
        PlacedComponentSpec(Ground("GND1", "0"), (140.0, 380.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V_CLK", "pos"), TerminalRef("U1", "CLK"))
    g = g.add_wire(TerminalRef("V_RST", "pos"), TerminalRef("U1", "RST"))
    g = g.add_wire(TerminalRef("U1", "Q0"), TerminalRef("DS1", "D0"))
    g = g.add_wire(TerminalRef("U1", "Q1"), TerminalRef("DS1", "D1"))
    g = g.add_wire(TerminalRef("U1", "Q2"), TerminalRef("DS1", "D2"))
    g = g.add_wire(TerminalRef("U1", "Q3"), TerminalRef("DS1", "D3"))
    g = g.add_junction((140.0, 340.0), "J_gnd")
    for t in (
        TerminalRef("V_CLK", "neg"),
        TerminalRef("V_RST", "neg"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


def _make_comparator_threshold_detector() -> CircuitDocumentV2:
    placed = [
        PlacedComponentSpec(ACVoltageSource("V_IN", "n_in", "0", amplitude=5.0, frequency=100.0), (100.0, 160.0), rotation_deg=90),
        PlacedComponentSpec(DCVoltageSource("V_REF", "n_ref", "0", voltage=2.5), (100.0, 280.0), rotation_deg=90),
        PlacedComponentSpec(Comparator("U1", "n_in", "n_ref", "n_out", v_high=5.0, v_low=0.0), (280.0, 200.0)),
        PlacedComponentSpec(Resistor("R_LOAD", "n_out", "0", resistance=10_000.0), (420.0, 240.0), rotation_deg=90),
        PlacedComponentSpec(Ground("GND1", "0"), (240.0, 360.0)),
    ]
    g = WireGraph()
    g = g.add_wire(TerminalRef("V_IN", "pos"), TerminalRef("U1", "in_pos"))
    g = g.add_wire(TerminalRef("V_REF", "pos"), TerminalRef("U1", "in_neg"))
    g = g.add_wire(TerminalRef("U1", "out"), TerminalRef("R_LOAD", "a"))
    g = g.add_junction((240.0, 320.0), "J_gnd")
    for t in (
        TerminalRef("V_IN", "neg"),
        TerminalRef("V_REF", "neg"),
        TerminalRef("R_LOAD", "b"),
        TerminalRef("GND1", "gnd"),
    ):
        g = g.add_wire(t, JunctionRef("J_gnd"))
    return CircuitDocumentV2(placed=placed, graph=g)


_JIG_BUILDERS = {
    "rc_lowpass": _make_rc_lowpass,
    "rlc_resonance": _make_rlc_resonance,
    "diode_clipper": _make_diode_clipper,
    "half_wave_rectifier": _make_half_wave_rectifier,
    "ring_oscillator": _make_ring_oscillator,
    "potentiometer_divider": _make_potentiometer_divider,
    "common_emitter_npn_amp": _make_common_emitter_npn_amp,
    "zener_5v1_regulator": _make_zener_5v1_regulator,
    "nmos_low_side_driver": _make_nmos_low_side_driver,
    "opamp_non_inverting_amp": _make_opamp_non_inverting_amp,
    "opamp_relaxation_oscillator": _make_opamp_relaxation_oscillator,
    "mcu_analog_threshold_controller": _make_mcu_analog_threshold_controller,
    "555_astable_oscillator": _make_555_astable_oscillator,
    "transformer_ac_supply": _make_transformer_ac_supply,
    "spdt_relay_driver": _make_spdt_relay_driver,
    "overcurrent_fuse_demo": _make_overcurrent_fuse_demo,
    "counter_4bit_seven_seg": _make_counter_4bit_seven_seg,
    "comparator_threshold_detector": _make_comparator_threshold_detector,
}


def build_test_jig(jig_id: str) -> CircuitDocumentV2:
    """Construct a solver-ready `CircuitDocumentV2` for one of the 18 built-in Test Jigs."""
    builder = _JIG_BUILDERS.get(jig_id)
    if builder is None:
        valid = ", ".join(sorted(_JIG_BUILDERS))
        raise KeyError(f"Unknown test jig {jig_id!r}; expected one of: {valid}")
    return builder()
