"""Qt-free factory that creates default component models from a type id.

Used by the canvas (placing a part from the palette) and by the `.circ` file
loader (rebuilding a part from its saved type name).
"""

from __future__ import annotations

from circuitsim.components.base import Component
from circuitsim.components.bjt import NPNBJT, PNPBJT
from circuitsim.components.capacitor import Capacitor
from circuitsim.components.comparator import Comparator
from circuitsim.components.ground import Ground
from circuitsim.components.inductor import Inductor
from circuitsim.components.diode import Diode, ZenerDiode
from circuitsim.components.function_generator import FunctionGenerator
from circuitsim.components.led import LED
from circuitsim.components.lightbulb import Lightbulb
from circuitsim.components.logic_gate import GATE_COMPONENT_TYPES, GATE_OPERATIONS, LogicGate
from circuitsim.components.mcu import MCUBlock
from circuitsim.components.meters import Ammeter, Voltmeter
from circuitsim.components.mosfet import NMOS, PMOS
from circuitsim.components.ne555 import NE555
from circuitsim.components.opamp import OpAmp
from circuitsim.components.oscilloscope import Oscilloscope
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.power import Fuse, SPDTRelay, Transformer
from circuitsim.components.resistor import Resistor
from circuitsim.components.sequential import Counter4Bit, DFlipFlop, JKFlipFlop, SevenSegDisplay
from circuitsim.components.sources import (
    ACVoltageSource,
    DCCurrentSource,
    DCVoltageSource,
    WavVoltageSource,
)
from circuitsim.components.switch import Switch

KNOWN_COMPONENT_TYPES: tuple[str, ...] = (
    "resistor",
    "capacitor",
    "inductor",
    "potentiometer",
    "dc_voltage",
    "ac_voltage",
    "function_generator",
    "wav_voltage",
    "dc_current",
    "ground",
    "switch",
    "transformer",
    "spdt_relay",
    "fuse",
    "lightbulb",
    "led",
    "diode",
    "zener_diode",
    "npn_bjt",
    "pnp_bjt",
    "nmos",
    "pmos",
    "opamp",
    "comparator",
    "ne555",
    "voltmeter",
    "ammeter",
    "mcu",
    "d_flipflop",
    "jk_flipflop",
    "counter_4bit",
    "seven_seg_display",
    "oscilloscope",
    *GATE_COMPONENT_TYPES,
)


def create_default_component_model(component_id: str, name: str) -> Component:
    """Create a default physics `Component` instance for `component_id`."""
    n0, n1 = f"{name}_a", f"{name}_b"
    if component_id == "resistor":
        return Resistor(name, n0, n1, resistance=1000.0)
    if component_id == "capacitor":
        return Capacitor(name, n0, n1, capacitance=1e-6)
    if component_id == "inductor":
        return Inductor(name, n0, n1, inductance=10e-3, dcr=1.0)
    if component_id == "potentiometer":
        return Potentiometer(name, f"{name}_a", f"{name}_w", f"{name}_b", resistance=10_000.0, alpha=0.5)
    if component_id == "dc_voltage":
        return DCVoltageSource(name, n0, n1, voltage=12.0)
    if component_id == "ac_voltage":
        return ACVoltageSource(name, n0, n1, amplitude=10.0, frequency=1000.0)
    if component_id == "function_generator":
        return FunctionGenerator(name, f"{name}_out", f"{name}_gnd", amplitude=5.0, frequency=100.0)
    if component_id == "wav_voltage":
        return WavVoltageSource(name, n0, n1)
    if component_id == "dc_current":
        return DCCurrentSource(name, n0, n1, current=0.01)
    if component_id == "ground":
        return Ground(name, "0")
    if component_id == "switch":
        return Switch(name, n0, n1, closed=False)
    if component_id == "transformer":
        return Transformer(name, f"{name}_p1", f"{name}_p2", f"{name}_s1", f"{name}_s2")
    if component_id == "spdt_relay":
        return SPDTRelay(
            name,
            f"{name}_coil_pos",
            f"{name}_coil_neg",
            f"{name}_com",
            f"{name}_no",
            f"{name}_nc",
        )
    if component_id == "fuse":
        return Fuse(name, n0, n1)
    if component_id == "lightbulb":
        return Lightbulb(name, n0, n1, rated_power=60.0, rated_voltage=120.0)
    if component_id == "led":
        return LED(name, n0, n1, forward_voltage=2.0, color_name="Red")
    if component_id == "diode":
        return Diode(name, n0, n1)
    if component_id == "zener_diode":
        return ZenerDiode(name, n0, n1)
    if component_id == "npn_bjt":
        return NPNBJT(name, f"{name}_B", f"{name}_C", f"{name}_E")
    if component_id == "pnp_bjt":
        return PNPBJT(name, f"{name}_B", f"{name}_C", f"{name}_E")
    if component_id == "nmos":
        return NMOS(name, f"{name}_G", f"{name}_D", f"{name}_S")
    if component_id == "pmos":
        return PMOS(name, f"{name}_G", f"{name}_D", f"{name}_S")
    if component_id == "opamp":
        return OpAmp(name, f"{name}_in_pos", f"{name}_in_neg", f"{name}_out")
    if component_id == "comparator":
        return Comparator(name, f"{name}_in_pos", f"{name}_in_neg", f"{name}_out")
    if component_id == "ne555":
        return NE555(name)
    if component_id == "voltmeter":
        return Voltmeter(name, n0, n1)
    if component_id == "ammeter":
        return Ammeter(name, n0, n1)
    if component_id == "mcu":
        return MCUBlock(name)
    if component_id == "d_flipflop":
        return DFlipFlop(name)
    if component_id == "jk_flipflop":
        return JKFlipFlop(name)
    if component_id == "counter_4bit":
        return Counter4Bit(name)
    if component_id == "seven_seg_display":
        return SevenSegDisplay(name)
    if component_id == "oscilloscope":
        return Oscilloscope(name)
    if component_id.startswith("gate_"):
        op = component_id.removeprefix("gate_")
        if op in GATE_OPERATIONS:
            if op == "not":
                return LogicGate(name, op, f"{name}_in", f"{name}_out")
            return LogicGate(name, op, (f"{name}_a", f"{name}_b"), f"{name}_out")
    raise ValueError(f"Unknown component_id {component_id!r}.")
