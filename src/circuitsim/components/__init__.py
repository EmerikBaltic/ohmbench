"""Component physics models for CircuitSim (no Qt dependencies)."""

from circuitsim.components.base import Component, ComponentParameter
from circuitsim.components.capacitor import Capacitor
from circuitsim.components.diode import Diode
from circuitsim.components.factory import (
    KNOWN_COMPONENT_TYPES,
    create_default_component_model,
)
from circuitsim.components.ground import Ground
from circuitsim.components.inductor import Inductor
from circuitsim.components.led import LED, LED_COLOUR_PRESETS
from circuitsim.components.lightbulb import Lightbulb
from circuitsim.components.logic_gate import (
    GATE_COMPONENT_TYPES,
    GATE_OPERATIONS,
    LOGIC_FAMILY_PRESETS,
    LogicGate,
    evaluate_gate_boolean,
    normalize_logic_family,
)
from circuitsim.components.meters import Ammeter, MeterMode, Voltmeter
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.resistor import Resistor
from circuitsim.components.sources import (
    ACVoltageSource,
    DCCurrentSource,
    DCVoltageSource,
    WavVoltageSource,
    WaveformType,
)
from circuitsim.components.switch import Switch

__all__ = [
    "ACVoltageSource",
    "Ammeter",
    "Capacitor",
    "Component",
    "ComponentParameter",
    "DCCurrentSource",
    "DCVoltageSource",
    "Diode",
    "GATE_COMPONENT_TYPES",
    "GATE_OPERATIONS",
    "Ground",
    "Inductor",
    "KNOWN_COMPONENT_TYPES",
    "LED",
    "LED_COLOUR_PRESETS",
    "LOGIC_FAMILY_PRESETS",
    "Lightbulb",
    "LogicGate",
    "MeterMode",
    "Potentiometer",
    "Resistor",
    "Switch",
    "Voltmeter",
    "WavVoltageSource",
    "WaveformType",
    "create_default_component_model",
    "evaluate_gate_boolean",
    "normalize_logic_family",
]
