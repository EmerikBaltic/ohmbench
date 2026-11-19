"""Explicit wiring model: attachments, junctions, netlist and routing (no Qt)."""

from circuitsim.wiring.model import (
    Attachment,
    FreeEnd,
    Junction,
    JunctionRef,
    SplitResult,
    TerminalRef,
    Wire,
    WireGraph,
    WireGraphError,
)
from circuitsim.wiring.netlist import GROUND_NODE, WiringNetlist, build_circuit
from circuitsim.wiring.overlay import (
    compute_current_dot_velocity,
    estimate_wire_currents,
    format_power_probe_text,
    format_watts_si,
    map_voltage_to_rgb,
)
from circuitsim.wiring.router import route
from circuitsim.wiring.serializer import (
    CircuitDocumentV2,
    SCHEMA_VERSION_V2,
    dict_to_document_v2,
    document_to_dict_v2,
    dumps_document_v2,
    load_document_v2,
    loads_document_v2,
    migrate_v1_to_v2,
    save_document_v2,
)

__all__ = [
    "Attachment",
    "CircuitDocumentV2",
    "FreeEnd",
    "GROUND_NODE",
    "Junction",
    "JunctionRef",
    "SCHEMA_VERSION_V2",
    "SplitResult",
    "TerminalRef",
    "Wire",
    "WireGraph",
    "WireGraphError",
    "WiringNetlist",
    "build_circuit",
    "compute_current_dot_velocity",
    "dict_to_document_v2",
    "document_to_dict_v2",
    "dumps_document_v2",
    "estimate_wire_currents",
    "format_power_probe_text",
    "format_watts_si",
    "load_document_v2",
    "loads_document_v2",
    "map_voltage_to_rgb",
    "migrate_v1_to_v2",
    "route",
    "save_document_v2",
]
