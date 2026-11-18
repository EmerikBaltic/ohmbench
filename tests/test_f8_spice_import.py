"""Tests for Item F8: 3rd-party SPICE .MODEL / .SUBCKT importer, pin mapper, encrypted guard."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import pytest

from circuitsim.components import DCVoltageSource, Ground, Resistor
from circuitsim.engine import Circuit, MNASolver
from circuitsim.io.spice_import import (
    EncryptedModelError,
    SpiceImportError,
    instantiate_diode_model,
    instantiate_subckt_elements,
    map_subckt_pins,
    parse_spice_file,
    parse_spice_library,
    parse_spice_number,
)


def test_f8_spice_import_is_qt_free() -> None:
    """Verify `circuitsim.io.spice_import` has no Qt dependencies."""
    path = Path(__file__).resolve().parent.parent / "src" / "circuitsim" / "io" / "spice_import.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f8_parse_1n4148_model_and_shockley_solver_match() -> None:
    """Verify multiline SPICE .MODEL D1N4148 imports with exact N/IS/RS/BV and solves identically."""
    lib_text = """
* Standard 1N4148 switching diode model
.MODEL D1N4148 D(
+ IS=2.52E-9
+ N=1.752
+ RS=0.568
+ BV=100
+ IBV=100u
+ )
.ENDS
"""
    lib = parse_spice_library(lib_text)
    assert "D1N4148" in lib.models
    assert math.isclose(parse_spice_number("1meg"), 1e6, rel_tol=1e-12)
    assert math.isclose(parse_spice_number("4.7k"), 4700.0, rel_tol=1e-12)
    assert math.isclose(parse_spice_number("100u"), 1e-4, rel_tol=1e-12)

    d_imp = instantiate_diode_model(lib.models["D1N4148"], "D1", "out", "0")
    assert math.isclose(d_imp.is_sat, 2.52e-9, rel_tol=1e-12)
    assert math.isclose(d_imp.n, 1.752, rel_tol=1e-12)
    assert math.isclose(d_imp.rs, 0.568, rel_tol=1e-12)
    assert math.isclose(d_imp.bv, 100.0, rel_tol=1e-12)
    assert math.isclose(d_imp.ibv, 1e-4, rel_tol=1e-12)

    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "in", "0", voltage=5.0))
    c.add(Resistor("R1", "in", "out", resistance=1000.0))
    c.add(d_imp)
    res = MNASolver(c).solve_dc()
    v_d = res.node_voltages["out"]
    i_d = res.branch_currents["D1"]
    assert 0.62 <= v_d <= 0.70
    assert math.isclose(i_d, (5.0 - v_d) / 1000.0, rel_tol=1e-8)


def test_f8_subckt_explicit_pin_order_mapper_and_expansion() -> None:
    """Verify `.SUBCKT` reversed pins remap cleanly via explicit pin mapper."""
    subckt_text = """
* Subcircuit with cathode K first, anode A second
.SUBCKT REV_CLAMP K A
D1 A K D_FAST
R1 A K 100k
.MODEL D_FAST D(IS=2.52n N=1.752 RS=0.568 BV=75 IBV=100u)
.ENDS REV_CLAMP
"""
    lib = parse_spice_library(subckt_text)
    sub = lib.subckts["REV_CLAMP"]
    assert sub.pins == ("K", "A")

    # Duplicate canonical target must fail loudly
    with pytest.raises(SpiceImportError, match="Duplicate"):
        map_subckt_pins(sub, {"K": "anode", "A": "anode"})

    # Ordered list remapping: K -> cathode, A -> anode
    pin_map = map_subckt_pins(sub, ["cathode", "anode"])
    assert pin_map == {"K": "cathode", "A": "anode"}

    # Forward test (+5 V applied to canonical anode): forward current through D1 + R1
    c_fwd = Circuit()
    c_fwd.add(Ground("GND", "0"))
    c_fwd.add(DCVoltageSource("V1", "src", "0", voltage=5.0))
    c_fwd.add(Resistor("Rser", "src", "v_an", resistance=1000.0))
    for comp in instantiate_subckt_elements(
        sub, "X1", pin_map, {"anode": "v_an", "cathode": "0"}
    ):
        c_fwd.add(comp)
    res_fwd = MNASolver(c_fwd).solve_dc()
    assert 0.60 < res_fwd.node_voltages["v_an"] < 0.72

    # Reverse test (-5 V on anode -> only R1 = 100k divider passes ~ -4.95 V)
    c_rev = Circuit()
    c_rev.add(Ground("GND", "0"))
    c_rev.add(DCVoltageSource("V1", "src", "0", voltage=-5.0))
    c_rev.add(Resistor("Rser", "src", "v_an", resistance=1000.0))
    for comp in instantiate_subckt_elements(
        sub, "X1", pin_map, {"anode": "v_an", "cathode": "0"}
    ):
        c_rev.add(comp)
    res_rev = MNASolver(c_rev).solve_dc()
    expected_v_rev = -5.0 * (100_000.0 / 101_000.0)
    assert math.isclose(res_rev.node_voltages["v_an"], expected_v_rev, rel_tol=1e-4)


@pytest.mark.parametrize(
    "payload",
    [
        "$CDNENCSTART\nabc123\n$CDNENCFINISH",
        "* Vendor model\n.prot\nABCXYZ==\n.unprot",
        "`pragma protect begin_protected\nabc\n`pragma protect end_protected",
        "*ENCRYPTED MODEL DATA",
        b"\x00\x01\x02\x03ENCRYPTED",
    ],
)
def test_f8_encrypted_vendor_detector_raises(tmp_path: Path, payload: str | bytes) -> None:
    """PSpice, HSPICE, IEEE-1735, and binary encrypted SPICE models raise EncryptedModelError."""
    with pytest.raises(EncryptedModelError):
        parse_spice_library(payload)

    fpath = tmp_path / "encrypted.lib"
    if isinstance(payload, bytes):
        fpath.write_bytes(payload)
    else:
        fpath.write_text(payload, encoding="utf-8")
    with pytest.raises(EncryptedModelError):
        parse_spice_file(fpath)


def test_f8_import_spice_dialog_ui(qapp) -> None:
    """Verify ImportSPICEDialog parses models, populates subckt pin table, and handles errors."""
    from circuitsim.dialogs.spice_import_dialog import ImportSPICEDialog

    dlg = ImportSPICEDialog()
    try:
        assert dlg.parse_text(
            ".SUBCKT MY_DIODE K A\nD1 A K DMOD\n.MODEL DMOD D(IS=1n N=1.5 RS=0.3 BV=50)\n.ENDS\n"
        )
        assert dlg.pin_table.rowCount() == 2
        assert dlg.pin_table.item(0, 0).text() == "K"
        assert dlg.pin_table.item(1, 0).text() == "A"

        # Encrypted content sets user-facing error without crashing
        assert dlg.parse_text("$CDNENCSTART\nENCRYPTED_BLOB\n") is False
        assert "encrypted" in dlg.last_error.lower()
    finally:
        dlg.close()
