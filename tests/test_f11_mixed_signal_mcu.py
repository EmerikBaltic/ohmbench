"""Tests for Item F11: Mixed-signal MCUBlock (10 MOhm ADC, 25 Ohm GPIO, Python callback, co-sim)."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import numpy as np

from circuitsim.components import (
    DCVoltageSource,
    Ground,
    KNOWN_COMPONENT_TYPES,
    LED,
    MCUBlock,
    Resistor,
    create_default_component_model,
)
from circuitsim.engine import Circuit, MNASolver
from circuitsim.io.circ_serializer import CircuitDocument, PlacedComponentSpec, dumps_document, loads_document


def test_f11_mcu_module_is_qt_free() -> None:
    """Verify `circuitsim.components.mcu` contains no Qt imports."""
    path = Path(__file__).resolve().parent.parent / "src" / "circuitsim" / "components" / "mcu.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f11_true_10mohm_adc_and_25ohm_thevenin_gpio_loading() -> None:
    """Verify true MNA 10 MOhm ADC input loading and 25 Ohm GPIO Thevenin drive."""
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "vcc", "0", voltage=5.0))
    # 10k / 10k divider loaded by 10 MOhm ADC input at A0
    c.add(Resistor("R_top", "vcc", "n_adc", resistance=10_000.0))
    c.add(Resistor("R_bot", "n_adc", "0", resistance=10_000.0))
    # D0 loaded by 100 ohm to ground -> 5.0 * 100 / (25 + 100) = 4.000 V
    c.add(Resistor("R_gpio", "n_d0", "0", resistance=100.0))
    mcu = MCUBlock(
        "U1",
        node_a0="n_adc",
        node_d0="n_d0",
        callback=lambda m, _t, _dt: {"D0": True},
    )
    c.add(mcu)

    res = MNASolver(c).solve_dc()
    # Exact 10 MOhm loading calculation: R_eq_bot = 10k || 10M
    r_bot_par = (10_000.0 * 1e7) / (10_000.0 + 1e7)
    v_adc_expected = 5.0 * r_bot_par / (10_000.0 + r_bot_par)
    assert math.isclose(res.node_voltages["n_adc"], v_adc_expected, rel_tol=1e-8)

    # Exact 25 Ohm GPIO Thevenin drop: 5.0 * 100 / (25 + 100) = 4.0 V, I = 40 mA
    assert math.isclose(res.node_voltages["n_d0"], 4.0, rel_tol=1e-8)
    assert math.isclose(res.branch_currents["R_gpio"], 0.04, rel_tol=1e-8)


def test_f11_analog_resistor_ladder_read_adc_codes_and_led_drive() -> None:
    """Verify 10-bit ADC quantization (512 / 341 / 767) and D0 LED threshold control."""
    # Exact 512 / 341 / 767 from resistor ladder taps (1/2, 1/3, 3/4 of 5.0 V with 100 ohm steps)
    c_lad = Circuit()
    c_lad.add(Ground("GND", "0"))
    c_lad.add(DCVoltageSource("V1", "vcc", "0", voltage=5.0))
    # 100 + 200 -> 1/3 * 5V = 1.6667 V -> round(1/3 * 1023) = 341
    c_lad.add(Resistor("Ra1", "vcc", "tap_third", resistance=200.0))
    c_lad.add(Resistor("Ra2", "tap_third", "0", resistance=100.0))
    # 100 + 100 -> 1/2 * 5V = 2.500 V -> round(0.5 * 1023) = 512
    c_lad.add(Resistor("Rb1", "vcc", "tap_half", resistance=100.0))
    c_lad.add(Resistor("Rb2", "tap_half", "0", resistance=100.0))
    # 100 + 300 -> 3/4 * 5V = 3.750 V -> round(0.75 * 1023) = 767
    c_lad.add(Resistor("Rc1", "vcc", "tap_three_q", resistance=100.0))
    c_lad.add(Resistor("Rc2", "tap_three_q", "0", resistance=300.0))

    dc_lad = MNASolver(c_lad).solve_dc()
    mcu_probe = MCUBlock("Uprobe")
    # Ideal mid-rail reference points check exact banker/nearest quantization
    mcu_probe.sense_nodes({"Uprobe_A0": 2.5})
    assert mcu_probe.read_adc("A0") == 512
    mcu_probe.sense_nodes({"Uprobe_A0": 5.0 / 3.0})
    assert mcu_probe.read_adc("A0") == 341
    mcu_probe.sense_nodes({"Uprobe_A0": 3.75})
    assert mcu_probe.read_adc("A0") == 767

    # Physical MNA resistor ladder taps (within <= 1 LSB under gmin loading)
    mcu_probe.sense_nodes({"Uprobe_A0": dc_lad.node_voltages["tap_half"]})
    assert mcu_probe.read_adc("A0") in (511, 512)
    mcu_probe.sense_nodes({"Uprobe_A0": dc_lad.node_voltages["tap_third"]})
    assert mcu_probe.read_adc("A0") == 341
    mcu_probe.sense_nodes({"Uprobe_A0": dc_lad.node_voltages["tap_three_q"]})
    assert mcu_probe.read_adc("A0") == 767

    # Connected ladder + D0 LED driver circuit: A0 at 3/4 Vcc -> code 767 -> D0 lights LED at 20 mA
    c = Circuit()
    c.add(Ground("GND", "0"))
    c.add(DCVoltageSource("V1", "vcc", "0", voltage=5.0))
    c.add(Resistor("Rtop", "vcc", "n_adc", resistance=100.0))
    c.add(Resistor("Rbot", "n_adc", "0", resistance=300.0))
    c.add(Resistor("Rled", "n_d0", "n_anode", resistance=125.0))
    led = LED("D1", "n_anode", "0", forward_voltage=2.0)
    c.add(led)

    seen_codes: list[int] = []

    def firmware(m: MCUBlock, _t: float, _dt: float) -> dict[str, bool]:
        code = m.read_adc("A0")
        seen_codes.append(code)
        return {"D0": code >= 512}

    mcu = MCUBlock("U1", node_a0="n_adc", node_d0="n_d0", callback=firmware)
    c.add(mcu)

    tr = MNASolver(c).solve_transient(duration=1e-4, dt=1e-5)
    assert seen_codes and all(code == 767 for code in seen_codes)
    # 5 V drive - 2 V LED = 3 V across (25 ohm D0 + 125 ohm Rled + 1 mOhm) = 20 mA
    assert math.isclose(float(tr.branch_currents["D1"][-1]), 0.020, rel_tol=1e-3)


def test_f11_two_mcu_co_simulation_tx_to_rx_handshake() -> None:
    """Two MCUBlocks connected TX->RX & D0->RX co-simulate accurately over shared wire nets."""
    c = Circuit()
    c.add(Ground("GND", "0"))

    # U1 drives TX on net_u1_tx alternating every 1 ms; listens on RX (net_u2_d0) and mirrors to D1.
    def fw1(m: MCUBlock, sim_time: float, _dt: float) -> dict[str, bool]:
        tx_bit = int(sim_time // 1e-3) % 2 == 1
        rx_in = m.digital_read("RX")
        return {"TX": tx_bit, "D1": rx_in}

    # U2 samples RX (net_u1_tx) and drives D0 = NOT RX, lighting LED load on net_u2_d0 -> GND.
    def fw2(m: MCUBlock, _t: float, _dt: float) -> dict[str, bool]:
        rx_bit = m.digital_read("RX")
        return {"D0": not rx_bit}

    u1 = MCUBlock("U1", node_tx="net_u1_tx", node_rx="net_u2_d0", node_d1="net_u1_d1", callback=fw1)
    u2 = MCUBlock("U2", node_rx="net_u1_tx", node_d0="net_u2_d0", callback=fw2)
    # Light resistive load on U2.D0 so MNA loads net_u2_d0 realistically (25 / (25 + 10k) drop)
    r_load = Resistor("R1", "net_u2_d0", "0", resistance=10_000.0)
    c.add(u1)
    c.add(u2)
    c.add(r_load)

    tr = MNASolver(c).solve_transient(duration=3.5e-3, dt=1e-4)
    times = tr.times
    v_tx = tr.node_voltages["net_u1_tx"]
    v_u2_d0 = tr.node_voltages["net_u2_d0"]
    v_u1_d1 = tr.node_voltages["net_u1_d1"]

    # At t = 0.5 ms (interval 0): U1 TX is LOW -> U2 D0 is HIGH (~5V * 10k/10025 = 4.9875 V) -> U1 D1 echoes HIGH
    idx_05 = int(np.argmin(np.abs(times - 0.5e-3)))
    assert float(v_tx[idx_05]) < 0.1
    assert math.isclose(float(v_u2_d0[idx_05]), 5.0 * 10_000.0 / 10_025.0, rel_tol=1e-4)
    assert float(v_u1_d1[idx_05]) > 4.9

    # At t = 1.5 ms (interval 1): U1 TX is HIGH (~5 V) -> U2 D0 goes LOW (< 0.1 V) -> U1 D1 goes LOW
    idx_15 = int(np.argmin(np.abs(times - 1.5e-3)))
    assert float(v_tx[idx_15]) > 4.9
    assert float(v_u2_d0[idx_15]) < 0.1
    assert float(v_u1_d1[idx_15]) < 0.1


def test_f11_mcu_catalog_g1_and_circ_round_trip(qapp) -> None:
    """Verify MCUBlock registers in KNOWN_COMPONENT_TYPES, factory, symbol, and .circ JSON."""
    assert "mcu" in KNOWN_COMPONENT_TYPES
    comp = create_default_component_model("mcu", "U1")
    assert isinstance(comp, MCUBlock)
    assert comp.get_editable_parameters()

    doc = CircuitDocument(
        placed=[PlacedComponentSpec(component=comp, center_px=(100.0, 100.0))]
    )
    loaded = loads_document(dumps_document(doc))
    assert isinstance(loaded.placed[0].component, MCUBlock)
