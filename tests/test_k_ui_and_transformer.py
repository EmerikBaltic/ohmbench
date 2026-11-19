"""Phase K Polish & Physics Fixes tests (K1-K5) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.components import (
    ACVoltageSource,
    DCVoltageSource,
    Ground,
    Resistor,
    Transformer,
    create_default_component_model,
)
from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog
from circuitsim.engine import MNASolver, SimulationSession
from circuitsim.engine.ac_analysis import run_ac_sweep
from circuitsim.engine.netlist import Circuit
from circuitsim.io.circ_serializer import dumps_document, loads_document
from circuitsim.wiring.model import TerminalRef


def test_k1_transformer_blocks_dc_and_scales_ac_ideal_and_non_ideal(qapp, tmp_path) -> None:
    """K1: Transformer blocks steady DC, supports Np/Ns windings + Ideal vs Real modes in AC/Transient/UI/.circ."""
    # (a) Steady DC blocking in both Ideal and Non-Ideal modes
    for ideal_mode in (True, False):
        circ_dc = Circuit()
        circ_dc.add(DCVoltageSource("V1", "p1", "0", voltage=12.0))
        circ_dc.add(Ground("G1", "0"))
        circ_dc.add(
            Transformer(
                "T1",
                "p1",
                "0",
                "s1",
                "0",
                primary_turns=100,
                secondary_turns=200,
                is_ideal=ideal_mode,
            )
        )
        circ_dc.add(Resistor("R_LOAD", "s1", "0", resistance=1_000.0))

        dc_res = MNASolver(circ_dc).solve_dc()
        assert abs(dc_res.node_voltages["s1"]) < 1e-5

        # Even starting interactive SimulationSession must stay 0 V on secondary under DC
        sess = SimulationSession(circuit=circ_dc)
        frame = sess.start()
        assert abs(frame.node_voltages["s1"]) < 1e-5

    # (b) Transient & AC frequency-domain scaling in Ideal vs Non-Ideal (Real) mode
    circ_ideal = Circuit()
    circ_ideal.add(ACVoltageSource("V_AC", "p1", "0", amplitude=10.0, frequency=100.0))
    circ_ideal.add(Ground("G1", "0"))
    t_ideal = Transformer(
        "T1",
        "p1",
        "0",
        "s1",
        "0",
        primary_turns=100,
        secondary_turns=200,
        is_ideal=True,
    )
    circ_ideal.add(t_ideal)
    circ_ideal.add(Resistor("R_LOAD", "s1", "0", resistance=1_000.0))

    tr_ideal = MNASolver(circ_ideal).solve_transient(duration=15e-3, dt=1e-4)
    v_peak_ideal = float(np.max(np.abs(tr_ideal.node_voltages["s1"])))
    assert v_peak_ideal == pytest.approx(20.0, rel=0.02)

    circ_real = Circuit()
    circ_real.add(ACVoltageSource("V_AC", "p1", "0", amplitude=10.0, frequency=100.0))
    circ_real.add(Ground("G1", "0"))
    t_real = Transformer(
        "T1",
        "p1",
        "0",
        "s1",
        "0",
        primary_turns=100,
        secondary_turns=200,
        is_ideal=False,
        l_primary_h=0.5,
        coupling_k=0.92,
        r_primary_ohm=5.0,
        r_secondary_ohm=10.0,
    )
    circ_real.add(t_real)
    circ_real.add(Resistor("R_LOAD", "s1", "0", resistance=1_000.0))

    tr_real = MNASolver(circ_real).solve_transient(duration=15e-3, dt=1e-4)
    v_peak_real = float(np.max(np.abs(tr_real.node_voltages["s1"])))
    assert 15.0 < v_peak_real < v_peak_ideal * 0.96

    ac_ideal = run_ac_sweep(
        circ_ideal,
        f_start=100.0,
        f_stop=100.0,
        num_points=1,
        stimulus_amplitude=10.0,
    )
    ac_real = run_ac_sweep(
        circ_real,
        f_start=100.0,
        f_stop=100.0,
        num_points=1,
        stimulus_amplitude=10.0,
    )
    assert float(ac_ideal.node_mag_v["s1"][0]) == pytest.approx(20.0, rel=0.02)
    assert float(ac_real.node_mag_v["s1"][0]) < float(ac_ideal.node_mag_v["s1"][0]) * 0.96

    # (c) .circ serializer round-trip (new settings + legacy turns_ratio compatibility)
    t_legacy = Transformer("T_LEG", "p1", "0", "s1", "0", turns_ratio=0.5, r_winding_ohm=2.0)
    assert t_legacy.primary_turns == pytest.approx(100.0)
    assert t_legacy.secondary_turns == pytest.approx(50.0)
    assert t_legacy.turns_ratio == pytest.approx(0.5)

    scene = CircuitScene(SymbolLibrary())
    try:
        t_item = scene.place_new_component("transformer", QPointF(0.0, 0.0))
        assert t_item is not None
        dlg = QuickValueDialog(t_item.component)
        try:
            assert dlg.transformer_ideal_cb is not None
            assert dlg.transformer_ideal_cb.isChecked() is True
            assert dlg.inputs["l_primary_h"].isEnabled() is False
            assert dlg.inputs["coupling_k"].isEnabled() is False

            dlg.transformer_ideal_cb.setChecked(False)
            qapp.processEvents()
            assert dlg.inputs["l_primary_h"].isEnabled() is True
            assert dlg.inputs["r_primary_ohm"].isEnabled() is True
            dlg.inputs["primary_turns"].setText("100")
            dlg.inputs["secondary_turns"].setText("400")
            assert dlg._apply_current_inputs() is True
            assert t_item.component.is_ideal is False  # type: ignore[attr-defined]
            assert t_item.component.turns_ratio == pytest.approx(4.0)  # type: ignore[attr-defined]
        finally:
            dlg.close()

        doc = scene.to_document()
        raw_json = dumps_document(doc)
        reloaded = loads_document(raw_json)
        t_reload = reloaded.placed[0].component
        assert isinstance(t_reload, Transformer)
        assert t_reload.is_ideal is False
        assert t_reload.primary_turns == pytest.approx(100.0)
        assert t_reload.secondary_turns == pytest.approx(400.0)
    finally:
        scene.shutdown_scene()

    # (d) Properties/Info panel displays DC-blocking hint for Transformer
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        t_item2 = win.circuit_scene.place_new_component("transformer", QPointF(0.0, 0.0))
        assert t_item2 is not None
        win.circuit_scene.clearSelection()
        t_item2.setSelected(True)
        qapp.processEvents()
        assert "Transformers block steady DC" in win.info_label.text()
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()


def test_k2_seven_seg_display_live_dc_and_session_updates(qapp) -> None:
    """K2: 7-Segment Display decodes live DC & SimulationSession frames even with unwired input pins."""
    from circuitsim.components import SevenSegDisplay

    scene = CircuitScene(SymbolLibrary())
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(-160.0, 120.0))
        ds1 = scene.place_new_component("seven_seg_display", QPointF(80.0, 0.0))
        assert v1 is not None and gnd is not None and ds1 is not None
        assert isinstance(ds1.component, SevenSegDisplay)
        assert ds1.component.v_threshold == pytest.approx(2.0)

        # Connect 12 V DC to D0 (bit 0 = 1) and D2 (bit 2 = 4) -> 5; leave D1 and D3 unwired
        v_pos = TerminalRef(v1.component.name, "pos")
        v_neg = TerminalRef(v1.component.name, "neg")
        g_pin = TerminalRef(gnd.component.name, "gnd")
        d0 = TerminalRef(ds1.component.name, "D0")
        d2 = TerminalRef(ds1.component.name, "D2")

        scene.set_wire_graph(
            scene.wire_graph.add_wire(v_neg, g_pin)
            .add_wire(v_pos, d0)
            .add_wire(v_pos, d2)
        )

        # (a) Direct DC solve + apply_solve_result syncs live canvas display
        dc_res = MNASolver(scene.build_circuit()).solve_dc()
        scene.apply_solve_result(dc_res)
        assert ds1.component.value_nibble == 5
        assert ds1.component.decoded_hex_char == "5"
        assert ds1.component.active_segments == ("a", "c", "d", "f", "g")

        # (b) Interactive SimulationSession start -> SimulationFrame -> apply_simulation_frame
        ds1.component.reset_state()
        assert ds1.component.decoded_hex_char == "0"

        conn = scene.extract_connectivity_result()
        sess = SimulationSession(circuit=conn.circuit, connectivity=conn)
        frame = sess.start()
        scene.apply_simulation_frame(frame)
        assert ds1.component.value_nibble == 5
        assert ds1.component.decoded_hex_char == "5"
        assert ds1.component.active_segments == ("a", "c", "d", "f", "g")
        assert ds1.live_display_value == "0x5"

        # (c) Wire remaining pins D1 and D3 high -> 0xF
        scene.set_wire_graph(
            scene.wire_graph.add_wire(v_pos, TerminalRef(ds1.component.name, "D1"))
            .add_wire(v_pos, TerminalRef(ds1.component.name, "D3"))
        )
        conn2 = scene.extract_connectivity_result()
        sess2 = SimulationSession(circuit=conn2.circuit, connectivity=conn2)
        frame2 = sess2.start()
        scene.apply_simulation_frame(frame2)
        assert ds1.component.value_nibble == 0xF
        assert ds1.component.decoded_hex_char == "F"
        assert ds1.component.active_segments == ("a", "e", "f", "g")
        assert ds1.live_display_value == "0xF"
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

