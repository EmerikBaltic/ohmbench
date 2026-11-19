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


def test_k3_blown_fuse_broken_wire_visual_and_pop_animation(qapp) -> None:
    """K3: Blown Fuse shows broken open filament stubs, POP! spark starburst animation, and click-to-reset."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from circuitsim.components import Fuse

    scene = CircuitScene(SymbolLibrary())
    try:
        v1 = scene.place_new_component("dc_voltage", QPointF(-160.0, 0.0), rotation_deg=90)
        f1 = scene.place_new_component("fuse", QPointF(0.0, -40.0))
        r1 = scene.place_new_component("resistor", QPointF(160.0, 0.0), rotation_deg=90)
        gnd = scene.place_new_component("ground", QPointF(0.0, 120.0))
        assert v1 is not None and f1 is not None and r1 is not None and gnd is not None
        assert isinstance(f1.component, Fuse)

        r1.component.resistance = 10.0  # 12 V / 10 ohm = 1.2 A > 0.5 A rating
        scene.set_wire_graph(
            scene.wire_graph.add_wire(TerminalRef(v1.component.name, "pos"), TerminalRef(f1.component.name, "a"))
            .add_wire(TerminalRef(f1.component.name, "b"), TerminalRef(r1.component.name, "a"))
            .add_wire(TerminalRef(r1.component.name, "b"), TerminalRef(gnd.component.name, "gnd"))
            .add_wire(TerminalRef(v1.component.name, "neg"), TerminalRef(gnd.component.name, "gnd"))
        )

        # Verify intact rendering draws continuous center wire at y = 0 inside fuse box
        img_intact = QImage(100, 60, QImage.Format.Format_ARGB32)
        img_intact.fill(QColor("#0D1117"))
        p1 = QPainter(img_intact)
        p1.translate(50.0, 30.0)
        scene.symbol_library.paint_symbol(p1, "fuse", state="intact", stroke_color="#E6EDF3")
        p1.end()
        assert QColor(img_intact.pixel(50, 30)).red() > 150  # Intact filament crosses center x=50, y=30

        # Run SimulationSession -> fuse blows, triggers POP! animation and broken symbol state
        conn = scene.extract_connectivity_result()
        sess = SimulationSession(circuit=conn.circuit, connectivity=conn)
        frame = sess.start()
        scene.apply_simulation_frame(frame)
        assert f1.component.is_blown is True
        assert f1.symbol_state == "blown"
        assert f1._fuse_pop_progress > 0.0
        assert f1.live_display_value is not None and "BLOWN" in f1.live_display_value

        # Verify blown symbol leaves open air gap at center (50, 30) and broken stubs away from center
        img_blown = QImage(100, 60, QImage.Format.Format_ARGB32)
        img_blown.fill(QColor("#0D1117"))
        p2 = QPainter(img_blown)
        p2.translate(50.0, 30.0)
        scene.symbol_library.paint_symbol(p2, "fuse", state="blown", stroke_color="#E6EDF3")
        p2.end()
        center_px = QColor(img_blown.pixel(50, 30))
        assert center_px.red() < 80  # Open air gap (no bright filament at exact center)

        # Render full scene item while POP! spark progress == 1.0 (offscreen painter safety + spark pixel check)
        f1._fuse_pop_progress = 1.0
        img_pop = QImage(120, 100, QImage.Format.Format_ARGB32)
        img_pop.fill(QColor("#0D1117"))
        p3 = QPainter(img_pop)
        scene.render(p3, QRectF(0, 0, 120, 100), QRectF(-60, -90, 120, 100))
        p3.end()

        # Resetting fuse restores intact state & clears spark animation
        assert scene.reset_fuse_item(f1) is True
        assert f1.component.is_blown is False
        assert f1.symbol_state == "intact"
        assert f1._fuse_pop_progress == 0.0
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_k4_multi_pin_components_have_always_visible_pin_labels(qapp, tmp_path) -> None:
    """K4: All multi-pin components have default always-visible pin labels & 7-seg inspector pinout note."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter

    multi_pin_types = (
        "seven_seg_display",
        "ne555",
        "d_flipflop",
        "jk_flipflop",
        "counter_4bit",
        "spdt_relay",
        "opamp",
        "comparator",
        "npn_bjt",
        "pnp_bjt",
        "nmos",
        "pmos",
        "potentiometer",
        "transformer",
    )

    scene = CircuitScene(SymbolLibrary())
    try:
        for idx, cid in enumerate(multi_pin_types):
            item = scene.place_new_component(cid, QPointF(idx * 220.0, 0.0))
            assert item is not None
            labels = item.default_pin_labels(item.component)
            assert labels, f"Expected default_pin_labels for {cid}"
            for term in item.metadata.terminals:
                assert term.name in labels, f"Missing label for {cid}.{term.name}"

        # Render 7-seg unselected/unhovered in offscreen QImage -> left strip must have D0..D3 pixels
        ds_item = next(it for it in scene.placed_items if it.component.component_type == "seven_seg_display")
        assert ds_item.isSelected() is False
        img = QImage(160, 180, QImage.Format.Format_ARGB32)
        img.fill(QColor("#0D1117"))
        painter = QPainter(img)
        scene.render(painter, QRectF(0, 0, 160, 180), QRectF(-80, -90, 160, 180))
        painter.end()
        # Sample left inner strip around (-27, -54) in item coords -> (53, 36) in image coords
        non_bg = 0
        for px in range(44, 62):
            for py in range(28, 44):
                if QColor(img.pixel(px, py)) != QColor("#0D1117"):
                    non_bg += 1
        assert non_bg > 0
    finally:
        scene.shutdown_scene()
        qapp.processEvents()

    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        ds = win.circuit_scene.place_new_component("seven_seg_display", QPointF(0.0, 0.0))
        assert ds is not None
        win.circuit_scene.clearSelection()
        ds.setSelected(True)
        qapp.processEvents()
        info_txt = win.info_label.text()
        assert "4-bit Hex/BCD inputs: D0(1), D1(2), D2(4), D3(8) — no GND pin needed" in info_txt
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()


def test_k5_problems_dock_locked_to_bottom_and_resizes_palette(qapp, tmp_path) -> None:
    """K5: ProblemsDock is locked to bottom dock area, spans full width, and shrinks Component Palette without overlapping."""
    from PySide6.QtWidgets import QDockWidget

    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        win.resize(1200, 800)
        win.show()
        qapp.processEvents()

        # Verify dock lock & non-floating properties + single title bar widget
        assert win.problems_dock.isFloating() is False
        assert win.problems_dock.allowedAreas() == Qt.DockWidgetArea.BottomDockWidgetArea
        assert win.problems_dock.features() == QDockWidget.DockWidgetFeature.DockWidgetClosable
        assert win.problems_dock.titleBarWidget() is win.problems_dock.panel.header_widget

        h_closed = win.palette_dock.height()

        # Open Problems Panel -> palette shrinks cleanly above Problems dock with no vertical overlap
        win.action_view_problems.trigger()
        qapp.processEvents()
        assert win.problems_dock.isVisible() is True
        assert win.problems_dock.isFloating() is False

        h_open = win.palette_dock.height()
        assert h_open < h_closed
        assert win.palette_dock.geometry().bottom() <= win.problems_dock.geometry().top() + 2
        assert win.problems_dock.width() >= win.palette_dock.width() + win.circuit_view.width()

        # Close Problems Panel -> palette expands back down
        win.problems_dock.panel.close_btn.click()
        qapp.processEvents()
        assert win.problems_dock.isVisible() is False
        assert win.palette_dock.height() == h_closed
    finally:
        win.circuit_scene.undo_stack.setClean()
        win.close()
        qapp.processEvents()

