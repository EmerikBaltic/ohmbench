"""Baseline audit and helper verification tests for Item F0 (Phase F)."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QSlider

from circuitsim.app.themes import DARK_PALETTE
from circuitsim.components import Capacitor, DCVoltageSource, Ground, Resistor
from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog
from circuitsim.engine import Circuit
from circuitsim.wiring.model import FreeEnd
from tests.ui_flow_helpers import create_window_and_view
from tests.workbench_helpers import (
    drag_slider_to_fraction,
    open_component_dialog,
    run_transient_check,
    sample_wire_color,
    trigger_placement_hotkey,
)


def test_f0_workbench_helpers_and_transient_baseline(tmp_path, qapp) -> None:
    """F0: verify offscreen helpers for hotkeys, modal dialog double-click, sliders, wire color, and RC transient."""
    win, view, scene = create_window_and_view(tmp_path, zoom=1.0)
    try:
        r1 = scene.place_new_component("resistor", QPointF(0.0, 0.0))
        scene.add_wire(FreeEnd(-100, 100), FreeEnd(100, 100))
        qapp.processEvents()

        # 1. Single-key hotkey invocation helper (W toggles wire mode & Esc exits cleanly)
        trigger_placement_hotkey(view, Qt.Key.Key_W, scene_pos=(-100.0, -100.0))
        assert view.wire_mode_enabled is True
        trigger_placement_hotkey(view, Qt.Key.Key_Escape)
        assert view.wire_mode_enabled is False

        # 2. Modal component dialog helper via real double-click
        def _set_2k(dlg: QuickValueDialog) -> str:
            dlg.inputs["resistance"].setText("2k")
            dlg._on_ok_clicked()
            return dlg.windowTitle()

        title = open_component_dialog(view, (0.0, 0.0), dialog_cls=QuickValueDialog, on_dialog=_set_2k)
        assert "R1" in title
        assert math.isclose(r1.component.resistance, 2000.0, rel_tol=1e-9)

        # 3. Slider drag helper
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.resize(200, 24)
        slider.setRange(0, 1000)
        slider.setValue(500)
        slider.show()
        qapp.processEvents()
        dragged = drag_slider_to_fraction(slider, 0.75)
        assert abs(dragged - 750) <= 5
        slider.close()

        # 4. Wire color sampler
        wcol = sample_wire_color(scene, (0.0, 100.0), bg_hex=DARK_PALETTE.canvas_bg)
        assert wcol != DARK_PALETTE.canvas_bg and wcol.name().lower() != DARK_PALETTE.canvas_bg.lower()

        # 5. Qt-free transient RC step helper: 10 V step across 1k + 1uF -> at t = tau = 1 ms, V_C = 10*(1 - e^-1) = 6.3212 V
        c = Circuit()
        c.add(DCVoltageSource("V1", "vin", "0", voltage=10.0))
        c.add(Resistor("R1", "vin", "vc", resistance=1000.0))
        c.add(Capacitor("C1", "vc", "0", capacitance=1e-6))
        c.add(Ground("G1", "0"))
        tres = run_transient_check(c, duration_s=0.001, dt_s=1e-6)
        v_end = float(tres.node_voltages["vc"][-1])
        expected_tau = 10.0 * (1.0 - math.exp(-1.0))
        assert math.isclose(v_end, expected_tau, rel_tol=5e-3)
    finally:
        scene.undo_stack.setClean()
        win.close()
