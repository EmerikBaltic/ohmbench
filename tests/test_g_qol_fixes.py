"""Tests for Phase G Owner QOL & Bug Fixes (G3-G8)."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.canvas.view import CircuitView
from tests.ui_flow_helpers import click_viewport, scene_to_viewport_point


def test_g3_scrollwheel_zoom_anchors_to_viewport_center(qapp) -> None:
    """G3: Scrollwheel zoom in/out stays centered strictly on viewport center without corner drift."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(800, 600)
    view.show()
    view.set_view_state(1.0, (100.0, -60.0))
    qapp.processEvents()

    try:
        c0 = view.view_center()
        assert math.isclose(c0[0], 100.0, abs_tol=2.0)
        assert math.isclose(c0[1], -60.0, abs_tol=2.0)

        # Send zoom-in wheel events with cursor near bottom-left corner (where naive AnchorUnderMouse drifts!)
        vp_corner = QPointF(25.0, 575.0)
        for _ in range(4):
            ev_in = QWheelEvent(
                vp_corner,
                view.viewport().mapToGlobal(vp_corner.toPoint()),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(view.viewport(), ev_in)
            qapp.processEvents()
            cx, cy = view.view_center()
            assert math.hypot(cx - 100.0, cy - (-60.0)) <= 2.0

        # Send zoom-out wheel events past initial zoom
        for _ in range(6):
            ev_out = QWheelEvent(
                vp_corner,
                view.viewport().mapToGlobal(vp_corner.toPoint()),
                QPoint(0, 0),
                QPoint(0, -120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            QApplication.sendEvent(view.viewport(), ev_out)
            qapp.processEvents()
            cx, cy = view.view_center()
            assert math.hypot(cx - 100.0, cy - (-60.0)) <= 2.0
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()


def test_g4_hotkey_autorepeat_ignored_and_no_overlapping_placement(qapp) -> None:
    """G4: Auto-repeat hotkeys are ignored and stacked non-voltmeter placements/drops offset cleanly."""
    scene = CircuitScene(SymbolLibrary())
    view = CircuitView(scene)
    view.resize(900, 650)
    view.show()
    view.set_view_state(1.0, (0.0, 0.0))
    qapp.processEvents()

    try:
        vp_origin = scene_to_viewport_point(view, 0.0, 0.0)
        QTest.mouseMove(view.viewport(), vp_origin)
        qapp.processEvents()

        # Auto-repeat R press when not placing must be ignored (cannot activate placement)
        rep_ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
            "r",
            True,
            1,
        )
        QApplication.sendEvent(view.viewport(), rep_ev)
        qapp.processEvents()
        assert view.placing_component_type is None

        # Non-repeat R enters placement; subsequent 9 auto-repeat R events stay ignored while 10 clicks stack cleanly
        first_ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
            "r",
            False,
            1,
        )
        QApplication.sendEvent(view.viewport(), first_ev)
        qapp.processEvents()
        assert view.placing_component_type == "resistor"

        for idx in range(10):
            # Send auto-repeat R presses too to simulate holding the key down
            hold_ev = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_R,
                Qt.KeyboardModifier.NoModifier,
                "r",
                idx > 0,
                1,
            )
            QApplication.sendEvent(view.viewport(), hold_ev)
            click_viewport(view, vp_origin)
        qapp.processEvents()

        assert len(scene.placed_items) == 10
        rects = [
            scene._component_body_rect_at(
                it.component.component_type, it.pos().x(), it.pos().y(), it.rotation_deg
            )
            for it in scene.placed_items
        ]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not rects[i].intersects(rects[j]), (
                    f"Components {i} and {j} overlap: {rects[i]} vs {rects[j]}"
                )

        # Also verify drag-drop onto an existing non-voltmeter body snaps away cleanly
        r0, r1 = scene.placed_items[0], scene.placed_items[1]
        scene.begin_component_move_tracking()
        r1.setPos(r0.pos())
        scene.commit_component_move_tracking()
        rect0 = scene._component_body_rect_at(r0.component.component_type, r0.pos().x(), r0.pos().y(), r0.rotation_deg)
        rect1 = scene._component_body_rect_at(r1.component.component_type, r1.pos().x(), r1.pos().y(), r1.rotation_deg)
        assert not rect0.intersects(rect1)
    finally:
        view.close()
        scene.shutdown_scene()
        qapp.processEvents()
