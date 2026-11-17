"""Tests for Phase G Owner QOL & Bug Fixes (G3-G8)."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.canvas.view import CircuitView


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
