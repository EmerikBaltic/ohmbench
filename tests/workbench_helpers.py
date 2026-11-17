"""Reusable offscreen workbench QTest & physics helpers for Phase F (F0-F12)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QSlider

from circuitsim.canvas.connectivity import ConnectivityResult
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.view import CircuitView
from circuitsim.engine import MNASolver, TransientResult
from circuitsim.engine.netlist import Circuit
from tests.ui_flow_helpers import press_key, scene_to_viewport_point


def trigger_placement_hotkey(
    view: CircuitView,
    key: Qt.Key,
    scene_pos: tuple[float, float] | QPointF | None = None,
) -> None:
    """Move cursor to `scene_pos` (optional) and press a single-key canvas hotkey."""
    app = QApplication.instance()
    if scene_pos is not None:
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])
        vp_pt = scene_to_viewport_point(view, sx, sy)
        QTest.mouseMove(view.viewport(), vp_pt)
        if app is not None:
            app.processEvents()
    press_key(view.viewport(), key, Qt.KeyboardModifier.NoModifier)


def open_component_dialog(
    view: CircuitView,
    scene_pos: tuple[float, float] | QPointF,
    *,
    dialog_cls: type[QDialog] = QDialog,
    on_dialog: Callable[[QDialog], Any] | None = None,
    timeout_ms: int = 2000,
) -> Any:
    """Double-click at `scene_pos`, intercept the modal QDialog, invoke `on_dialog`, and return its result."""
    app = QApplication.instance()
    sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
    sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])
    vp_pt = scene_to_viewport_point(view, sx, sy)

    result_box: dict[str, Any] = {"handled": False, "value": None, "error": None}
    deadline_timer = QTimer()
    deadline_timer.setSingleShot(True)

    def _poll_dialog() -> None:
        if result_box["handled"] or app is None:
            return
        for widget in app.topLevelWidgets():
            if isinstance(widget, dialog_cls) and widget.isVisible():
                result_box["handled"] = True
                try:
                    if on_dialog is not None:
                        result_box["value"] = on_dialog(widget)
                    else:
                        widget.accept()
                except BaseException as exc:  # pragma: no cover - surfaced below
                    result_box["error"] = exc
                    try:
                        widget.reject()
                    except RuntimeError:
                        pass
                return
        QTimer.singleShot(5, _poll_dialog)

    QTimer.singleShot(0, _poll_dialog)
    deadline_timer.start(timeout_ms)
    vp = view.viewport()
    QTest.mouseMove(vp, vp_pt)
    if app is not None:
        app.processEvents()
    QTest.mouseDClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp_pt)
    if app is not None:
        app.processEvents()
    deadline_timer.stop()

    if result_box["error"] is not None:
        raise result_box["error"]
    if not result_box["handled"]:
        raise TimeoutError(f"Expected {dialog_cls.__name__} did not open on double-click at ({sx}, {sy})")
    return result_box["value"]


def drag_slider_to_fraction(slider: QSlider, fraction: float) -> int:
    """Drag a horizontal `QSlider` via real QTest mouse events to `fraction` in [0.0, 1.0]."""
    frac = min(max(float(fraction), 0.0), 1.0)
    app = QApplication.instance()
    w = max(slider.width(), 40)
    h = max(slider.height(), 16)
    cur_frac = 0.5
    if slider.maximum() > slider.minimum():
        cur_frac = (slider.value() - slider.minimum()) / (slider.maximum() - slider.minimum())
    start_x = int(round(4 + cur_frac * (w - 8)))
    end_x = int(round(4 + frac * (w - 8)))
    mid_x = int(round((start_x + end_x) / 2.0))
    y = h // 2

    QTest.mousePress(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(start_x, y))
    if app is not None:
        app.processEvents()
    QTest.mouseMove(slider, QPoint(mid_x, y))
    if app is not None:
        app.processEvents()
    QTest.mouseMove(slider, QPoint(end_x, y))
    if app is not None:
        app.processEvents()
    # Set exact normalized slider position after move if groove pixel rounding differed by 1 step
    target_val = int(round(slider.minimum() + frac * (slider.maximum() - slider.minimum())))
    if slider.value() != target_val:
        slider.setValue(target_val)
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(end_x, y))
    if app is not None:
        app.processEvents()
    return slider.value()


def sample_wire_color(
    scene: CircuitScene,
    scene_pt: tuple[float, float] | QPointF,
    *,
    bg_hex: str = "#0D1117",
    radius_px: int = 10,
) -> QColor:
    """Render a small patch centered at `scene_pt` offscreen and return the sampled pixel color."""
    sx = scene_pt.x() if isinstance(scene_pt, QPointF) else float(scene_pt[0])
    sy = scene_pt.y() if isinstance(scene_pt, QPointF) else float(scene_pt[1])
    size = 2 * radius_px + 1
    rect = QRectF(sx - radius_px, sy - radius_px, float(size), float(size))
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(bg_hex))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scene.render(painter, QRectF(0.0, 0.0, float(size), float(size)), rect)
    painter.end()
    return img.pixelColor(radius_px, radius_px)


def run_transient_check(
    circuit_or_conn: Circuit | ConnectivityResult,
    *,
    duration_s: float,
    dt_s: float | None = None,
) -> TransientResult:
    """Run a Qt-free transient solve via `MNASolver` on a `Circuit` or `ConnectivityResult`."""
    circuit = (
        circuit_or_conn.circuit
        if isinstance(circuit_or_conn, ConnectivityResult)
        else circuit_or_conn
    )
    solver = MNASolver(circuit)
    step = dt_s if dt_s is not None else min(duration_s / 200.0, solver.choose_default_dt())
    return solver.solve_transient(duration=duration_s, dt=step)


def run_ac_sweep_check(
    circuit_or_conn: Circuit | ConnectivityResult,
    *,
    source_name: str,
    out_node: str,
    freqs_hz: Sequence[float],
    in_node: str | None = None,
):
    """Run a Qt-free AC frequency sweep via `circuitsim.engine.ac_analysis`."""
    from circuitsim.engine.ac_analysis import run_ac_sweep

    circuit = (
        circuit_or_conn.circuit
        if isinstance(circuit_or_conn, ConnectivityResult)
        else circuit_or_conn
    )
    return run_ac_sweep(
        circuit,
        source_name=source_name,
        out_node=out_node,
        freqs_hz=freqs_hz,
        in_node=in_node,
    )
