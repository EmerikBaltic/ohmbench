"""Offscreen QTest UI interaction helpers for real mouse/keyboard workflow tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.component_item import ComponentItem
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_geometry import GRID_PITCH_PX, transform_terminal_position
from circuitsim.canvas.view import CircuitView


def create_window_and_view(
    tmp_path: Path,
    *,
    zoom: float = 1.0,
    theme: str = "dark",
    width: int = 1100,
    height: int = 800,
) -> tuple[MainWindow, CircuitView, CircuitScene]:
    """Create and show an offscreen MainWindow at the requested zoom level."""
    settings = AppSettings(data_dir=tmp_path / f"settings_{id(tmp_path)}_{zoom:.2f}")
    win = MainWindow(settings=settings)
    if win.theme != theme:
        win.set_theme(theme)  # type: ignore[arg-type]
    win.resize(width, height)
    win.show()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()

    view = win.circuit_view
    scene = win.circuit_scene
    view.set_view_state(zoom, (0.0, 0.0))
    if app is not None:
        app.processEvents()
    return win, view, scene


def terminal_scene_point(item: ComponentItem, terminal_name: str) -> tuple[int, int]:
    """Compute terminal scene pixel coords using the shared transform_terminal_position helper."""
    for term in item.metadata.terminals:
        if term.name == terminal_name:
            tx, ty = transform_terminal_position(
                term.grid_pos,
                (item.pos().x(), item.pos().y()),
                rotation_deg=item.rotation_deg,
                flipped_h=item.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
            return (int(round(tx)), int(round(ty)))
    raise KeyError(f"Unknown terminal {terminal_name!r} on {item.component.name}")


def scene_to_viewport_point(
    view: CircuitView,
    sx: float,
    sy: float,
    offset_view_px: tuple[int, int] = (0, 0),
) -> QPoint:
    """Map scene coordinates to viewport coordinates plus an optional view-pixel offset."""
    base = view.mapFromScene(QPointF(sx, sy))
    return QPoint(base.x() + int(offset_view_px[0]), base.y() + int(offset_view_px[1]))


def click_viewport(
    view: CircuitView,
    vp_pt: QPoint,
    *,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Perform a real QTest hover move + press + release at *vp_pt* on the viewport."""
    vp = view.viewport()
    app = QApplication.instance()
    QTest.mouseMove(vp, vp_pt)
    if app is not None:
        app.processEvents()
    QTest.mousePress(vp, button, modifiers, vp_pt)
    if app is not None:
        app.processEvents()
    QTest.mouseRelease(vp, button, modifiers, vp_pt)
    if app is not None:
        app.processEvents()


def double_click_viewport(
    view: CircuitView,
    vp_pt: QPoint,
    *,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Perform a real QTest double-click sequence on the viewport."""
    vp = view.viewport()
    app = QApplication.instance()
    QTest.mouseMove(vp, vp_pt)
    if app is not None:
        app.processEvents()
    QTest.mousePress(vp, button, modifiers, vp_pt)
    QTest.mouseRelease(vp, button, modifiers, vp_pt)
    QTest.mouseDClick(vp, button, modifiers, vp_pt)
    QTest.mouseRelease(vp, button, modifiers, vp_pt)
    if app is not None:
        app.processEvents()


def drag_viewport(
    view: CircuitView,
    start_vp: QPoint,
    end_vp: QPoint,
    *,
    steps: int = 8,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Perform a real QTest press -> interpolated moves -> release drag on the viewport."""
    vp = view.viewport()
    app = QApplication.instance()
    QTest.mouseMove(vp, start_vp)
    if app is not None:
        app.processEvents()
    QTest.mousePress(vp, button, modifiers, start_vp)
    if app is not None:
        app.processEvents()
    n = max(2, int(steps))
    for i in range(1, n + 1):
        t = i / n
        ix = int(round(start_vp.x() + (end_vp.x() - start_vp.x()) * t))
        iy = int(round(start_vp.y() + (end_vp.y() - start_vp.y()) * t))
        QTest.mouseMove(vp, QPoint(ix, iy))
        if app is not None:
            app.processEvents()
    QTest.mouseRelease(vp, button, modifiers, end_vp)
    if app is not None:
        app.processEvents()


def draw_wire_click_to_click(
    view: CircuitView,
    start_scene: tuple[float, float],
    end_scene: tuple[float, float],
    *,
    waypoints_scene: Sequence[tuple[float, float]] = (),
    start_offset_vpx: tuple[int, int] = (0, 0),
    end_offset_vpx: tuple[int, int] = (0, 0),
) -> None:
    """Draw a wire in click-move-click style via real viewport mouse clicks."""
    start_vp = scene_to_viewport_point(view, start_scene[0], start_scene[1], start_offset_vpx)
    click_viewport(view, start_vp)
    for wx, wy in waypoints_scene:
        wp_vp = scene_to_viewport_point(view, wx, wy)
        click_viewport(view, wp_vp)
    end_vp = scene_to_viewport_point(view, end_scene[0], end_scene[1], end_offset_vpx)
    click_viewport(view, end_vp)


def press_key(
    widget: QWidget,
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Send a real QTest keyClick to *widget* and process events."""
    QTest.keyClick(widget, key, modifiers)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def select_all_via_rubberband(view: CircuitView, margin_px: int = 80) -> None:
    """Select all items on the canvas via a real mouse rubber-band drag."""
    scene = view.circuit_scene
    items = list(scene.items())
    if not items:
        return
    rect = items[0].sceneBoundingRect()
    for it in items[1:]:
        rect = rect.united(it.sceneBoundingRect())
    rect = rect.adjusted(-margin_px, -margin_px, margin_px, margin_px)
    tl_vp = view.mapFromScene(rect.topLeft())
    br_vp = view.mapFromScene(rect.bottomRight())
    drag_viewport(view, tl_vp, br_vp, steps=6)


def select_all_and_delete(view: CircuitView, *, use_rubberband: bool = False) -> None:
    """Select all canvas items (Ctrl+A or rubberband) and press Delete."""
    if use_rubberband:
        select_all_via_rubberband(view)
    else:
        press_key(view.viewport(), Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        if len(view.circuit_scene.selectedItems()) < len(view.circuit_scene.placed_items):
            select_all_via_rubberband(view)
    press_key(view.viewport(), Qt.Key.Key_Delete)


def render_scene_image(scene: CircuitScene, rect_scene=None, *, bg_hex: str = "#0D1117") -> QImage:
    """Render *scene* offscreen to an ARGB32 QImage covering *rect_scene* (or sceneRect)."""
    target_rect = rect_scene if rect_scene is not None else scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
    w = max(1, int(round(target_rect.width())))
    h = max(1, int(round(target_rect.height())))
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor(bg_hex))
    painter = QPainter(img)
    scene.render(painter, target=img.rect().toRectF(), source=target_rect)
    painter.end()
    return img
