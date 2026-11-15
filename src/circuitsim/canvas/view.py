"""Interactive QGraphicsView with zoom, pan, wire routing preview, switch click, and quick edit."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QContextMenuEvent,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QHelpEvent,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsView, QMenu, QToolTip

from circuitsim.canvas.component_item import ComponentItem
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_geometry import GRID_PITCH_PX, snap_to_grid
from circuitsim.canvas.wire_item import JunctionItem, WireItem
from circuitsim.components.switch import Switch
from circuitsim.dialogs.quick_edit_dialog import QuickValueDialog

MIME_COMPONENT_ID = "application/x-circuitsim-component-id"


class CircuitView(QGraphicsView):
    """Schematic viewport supporting zoom, pan, wiring, switch click, and double-click value editing."""

    cursor_grid_moved = Signal(float, float, int, int)  # (x_px, y_px, gx, gy)
    zoom_changed = Signal(float)  # Zoom factor (1.0 = 100%)
    wire_mode_changed = Signal(bool)  # Synced with toolbar Wire button

    MIN_ZOOM: float = 0.25
    MAX_ZOOM: float = 4.0
    ZOOM_STEP: float = 1.15

    def __init__(self, circuit_scene: CircuitScene) -> None:
        super().__init__(circuit_scene)
        self.circuit_scene: CircuitScene = circuit_scene
        self._zoom_factor: float = 1.0
        self._is_panning: bool = False
        self._space_held: bool = False
        self._pan_start: QPoint = QPoint()
        self._last_scene_pos: QPointF = QPointF(0.0, 0.0)
        self._wire_drag_active: bool = False
        self._wire_mode_enabled: bool = False
        self._suppress_next_context_menu: bool = False
        self._pressed_switch_item: ComponentItem | None = None
        self._pressed_switch_pos: QPointF = QPointF()

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.centerOn(0.0, 0.0)

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    @property
    def wire_mode_enabled(self) -> bool:
        return self._wire_mode_enabled

    def set_wire_mode(self, enabled: bool) -> None:
        """Toggle dedicated Wire Drawing mode."""
        new_val = bool(enabled)
        if new_val != self._wire_mode_enabled:
            self._wire_mode_enabled = new_val
            if not self._wire_mode_enabled:
                self._wire_drag_active = False
                self.circuit_scene.cancel_wire_preview()
            self._update_cursor_for_scene_pos(self._last_scene_pos)
            self.wire_mode_changed.emit(self._wire_mode_enabled)

    def reset_zoom(self) -> None:
        """Reset viewport zoom to 100% and center on `(0, 0)`."""
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_to_fit(self, margin_px: float = 40.0) -> None:
        """Fit all components and wires into the viewport with a margin; reset to 100% if empty."""
        items = self.circuit_scene.placed_items
        if not items:
            self.reset_zoom()
            self.centerOn(0.0, 0.0)
            return

        rect = items[0].sceneBoundingRect()
        for it in items[1:]:
            rect = rect.united(it.sceneBoundingRect())
        for w in self.circuit_scene.placed_wires:
            if len(w.points) >= 2:
                rect = rect.united(w.sceneBoundingRect())
        for jitem in self.circuit_scene.junction_items:
            rect = rect.united(jitem.sceneBoundingRect())

        rect = rect.adjusted(-margin_px, -margin_px, margin_px, margin_px)
        vp_size = self.viewport().size()
        if vp_size.width() <= 10 or vp_size.height() <= 10:
            return

        scale_x = vp_size.width() / max(rect.width(), 1.0)
        scale_y = vp_size.height() / max(rect.height(), 1.0)
        target_zoom = min(scale_x, scale_y)
        target_zoom = min(max(target_zoom, self.MIN_ZOOM), self.MAX_ZOOM)

        self.resetTransform()
        self.scale(target_zoom, target_zoom)
        self._zoom_factor = target_zoom
        self.centerOn(rect.center())
        self.zoom_changed.emit(self._zoom_factor)

    def view_center(self) -> tuple[float, float]:
        """Return the scene point shown in the middle of the viewport (saved as 'pan')."""
        c = self.mapToScene(self.viewport().rect().center())
        return (c.x(), c.y())

    def set_view_state(self, zoom: float, center: tuple[float, float]) -> None:
        """Restore a saved zoom level (clamped to the allowed range) and view centre."""
        zoom = min(max(float(zoom), self.MIN_ZOOM), self.MAX_ZOOM)
        self.resetTransform()
        self.scale(zoom, zoom)
        self._zoom_factor = zoom
        self.centerOn(float(center[0]), float(center[1]))
        self.zoom_changed.emit(self._zoom_factor)

    def zoom_in(self) -> None:
        self._apply_zoom(self.ZOOM_STEP)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / self.ZOOM_STEP)

    def _apply_zoom(self, factor: float) -> None:
        new_zoom = self._zoom_factor * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom_factor)

    def _item_at_is_component_or_junction(self, scene_pt: QPointF) -> bool:
        curr = self.circuit_scene.itemAt(scene_pt, self.transform())
        while curr is not None:
            if isinstance(curr, (ComponentItem, JunctionItem)):
                return True
            curr = curr.parentItem()
        return False

    def _update_cursor_for_scene_pos(self, scene_pt: QPointF) -> None:
        if self._is_panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._space_held:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if self.circuit_scene.is_dragging_wire_endpoint:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self.circuit_scene.is_dragging_wire_segment:
            orient = self.circuit_scene.active_wire_segment_orientation
            self.setCursor(
                Qt.CursorShape.SizeVerCursor
                if orient == "H"
                else Qt.CursorShape.SizeHorCursor
            )
            return
        hovered_pt = self.circuit_scene.update_hover_highlight(scene_pt)
        if (
            self._wire_mode_enabled
            or self.circuit_scene.is_drawing_wire
            or hovered_pt is not None
            or self.circuit_scene.hovered_wire_end_pt is not None
        ):
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if not self._item_at_is_component_or_junction(scene_pt):
            seg_hit = self.circuit_scene.find_wire_segment_near(scene_pt)
            if seg_hit is not None:
                self.setCursor(
                    Qt.CursorShape.SizeVerCursor
                    if seg_hit[2] == "H"
                    else Qt.CursorShape.SizeHorCursor
                )
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def open_quick_edit_for_item(self, item: ComponentItem) -> bool:
        """Open the double-click `QuickValueDialog` for `item` and apply changes live."""
        comp = item.component
        if not comp.get_editable_parameters() and not isinstance(comp, Switch):
            return False

        def _on_apply(parsed_map: dict[str, float], show_val: bool) -> None:
            self.circuit_scene.edit_component_parameters(
                item, parsed_map, show_value=show_val
            )

        dlg = QuickValueDialog(comp, on_apply=_on_apply, parent=self.window())
        dlg.exec()
        if isinstance(comp, Switch):
            item.animate_switch_state(comp.closed)
            self.circuit_scene.switch_toggled_by_user.emit(
                comp.name, comp.closed
            )
        return True

    def viewportEvent(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            if isinstance(event, QHelpEvent):
                scene_pt = self.mapToScene(event.pos())
                term_hit = self.circuit_scene.find_terminal_near(scene_pt)
                if term_hit is not None:
                    item, tname, _ = term_hit
                    tip = self.circuit_scene.get_terminal_tooltip(item, tname)
                    QToolTip.showText(event.globalPos(), tip, self.viewport())
                    return True
                hit = self.itemAt(event.pos())
                curr = hit
                while curr is not None and not isinstance(curr, ComponentItem) and curr.parentItem() is not None:
                    curr = curr.parentItem()
                if isinstance(curr, ComponentItem):
                    tip = self.circuit_scene.get_component_tooltip(curr)
                    QToolTip.showText(event.globalPos(), tip, self.viewport())
                    return True
        return super().viewportEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom smoothly around the cursor using the mouse wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(self.ZOOM_STEP)
        elif delta < 0:
            self._apply_zoom(1.0 / self.ZOOM_STEP)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        event_pos = event.position().toPoint()
        scene_pt: QPointF = self.mapToScene(event_pos)
        self._last_scene_pos = scene_pt
        self._pressed_switch_item = None

        # 1. Middle-click or Space + Left-click panning
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_held
        ):
            self._is_panning = True
            self._pan_start = event_pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # 2. Right-click cancels active wire drawing or active wire segment drag
        if event.button() == Qt.MouseButton.RightButton:
            if self.circuit_scene.is_drawing_wire:
                self._wire_drag_active = False
                self._suppress_next_context_menu = True
                self.circuit_scene.cancel_wire_preview()
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return
            if self.circuit_scene.is_dragging_wire_endpoint:
                self._suppress_next_context_menu = True
                self.circuit_scene.cancel_wire_endpoint_drag()
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return
            if self.circuit_scene.is_dragging_wire_segment:
                self._suppress_next_context_menu = True
                self.circuit_scene.cancel_wire_segment_drag()
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

        # 3. Left-click wire drawing / endpoint drag / segment drag / component selection
        if event.button() == Qt.MouseButton.LeftButton:
            if self.circuit_scene.is_drawing_wire:
                self._wire_drag_active = False
                if self.circuit_scene.is_snap_target_at(scene_pt):
                    self.circuit_scene.finish_wire_preview(scene_pt)
                elif not self.circuit_scene.add_wire_preview_waypoint(scene_pt):
                    self.circuit_scene.finish_wire_preview(scene_pt)
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

            if not self._wire_mode_enabled:
                end_hit = self.circuit_scene.find_wire_endpoint_near(
                    scene_pt, tolerance_px=self.circuit_scene.TERMINAL_SNAP_RADIUS_PX
                )
                if end_hit is not None:
                    wire_item, which_end, _end_pt = end_hit
                    if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        self.circuit_scene.clearSelection()
                    wire_item.setSelected(True)
                    self.circuit_scene.begin_wire_endpoint_drag(wire_item, which_end)
                    self._update_cursor_for_scene_pos(scene_pt)
                    event.accept()
                    return

            term_hit = self.circuit_scene.find_terminal_near(scene_pt)
            if term_hit is not None or self._wire_mode_enabled:
                self._wire_drag_active = True
                self.circuit_scene.start_wire_preview(scene_pt)
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

            if not self._item_at_is_component_or_junction(scene_pt):
                seg_hit = self.circuit_scene.find_wire_segment_near(scene_pt)
                if seg_hit is not None:
                    wire_item, seg_idx, orient = seg_hit
                    if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        self.circuit_scene.clearSelection()
                    wire_item.setSelected(True)
                    self.circuit_scene.begin_wire_segment_drag(
                        wire_item, seg_idx, orient
                    )
                    self._update_cursor_for_scene_pos(scene_pt)
                    event.accept()
                    return

            # Check if clicking a Switch component body (to toggle on release if not dragged)
            hit_item = self.circuit_scene.itemAt(scene_pt, self.transform())
            if isinstance(hit_item, ComponentItem) and isinstance(
                hit_item.component, Switch
            ):
                self._pressed_switch_item = hit_item
                self._pressed_switch_pos = hit_item.pos()

            # Prepare for component selection / drag-move
            self.circuit_scene.begin_component_move_tracking()
            self._left_press_pos = event_pos
            self._drag_threshold_reached = False

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        event_pos = event.position().toPoint()
        scene_pt: QPointF = self.mapToScene(event_pos)
        self._last_scene_pos = scene_pt
        sx, sy = snap_to_grid(scene_pt.x(), scene_pt.y(), GRID_PITCH_PX)
        gx = int(round(sx / GRID_PITCH_PX))
        gy = int(round(sy / GRID_PITCH_PX))
        self.cursor_grid_moved.emit(sx, sy, gx, gy)

        if self._is_panning:
            delta = event_pos - self._pan_start
            self._pan_start = event_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        if self.circuit_scene.is_drawing_wire:
            self.circuit_scene.update_wire_preview(scene_pt)
            self._update_cursor_for_scene_pos(scene_pt)
            event.accept()
            return

        if self.circuit_scene.is_dragging_wire_endpoint:
            self.circuit_scene.update_wire_endpoint_drag(scene_pt)
            self._update_cursor_for_scene_pos(scene_pt)
            event.accept()
            return

        if self.circuit_scene.is_dragging_wire_segment:
            self.circuit_scene.update_wire_segment_drag(scene_pt)
            self._update_cursor_for_scene_pos(scene_pt)
            event.accept()
            return

        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and not self._is_panning
            and not self.circuit_scene.is_drawing_wire
            and hasattr(self, "_left_press_pos")
            and not self._drag_threshold_reached
        ):
            if (event_pos - self._left_press_pos).manhattanLength() < 4:
                event.accept()
                return
            self._drag_threshold_reached = True

        self._update_cursor_for_scene_pos(scene_pt)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        scene_pt: QPointF = self.mapToScene(event.position().toPoint())
        self._last_scene_pos = scene_pt

        if hasattr(self, "_left_press_pos"):
            del self._left_press_pos
        self._drag_threshold_reached = False

        if self._is_panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._is_panning = False
            self._update_cursor_for_scene_pos(scene_pt)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._wire_drag_active and self.circuit_scene.is_drawing_wire:
                self._wire_drag_active = False
                pts = self.circuit_scene.update_wire_preview(scene_pt)
                if len(pts) >= 2:
                    self.circuit_scene.finish_wire_preview(scene_pt)
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

            if self.circuit_scene.is_dragging_wire_endpoint:
                self.circuit_scene.commit_wire_endpoint_drag(scene_pt)
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

            if self.circuit_scene.is_dragging_wire_segment:
                self.circuit_scene.commit_wire_segment_drag()
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return

            super().mouseReleaseEvent(event)
            moved = self.circuit_scene.commit_component_move_tracking()
            if (
                not moved
                and self._pressed_switch_item is not None
                and self._pressed_switch_item.pos() == self._pressed_switch_pos
            ):
                self.circuit_scene.toggle_switch_item(self._pressed_switch_item)
            self._pressed_switch_item = None
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Finish wire on double-click while drawing, otherwise open Quick Value Dialog."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            if self.circuit_scene.is_drawing_wire:
                self._wire_drag_active = False
                self.circuit_scene.finish_wire_preview(scene_pt)
                self._update_cursor_for_scene_pos(scene_pt)
                event.accept()
                return
            self._pressed_switch_item = None
            hit_item = self.circuit_scene.itemAt(scene_pt, self.transform())
            if isinstance(hit_item, ComponentItem):
                if self.open_quick_edit_for_item(hit_item):
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Show right-click context menu on components, wires, or canvas (Requirement 7)."""
        if self._suppress_next_context_menu:
            self._suppress_next_context_menu = False
            event.accept()
            return
        scene_pt = self.mapToScene(event.pos())
        self._last_scene_pos = scene_pt

        clicked_item = self.circuit_scene.itemAt(scene_pt, self.transform())
        if isinstance(clicked_item, (ComponentItem, WireItem)) and not clicked_item.isSelected():
            self.circuit_scene.clearSelection()
            clicked_item.setSelected(True)

        selected_comps = [
            it
            for it in self.circuit_scene.selectedItems()
            if isinstance(it, ComponentItem)
        ]
        has_comp = bool(selected_comps)
        has_sel = bool(self.circuit_scene.selectedItems())

        menu = QMenu(self)

        if len(selected_comps) == 1:
            target_comp = selected_comps[0]
            act_edit = QAction("Properties...\tDouble-Click", menu)
            act_edit.triggered.connect(
                lambda: self.open_quick_edit_for_item(target_comp)
            )
            menu.addAction(act_edit)
            if isinstance(target_comp.component, Switch):
                act_toggle_sw = QAction("Toggle Switch (ON/OFF)\tClick", menu)
                act_toggle_sw.triggered.connect(
                    lambda: self.circuit_scene.toggle_switch_item(target_comp)
                )
                menu.addAction(act_toggle_sw)
            menu.addSeparator()

        act_rotate = QAction("Rotate 90° CW\tR", menu)
        act_rotate.setShortcut(QKeySequence("R"))
        act_rotate.setEnabled(has_comp)
        act_rotate.triggered.connect(self.circuit_scene.rotate_selected)
        menu.addAction(act_rotate)

        act_flip = QAction("Flip Horizontal\tF", menu)
        act_flip.setShortcut(QKeySequence("F"))
        act_flip.setEnabled(has_comp)
        act_flip.triggered.connect(self.circuit_scene.flip_selected)
        menu.addAction(act_flip)

        menu.addSeparator()

        act_cut = QAction("Cut\tCtrl+X", menu)
        act_cut.setShortcut(QKeySequence.StandardKey.Cut)
        act_cut.setEnabled(has_sel)
        act_cut.triggered.connect(self.circuit_scene.cut_selected)
        menu.addAction(act_cut)

        act_copy = QAction("Copy\tCtrl+C", menu)
        act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        act_copy.setEnabled(has_sel)
        act_copy.triggered.connect(self.circuit_scene.copy_selected)
        menu.addAction(act_copy)

        act_paste = QAction("Paste\tCtrl+V", menu)
        act_paste.setShortcut(QKeySequence.StandardKey.Paste)
        act_paste.setEnabled(self.circuit_scene._clipboard is not None)
        act_paste.triggered.connect(self.circuit_scene.paste_clipboard)
        menu.addAction(act_paste)

        act_delete = QAction("Delete\tDel", menu)
        act_delete.setShortcut(QKeySequence.StandardKey.Delete)
        act_delete.setEnabled(has_sel)
        act_delete.triggered.connect(self.circuit_scene.delete_selected)
        menu.addAction(act_delete)

        menu.addSeparator()

        act_junction = QAction("Toggle Junction Dot Here\tJ", menu)
        act_junction.setShortcut(QKeySequence("J"))
        act_junction.triggered.connect(
            lambda: self.circuit_scene.toggle_junction_at(scene_pt)
        )
        menu.addAction(act_junction)

        menu.exec(event.globalPos())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self._update_cursor_for_scene_pos(self._last_scene_pos)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self.circuit_scene.is_dragging_wire_endpoint:
                self.circuit_scene.cancel_wire_endpoint_drag()
            elif self.circuit_scene.is_dragging_wire_segment:
                self.circuit_scene.cancel_wire_segment_drag()
            elif self.circuit_scene.is_drawing_wire:
                self._wire_drag_active = False
                self.circuit_scene.cancel_wire_preview()
            elif self._wire_mode_enabled:
                self.set_wire_mode(False)
            else:
                self.circuit_scene.clearSelection()
            self._update_cursor_for_scene_pos(self._last_scene_pos)
            event.accept()
            return
        if event.key() == Qt.Key.Key_R and not event.modifiers():
            self.circuit_scene.rotate_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F and not event.modifiers():
            self.circuit_scene.flip_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_J and not event.modifiers():
            self.circuit_scene.toggle_junction_at(self._last_scene_pos)
            event.accept()
            return
        if event.key() == Qt.Key.Key_W and not event.modifiers():
            self.set_wire_mode(not self._wire_mode_enabled)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Backspace and self.circuit_scene.is_drawing_wire:
            self.circuit_scene.pop_wire_preview_waypoint(self._last_scene_pos)
            self._update_cursor_for_scene_pos(self._last_scene_pos)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.circuit_scene.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor_for_scene_pos(self._last_scene_pos)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat(MIME_COMPONENT_ID):
            comp_id = bytes(event.mimeData().data(MIME_COMPONENT_ID)).decode("utf-8")
            scene_pos = self.mapToScene(event.position().toPoint())
            self.circuit_scene.show_ghost_preview(comp_id, scene_pos)
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasFormat(MIME_COMPONENT_ID):
            comp_id = bytes(event.mimeData().data(MIME_COMPONENT_ID)).decode("utf-8")
            scene_pos = self.mapToScene(event.position().toPoint())
            self.circuit_scene.show_ghost_preview(comp_id, scene_pos)
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self.circuit_scene.hide_ghost_preview()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasFormat(MIME_COMPONENT_ID):
            comp_id = bytes(event.mimeData().data(MIME_COMPONENT_ID)).decode("utf-8")
            scene_pos = self.mapToScene(event.position().toPoint())
            self.circuit_scene.hide_ghost_preview()
            self.circuit_scene.place_new_component(comp_id, scene_pos)
            event.acceptProposedAction()
            self.setFocus()
            return
        super().dropEvent(event)
