"""QGraphicsObject for orthogonal schematic wires and live wire-routing preview."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)

from circuitsim.app.themes import ThemeMode, ThemePalette, get_theme_palette
from circuitsim.canvas.connectivity import (
    PointPx,
    WireSpec,
    simplify_orthogonal_points,
)
from circuitsim.wiring.model import FreeEnd

if TYPE_CHECKING:
    from circuitsim.wiring.model import Attachment


class WireItem(QGraphicsObject):
    """Orthogonal wire polyline on the schematic canvas.

    Stores its vertices in scene pixel coordinates (`points`) while keeping
    `pos() == (0, 0)`. Supports selection, theme switching, attachment metadata
    (`end_a`, `end_b`, `hints`) from `WireGraph`, and a dashed live-preview mode
    while the user is routing a new wire.
    """

    def __init__(
        self,
        wire_id: str,
        points: Sequence[tuple[float, float] | PointPx],
        *,
        theme: ThemeMode = ThemeMode.DARK,
        is_preview: bool = False,
        end_a: Attachment | None = None,
        end_b: Attachment | None = None,
        hints: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__()
        self.wire_id: str = wire_id
        self.theme: ThemeMode = theme
        self.is_preview: bool = is_preview
        self.end_a: Attachment | None = end_a
        self.end_b: Attachment | None = end_b
        self.hints: tuple[PointPx, ...] = tuple((int(p[0]), int(p[1])) for p in hints)
        self._points: tuple[PointPx, ...] = simplify_orthogonal_points(
            points, snap=True
        )
        self._path: QPainterPath = QPainterPath()
        self._hit_shape: QPainterPath = QPainterPath()
        self.diagnostic_severity: str | None = None
        self.diagnostic_message: str = ""
        self.voltage_color_enabled: bool = False
        self.current_dots_enabled: bool = False
        self.overlay_rgb: tuple[int, int, int] | None = None
        self.dot_velocity: float = 0.0
        self.dot_phase: float = 0.0
        self._rebuild_paths()

        if not is_preview:
            self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setZValue(5.0)
        else:
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.setZValue(95.0)

    @property
    def points(self) -> tuple[PointPx, ...]:
        """Return the normalized orthogonal polyline points in scene coordinates."""
        return self._points

    def set_points(
        self, points: Sequence[tuple[float, float] | PointPx]
    ) -> None:
        """Update the wire's polyline points and trigger a geometry/scene repaint."""
        norm = simplify_orthogonal_points(points, snap=True)
        if norm == self._points:
            return
        self.prepareGeometryChange()
        self._points = norm
        self._rebuild_paths()
        self.update()

    def set_theme(self, theme: ThemeMode) -> None:
        """Update theme color and repaint."""
        self.theme = theme
        self.update()

    def set_diagnostic(
        self, severity: str | None, message: str = ""
    ) -> None:
        """Set or clear canvas problem highlighting on this wire."""
        if self.diagnostic_severity == severity and self.diagnostic_message == message:
            return
        self.diagnostic_severity = severity
        self.diagnostic_message = message
        self.setToolTip(message)
        self.update()

    def effective_highlight_color(self) -> str | None:
        """Return the active problem-highlight hex color ('#FF5252' for error, '#FFA726' for warning), or None."""
        if self.diagnostic_severity == "error":
            return "#FF5252"
        if self.diagnostic_severity == "warning":
            return "#FFA726"
        return None

    def set_overlay_state(
        self,
        *,
        voltage_color_enabled: bool | None = None,
        current_dots_enabled: bool | None = None,
        overlay_rgb: tuple[int, int, int] | None = None,
        dot_velocity: float | None = None,
        dot_phase: float | None = None,
        clear_overlay: bool = False,
    ) -> None:
        """Update live voltage color and current dot animation parameters on this wire."""
        if voltage_color_enabled is not None:
            self.voltage_color_enabled = bool(voltage_color_enabled)
        if current_dots_enabled is not None:
            self.current_dots_enabled = bool(current_dots_enabled)
        if clear_overlay:
            self.overlay_rgb = None
            self.dot_velocity = 0.0
            self.dot_phase = 0.0
        else:
            if overlay_rgb is not None:
                self.overlay_rgb = (int(overlay_rgb[0]), int(overlay_rgb[1]), int(overlay_rgb[2]))
            if dot_velocity is not None:
                self.dot_velocity = float(dot_velocity)
            if dot_phase is not None:
                self.dot_phase = float(dot_phase) % 1.0
        self.update()

    def advance_dot_phase(self, step_scale: float = 0.12) -> None:
        """Advance current dot phase along the polyline according to signed `dot_velocity`."""
        if not self.current_dots_enabled or abs(self.dot_velocity) <= 0.0:
            return
        self.dot_phase = (self.dot_phase + self.dot_velocity * float(step_scale)) % 1.0
        self.update()

    def _sample_dot_positions(self, spacing_px: float = 28.0) -> list[QPointF]:
        """Return animated dot positions along `self._points` shifted by normalized `dot_phase`."""
        if len(self._points) < 2:
            return []
        segments: list[tuple[float, float, float, float, float]] = []
        total_len = 0.0
        for idx in range(len(self._points) - 1):
            x0, y0 = float(self._points[idx][0]), float(self._points[idx][1])
            x1, y1 = float(self._points[idx + 1][0]), float(self._points[idx + 1][1])
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len > 1e-6:
                segments.append((x0, y0, x1, y1, seg_len))
                total_len += seg_len
        if total_len < 6.0:
            return []
        step = max(12.0, float(spacing_px))
        offset = (self.dot_phase % 1.0) * step
        positions: list[QPointF] = []
        s = offset
        seg_idx = 0
        seg_start = 0.0
        while s <= total_len and seg_idx < len(segments):
            x0, y0, x1, y1, sl = segments[seg_idx]
            if s > seg_start + sl:
                seg_start += sl
                seg_idx += 1
                continue
            t = (s - seg_start) / sl
            positions.append(QPointF(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            s += step
        return positions

    def to_wire_spec(self) -> WireSpec:
        """Convert to a pure-Python `WireSpec` for connectivity extraction."""
        return WireSpec(wire_id=self.wire_id, points=self._points)

    def _rebuild_paths(self) -> None:
        path = QPainterPath()
        if len(self._points) >= 2:
            p0 = self._points[0]
            path.moveTo(float(p0[0]), float(p0[1]))
            for px, py in self._points[1:]:
                path.lineTo(float(px), float(py))
        self._path = path

        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._hit_shape = stroker.createStroke(path)

    def boundingRect(self) -> QRectF:
        if self._path.isEmpty():
            return QRectF(0.0, 0.0, 1.0, 1.0)
        return self._hit_shape.boundingRect().adjusted(-4.0, -4.0, 4.0, 4.0)

    def shape(self) -> QPainterPath:
        return self._hit_shape

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        try:
            self._paint_impl(painter, option, widget)
        except Exception as exc:
            self._handle_paint_error(painter, exc)

    def _handle_paint_error(self, painter: QPainter, exc: Exception) -> None:
        """Handle wire paint failure: log once per item and draw red dashed rectangle with '!'."""
        if not getattr(self, "_paint_error_logged", False):
            self._paint_error_logged = True
            try:
                from circuitsim.app.crash_handler import write_crash_log

                write_crash_log(type(exc), exc, exc.__traceback__, count=1)
            except Exception:
                pass

        try:
            painter.save()
            red_col = QColor("#FF5252")
            pen = QPen(red_col, 1.8, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 82, 82, 35)))
            rect = self.boundingRect()
            if rect.width() < 14:
                rect = rect.adjusted(-7, 0, 7, 0)
            if rect.height() < 14:
                rect = rect.adjusted(0, -7, 0, 7)
            painter.drawRoundedRect(rect, 3.0, 3.0)

            font = QFont("Segoe UI", 10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(red_col)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "!")
            painter.restore()
        except Exception:
            pass

    def _paint_impl(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        _ = (option, widget)
        if len(self._points) < 2:
            return

        palette: ThemePalette = get_theme_palette(self.theme)
        painter.save()

        active_color: QColor | None = None
        if self.is_preview:
            pen = QPen(
                QColor(palette.terminal_hover),
                2.4,
                Qt.PenStyle.DashLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.MiterJoin,
            )
            painter.setPen(pen)
            painter.drawPath(self._path)
            # Draw start and end preview rings
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for ep in (self._points[0], self._points[-1]):
                painter.drawEllipse(
                    QPointF(float(ep[0]), float(ep[1])), 5.0, 5.0
                )
        elif self.isSelected():
            active_color = QColor(palette.selection_accent)
            # Subtle outer glow + accent wire stroke
            glow_pen = QPen(
                active_color,
                5.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            glow_col = QColor(active_color)
            glow_col.setAlpha(90)
            glow_pen.setColor(glow_col)
            painter.setPen(glow_pen)
            painter.drawPath(self._path)

            main_pen = QPen(
                active_color,
                2.4,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.MiterJoin,
            )
            painter.setPen(main_pen)
            painter.drawPath(self._path)
        elif self.diagnostic_severity in ("error", "warning"):
            active_color = (
                QColor("#FF5252")
                if self.diagnostic_severity == "error"
                else QColor("#FFA726")
            )
            glow_col = QColor(active_color)
            glow_col.setAlpha(85)
            painter.setPen(
                QPen(
                    glow_col,
                    5.5,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawPath(self._path)
            painter.setPen(
                QPen(
                    active_color,
                    2.4,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.MiterJoin,
                )
            )
            painter.drawPath(self._path)
        else:
            if self.voltage_color_enabled and self.overlay_rgb is not None:
                active_color = QColor(*self.overlay_rgb)
                pen_w = 2.6
            else:
                active_color = QColor(palette.symbol_stroke)
                pen_w = 2.2
            pen = QPen(
                active_color,
                pen_w,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.MiterJoin,
            )
            painter.setPen(pen)
            painter.drawPath(self._path)

        if not self.is_preview and self.current_dots_enabled and abs(self.dot_velocity) > 0.0:
            dot_pts = self._sample_dot_positions()
            if dot_pts:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#FFD166")))
                for dp in dot_pts:
                    painter.drawEllipse(dp, 2.5, 2.5)

        if not self.is_preview and active_color is not None:
            marker_pen = QPen(active_color, 1.6)
            painter.setPen(marker_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if isinstance(self.end_a, FreeEnd):
                p0 = self._points[0]
                painter.drawEllipse(QPointF(float(p0[0]), float(p0[1])), 3.5, 3.5)
            if isinstance(self.end_b, FreeEnd):
                pn = self._points[-1]
                painter.drawEllipse(QPointF(float(pn[0]), float(pn[1])), 3.5, 3.5)

        painter.restore()


class JunctionItem(QGraphicsObject):
    """Visual junction dot item with paint safety guard."""

    def __init__(
        self,
        position: QPointF,
        theme: ThemeMode | str = ThemeMode.DARK,
        parent: Any = None,
        *,
        junction_id: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(position)
        self.theme: ThemeMode | str = theme
        self.junction_id: str | None = junction_id
        self._paint_error_logged: bool = False
        self.setZValue(6.0)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def set_theme(self, theme: ThemeMode | str) -> None:
        """Update theme color and repaint."""
        self.theme = theme
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-7.0, -7.0, 14.0, 14.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        try:
            self._paint_impl(painter, option, widget)
        except Exception as exc:
            self._handle_paint_error(painter, exc)

    def _handle_paint_error(self, painter: QPainter, exc: Exception) -> None:
        if not getattr(self, "_paint_error_logged", False):
            self._paint_error_logged = True
            try:
                from circuitsim.app.crash_handler import write_crash_log

                write_crash_log(type(exc), exc, exc.__traceback__, count=1)
            except Exception:
                pass

        try:
            painter.save()
            red_col = QColor("#FF5252")
            painter.setPen(QPen(red_col, 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(255, 82, 82, 35)))
            rect = self.boundingRect()
            painter.drawRect(rect)
            font = QFont("Segoe UI", 8)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(red_col)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "!")
            painter.restore()
        except Exception:
            pass

    def _paint_impl(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        _ = (option, widget)
        palette = get_theme_palette(self.theme)
        painter.save()
        if self.isSelected():
            painter.setPen(QPen(QColor(palette.selection_accent), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0.0, 0.0), 5.8, 5.8)
        junc_color = QColor(palette.symbol_stroke)
        painter.setPen(QPen(junc_color, 1.2))
        painter.setBrush(QBrush(junc_color))
        painter.drawEllipse(QPointF(0.0, 0.0), 4.2, 4.2)
        painter.restore()

