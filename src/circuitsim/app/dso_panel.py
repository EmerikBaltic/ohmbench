"""Voltcraft DSO-1062D styled Benchtop Oscilloscope front-panel dialog (Qt 6)."""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDial,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components.oscilloscope import Oscilloscope
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.simulation_session import SimulationFrame, SimulationState
from circuitsim.engine.solver import MNASolver
from circuitsim.scope.dso_engine import (
    DSO_AVG_COUNTS,
    DSO_HORIZONTAL_DIVS,
    DSO_MEM_DEPTHS,
    DSO_SEC_PER_DIV,
    DSO_VERTICAL_DIVS,
    DSO_VOLTS_PER_DIV,
    DSOFrame,
    DSOState,
    compute_auto_set,
    compute_set_to_50_percent,
    process_dso_capture,
    recall_setup_json,
    reset_default_setup,
    save_setup_json,
    snap_dso_step,
)

DSO_MENU_NAMES: tuple[str, ...] = (
    "CH1",
    "CH2",
    "MATH",
    "ACQUIRE",
    "MEASURE",
    "CURSOR",
    "DISPLAY",
    "HORIZ",
    "TRIG",
    "SAVE/RECALL",
    "UTILITY",
)


def _step_table(current: float, table: tuple[float, ...], delta: int) -> float:
    snapped = snap_dso_step(current, table)
    idx = table.index(snapped)
    nxt = max(0, min(len(table) - 1, idx + int(delta)))
    return float(table[nxt])


def _cycle_choice(current: object, choices: tuple[object, ...]) -> object:
    try:
        idx = choices.index(current)
        return choices[(idx + 1) % len(choices)]
    except ValueError:
        return choices[0]


class _DSOScreenWidget(QWidget):
    """High-contrast 12x8 DSO LCD viewport with graticule, traces, cursors & softkey menu."""

    cursors_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(QSize(580, 380))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.state: DSOState = DSOState()
        self.frame: DSOFrame | None = None
        self.active_menu_title: str = "CH1"
        self.menu_rows: list[tuple[str, str]] = []
        self.menu_visible: bool = True
        self._dragging_cursor: int | None = None
        self._plot_rect: QRectF = QRectF()

    def set_view_data(
        self,
        *,
        state: DSOState,
        frame: DSOFrame | None,
        menu_title: str,
        menu_rows: list[tuple[str, str]],
        menu_visible: bool,
    ) -> None:
        self.state = state
        self.frame = frame
        self.active_menu_title = menu_title
        self.menu_rows = list(menu_rows[:7])
        self.menu_visible = bool(menu_visible)
        self.update()

    def _compute_plot_rect(self) -> QRectF:
        w = float(self.width())
        h = float(self.height())
        top_h = 26.0
        bot_h = 26.0
        return QRectF(8.0, top_h + 4.0, max(120.0, w - 16.0), max(120.0, h - top_h - bot_h - 8.0))

    def _y_from_div(self, y_div: float, plot: QRectF) -> float:
        return plot.center().y() - (y_div / (DSO_VERTICAL_DIVS / 2.0)) * (plot.height() / 2.0)

    def _div_from_y(self, y_px: float, plot: QRectF) -> float:
        return -((y_px - plot.center().y()) / max(1.0, plot.height() / 2.0)) * (DSO_VERTICAL_DIVS / 2.0)

    def _cursor_ref_channel(self):
        return self.state.ch2 if self.state.cursors.cursor_source.upper() == "CH2" else self.state.ch1

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.state.cursors.cursor_mode != "OFF":
            plot = self._compute_plot_rect()
            if plot.contains(event.position()):
                mode = self.state.cursors.cursor_mode.upper()
                if mode == "VOLTAGE":
                    ch = self._cursor_ref_channel()
                    y1 = self._y_from_div(self.state.cursors.v1 / ch.volts_per_div + ch.position_div, plot)
                    y2 = self._y_from_div(self.state.cursors.v2 / ch.volts_per_div + ch.position_div, plot)
                    self._dragging_cursor = 1 if abs(event.position().y() - y1) <= abs(event.position().y() - y2) else 2
                elif mode == "TIME":
                    span_s = max(self.state.horizontal.sec_per_div * DSO_HORIZONTAL_DIVS, 1e-12)
                    t_min = -0.5 * span_s + self.state.horizontal.position_s
                    x1 = plot.left() + ((self.state.cursors.t1 - t_min) / span_s) * plot.width()
                    x2 = plot.left() + ((self.state.cursors.t2 - t_min) / span_s) * plot.width()
                    self._dragging_cursor = 1 if abs(event.position().x() - x1) <= abs(event.position().x() - x2) else 2
                self._update_cursor_from_pos(event.position(), plot)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_cursor in (1, 2):
            plot = self._compute_plot_rect()
            self._update_cursor_from_pos(event.position(), plot)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_cursor = None
        super().mouseReleaseEvent(event)

    def _update_cursor_from_pos(self, pos: QPointF, plot: QRectF) -> None:
        mode = self.state.cursors.cursor_mode.upper()
        if mode == "VOLTAGE":
            ch = self._cursor_ref_channel()
            y_div = self._div_from_y(max(plot.top(), min(plot.bottom(), pos.y())), plot)
            val_v = (y_div - ch.position_div) * ch.volts_per_div
            if self._dragging_cursor == 1:
                self.state.cursors.v1 = val_v
            else:
                self.state.cursors.v2 = val_v
            self.cursors_changed.emit()
        elif mode == "TIME":
            span_s = max(self.state.horizontal.sec_per_div * DSO_HORIZONTAL_DIVS, 1e-12)
            t_min = -0.5 * span_s + self.state.horizontal.position_s
            frac = max(0.0, min(1.0, (pos.x() - plot.left()) / max(1.0, plot.width())))
            t_val = t_min + frac * span_s
            if self._dragging_cursor == 1:
                self.state.cursors.t1 = t_val
            else:
                self.state.cursors.t2 = t_val
            self.cursors_changed.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        # Dark high-contrast LCD chassis background
        painter.fillRect(QRectF(0.0, 0.0, w, h), QColor("#0F172A"))
        plot = self._compute_plot_rect()
        self._plot_rect = plot

        # Top status bar
        self._paint_top_bar(painter, w)
        # Bottom status bar
        self._paint_bottom_bar(painter, w, h)
        # 12x8 graticule with sub-division ticks
        self._paint_graticule(painter, plot)

        # Waveforms + markers inside plot area
        if self.frame is not None:
            self._paint_waveforms(painter, plot, self.frame)
            if self.state.auto_measure_enabled:
                self._paint_auto_measure_strip(painter, plot, self.frame)
        self._paint_markers_and_cursors(painter, plot)

        # Right-edge blue softkey menu
        if self.menu_visible:
            self._paint_softkey_menu(painter, plot, w)

        painter.end()

    def _paint_top_bar(self, painter: QPainter, w: float) -> None:
        bar = QRectF(0.0, 0.0, w, 26.0)
        painter.fillRect(bar, QColor("#141923"))
        painter.setPen(QColor("#334155"))
        painter.drawLine(QPointF(0.0, 26.0), QPointF(w, 26.0))

        trig_st = self.frame.trigger_status if self.frame is not None else ("Auto" if self.state.running else "Stop")
        if trig_st == "Trig'd":
            pill_col = QColor("#16A34A")
        elif trig_st == "Wait":
            pill_col = QColor("#D97706")
        elif trig_st == "Stop":
            pill_col = QColor("#DC2626")
        else:
            pill_col = QColor("#2563EB")

        pill_rect = QRectF(8.0, 4.0, 58.0, 18.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pill_col)
        painter.drawRoundedRect(pill_rect, 3.0, 3.0)
        painter.setPen(QColor("#FFFFFF"))
        f_bold = QFont("Segoe UI", 8)
        f_bold.setBold(True)
        painter.setFont(f_bold)
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, trig_st)

        # Main timebase
        m_str = format_engineering_value(self.state.horizontal.sec_per_div, "s", sig_figs=3)
        painter.setPen(QColor("#E2E8F0"))
        f_reg = QFont("Segoe UI", 8)
        painter.setFont(f_reg)
        painter.drawText(QRectF(74.0, 2.0, 96.0, 22.0), Qt.AlignmentFlag.AlignVCenter, f"M {m_str}/")

        # Miniature memory overview bar
        mem_box = QRectF(176.0, 7.0, 94.0, 12.0)
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QColor("#0B1120"))
        painter.drawRect(mem_box)
        painter.fillRect(QRectF(mem_box.center().x() - 12.0, 8.0, 24.0, 10.0), QColor("#38BDF8"))
        painter.setPen(QColor("#FACC15"))
        painter.drawLine(QPointF(mem_box.center().x(), 6.0), QPointF(mem_box.center().x(), 20.0))

        # Window zoom & trigger horizontal delay
        w_str = (
            f"W {format_engineering_value(self.state.horizontal.zoom_sec_per_div, 's', sig_figs=3)}"
            if self.state.horizontal.zoom_enabled
            else f"Acq {self.state.acquire.acq_mode[:4]}"
        )
        painter.setPen(QColor("#94A3B8"))
        painter.drawText(QRectF(280.0, 2.0, 100.0, 22.0), Qt.AlignmentFlag.AlignVCenter, w_str)

        delay_str = format_engineering_value(self.state.horizontal.position_s, "s", sig_figs=3)
        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(QRectF(max(390.0, w - 190.0), 2.0, 180.0, 22.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"Delay: {delay_str}")

    def _paint_bottom_bar(self, painter: QPainter, w: float, h: float) -> None:
        y0 = h - 26.0
        painter.fillRect(QRectF(0.0, y0, w, 26.0), QColor("#141923"))
        painter.setPen(QColor("#334155"))
        painter.drawLine(QPointF(0.0, y0), QPointF(w, y0))

        f_bold = QFont("Segoe UI", 8)
        f_bold.setBold(True)
        painter.setFont(f_bold)

        # CH1 badge
        c1_str = f"CH1 {'OFF' if not self.state.ch1.enabled else self.state.ch1.coupling + ' ' + format_engineering_value(self.state.ch1.volts_per_div, 'V', sig_figs=3)}"
        c1_box = QRectF(8.0, y0 + 4.0, 118.0, 18.0)
        painter.fillRect(c1_box, QColor("#FACC15") if self.state.ch1.enabled else QColor("#334155"))
        painter.setPen(QColor("#0F172A") if self.state.ch1.enabled else QColor("#94A3B8"))
        painter.drawText(c1_box, Qt.AlignmentFlag.AlignCenter, c1_str)

        # CH2 badge
        c2_str = f"CH2 {'OFF' if not self.state.ch2.enabled else self.state.ch2.coupling + ' ' + format_engineering_value(self.state.ch2.volts_per_div, 'V', sig_figs=3)}"
        c2_box = QRectF(132.0, y0 + 4.0, 118.0, 18.0)
        painter.fillRect(c2_box, QColor("#22D3EE") if self.state.ch2.enabled else QColor("#334155"))
        painter.setPen(QColor("#0F172A") if self.state.ch2.enabled else QColor("#94A3B8"))
        painter.drawText(c2_box, Qt.AlignmentFlag.AlignCenter, c2_str)

        # Trigger badge
        slope_icon = "↗" if self.state.trigger.slope == "RISING" else ("↘" if self.state.trigger.slope == "FALLING" else "⇅")
        lvl_str = format_engineering_value(self.state.trigger.level_v, "V", sig_figs=3)
        trig_txt = f"{self.state.trigger.source} {slope_icon} {lvl_str}"
        painter.setPen(QColor("#F8FAFC"))
        painter.drawText(QRectF(258.0, y0 + 2.0, 165.0, 22.0), Qt.AlignmentFlag.AlignVCenter, trig_txt)

        # Hardware trigger frequency counter
        freq = self.frame.trig_freq_hz if self.frame is not None else None
        freq_txt = f"{freq:.5f}Hz" if (freq is not None and freq < 100.0) else (
            format_engineering_value(freq, "Hz", sig_figs=5) if freq is not None else "0.00000Hz"
        )
        painter.setPen(QColor("#38BDF8"))
        painter.drawText(QRectF(max(430.0, w - 150.0), y0 + 2.0, 140.0, 22.0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, freq_txt)

    def _paint_graticule(self, painter: QPainter, plot: QRectF) -> None:
        alpha = int(max(40, min(220, round(self.state.display.grid_brightness * 180))))
        painter.setPen(QPen(QColor(71, 85, 105, alpha), 1.0))
        painter.drawRect(plot)

        grid_pen = QPen(QColor(71, 85, 105, max(30, alpha // 2)), 0.8, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, DSO_HORIZONTAL_DIVS):
            x = plot.left() + i * plot.width() / DSO_HORIZONTAL_DIVS
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        for j in range(1, DSO_VERTICAL_DIVS):
            y = plot.top() + j * plot.height() / DSO_VERTICAL_DIVS
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        # Center crosshairs with sub-division tick marks
        axis_pen = QPen(QColor(148, 163, 184, alpha), 1.0)
        painter.setPen(axis_pen)
        cx = plot.center().x()
        cy = plot.center().y()
        painter.drawLine(QPointF(plot.left(), cy), QPointF(plot.right(), cy))
        painter.drawLine(QPointF(cx, plot.top()), QPointF(cx, plot.bottom()))

        sub_x = plot.width() / (DSO_HORIZONTAL_DIVS * 5.0)
        for k in range(DSO_HORIZONTAL_DIVS * 5 + 1):
            tx = plot.left() + k * sub_x
            painter.drawLine(QPointF(tx, cy - 2.5), QPointF(tx, cy + 2.5))
        sub_y = plot.height() / (DSO_VERTICAL_DIVS * 5.0)
        for k in range(DSO_VERTICAL_DIVS * 5 + 1):
            ty = plot.top() + k * sub_y
            painter.drawLine(QPointF(cx - 2.5, ty), QPointF(cx + 2.5, ty))

    def _draw_trace(
        self,
        painter: QPainter,
        pts: list[QPointF] | list[list[QPointF]],
        color: QColor,
        width: float = 1.7,
    ) -> None:
        if not pts:
            return
        segments: list[list[QPointF]]
        if isinstance(pts[0], list):
            segments = [seg for seg in pts if seg]  # type: ignore[assignment]
        else:
            segments = [pts]  # type: ignore[list-item]
        painter.setPen(QPen(color, width))
        is_dots = self.state.display.render_style.upper() == "DOTS"
        for seg in segments:
            if is_dots or len(seg) == 1:
                for pt in seg:
                    painter.drawPoint(pt)
            else:
                painter.drawPolyline(QPolygonF(seg))

    def _map_yt_segments(
        self,
        t_rel: np.ndarray,
        vals: np.ndarray,
        v_div: float,
        pos_div: float,
        plot: QRectF,
    ) -> list[list[QPointF]]:
        n = min(len(t_rel), len(vals))
        if n < 2:
            return []
        eff_sec_div = (
            self.state.horizontal.zoom_sec_per_div
            if self.state.horizontal.zoom_enabled
            else self.state.horizontal.sec_per_div
        )
        t_span = max(eff_sec_div * DSO_HORIZONTAL_DIVS, 1e-12)
        frame_x = (
            self.frame.x_norm
            if (self.frame is not None and len(self.frame.x_norm) == n)
            else None
        )

        diffs = np.diff(t_rel[:n])
        pos_diffs = diffs[diffs > 0.0]
        med_dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0

        step = max(1, n // 1200)
        segments: list[list[QPointF]] = [[]]
        prev_t: float | None = None
        gap_limit = 10.0 * med_dt * step if med_dt > 0.0 else 0.0
        for idx in range(0, n, step):
            cur_t = float(t_rel[idx])
            if prev_t is not None and (cur_t <= prev_t or (gap_limit > 0.0 and (cur_t - prev_t) > gap_limit)):
                if segments[-1]:
                    segments.append([])
            prev_t = cur_t
            if frame_x is not None:
                fx = float(frame_x[idx])
            else:
                fx = (cur_t + 0.5 * t_span) / t_span
            px = plot.left() + fx * plot.width()
            py = self._y_from_div(float(vals[idx]) / max(v_div, 1e-9) + pos_div, plot)
            segments[-1].append(QPointF(px, py))
        return [seg for seg in segments if len(seg) >= 1]

    def _map_yt_points(
        self,
        t_rel: np.ndarray,
        vals: np.ndarray,
        v_div: float,
        pos_div: float,
        plot: QRectF,
    ) -> list[QPointF]:
        return [pt for seg in self._map_yt_segments(t_rel, vals, v_div, pos_div, plot) for pt in seg]

    def _paint_waveforms(self, painter: QPainter, plot: QRectF, frame: DSOFrame) -> None:
        painter.save()
        painter.setClipRect(plot)

        if self.state.horizontal.format_mode.upper() == "XY":
            n = min(len(frame.ch1_v), len(frame.ch2_v))
            step = max(1, n // 900)
            pts: list[QPointF] = []
            for idx in range(0, n, step):
                x_div = float(frame.ch1_v[idx]) / max(self.state.ch1.volts_per_div, 1e-9) + self.state.ch1.position_div
                y_div = float(frame.ch2_v[idx]) / max(self.state.ch2.volts_per_div, 1e-9) + self.state.ch2.position_div
                px = plot.center().x() + (x_div / (DSO_HORIZONTAL_DIVS / 2.0)) * (plot.width() / 2.0)
                py = self._y_from_div(y_div, plot)
                pts.append(QPointF(px, py))
            self._draw_trace(painter, pts, QColor("#38BDF8"), 1.6)
            painter.restore()
            return

        # Persistence ghosts
        for idx_p, (p1, p2) in enumerate(frame.persistence_traces[:-1]):
            alpha = 40 + idx_p * 25
            if self.state.ch1.enabled:
                segs1 = self._map_yt_segments(frame.times_rel_s, p1, self.state.ch1.volts_per_div, self.state.ch1.position_div, plot)
                self._draw_trace(painter, segs1, QColor(250, 204, 21, alpha), 1.2)
            if self.state.ch2.enabled:
                segs2 = self._map_yt_segments(frame.times_rel_s, p2, self.state.ch2.volts_per_div, self.state.ch2.position_div, plot)
                self._draw_trace(painter, segs2, QColor(34, 211, 238, alpha), 1.2)

        if self.state.ch1.enabled:
            segs1 = self._map_yt_segments(frame.times_rel_s, frame.ch1_v, self.state.ch1.volts_per_div, self.state.ch1.position_div, plot)
            self._draw_trace(painter, segs1, QColor("#FACC15"), 1.8)

        if self.state.ch2.enabled:
            segs2 = self._map_yt_segments(frame.times_rel_s, frame.ch2_v, self.state.ch2.volts_per_div, self.state.ch2.position_div, plot)
            self._draw_trace(painter, segs2, QColor("#22D3EE"), 1.8)

        if self.state.math.enabled:
            if self.state.math.operation.upper() == "FFT":
                freqs = frame.fft_freqs_hz
                db = frame.fft_mag_db
                if len(freqs) >= 2:
                    f_max = max(float(freqs[-1]), 1.0)
                    # Cap display span neatly so 1kHz tones stay clearly visible
                    if frame.fft_peak_hz is not None and frame.fft_peak_hz > 0.0:
                        f_max = min(f_max, max(frame.fft_peak_hz * 5.0, 5000.0))
                    pts_fft: list[QPointF] = []
                    for f_val, db_val in zip(freqs, db):
                        if f_val > f_max:
                            break
                        px = plot.left() + (float(f_val) / f_max) * plot.width()
                        norm_y = (float(db_val) - (-80.0)) / 100.0
                        py = plot.bottom() - max(0.0, min(1.0, norm_y)) * plot.height()
                        pts_fft.append(QPointF(px, py))
                    self._draw_trace(painter, pts_fft, QColor("#E879F9"), 1.6)
                    if frame.fft_peak_hz is not None:
                        painter.setPen(QColor("#F0ABFC"))
                        painter.setFont(QFont("Segoe UI", 8))
                        painter.drawText(
                            QRectF(plot.left() + 8.0, plot.top() + 6.0, 220.0, 18.0),
                            Qt.AlignmentFlag.AlignLeft,
                            f"FFT Peak: {format_engineering_value(frame.fft_peak_hz, 'Hz')}",
                        )
            else:
                segs_m = self._map_yt_segments(frame.times_rel_s, frame.math_v, self.state.math.scale, self.state.math.position_div, plot)
                self._draw_trace(painter, segs_m, QColor("#E879F9"), 1.6)

        painter.restore()

    def _paint_auto_measure_strip(self, painter: QPainter, plot: QRectF, frame: DSOFrame) -> None:
        strip = QRectF(plot.left() + 4.0, plot.bottom() - 20.0, plot.width() - 8.0, 18.0)
        painter.fillRect(strip, QColor(15, 23, 42, 185))
        painter.setFont(QFont("Segoe UI", 7))
        m1 = frame.ch1_meas
        f1 = format_engineering_value(m1.freq_hz, "Hz", sig_figs=3) if m1.freq_hz else "--"
        txt1 = f"CH1 Vpp:{format_engineering_value(m1.v_pp, 'V', 3)}  Vrms:{format_engineering_value(m1.v_rms, 'V', 3)}  Vavg:{format_engineering_value(m1.v_avg, 'V', 3)}  f:{f1}"
        painter.setPen(QColor("#FDE047"))
        painter.drawText(strip.adjusted(4.0, 0.0, -4.0, 0.0), Qt.AlignmentFlag.AlignVCenter, txt1)

    def _paint_markers_and_cursors(self, painter: QPainter, plot: QRectF) -> None:
        f_tag = QFont("Segoe UI", 7)
        f_tag.setBold(True)
        painter.setFont(f_tag)

        # Left-edge CH1/CH2 ground reference markers
        for ch_cfg, lbl, col in (
            (self.state.ch1, "1", QColor("#FACC15")),
            (self.state.ch2, "2", QColor("#22D3EE")),
        ):
            if not ch_cfg.enabled:
                continue
            y_g = max(plot.top() + 8.0, min(plot.bottom() - 8.0, self._y_from_div(ch_cfg.position_div, plot)))
            painter.fillRect(QRectF(plot.left() + 1.0, y_g - 6.0, 13.0, 12.0), col)
            painter.setPen(QColor("#0F172A"))
            painter.drawText(QRectF(plot.left() + 1.0, y_g - 6.0, 13.0, 12.0), Qt.AlignmentFlag.AlignCenter, lbl)

        # Right-edge trigger level arrow
        trig_ch = self.state.ch2 if self.state.trigger.source.upper() == "CH2" else self.state.ch1
        y_trig = self._y_from_div(self.state.trigger.level_v / max(trig_ch.volts_per_div, 1e-9) + trig_ch.position_div, plot)
        y_trig = max(plot.top() + 6.0, min(plot.bottom() - 6.0, y_trig))
        arrow = QPolygonF(
            [
                QPointF(plot.right() - 12.0, y_trig),
                QPointF(plot.right() - 2.0, y_trig - 5.0),
                QPointF(plot.right() - 2.0, y_trig + 5.0),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#F97316"))
        painter.drawPolygon(arrow)

        # Cursors
        cmode = self.state.cursors.cursor_mode.upper()
        if cmode == "VOLTAGE":
            ch = self._cursor_ref_channel()
            pen_c = QPen(QColor("#FDE047"), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(pen_c)
            for v_val in (self.state.cursors.v1, self.state.cursors.v2):
                yc = max(plot.top(), min(plot.bottom(), self._y_from_div(v_val / max(ch.volts_per_div, 1e-9) + ch.position_div, plot)))
                painter.drawLine(QPointF(plot.left(), yc), QPointF(plot.right(), yc))
        elif cmode == "TIME":
            pen_c = QPen(QColor("#FDE047"), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(pen_c)
            span_s = max(self.state.horizontal.sec_per_div * DSO_HORIZONTAL_DIVS, 1e-12)
            t_min = -0.5 * span_s + self.state.horizontal.position_s
            for t_val in (self.state.cursors.t1, self.state.cursors.t2):
                xc = max(plot.left(), min(plot.right(), plot.left() + ((t_val - t_min) / span_s) * plot.width()))
                painter.drawLine(QPointF(xc, plot.top()), QPointF(xc, plot.bottom()))

        if cmode != "OFF" and self.frame is not None:
            ro = self.frame.cursor_readout
            right_offset = 132.0 if self.menu_visible else 6.0
            box = QRectF(plot.right() - right_offset - 166.0, plot.top() + 6.0, 166.0, 38.0)
            painter.fillRect(box, QColor(15, 23, 42, 220))
            painter.setPen(QColor("#60A5FA"))
            painter.drawRect(box)
            painter.setPen(QColor("#F8FAFC"))
            painter.setFont(QFont("Segoe UI", 7))
            if cmode == "VOLTAGE":
                txt = f"V1:{format_engineering_value(ro.v1, 'V', 3)}  V2:{format_engineering_value(ro.v2, 'V', 3)}\nΔV:{format_engineering_value(ro.delta_v, 'V', 4)}"
            else:
                inv = format_engineering_value(ro.inv_delta_t_hz, "Hz", 4) if ro.inv_delta_t_hz else "--"
                txt = f"Δt:{format_engineering_value(ro.delta_t, 's', 3)}\n1/Δt:{inv}"
            painter.drawText(box.adjusted(4.0, 2.0, -4.0, -2.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)

    def _paint_softkey_menu(self, painter: QPainter, plot: QRectF, w: float) -> None:
        menu_w = min(126.0, max(84.0, plot.width() * 0.24))
        menu_x = plot.right() - menu_w
        menu_rect = QRectF(menu_x, plot.top(), menu_w, plot.height())
        painter.fillRect(menu_rect, QColor(30, 58, 138, 195))
        painter.setPen(QPen(QColor("#60A5FA"), 1.2))
        painter.drawRect(menu_rect)

        hdr = QRectF(menu_x, plot.top(), menu_w, 22.0)
        painter.fillRect(hdr, QColor("#2563EB"))
        f_hdr = QFont("Segoe UI", 8)
        f_hdr.setBold(True)
        painter.setFont(f_hdr)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(hdr, Qt.AlignmentFlag.AlignCenter, self.active_menu_title)

        row_h = (menu_rect.height() - 22.0) / 7.0
        f_lbl = QFont("Segoe UI", 7)
        f_val = QFont("Segoe UI", 8)
        f_val.setBold(True)
        for idx in range(7):
            ry = menu_rect.top() + 22.0 + idx * row_h
            cell = QRectF(menu_x + 2.0, ry + 1.0, menu_w - 4.0, row_h - 2.0)
            painter.fillRect(cell, QColor(15, 23, 42, 160 if idx < len(self.menu_rows) else 65))
            if idx < len(self.menu_rows):
                opt_name, opt_val = self.menu_rows[idx]
                painter.setFont(f_lbl)
                painter.setPen(QColor("#93C5FD"))
                painter.drawText(cell.adjusted(4.0, 2.0, -4.0, -cell.height() * 0.45), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"F{idx} {opt_name}")
                painter.setFont(f_val)
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(cell.adjusted(4.0, cell.height() * 0.42, -4.0, -2.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, opt_val)


class DSOOscilloscopeDialog(QDialog):
    """Voltcraft DSO-1062D Benchtop Oscilloscope dialog bound to an `Oscilloscope` item."""

    def __init__(
        self,
        osc_item_or_model: object,
        *,
        circuit_provider: Callable[[], Circuit] | None = None,
        nets_provider: Callable[[], dict[str, str]] | None = None,
        timebase_changed_callback: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(osc_item_or_model, Oscilloscope):
            self.osc_item = None
            self.osc_component: Oscilloscope = osc_item_or_model
        else:
            self.osc_item = osc_item_or_model
            self.osc_component = osc_item_or_model.component  # type: ignore[attr-defined]

        self._circuit_provider = circuit_provider
        self._nets_provider = nets_provider
        self._timebase_changed_callback = timebase_changed_callback
        self.state: DSOState = recall_setup_json(self.osc_component.dso_state_json)
        self.latest_frame: DSOFrame | None = None
        self.active_menu: str = "CH1"
        self.menu_visible: bool = True
        self._saved_setup_slot: str = ""
        self._syncing_dials: bool = False

        self._live_times: deque[float] = deque(maxlen=65536)
        self._live_ch1: deque[float] = deque(maxlen=65536)
        self._live_ch2: deque[float] = deque(maxlen=65536)
        self._live_ext: deque[float] = deque(maxlen=65536)
        self._last_live_t: float | None = None
        self._live_step_counter: int = 0
        self._received_sample_blocks: bool = False
        self._cached_nets: dict[str, str] = {}

        self.setWindowTitle(f"{self.osc_component.name} — DSO-1062D Benchtop Oscilloscope")
        self.resize(1080, 620)
        self._build_ui()
        self._sync_ui_from_state()
        self.refresh_net_labels()

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background-color: #D8DCE3; color: #0F172A; font-family: "Segoe UI"; }
            QGroupBox {
                background-color: #E5E7EB;
                border: 1px solid #94A3B8;
                border-radius: 5px;
                margin-top: 14px;
                font-weight: bold;
                font-size: 10px;
                color: #1E293B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #F1F5F9;
                color: #0F172A;
                border: 1px solid #94A3B8;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #E2E8F0; }
            QPushButton:pressed { background-color: #CBD5E1; }
            QDial { background-color: #CBD5E1; }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # Top dark nameplate banner
        header = QFrame(self)
        header.setStyleSheet("QFrame { background-color: #0F172A; border-radius: 4px; }")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(10, 5, 10, 5)
        name_lbl = QLabel(f"OHMBENCH DSO-1062D  |  {self.osc_component.name}  —  2 CHANNEL DIGITAL STORAGE OSCILLOSCOPE")
        name_lbl.setStyleSheet("color: #F8FAFC; font-weight: bold; font-size: 11px;")
        h_lay.addWidget(name_lbl)
        h_lay.addStretch(1)
        self.status_badge = QLabel("60MHz  1GSa/s")
        self.status_badge.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 10px;")
        h_lay.addWidget(self.status_badge)
        root.addWidget(header)

        # Center split: Left LCD + Softkey F0..F7 column | Right chassis controls
        body = QHBoxLayout()
        body.setSpacing(8)

        # Left LCD + softkey column + bottom BNC strip
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        lcd_row = QHBoxLayout()
        lcd_row.setSpacing(4)

        self.screen_widget = _DSOScreenWidget(self)
        self.screen_widget.cursors_changed.connect(self._on_screen_cursors_changed)
        lcd_row.addWidget(self.screen_widget, 1)

        soft_col = QVBoxLayout()
        soft_col.setSpacing(4)
        self.multipurpose_dial = QDial(self)
        self.multipurpose_dial.setObjectName("dso_multipurpose_dial")
        self.multipurpose_dial.setFixedSize(36, 36)
        self.multipurpose_dial.setWrapping(True)
        self.multipurpose_dial.setRange(0, 23)
        self.multipurpose_dial.valueChanged.connect(self._on_multipurpose_dial)
        soft_col.addWidget(self.multipurpose_dial, alignment=Qt.AlignmentFlag.AlignCenter)

        self.softkey_buttons: list[QPushButton] = []
        for idx in range(7):
            btn = QPushButton(f"F{idx}", self)
            btn.setObjectName(f"dso_softkey_f{idx}")
            btn.setFixedSize(36, 36)
            btn.clicked.connect(lambda _=False, k=idx: self.trigger_softkey(k))
            self.softkey_buttons.append(btn)
            soft_col.addWidget(btn)

        self.btn_menu_hide = QPushButton("F7\nHide", self)
        self.btn_menu_hide.setObjectName("dso_btn_menu_hide")
        self.btn_menu_hide.setFixedSize(36, 36)
        self.btn_menu_hide.clicked.connect(self._toggle_menu_visible)
        soft_col.addWidget(self.btn_menu_hide)
        lcd_row.addLayout(soft_col)
        left_col.addLayout(lcd_row, 1)

        # Bottom BNC jack visual strip
        bnc_bar = QFrame(self)
        bnc_bar.setStyleSheet("QFrame { background-color: #CBD5E1; border: 1px solid #94A3B8; border-radius: 4px; }")
        bnc_lay = QHBoxLayout(bnc_bar)
        bnc_lay.setContentsMargins(10, 4, 10, 4)
        self.bnc_labels: dict[str, QLabel] = {}
        for pin_name, ring_col, cap in (
            ("CH1", "#EAB308", "CH1"),
            ("CH2", "#06B6D4", "CH2"),
            ("GND", "#334155", "GND (300V CAT II)"),
            ("EXT_TRIG", "#0F172A", "EXT TRIG"),
        ):
            chip = QLabel(f"◉ {cap}: --", self)
            chip.setStyleSheet(
                f"color: #0F172A; background: #F8FAFC; border: 2px solid {ring_col}; border-radius: 10px; padding: 2px 8px; font-size: 10px; font-weight: 600;"
            )
            self.bnc_labels[pin_name] = chip
            bnc_lay.addWidget(chip)
        bnc_lay.addStretch(1)
        left_col.addWidget(bnc_bar)

        body.addLayout(left_col, 3)

        # Right front-panel matrix & rotary knobs
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        # Top control matrix + Run cluster
        top_grid = QGridLayout()
        top_grid.setSpacing(4)
        self.btn_save_recall = self._make_menu_btn("SAVE/\nRECALL", "SAVE/RECALL", top_grid, 0, 0)
        self.btn_measure = self._make_menu_btn("MEASURE", "MEASURE", top_grid, 0, 1)
        self.btn_acquire = self._make_menu_btn("ACQUIRE", "ACQUIRE", top_grid, 0, 2)
        self.btn_auto_set = QPushButton("AUTO\nSET", self)
        self.btn_auto_set.setStyleSheet("background-color: #FACC15; color: #0F172A; font-weight: bold;")
        self.btn_auto_set.clicked.connect(self._on_auto_set)
        top_grid.addWidget(self.btn_auto_set, 0, 3)

        self.btn_utility = self._make_menu_btn("UTILITY", "UTILITY", top_grid, 1, 0)
        self.btn_cursor = self._make_menu_btn("CURSOR", "CURSOR", top_grid, 1, 1)
        self.btn_display = self._make_menu_btn("DISPLAY", "DISPLAY", top_grid, 1, 2)
        self.btn_run_stop = QPushButton("RUN /\nSTOP", self)
        self.btn_run_stop.clicked.connect(self._on_run_stop)
        top_grid.addWidget(self.btn_run_stop, 1, 3)

        self.btn_help = QPushButton("HELP", self)
        self.btn_help.clicked.connect(self._on_help)
        top_grid.addWidget(self.btn_help, 2, 0)
        self.btn_default_setup = QPushButton("DEFAULT\nSETUP", self)
        self.btn_default_setup.clicked.connect(self._on_default_setup)
        top_grid.addWidget(self.btn_default_setup, 2, 1)
        self.btn_save_usb = QPushButton("SAVE TO\nUSB", self)
        self.btn_save_usb.clicked.connect(self._on_save_to_usb)
        top_grid.addWidget(self.btn_save_usb, 2, 2)
        self.btn_single_seq = QPushButton("SINGLE\nSEQ", self)
        self.btn_single_seq.clicked.connect(self._on_single_seq)
        top_grid.addWidget(self.btn_single_seq, 2, 3)
        right_panel.addLayout(top_grid)

        # VERTICAL chassis group
        vert_grp = QGroupBox("VERTICAL", self)
        v_grid = QGridLayout(vert_grp)
        v_grid.setSpacing(4)
        self.ch1_pos_dial = self._make_rotary(34, -40, 40, self._on_ch1_pos_dial)
        self.ch2_pos_dial = self._make_rotary(34, -40, 40, self._on_ch2_pos_dial)
        self.ch1_vdiv_dial = self._make_rotary(46, 0, len(DSO_VOLTS_PER_DIV) - 1, self._on_ch1_vdiv_dial)
        self.ch2_vdiv_dial = self._make_rotary(46, 0, len(DSO_VOLTS_PER_DIV) - 1, self._on_ch2_vdiv_dial)

        v_grid.addWidget(QLabel("CH1 Pos", self), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(QLabel("Math", self), 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(QLabel("CH2 Pos", self), 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(self.ch1_pos_dial, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_math_menu = QPushButton("MATH\nMENU", self)
        self.btn_math_menu.setStyleSheet("background-color: #F472B6; color: #0F172A; font-weight: bold;")
        self.btn_math_menu.clicked.connect(lambda: self.select_menu("MATH"))
        v_grid.addWidget(self.btn_math_menu, 1, 1)
        v_grid.addWidget(self.ch2_pos_dial, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn_ch1_menu = QPushButton("CH1\nMENU", self)
        self.btn_ch1_menu.setStyleSheet("background-color: #FACC15; color: #0F172A; font-weight: bold;")
        self.btn_ch1_menu.clicked.connect(lambda: self.select_menu("CH1"))
        v_grid.addWidget(self.btn_ch1_menu, 2, 0)
        self.btn_ch2_menu = QPushButton("CH2\nMENU", self)
        self.btn_ch2_menu.setStyleSheet("background-color: #22D3EE; color: #0F172A; font-weight: bold;")
        self.btn_ch2_menu.clicked.connect(lambda: self.select_menu("CH2"))
        v_grid.addWidget(self.btn_ch2_menu, 2, 2)

        v_grid.addWidget(self.ch1_vdiv_dial, 3, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(QLabel("VOLTS/DIV", self), 3, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(self.ch2_vdiv_dial, 3, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.ch1_vdiv_readout = QLabel("2 V/div", self)
        self.ch2_vdiv_readout = QLabel("2 V/div", self)
        v_grid.addWidget(self.ch1_vdiv_readout, 4, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        v_grid.addWidget(self.ch2_vdiv_readout, 4, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(vert_grp)

        # HORIZONTAL & TRIGGER side-by-side groups
        ht_row = QHBoxLayout()
        horiz_grp = QGroupBox("HORIZONTAL", self)
        h_grid = QVBoxLayout(horiz_grp)
        self.horiz_pos_dial = self._make_rotary(34, -50, 50, self._on_horiz_pos_dial)
        h_grid.addWidget(QLabel("Position", self), alignment=Qt.AlignmentFlag.AlignCenter)
        h_grid.addWidget(self.horiz_pos_dial, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_horiz_menu = QPushButton("HORIZ MENU", self)
        self.btn_horiz_menu.clicked.connect(lambda: self.select_menu("HORIZ"))
        h_grid.addWidget(self.btn_horiz_menu)
        self.sec_div_dial = self._make_rotary(46, 0, len(DSO_SEC_PER_DIV) - 1, self._on_sec_div_dial)
        h_grid.addWidget(self.sec_div_dial, alignment=Qt.AlignmentFlag.AlignCenter)
        self.sec_div_readout = QLabel("1 ms/div", self)
        h_grid.addWidget(self.sec_div_readout, alignment=Qt.AlignmentFlag.AlignCenter)
        ht_row.addWidget(horiz_grp)

        trig_grp = QGroupBox("TRIGGER", self)
        t_grid = QVBoxLayout(trig_grp)
        self.trig_level_dial = self._make_rotary(40, -100, 100, self._on_trig_level_dial)
        t_grid.addWidget(QLabel("LEVEL", self), alignment=Qt.AlignmentFlag.AlignCenter)
        t_grid.addWidget(self.trig_level_dial, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_trig_menu = QPushButton("TRIG MENU", self)
        self.btn_trig_menu.clicked.connect(lambda: self.select_menu("TRIG"))
        t_grid.addWidget(self.btn_trig_menu)
        self.btn_set_50 = QPushButton("SET TO 50%", self)
        self.btn_set_50.clicked.connect(self._on_set_to_50)
        t_grid.addWidget(self.btn_set_50)
        self.btn_force_trig = QPushButton("FORCE TRIG", self)
        self.btn_force_trig.clicked.connect(self._on_force_trig)
        t_grid.addWidget(self.btn_force_trig)
        self.trig_lvl_readout = QLabel("0.00 V", self)
        t_grid.addWidget(self.trig_lvl_readout, alignment=Qt.AlignmentFlag.AlignCenter)
        ht_row.addWidget(trig_grp)
        right_panel.addLayout(ht_row)

        self.knob_status_label = QLabel("Ready — DSO-1062D", self)
        self.knob_status_label.setStyleSheet("color: #334155; font-size: 10px;")
        right_panel.addWidget(self.knob_status_label)
        right_panel.addStretch(1)

        body.addLayout(right_panel, 2)
        root.addLayout(body, 1)

    def _make_menu_btn(self, text: str, menu_name: str, grid: QGridLayout, r: int, c: int) -> QPushButton:
        btn = QPushButton(text, self)
        btn.clicked.connect(lambda _=False, m=menu_name: self.select_menu(m))
        grid.addWidget(btn, r, c)
        return btn

    def _make_rotary(self, size: int, min_v: int, max_v: int, slot) -> QDial:
        d = QDial(self)
        d.setFixedSize(size, size)
        d.setRange(min_v, max_v)
        d.setNotchesVisible(True)
        d.valueChanged.connect(slot)
        return d

    def select_menu(self, menu_name: str) -> None:
        key = menu_name.strip().upper()
        if key not in DSO_MENU_NAMES:
            return
        self.active_menu = key
        self.menu_visible = True
        self._sync_ui_from_state()

    def _toggle_menu_visible(self) -> None:
        self.menu_visible = not self.menu_visible
        self._sync_ui_from_state()

    def _menu_options(self) -> list[tuple[str, str]]:
        m = self.active_menu
        s = self.state
        if m in ("CH1", "CH2"):
            ch = s.ch1 if m == "CH1" else s.ch2
            return [
                ("Channel", "ON" if ch.enabled else "OFF"),
                ("Coupling", ch.coupling),
                ("BW Limit", "20 MHz" if ch.bw_limit else "OFF"),
                ("Probe", f"{int(ch.probe_atten)}X"),
                ("Invert", "ON" if ch.invert else "OFF"),
                ("V/div +", format_engineering_value(_step_table(ch.volts_per_div, DSO_VOLTS_PER_DIV, 1), "V", 2)),
                ("V/div -", format_engineering_value(_step_table(ch.volts_per_div, DSO_VOLTS_PER_DIV, -1), "V", 2)),
            ]
        if m == "MATH":
            return [
                ("Math", "ON" if s.math.enabled else "OFF"),
                ("Operator", s.math.operation),
                ("FFT Src", s.math.fft_source),
                ("Window", s.math.fft_window),
                ("Scale +", format_engineering_value(s.math.scale, "V", 2)),
                ("Scale -", format_engineering_value(max(0.01, s.math.scale * 0.5), "V", 2)),
                ("Zero Pos", f"{s.math.position_div:+.1f} div"),
            ]
        if m == "ACQUIRE":
            return [
                ("Acq Type", s.acquire.acq_type),
                ("Mode", s.acquire.acq_mode),
                ("Averages", str(s.acquire.avg_count)),
                ("MemDepth", f"{s.acquire.mem_depth // 1000}k"),
                ("Restore", "Defaults"),
            ]
        if m == "MEASURE":
            m1 = self.latest_frame.ch1_meas if self.latest_frame else None
            return [
                ("Show All", "ON" if s.auto_measure_enabled else "OFF"),
                ("Source", s.trigger.source if s.trigger.source in ("CH1", "CH2") else "CH1"),
                ("Vpp", format_engineering_value(m1.v_pp, "V", 3) if m1 else "--"),
                ("Vrms", format_engineering_value(m1.v_rms, "V", 3) if m1 else "--"),
                ("Freq", format_engineering_value(m1.freq_hz, "Hz", 3) if (m1 and m1.freq_hz) else "--"),
            ]
        if m == "CURSOR":
            return [
                ("Mode", s.cursors.cursor_mode),
                ("Source", s.cursors.cursor_source),
                ("C1 +", f"{s.cursors.v1:.2f}V" if s.cursors.cursor_mode == "VOLTAGE" else format_engineering_value(s.cursors.t1, "s", 3)),
                ("C1 -", "Step Down"),
                ("C2 +", f"{s.cursors.v2:.2f}V" if s.cursors.cursor_mode == "VOLTAGE" else format_engineering_value(s.cursors.t2, "s", 3)),
                ("C2 -", "Step Down"),
                ("Center", "Reset"),
            ]
        if m == "DISPLAY":
            return [
                ("Type", s.display.render_style),
                ("Persist", "ON" if s.display.persistence else "OFF"),
                ("Grid", f"{int(round(s.display.grid_brightness * 100))}%"),
                ("Format", s.horizontal.format_mode),
            ]
        if m == "HORIZ":
            return [
                ("Format", s.horizontal.format_mode),
                ("Window", "ZOOM ON" if s.horizontal.zoom_enabled else "MAIN"),
                ("Sec/div -", "Faster"),
                ("Sec/div +", "Slower"),
                ("Delay 0", f"{s.horizontal.position_s:.3g}s"),
            ]
        if m == "TRIG":
            return [
                ("Source", s.trigger.source),
                ("Slope", s.trigger.slope),
                ("Mode", s.trigger.mode),
                ("Coupling", s.trigger.coupling),
                ("Set 50%", format_engineering_value(s.trigger.level_v, "V", 3)),
                ("Force", "Trigger Now"),
                ("Holdoff", format_engineering_value(s.trigger.holdoff_s, "s", 2)),
            ]
        if m == "SAVE/RECALL":
            return [
                ("Save PNG", "To File"),
                ("Save CSV", "Waveform"),
                ("Save Slot", "Slot 1"),
                ("Recall", "Slot 1" if self._saved_setup_slot else "Empty"),
                ("Factory", "Reset"),
            ]
        # UTILITY
        return [
            ("Net Sync", "Refresh"),
            ("Capture", "Burst Now"),
            ("SelfTest", "OK"),
            ("Defaults", "Reset"),
        ]

    def trigger_softkey(self, index: int) -> None:
        m = self.active_menu
        s = self.state
        if m in ("CH1", "CH2"):
            ch = s.ch1 if m == "CH1" else s.ch2
            if index == 0:
                ch.enabled = not ch.enabled
            elif index == 1:
                ch.coupling = str(_cycle_choice(ch.coupling, ("DC", "AC", "GND")))
            elif index == 2:
                ch.bw_limit = not ch.bw_limit
            elif index == 3:
                ch.probe_atten = float(_cycle_choice(ch.probe_atten, (1.0, 10.0)))
            elif index == 4:
                ch.invert = not ch.invert
            elif index == 5:
                ch.volts_per_div = _step_table(ch.volts_per_div, DSO_VOLTS_PER_DIV, 1)
            elif index == 6:
                ch.volts_per_div = _step_table(ch.volts_per_div, DSO_VOLTS_PER_DIV, -1)
        elif m == "MATH":
            if index == 0:
                s.math.enabled = not s.math.enabled
            elif index == 1:
                s.math.operation = str(_cycle_choice(s.math.operation, ("CH1+CH2", "CH1-CH2", "CH1*CH2", "CH1/CH2", "FFT")))
                s.math.enabled = True
            elif index == 2:
                s.math.fft_source = "CH2" if s.math.fft_source == "CH1" else "CH1"
            elif index == 3:
                s.math.fft_window = str(_cycle_choice(s.math.fft_window, ("HANNING", "HAMMING", "BLACKMAN", "RECTANGLE")))
            elif index == 4:
                s.math.scale = _step_table(s.math.scale, DSO_VOLTS_PER_DIV, 1)
            elif index == 5:
                s.math.scale = _step_table(s.math.scale, DSO_VOLTS_PER_DIV, -1)
            elif index == 6:
                s.math.position_div = 0.0
        elif m == "ACQUIRE":
            if index == 0:
                s.acquire.acq_type = "EQU_TIME" if s.acquire.acq_type == "REAL_TIME" else "REAL_TIME"
            elif index == 1:
                s.acquire.acq_mode = str(_cycle_choice(s.acquire.acq_mode, ("NORMAL", "PEAK", "AVERAGE")))
            elif index == 2:
                s.acquire.avg_count = int(_cycle_choice(s.acquire.avg_count, DSO_AVG_COUNTS))
            elif index == 3:
                s.acquire.mem_depth = int(_cycle_choice(s.acquire.mem_depth, DSO_MEM_DEPTHS))
            elif index == 4:
                s.acquire = reset_default_setup().acquire
        elif m == "MEASURE":
            if index == 0:
                s.auto_measure_enabled = not s.auto_measure_enabled
            elif index == 1:
                s.trigger.source = "CH2" if s.trigger.source == "CH1" else "CH1"
        elif m == "CURSOR":
            if index == 0:
                s.cursors.cursor_mode = str(_cycle_choice(s.cursors.cursor_mode, ("OFF", "VOLTAGE", "TIME")))
            elif index == 1:
                s.cursors.cursor_source = "CH2" if s.cursors.cursor_source == "CH1" else "CH1"
            elif index in (2, 3):
                sign = 1.0 if index == 2 else -1.0
                if s.cursors.cursor_mode == "TIME":
                    s.cursors.t1 += sign * s.horizontal.sec_per_div * 0.5
                else:
                    s.cursors.v1 += sign * s.ch1.volts_per_div * 0.25
            elif index in (4, 5):
                sign = 1.0 if index == 4 else -1.0
                if s.cursors.cursor_mode == "TIME":
                    s.cursors.t2 += sign * s.horizontal.sec_per_div * 0.5
                else:
                    s.cursors.v2 += sign * s.ch1.volts_per_div * 0.25
            elif index == 6:
                s.cursors.v1, s.cursors.v2 = -1.0, 1.0
                s.cursors.t1, s.cursors.t2 = -s.horizontal.sec_per_div, s.horizontal.sec_per_div
        elif m == "DISPLAY":
            if index == 0:
                s.display.render_style = "DOTS" if s.display.render_style == "VECTORS" else "VECTORS"
            elif index == 1:
                s.display.persistence = not s.display.persistence
            elif index == 2:
                s.display.grid_brightness = float(_cycle_choice(round(s.display.grid_brightness, 1), (0.8, 1.0, 0.4)))
            elif index == 3:
                s.horizontal.format_mode = "XY" if s.horizontal.format_mode == "YT" else "YT"
        elif m == "HORIZ":
            if index == 0:
                s.horizontal.format_mode = "XY" if s.horizontal.format_mode == "YT" else "YT"
            elif index == 1:
                s.horizontal.zoom_enabled = not s.horizontal.zoom_enabled
            elif index == 2:
                s.horizontal.sec_per_div = _step_table(s.horizontal.sec_per_div, DSO_SEC_PER_DIV, -1)
                self._notify_timebase_changed()
            elif index == 3:
                s.horizontal.sec_per_div = _step_table(s.horizontal.sec_per_div, DSO_SEC_PER_DIV, 1)
                self._notify_timebase_changed()
            elif index == 4:
                s.horizontal.position_s = 0.0
        elif m == "TRIG":
            if index == 0:
                s.trigger.source = str(_cycle_choice(s.trigger.source, ("CH1", "CH2", "EXT")))
            elif index == 1:
                s.trigger.slope = str(_cycle_choice(s.trigger.slope, ("RISING", "FALLING", "EITHER")))
            elif index == 2:
                s.trigger.mode = str(_cycle_choice(s.trigger.mode, ("AUTO", "NORMAL", "SINGLE")))
                s.trigger.single_armed = s.trigger.mode == "SINGLE"
            elif index == 3:
                s.trigger.coupling = "AC" if s.trigger.coupling == "DC" else "DC"
            elif index == 4:
                self._on_set_to_50()
                return
            elif index == 5:
                self._on_force_trig()
                return
            elif index == 6:
                s.trigger.holdoff_s = float(_cycle_choice(s.trigger.holdoff_s, (0.0, 1e-4, 1e-3, 1e-2)))
        elif m == "SAVE/RECALL":
            if index == 0:
                self._on_save_to_usb()
                return
            if index == 1:
                path, _ = QFileDialog.getSaveFileName(self, "Save Waveform CSV", "dso_waveform.csv", "CSV (*.csv)")
                if path:
                    self.export_waveform_csv(path)
                return
            if index == 2:
                self._saved_setup_slot = save_setup_json(s)
                self.knob_status_label.setText("Setup saved to Slot 1")
            elif index == 3 and self._saved_setup_slot:
                self.state = recall_setup_json(self._saved_setup_slot)
                self.knob_status_label.setText("Recalled Slot 1 setup")
                self._notify_timebase_changed()
            elif index == 4:
                self._on_default_setup()
                return
        else:  # UTILITY
            if index in (0, 1):
                self.refresh_net_labels()
                self.capture_from_scene()
            elif index == 2:
                self.knob_status_label.setText("DSO-1062D Self-Test: PASS")
            elif index == 3:
                self._on_default_setup()
                return

        self._persist_state_to_component()
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _on_multipurpose_dial(self, val: int) -> None:
        if self._syncing_dials:
            return
        if self.active_menu == "CURSOR" and self.state.cursors.cursor_mode == "VOLTAGE":
            self.state.cursors.v1 = (val - 12) * self.state.ch1.volts_per_div * 0.25
        elif self.active_menu == "ACQUIRE":
            self.state.acquire.avg_count = DSO_AVG_COUNTS[val % len(DSO_AVG_COUNTS)]
        else:
            self.state.trigger.holdoff_s = (val % 10) * 1e-4
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _on_ch1_pos_dial(self, val: int) -> None:
        if self._syncing_dials:
            return
        self.state.ch1.position_div = val / 10.0
        self.knob_status_label.setText(f"CH1 Position: {self.state.ch1.position_div:+.1f} div")
        self._persist_state_to_component()
        self._sync_ui_from_state()

    def _on_ch2_pos_dial(self, val: int) -> None:
        if self._syncing_dials:
            return
        self.state.ch2.position_div = val / 10.0
        self.knob_status_label.setText(f"CH2 Position: {self.state.ch2.position_div:+.1f} div")
        self._persist_state_to_component()
        self._sync_ui_from_state()

    def _on_ch1_vdiv_dial(self, idx: int) -> None:
        if self._syncing_dials:
            return
        self.state.ch1.volts_per_div = float(DSO_VOLTS_PER_DIV[max(0, min(len(DSO_VOLTS_PER_DIV) - 1, idx))])
        self.knob_status_label.setText(f"CH1 Scale: {format_engineering_value(self.state.ch1.volts_per_div, 'V/div')}")
        self._persist_state_to_component()
        self._sync_ui_from_state()

    def _on_ch2_vdiv_dial(self, idx: int) -> None:
        if self._syncing_dials:
            return
        self.state.ch2.volts_per_div = float(DSO_VOLTS_PER_DIV[max(0, min(len(DSO_VOLTS_PER_DIV) - 1, idx))])
        self.knob_status_label.setText(f"CH2 Scale: {format_engineering_value(self.state.ch2.volts_per_div, 'V/div')}")
        self._persist_state_to_component()
        self._sync_ui_from_state()

    def _on_horiz_pos_dial(self, val: int) -> None:
        if self._syncing_dials:
            return
        self.state.horizontal.position_s = (val / 10.0) * self.state.horizontal.sec_per_div
        self.knob_status_label.setText(f"Horiz Pos: {format_engineering_value(self.state.horizontal.position_s, 's')}")
        self._persist_state_to_component()
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _notify_timebase_changed(self) -> None:
        if self._timebase_changed_callback is not None:
            try:
                self._timebase_changed_callback()
            except Exception:
                pass

    def _on_sec_div_dial(self, idx: int) -> None:
        if self._syncing_dials:
            return
        self.state.horizontal.sec_per_div = float(DSO_SEC_PER_DIV[max(0, min(len(DSO_SEC_PER_DIV) - 1, idx))])
        self.knob_status_label.setText(f"Timebase: {format_engineering_value(self.state.horizontal.sec_per_div, 's/div')}")
        self._persist_state_to_component()
        self._notify_timebase_changed()
        self.capture_from_scene()
        self._sync_ui_from_state()

    def _on_trig_level_dial(self, val: int) -> None:
        if self._syncing_dials:
            return
        ref_ch = self.state.ch2 if self.state.trigger.source == "CH2" else self.state.ch1
        self.state.trigger.level_v = (val / 20.0) * ref_ch.volts_per_div
        self.knob_status_label.setText(f"Trigger Level: {format_engineering_value(self.state.trigger.level_v, 'V')}")
        self._persist_state_to_component()
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _on_auto_set(self) -> None:
        t_arr, c1, c2, _ = self._raw_arrays_for_processing()
        if len(t_arr) >= 4:
            self.state = compute_auto_set(t_arr, c1, c2, self.state)
        else:
            self.capture_from_scene()
            t_arr, c1, c2, _ = self._raw_arrays_for_processing()
            if len(t_arr) >= 4:
                self.state = compute_auto_set(t_arr, c1, c2, self.state)
        self.knob_status_label.setText("AUTO SET complete")
        self._persist_state_to_component()
        self._notify_timebase_changed()
        if len(self._live_times) >= 16:
            self._reprocess_or_capture()
        else:
            self.capture_from_scene()
        self._sync_ui_from_state()

    def _on_run_stop(self) -> None:
        self.state.running = not self.state.running
        if self.state.running and self.state.trigger.mode == "SINGLE":
            self.state.trigger.single_armed = True
        self.knob_status_label.setText("Running" if self.state.running else "Stopped (Hold)")
        self._persist_state_to_component()
        if not self.state.running:
            if self.latest_frame is not None or len(self._live_times) >= 16:
                self._reprocess_or_capture()
        else:
            if len(self._live_times) >= 16:
                self._reprocess_or_capture()
            elif self._circuit_provider is not None:
                self.capture_from_scene()
        self._sync_ui_from_state()

    def _on_single_seq(self) -> None:
        self.state.trigger.mode = "SINGLE"
        self.state.trigger.single_armed = True
        self.state.running = True
        self.knob_status_label.setText("SINGLE SEQ armed")
        self._persist_state_to_component()
        self.capture_from_scene()
        self._sync_ui_from_state()

    def _on_default_setup(self) -> None:
        self.state = reset_default_setup()
        self.knob_status_label.setText("Factory default setup restored")
        self._persist_state_to_component()
        self._notify_timebase_changed()
        self.capture_from_scene()
        self._sync_ui_from_state()

    def _on_set_to_50(self) -> None:
        _, c1, c2, ext = self._raw_arrays_for_processing()
        src = c2 if self.state.trigger.source == "CH2" else (ext if self.state.trigger.source == "EXT" else c1)
        if len(src) > 0:
            self.state.trigger.level_v = compute_set_to_50_percent(src)
        elif self.latest_frame is not None:
            sig = self.latest_frame.ch2_v if self.state.trigger.source == "CH2" else self.latest_frame.ch1_v
            self.state.trigger.level_v = compute_set_to_50_percent(sig)
        self.knob_status_label.setText(f"Trigger set to 50%: {format_engineering_value(self.state.trigger.level_v, 'V')}")
        self._persist_state_to_component()
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _on_force_trig(self) -> None:
        self.state.trigger.forced = True
        self.knob_status_label.setText("Trigger forced")
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _on_help(self) -> None:
        QMessageBox.information(
            self,
            "DSO-1062D Quick Guide",
            "• Connect CH1 (Yellow), CH2 (Cyan), EXT TRIG, and GND (optional) on the schematic.\n"
            "• Click AUTO SET to auto-scale V/div, timebase (~2–4 cycles), and 50% trigger.\n"
            "• Use CH1/CH2/MATH/CURSOR/MEASURE/DISPLAY/TRIG buttons + F0–F6 softkeys.\n"
            "• In CURSOR mode, drag the dashed cursor lines directly on the LCD.\n"
            "• Click SAVE TO USB to export a PNG screenshot and CSV waveform.",
        )

    def _on_save_to_usb(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save DSO Screenshot (.png)", f"{self.osc_component.name}_screen.png", "PNG Image (*.png)"
        )
        if not path:
            return
        self.export_screen_png(path)
        csv_path = str(Path(path).with_suffix(".csv"))
        self.export_waveform_csv(csv_path)
        self.knob_status_label.setText(f"Saved PNG + CSV: {Path(path).name}")

    def _on_screen_cursors_changed(self) -> None:
        self._persist_state_to_component()
        self._reprocess_or_capture()
        self._sync_ui_from_state()

    def _persist_state_to_component(self) -> None:
        self.osc_component.dso_state_json = save_setup_json(self.state)

    def _raw_arrays_for_processing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self._live_times) >= 4:
            return (
                np.array(self._live_times, dtype=np.float64),
                np.array(self._live_ch1, dtype=np.float64),
                np.array(self._live_ch2, dtype=np.float64),
                np.array(self._live_ext, dtype=np.float64),
            )
        if self.latest_frame is not None and len(self.latest_frame.times_rel_s) >= 2:
            return (
                self.latest_frame.times_rel_s,
                self.latest_frame.ch1_v,
                self.latest_frame.ch2_v,
                np.zeros_like(self.latest_frame.ch1_v),
            )
        return tuple(np.empty(0, dtype=np.float64) for _ in range(4))  # type: ignore[return-value]

    def _reprocess_or_capture(self) -> None:
        t_arr, c1, c2, ext = self._raw_arrays_for_processing()
        if len(t_arr) >= 4:
            self.latest_frame = process_dso_capture(t_arr, c1, c2, ext, self.state, prev_frame=self.latest_frame)
            self._push_mini_preview()
        elif self._circuit_provider is not None:
            self.capture_from_scene()

    def _push_mini_preview(self) -> None:
        if self.latest_frame is None:
            return
        for attr, arr, vdiv in (
            ("mini_preview_ch1", self.latest_frame.ch1_v, self.state.ch1.volts_per_div),
            ("mini_preview_ch2", self.latest_frame.ch2_v, self.state.ch2.volts_per_div),
        ):
            if len(arr) >= 2:
                idxs = np.linspace(0, len(arr) - 1, min(32, len(arr)), dtype=int)
                sub = arr[idxs]
                scale = max(float(np.max(np.abs(sub))), vdiv, 1e-6)
                setattr(self.osc_component, attr, tuple(float(v / scale) for v in sub))
        if self.osc_item is not None:
            self.osc_item.update()

    def _resolve_active_nets(self) -> dict[str, str]:
        nets: dict[str, str] = {}
        if self._nets_provider is not None:
            try:
                provided = self._nets_provider()
                if provided:
                    nets = dict(provided)
                    self._cached_nets = dict(nets)
            except Exception:
                nets = {}
        if not nets and self._cached_nets:
            nets = dict(self._cached_nets)
        if not nets and len(self.osc_component.nodes) >= 4:
            nets = dict(zip(Oscilloscope.PIN_NAMES, self.osc_component.nodes))
        return nets

    def refresh_net_labels(self) -> None:
        """Update bottom BNC jack badges with active schematic node/net names."""
        nets = self._resolve_active_nets()

        for pin_name, ring_col, cap in (
            ("CH1", "#EAB308", "CH1"),
            ("CH2", "#06B6D4", "CH2"),
            ("GND", "#334155", "GND 300V"),
            ("EXT_TRIG", "#0F172A", "EXT"),
        ):
            lbl = self.bnc_labels.get(pin_name)
            if lbl is not None:
                net_txt = nets.get(pin_name, "--")
                lbl.setText(f"◉ {cap}: {net_txt}")
                lbl.setStyleSheet(
                    f"color: #0F172A; background: #F8FAFC; border: 2px solid {ring_col}; border-radius: 10px; padding: 2px 8px; font-size: 10px; font-weight: 600;"
                )

    def capture_from_scene(self) -> DSOFrame | None:
        """Run a transient capture on the active schematic over the visible time window."""
        if self._circuit_provider is None:
            return None
        try:
            circuit = self._circuit_provider()
            circuit.validate_topology()
        except Exception as exc:
            self.knob_status_label.setText(f"Capture skipped: {exc}")
            return None

        osc_in_circuit: Oscilloscope | None = None
        ac_freqs: list[float] = []
        for comp in circuit.flat_components():
            if isinstance(comp, Oscilloscope) and comp.name == self.osc_component.name:
                osc_in_circuit = comp
            freq_val = getattr(comp, "frequency", None)
            if isinstance(freq_val, (int, float)) and float(freq_val) > 0.0:
                ac_freqs.append(float(freq_val))
        if osc_in_circuit is None:
            self.knob_status_label.setText(f"{self.osc_component.name} not wired to active circuit")
            return None

        sec_div = max(float(self.state.horizontal.sec_per_div), 1e-9)
        span = max(sec_div * DSO_HORIZONTAL_DIVS, 1e-7)
        if ac_freqs:
            min_f = min(ac_freqs)
            max_f = max(ac_freqs)
            duration = min(max(2.0 * span, 6.0 / max(min_f, 1.0)), 1.5)
            dt_target = min(sec_div / 50.0, 1.0 / (64.0 * max(max_f, 50.0)))
        else:
            duration = min(max(span, 1e-6), 1.0)
            dt_target = sec_div / 50.0
        dt = max(dt_target, duration / 12000.0, 1e-8)

        tr = MNASolver(circuit).solve_transient(duration=duration, dt=dt)
        gnd_wired = circuit.is_oscilloscope_gnd_wired(osc_in_circuit)
        vg = tr.node_voltages[osc_in_circuit.node_gnd] if (gnd_wired and osc_in_circuit.node_gnd in tr.node_voltages) else np.zeros_like(tr.times)
        ch1 = tr.node_voltages.get(osc_in_circuit.node_ch1, np.zeros_like(tr.times)) - vg
        ch2 = tr.node_voltages.get(osc_in_circuit.node_ch2, np.zeros_like(tr.times)) - vg
        ext = tr.node_voltages.get(osc_in_circuit.node_ext_trig, np.zeros_like(tr.times)) - vg

        self._live_times.clear()
        self._live_ch1.clear()
        self._live_ch2.clear()
        self._live_ext.clear()
        self._live_times.extend(float(v) for v in tr.times)
        self._live_ch1.extend(float(v) for v in ch1)
        self._live_ch2.extend(float(v) for v in ch2)
        self._live_ext.extend(float(v) for v in ext)
        self._last_live_t = float(tr.times[-1]) if len(tr.times) > 0 else None

        self.latest_frame = process_dso_capture(tr.times, ch1, ch2, ext, self.state, prev_frame=self.latest_frame)
        self._push_mini_preview()
        self._sync_ui_from_state()
        return self.latest_frame

    def __call__(self, block: object) -> None:
        """Allow direct registration as a `SimulationSession.attach_scope_sink` callable."""
        self.on_sample_block(block)

    def on_sample_block(self, block: object) -> None:
        """Ingest high-rate `SampleBlock` chunks without dropping solver substeps."""
        if not self.state.running and not self.state.trigger.single_armed:
            return
        n_steps = int(getattr(block, "n_steps", 0))
        if n_steps <= 0:
            return
        t_arr = block.timestamps()  # type: ignore[attr-defined]
        if len(t_arr) == 0:
            return

        t_first = float(t_arr[0])
        t_last = float(t_arr[-1])
        if self._last_live_t is not None:
            if t_first < self._last_live_t - 1e-12 and t_last < self._last_live_t - 1e-12:
                # Simulation restarted from an earlier t -> reset live rolling buffer cleanly
                self._live_times.clear()
                self._live_ch1.clear()
                self._live_ch2.clear()
                self._live_ext.clear()
                self._last_live_t = None
            elif t_last <= self._last_live_t + 1e-15:
                return

        slice_mask: np.ndarray | None = None
        if self._last_live_t is not None and t_first <= self._last_live_t + 1e-15:
            slice_mask = t_arr > (self._last_live_t + 1e-15)
            if not np.any(slice_mask):
                return
            t_arr = t_arr[slice_mask]

        nets = self._resolve_active_nets()
        ch1_n = nets.get("CH1", self.osc_component.node_ch1)
        ch2_n = nets.get("CH2", self.osc_component.node_ch2)
        ext_n = nets.get("EXT_TRIG", self.osc_component.node_ext_trig)
        gnd_n = nets.get("GND", self.osc_component.node_gnd)
        node_names = tuple(getattr(block, "node_names", ()))

        def _col_or_zero(name: str) -> np.ndarray:
            if name == "0" or name in node_names:
                arr = np.asarray(block.column(name), dtype=np.float64)  # type: ignore[attr-defined]
            else:
                arr = np.zeros(n_steps, dtype=np.float64)
            return arr[slice_mask] if slice_mask is not None else arr

        vg = _col_or_zero(gnd_n)
        v1 = _col_or_zero(ch1_n) - vg
        v2 = _col_or_zero(ch2_n) - vg
        ve = _col_or_zero(ext_n) - vg

        self._live_times.extend(t_arr.tolist())
        self._live_ch1.extend(v1.tolist())
        self._live_ch2.extend(v2.tolist())
        self._live_ext.extend(ve.tolist())
        self._last_live_t = float(t_arr[-1])
        self._received_sample_blocks = True
        self._live_step_counter += len(t_arr)

        if len(v1) > 0:
            self.osc_component.v_gnd = float(vg[-1])
            self.osc_component.v_ch1 = float(v1[-1])
            self.osc_component.v_ch2 = float(v2[-1])
            self.osc_component.v_ext = float(ve[-1])

        if len(self._live_times) >= 8 and (n_steps >= 2 or self._live_step_counter % 8 == 0):
            t_buf, c1_buf, c2_buf, ext_buf = self._raw_arrays_for_processing()
            self.latest_frame = process_dso_capture(
                t_buf, c1_buf, c2_buf, ext_buf, self.state, prev_frame=self.latest_frame
            )
            self._push_mini_preview()
            self._sync_ui_from_state()

    def on_simulation_frame(self, frame: SimulationFrame) -> None:
        """Ingest live `SimulationFrame` telemetry without blocking the UI thread."""
        if frame.state == SimulationState.STOPPED:
            self._last_live_t = None
            self._received_sample_blocks = False
            return
        if not self.state.running and not self.state.trigger.single_armed:
            return
        if not frame.node_voltages or frame.dt <= 0.0:
            return
        if (
            self._received_sample_blocks
            and self._last_live_t is not None
            and frame.sim_time <= self._last_live_t + 1e-12
        ):
            return
        if self._last_live_t is not None and frame.sim_time <= self._last_live_t:
            return
        self._last_live_t = frame.sim_time

        nets = self._resolve_active_nets()
        ch1_n = nets.get("CH1", self.osc_component.node_ch1)
        ch2_n = nets.get("CH2", self.osc_component.node_ch2)
        ext_n = nets.get("EXT_TRIG", self.osc_component.node_ext_trig)
        gnd_n = nets.get("GND", self.osc_component.node_gnd)
        vg = float(frame.node_voltages.get(gnd_n, 0.0)) if gnd_n in frame.node_voltages else 0.0

        self._live_times.append(float(frame.sim_time))
        self._live_ch1.append(float(frame.node_voltages.get(ch1_n, 0.0)) - vg)
        self._live_ch2.append(float(frame.node_voltages.get(ch2_n, 0.0)) - vg)
        self._live_ext.append(float(frame.node_voltages.get(ext_n, 0.0)) - vg)
        self._live_step_counter += 1
        if len(self._live_times) >= 8 and self._live_step_counter % 4 == 0:
            t_arr, c1, c2, ext = self._raw_arrays_for_processing()
            self.latest_frame = process_dso_capture(t_arr, c1, c2, ext, self.state, prev_frame=self.latest_frame)
            self._push_mini_preview()
            self._sync_ui_from_state()

    def _sync_ui_from_state(self) -> None:
        self._syncing_dials = True
        try:
            s = self.state
            self.ch1_pos_dial.setValue(max(-40, min(40, int(round(s.ch1.position_div * 10.0)))))
            self.ch2_pos_dial.setValue(max(-40, min(40, int(round(s.ch2.position_div * 10.0)))))
            c1_idx = DSO_VOLTS_PER_DIV.index(snap_dso_step(s.ch1.volts_per_div, DSO_VOLTS_PER_DIV))
            c2_idx = DSO_VOLTS_PER_DIV.index(snap_dso_step(s.ch2.volts_per_div, DSO_VOLTS_PER_DIV))
            self.ch1_vdiv_dial.setValue(c1_idx)
            self.ch2_vdiv_dial.setValue(c2_idx)
            self.ch1_vdiv_readout.setText(format_engineering_value(s.ch1.volts_per_div, "V/div", 3))
            self.ch2_vdiv_readout.setText(format_engineering_value(s.ch2.volts_per_div, "V/div", 3))

            sd_idx = DSO_SEC_PER_DIV.index(snap_dso_step(s.horizontal.sec_per_div, DSO_SEC_PER_DIV))
            self.sec_div_dial.setValue(sd_idx)
            self.sec_div_readout.setText(format_engineering_value(s.horizontal.sec_per_div, "s/div", 3))
            self.horiz_pos_dial.setValue(
                max(-50, min(50, int(round((s.horizontal.position_s / max(s.horizontal.sec_per_div, 1e-12)) * 10.0))))
            )

            ref_ch = s.ch2 if s.trigger.source == "CH2" else s.ch1
            self.trig_level_dial.setValue(
                max(-100, min(100, int(round((s.trigger.level_v / max(ref_ch.volts_per_div, 1e-9)) * 20.0))))
            )
            self.trig_lvl_readout.setText(format_engineering_value(s.trigger.level_v, "V", 3))

            if s.running:
                self.btn_run_stop.setStyleSheet("background-color: #22C55E; color: #0F172A; font-weight: bold;")
            else:
                self.btn_run_stop.setStyleSheet("background-color: #EF4444; color: #FFFFFF; font-weight: bold;")
        finally:
            self._syncing_dials = False

        opts = self._menu_options()
        self.screen_widget.set_view_data(
            state=self.state,
            frame=self.latest_frame,
            menu_title=self.active_menu,
            menu_rows=opts,
            menu_visible=self.menu_visible,
        )

    def export_screen_png(self, path: str | Path) -> Path:
        """Render the DSO-1062D LCD widget into a high-contrast PNG screenshot."""
        out_path = Path(path)
        if self.latest_frame is None:
            self.capture_from_scene()
        target_size = self.screen_widget.size()
        if target_size.width() < 580 or target_size.height() < 380:
            self.screen_widget.resize(640, 400)
            target_size = self.screen_widget.size()
        img = QImage(target_size, QImage.Format.Format_ARGB32)
        img.fill(QColor("#0F172A"))
        self.screen_widget.render(img)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path), "PNG")
        return out_path

    def export_waveform_csv(self, path: str | Path) -> Path:
        """Export the active DSO frame (`time_s, ch1_v, ch2_v, math_v`) to CSV."""
        out_path = Path(path)
        if self.latest_frame is None:
            self.capture_from_scene()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "ch1_v", "ch2_v", "math_v"])
            if self.latest_frame is not None:
                f = self.latest_frame
                n = min(len(f.times_rel_s), len(f.ch1_v), len(f.ch2_v), len(f.math_v))
                for idx in range(n):
                    writer.writerow([
                        f"{float(f.times_rel_s[idx]):.9g}",
                        f"{float(f.ch1_v[idx]):.9g}",
                        f"{float(f.ch2_v[idx]):.9g}",
                        f"{float(f.math_v[idx]):.9g}",
                    ])
        return out_path
