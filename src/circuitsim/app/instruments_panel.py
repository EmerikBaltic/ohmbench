"""Benchtop Virtual Instruments dialog: Multi-Channel Waveform Analyzer & AC Bode/Z_in panel."""

from __future__ import annotations

import math
from typing import Callable
import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from circuitsim.engine.ac_analysis import ACAnalysisResult, run_ac_sweep
from circuitsim.engine.dmm import DMMReading, measure_dmm
from circuitsim.engine.fft_analysis import FFTSpectrumResult, compute_fft_spectrum
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.simulation_session import SimulationFrame, SimulationState
from circuitsim.engine.solver import MNASolver, TransientResult
from circuitsim.scope import (
    CHANNEL_COLOR_HEX,
    CHANNEL_NAMES,
    SampleBlock,
    ScopeCore,
    ScopeFrame,
    TriggerEdge,
    TriggerMode,
    measure_channel,
)
from circuitsim.scope.math_fft import CursorReadout, compute_cursors


class _WaveformCanvas(QWidget):
    """Dark-theme graticule plotter for up to 4 scope channels + dual time cursors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 260)
        self._frame: ScopeFrame | None = None
        self._raw_times: np.ndarray | None = None
        self._raw_traces: dict[str, tuple[str, np.ndarray]] = {}
        self._cursor_t1: float = 0.0
        self._cursor_t2: float = 0.0
        self._cursors_enabled: bool = True
        self._xy_mode: bool = False

    def set_xy_mode(self, enabled: bool) -> None:
        self._xy_mode = bool(enabled)
        self.update()

    def get_xy_points(self) -> list[tuple[float, float]]:
        """Return paired ``(CH1_V, CH2_V)`` sample points for X-Y Lissajous mode."""
        t1 = self._raw_traces.get("CH1")
        t2 = self._raw_traces.get("CH2")
        if t1 is None or t2 is None:
            return []
        a1 = np.asarray(t1[1], dtype=np.float64)
        a2 = np.asarray(t2[1], dtype=np.float64)
        n = min(len(a1), len(a2))
        return [(float(a1[i]), float(a2[i])) for i in range(n)]

    def set_capture(
        self,
        frame: ScopeFrame,
        times: np.ndarray | None = None,
        traces: dict[str, tuple[str, np.ndarray]] | None = None,
        cursor_t1: float = 0.0,
        cursor_t2: float = 0.0,
    ) -> None:
        self._frame = frame
        self._raw_times = times
        self._raw_traces = dict(traces) if traces else {}
        self._cursor_t1 = float(cursor_t1)
        self._cursor_t2 = float(cursor_t2)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#11161C"))

        pad_l, pad_r, pad_t, pad_b = 52.0, 16.0, 14.0, 26.0
        plot = QRectF(
            pad_l,
            pad_t,
            max(40.0, rect.width() - pad_l - pad_r),
            max(40.0, rect.height() - pad_t - pad_b),
        )

        # Graticule
        grid_pen = QPen(QColor("#24303D"), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(11):
            x = plot.left() + plot.width() * (i / 10.0)
            painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))
        for j in range(9):
            y = plot.top() + plot.height() * (j / 8.0)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))

        painter.setPen(QPen(QColor("#3E5063"), 1.2))
        painter.drawRect(plot)

        times = self._raw_times
        traces = self._raw_traces
        if self._xy_mode:
            pts = self.get_xy_points()
            if len(pts) < 2:
                painter.setPen(QColor("#8CA0B3"))
                painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Assign CH1 (X) and CH2 (Y) and run capture for X-Y plot.")
                return
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if abs(x_max - x_min) < 1e-9:
                x_min -= 1.0
                x_max += 1.0
            if abs(y_max - y_min) < 1e-9:
                y_min -= 1.0
                y_max += 1.0
            pad_x = 0.08 * (x_max - x_min)
            pad_y = 0.08 * (y_max - y_min)
            x_min -= pad_x
            x_max += pad_x
            y_min -= pad_y
            y_max += pad_y
            x_span = max(x_max - x_min, 1e-12)
            y_span = max(y_max - y_min, 1e-12)
            path = QPainterPath()
            stride = max(1, len(pts) // 1200)
            for idx in range(0, len(pts), stride):
                xv, yv = pts[idx]
                px = plot.left() + ((xv - x_min) / x_span) * plot.width()
                py = plot.bottom() - ((yv - y_min) / y_span) * plot.height()
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.setPen(QPen(QColor("#38BDF8"), 1.8))
            painter.drawPath(path)
            painter.setPen(QColor("#A9BACB"))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(4, int(plot.top() + 10), f"Y:{y_max:.3g}V")
            painter.drawText(4, int(plot.bottom()), f"Y:{y_min:.3g}V")
            painter.drawText(int(plot.left()), int(rect.bottom() - 4), f"X:{x_min:.3g}V")
            painter.drawText(int(plot.right() - 62), int(rect.bottom() - 4), f"X:{x_max:.3g}V")
            return

        if times is None or len(times) < 2 or not traces:
            painter.setPen(QColor("#8CA0B3"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "No transient waveform captured yet.")
            return

        t_min = float(times[0])
        t_max = float(times[-1])
        dt_span = max(t_max - t_min, 1e-12)

        y_vals = [float(np.min(arr)) for _, arr in traces.values() if len(arr) > 0]
        y_vals.extend(float(np.max(arr)) for _, arr in traces.values() if len(arr) > 0)
        v_min = min(y_vals) if y_vals else -1.0
        v_max = max(y_vals) if y_vals else 1.0
        if abs(v_max - v_min) < 1e-9:
            v_min -= 1.0
            v_max += 1.0
        pad_y = 0.08 * (v_max - v_min)
        v_min -= pad_y
        v_max += pad_y
        dv_span = max(v_max - v_min, 1e-12)

        def to_px(t_val: float, v_val: float) -> tuple[float, float]:
            px = plot.left() + ((t_val - t_min) / dt_span) * plot.width()
            py = plot.bottom() - ((v_val - v_min) / dv_span) * plot.height()
            return px, py

        # Cursors
        if self._cursors_enabled:
            for c_val, col in ((self._cursor_t1, "#FFD166"), (self._cursor_t2, "#06D6A0")):
                cx, _ = to_px(c_val, v_min)
                if plot.left() <= cx <= plot.right():
                    painter.setPen(QPen(QColor(col), 1.2, Qt.PenStyle.DashLine))
                    painter.drawLine(int(cx), int(plot.top()), int(cx), int(plot.bottom()))

        # Traces
        for ch_name, (_, arr) in traces.items():
            if len(arr) != len(times):
                continue
            color_hex = CHANNEL_COLOR_HEX.get(ch_name, "#58A6FF")
            path = QPainterPath()
            stride = max(1, len(times) // 800)
            for idx in range(0, len(times), stride):
                px, py = to_px(float(times[idx]), float(arr[idx]))
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.setPen(QPen(QColor(color_hex), 1.8))
            painter.drawPath(path)

        # Axis labels
        painter.setPen(QColor("#A9BACB"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(4, int(plot.top() + 10), f"{v_max:.3g}V")
        painter.drawText(4, int(plot.bottom()), f"{v_min:.3g}V")
        painter.drawText(int(plot.left()), int(rect.bottom() - 4), f"{t_min * 1e3:.3g} ms")
        painter.drawText(int(plot.right() - 56), int(rect.bottom() - 4), f"{t_max * 1e3:.3g} ms")


class _BodeCanvas(QWidget):
    """Dark-theme Bode magnitude (dB) + phase (deg) plotter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 260)
        self._freqs: np.ndarray | None = None
        self._db: np.ndarray | None = None
        self._deg: np.ndarray | None = None
        self._cursor_f: float | None = None

    def set_bode(
        self,
        freqs: np.ndarray | None,
        db: np.ndarray | None,
        deg: np.ndarray | None,
        cursor_f: float | None = None,
    ) -> None:
        self._freqs = freqs
        self._db = db
        self._deg = deg
        self._cursor_f = cursor_f
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#11161C"))

        pad_l, pad_r, pad_t, pad_b = 54.0, 54.0, 14.0, 26.0
        plot = QRectF(
            pad_l,
            pad_t,
            max(40.0, rect.width() - pad_l - pad_r),
            max(40.0, rect.height() - pad_t - pad_b),
        )
        painter.setPen(QPen(QColor("#24303D"), 1.0, Qt.PenStyle.DotLine))
        for i in range(9):
            y = plot.top() + plot.height() * (i / 8.0)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
        painter.setPen(QPen(QColor("#3E5063"), 1.2))
        painter.drawRect(plot)

        if self._freqs is None or self._db is None or len(self._freqs) < 2:
            painter.setPen(QColor("#8CA0B3"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Run AC Sweep to plot Bode response.")
            return

        logf = np.log10(np.maximum(self._freqs, 1e-12))
        lf_min, lf_max = float(logf[0]), float(logf[-1])
        lf_span = max(lf_max - lf_min, 1e-9)

        db_min = min(float(np.min(self._db)), -6.0)
        db_max = max(float(np.max(self._db)), 3.0)
        db_span = max(db_max - db_min, 1e-6)

        def x_of(f_val: float) -> float:
            return plot.left() + ((math.log10(max(f_val, 1e-12)) - lf_min) / lf_span) * plot.width()

        # Magnitude trace (cyan)
        mag_path = QPainterPath()
        for idx, (f_val, m_val) in enumerate(zip(self._freqs, self._db)):
            px = x_of(float(f_val))
            py = plot.bottom() - ((float(m_val) - db_min) / db_span) * plot.height()
            if idx == 0:
                mag_path.moveTo(px, py)
            else:
                mag_path.lineTo(px, py)
        painter.setPen(QPen(QColor("#4ECDC4"), 2.0))
        painter.drawPath(mag_path)

        # Phase trace (amber, -180..+180)
        if self._deg is not None and len(self._deg) == len(self._freqs):
            ph_path = QPainterPath()
            for idx, (f_val, p_val) in enumerate(zip(self._freqs, self._deg)):
                px = x_of(float(f_val))
                py = plot.bottom() - ((float(p_val) + 180.0) / 360.0) * plot.height()
                if idx == 0:
                    ph_path.moveTo(px, py)
                else:
                    ph_path.lineTo(px, py)
            painter.setPen(QPen(QColor("#FF9F1C"), 1.6, Qt.PenStyle.DashLine))
            painter.drawPath(ph_path)

        if self._cursor_f is not None and self._cursor_f > 0.0:
            cx = x_of(self._cursor_f)
            if plot.left() <= cx <= plot.right():
                painter.setPen(QPen(QColor("#FFFFFF"), 1.2, Qt.PenStyle.DotLine))
                painter.drawLine(int(cx), int(plot.top()), int(cx), int(plot.bottom()))

        painter.setPen(QColor("#4ECDC4"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(4, int(plot.top() + 10), f"{db_max:.1f}dB")
        painter.drawText(4, int(plot.bottom()), f"{db_min:.1f}dB")
        painter.setPen(QColor("#FF9F1C"))
        painter.drawText(int(plot.right() + 4), int(plot.top() + 10), "+180°")
        painter.drawText(int(plot.right() + 4), int(plot.bottom()), "-180°")


class _FFTCanvas(QWidget):
    """Dark-theme spectrum plotter displaying magnitude in dBV vs Frequency (Hz)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 260)
        self._result: FFTSpectrumResult | None = None

    def set_spectrum(self, result: FFTSpectrumResult | None) -> None:
        self._result = result
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11161C"))
        plot = QRectF(56.0, 20.0, max(80.0, self.width() - 78.0), max(80.0, self.height() - 52.0))
        painter.setPen(QPen(QColor("#24303D"), 1.0, Qt.PenStyle.DotLine))
        for i in range(9):
            y = plot.top() + plot.height() * (i / 8.0)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
        painter.setPen(QPen(QColor("#3E5063"), 1.2))
        painter.drawRect(plot)

        if self._result is None or len(self._result.freqs_hz) < 2:
            painter.setPen(QColor("#8CA0B3"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Click 'Compute FFT' to plot spectrum.")
            return

        freqs = np.asarray(self._result.freqs_hz, dtype=np.float64)
        dbv = np.asarray(self._result.magnitude_dbv, dtype=np.float64)
        f_max = max(float(freqs[-1]), 1.0)
        db_max = max(float(np.max(dbv)), 0.0) + 6.0
        db_min = max(float(np.min(dbv)), db_max - 120.0)
        db_span = max(db_max - db_min, 1.0)

        path = QPainterPath()
        for idx, (f_val, mag) in enumerate(zip(freqs, dbv)):
            px = plot.left() + (float(f_val) / f_max) * plot.width()
            clamped = max(db_min, min(db_max, float(mag)))
            py = plot.bottom() - ((clamped - db_min) / db_span) * plot.height()
            if idx == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        painter.setPen(QPen(QColor("#60A5FA"), 1.8))
        painter.drawPath(path)

        # Fundamental peak marker
        if self._result.fundamental_freq_hz > 0.0:
            fx = plot.left() + (min(self._result.fundamental_freq_hz, f_max) / f_max) * plot.width()
            fy = plot.bottom() - ((max(db_min, min(db_max, self._result.peak_dbv)) - db_min) / db_span) * plot.height()
            painter.setBrush(QColor("#FBBF24"))
            painter.setPen(QPen(QColor("#11161C"), 1.0))
            painter.drawEllipse(QRectF(fx - 3.5, fy - 3.5, 7.0, 7.0))

        painter.setPen(QColor("#8CA0B3"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(4, int(plot.top() + 10), f"{db_max:.0f} dBV")
        painter.drawText(4, int(plot.bottom()), f"{db_min:.0f} dBV")
        painter.drawText(int(plot.left()), int(plot.bottom() + 16), "0 Hz")
        painter.drawText(int(plot.right() - 70), int(plot.bottom() + 16), f"{f_max:.0f} Hz")


class BenchtopInstrumentsDialog(QDialog):
    """Non-modal benchtop Waveform Analyzer (4-Ch) + AC Bode & Z_in instrument suite."""

    def __init__(
        self,
        circuit_provider: Callable[[], Circuit] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Benchtop Virtual Instruments — Waveform & AC Bode / Z_in")
        self.setObjectName("BenchtopInstrumentsDialog")
        self.resize(780, 580)

        self._circuit_provider = circuit_provider
        self.scope_core = ScopeCore()
        self.latest_transient: TransientResult | None = None
        self.latest_scope_frame: ScopeFrame | None = None
        self.latest_cursor_readout: CursorReadout | None = None
        self.latest_ac_result: ACAnalysisResult | None = None
        self.latest_fft_result: FFTSpectrumResult | None = None
        self._last_live_t: float | None = None

        root = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)

        self._build_waveform_tab()
        self._build_bode_tab()
        self._build_fft_tab()
        self._build_dmm_tab()
        self.refresh_nodes()

    def _build_waveform_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        ctrl = QGridLayout()
        self.ch_combos: dict[str, QComboBox] = {}
        for idx, ch_name in enumerate(CHANNEL_NAMES):
            lbl = QLabel(f"{ch_name}:")
            lbl.setStyleSheet(f"color: {CHANNEL_COLOR_HEX[ch_name]}; font-weight: 600;")
            cmb = QComboBox()
            cmb.addItem("(None)", "")
            cmb.currentIndexChanged.connect(self._on_channel_assignment_changed)
            self.ch_combos[ch_name] = cmb
            ctrl.addWidget(lbl, 0, idx * 2)
            ctrl.addWidget(cmb, 0, idx * 2 + 1)

        self.trig_edge_combo = QComboBox()
        self.trig_edge_combo.addItem("Rising", TriggerEdge.RISING)
        self.trig_edge_combo.addItem("Falling", TriggerEdge.FALLING)
        self.trig_level_spin = QDoubleSpinBox()
        self.trig_level_spin.setRange(-1000.0, 1000.0)
        self.trig_level_spin.setValue(0.0)
        self.trig_level_spin.setSuffix(" V")

        self.duration_ms_spin = QDoubleSpinBox()
        self.duration_ms_spin.setRange(0.01, 10000.0)
        self.duration_ms_spin.setValue(10.0)
        self.duration_ms_spin.setSuffix(" ms")

        self.dt_us_spin = QDoubleSpinBox()
        self.dt_us_spin.setRange(0.05, 10000.0)
        self.dt_us_spin.setValue(25.0)
        self.dt_us_spin.setSuffix(" us")

        self.btn_capture_wave = QPushButton("Run Transient Capture")
        self.btn_capture_wave.clicked.connect(self._on_run_capture_clicked)

        ctrl.addWidget(QLabel("Trigger Edge:"), 1, 0)
        ctrl.addWidget(self.trig_edge_combo, 1, 1)
        ctrl.addWidget(QLabel("Level:"), 1, 2)
        ctrl.addWidget(self.trig_level_spin, 1, 3)
        ctrl.addWidget(QLabel("Duration:"), 1, 4)
        ctrl.addWidget(self.duration_ms_spin, 1, 5)
        ctrl.addWidget(QLabel("dt:"), 1, 6)
        ctrl.addWidget(self.dt_us_spin, 1, 7)
        layout.addLayout(ctrl)

        cursor_row = QHBoxLayout()
        self.cursor_t1_spin = QDoubleSpinBox()
        self.cursor_t1_spin.setRange(0.0, 10000.0)
        self.cursor_t1_spin.setValue(1.0)
        self.cursor_t1_spin.setSuffix(" ms")
        self.cursor_t2_spin = QDoubleSpinBox()
        self.cursor_t2_spin.setRange(0.0, 10000.0)
        self.cursor_t2_spin.setValue(5.0)
        self.cursor_t2_spin.setSuffix(" ms")
        self.cursor_t1_spin.valueChanged.connect(self._update_cursor_readout)
        self.cursor_t2_spin.valueChanged.connect(self._update_cursor_readout)

        self.cursor_readout_label = QLabel("Cursors: dt = 4.000 ms | dV = 0.000 V")
        self.xy_mode_checkbox = QCheckBox("X-Y Mode (CH1 vs CH2)", self)
        self.xy_mode_checkbox.toggled.connect(self._on_xy_mode_toggled)
        cursor_row.addWidget(QLabel("T1:"))
        cursor_row.addWidget(self.cursor_t1_spin)
        cursor_row.addWidget(QLabel("T2:"))
        cursor_row.addWidget(self.cursor_t2_spin)
        cursor_row.addWidget(self.cursor_readout_label, 1)
        cursor_row.addWidget(self.xy_mode_checkbox)
        cursor_row.addWidget(self.btn_capture_wave)
        layout.addLayout(cursor_row)

        self.wave_canvas = _WaveformCanvas(self)
        layout.addWidget(self.wave_canvas, 1)

        self.meas_table = QTableWidget(4, 5, self)
        self.meas_table.setHorizontalHeaderLabels(["Channel (Node)", "V_pp", "V_rms", "V_avg", "Freq (Hz)"])
        self.meas_table.verticalHeader().setVisible(False)
        self.meas_table.setMaximumHeight(138)
        for r, ch in enumerate(CHANNEL_NAMES):
            self.meas_table.setItem(r, 0, QTableWidgetItem(ch))
            for c in range(1, 5):
                self.meas_table.setItem(r, c, QTableWidgetItem("—"))
        layout.addWidget(self.meas_table)

        self.tabs.addTab(tab, "Waveform Capture (4-Ch)")

    def _build_bode_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        form_row = QHBoxLayout()
        left_form = QFormLayout()
        self.bode_node_combo = QComboBox()
        self.bode_f_start_spin = QDoubleSpinBox()
        self.bode_f_start_spin.setRange(0.01, 1e9)
        self.bode_f_start_spin.setDecimals(2)
        self.bode_f_start_spin.setValue(10.0)
        self.bode_f_start_spin.setSuffix(" Hz")
        self.bode_f_stop_spin = QDoubleSpinBox()
        self.bode_f_stop_spin.setRange(1.0, 1e9)
        self.bode_f_stop_spin.setDecimals(1)
        self.bode_f_stop_spin.setValue(100000.0)
        self.bode_f_stop_spin.setSuffix(" Hz")
        left_form.addRow("Output Probe Node:", self.bode_node_combo)
        left_form.addRow("f_start:", self.bode_f_start_spin)
        left_form.addRow("f_stop:", self.bode_f_stop_spin)

        right_form = QFormLayout()
        self.bode_ppd_spin = QSpinBox()
        self.bode_ppd_spin.setRange(5, 200)
        self.bode_ppd_spin.setValue(25)
        self.bode_z0_spin = QDoubleSpinBox()
        self.bode_z0_spin.setRange(1.0, 10000.0)
        self.bode_z0_spin.setValue(50.0)
        self.bode_z0_spin.setSuffix(" Ω")
        self.bode_cursor_f_spin = QDoubleSpinBox()
        self.bode_cursor_f_spin.setRange(0.01, 1e9)
        self.bode_cursor_f_spin.setDecimals(2)
        self.bode_cursor_f_spin.setValue(1000.0)
        self.bode_cursor_f_spin.setSuffix(" Hz")
        self.bode_cursor_f_spin.valueChanged.connect(self._update_bode_cursor_readout)
        right_form.addRow("Points / decade:", self.bode_ppd_spin)
        right_form.addRow("Reference Z0:", self.bode_z0_spin)
        right_form.addRow("Marker Frequency:", self.bode_cursor_f_spin)

        self.btn_run_ac = QPushButton("Run AC Sweep")
        self.btn_run_ac.clicked.connect(self._on_run_ac_clicked)

        form_row.addLayout(left_form, 1)
        form_row.addLayout(right_form, 1)
        form_row.addWidget(self.btn_run_ac)
        layout.addLayout(form_row)

        self.bode_canvas = _BodeCanvas(self)
        layout.addWidget(self.bode_canvas, 1)

        grp = QGroupBox("Bode / Z_in / Reflection Coefficient Readout", self)
        g_layout = QHBoxLayout(grp)
        self.bode_readout_label = QLabel(
            "Run AC Sweep to inspect |V(jw)| (dB), Phase (deg), Z_in (Ω), and Gamma."
        )
        self.bode_readout_label.setWordWrap(True)
        g_layout.addWidget(self.bode_readout_label)
        layout.addWidget(grp)

        self.tabs.addTab(tab, "AC Bode & Z_in")

    def _build_fft_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        top.addWidget(QLabel("Node / Channel:"))
        self.fft_channel_combo = QComboBox(tab)
        top.addWidget(self.fft_channel_combo, 1)

        self.compute_fft_btn = QPushButton("Compute FFT", tab)
        self.compute_fft_btn.clicked.connect(self._on_run_fft_clicked)
        top.addWidget(self.compute_fft_btn)
        layout.addLayout(top)

        self.fft_canvas = _FFTCanvas(tab)
        layout.addWidget(self.fft_canvas, 1)

        grp = QGroupBox("Spectrum & Harmonic Distortion Readout", tab)
        g_layout = QHBoxLayout(grp)
        self.fft_readout_label = QLabel(
            "Click 'Compute FFT' (after Waveform Capture) to inspect Fundamental, Peak dBV, and THD%.",
            tab,
        )
        self.fft_readout_label.setWordWrap(True)
        g_layout.addWidget(self.fft_readout_label)
        layout.addWidget(grp)

        self.tabs.addTab(tab, "FFT Spectrum")

    def _build_dmm_tab(self) -> None:
        """Build the 4th tab: Digital Multimeter (DC V, AC V RMS, DC A, Resistance/Continuity)."""
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        form = QFormLayout()
        self.dmm_mode_combo = QComboBox(self)
        self.dmm_mode_combo.addItem("DC Voltage (V)", "dc_v")
        self.dmm_mode_combo.addItem("AC Voltage RMS (V)", "ac_v_rms")
        self.dmm_mode_combo.addItem("DC Current (A)", "dc_a")
        self.dmm_mode_combo.addItem("Resistance / Continuity (Ω)", "resistance")

        self.dmm_pos_combo = QComboBox(self)
        self.dmm_neg_combo = QComboBox(self)
        self.dmm_neg_combo.addItem("0 (GND)", "0")

        form.addRow("DMM Mode:", self.dmm_mode_combo)
        form.addRow("Red Probe (+ / Component):", self.dmm_pos_combo)
        form.addRow("Black Probe (- Reference Node):", self.dmm_neg_combo)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_dmm_measure = QPushButton("Measure", self)
        self.btn_dmm_measure.clicked.connect(self._on_dmm_measure_clicked)
        btn_row.addWidget(self.btn_dmm_measure)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        read_grp = QGroupBox("Multimeter Readout", self)
        read_lay = QVBoxLayout(read_grp)
        self.dmm_readout_label = QLabel("0.000 V", self)
        self.dmm_readout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dmm_readout_label.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        self.dmm_readout_label.setStyleSheet(
            "background-color: #11161C; color: #3FB950; padding: 18px; border-radius: 6px;"
        )
        self.dmm_continuity_label = QLabel("Continuity (< 50 Ω): Inactive", self)
        self.dmm_continuity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        read_lay.addWidget(self.dmm_readout_label)
        read_lay.addWidget(self.dmm_continuity_label)
        layout.addWidget(read_grp)
        layout.addStretch(1)

        self.latest_dmm_reading: DMMReading | None = None
        self.tabs.addTab(tab, "Digital Multimeter (DMM)")

    def _on_xy_mode_toggled(self, checked: bool) -> None:
        self.wave_canvas.set_xy_mode(checked)

    def get_xy_points(self) -> list[tuple[float, float]]:
        """Return paired ``(CH1_V, CH2_V)`` sample points from the Waveform canvas."""
        return self.wave_canvas.get_xy_points()

    def _on_dmm_measure_clicked(self) -> None:
        if self._circuit_provider is None:
            return
        self.run_dmm_measurement(self._circuit_provider())

    def run_dmm_measurement(
        self,
        circuit: Circuit,
        *,
        mode: str | None = None,
        pos_target: str | None = None,
        neg_target: str | None = None,
    ) -> DMMReading:
        """Execute a DMM reading against ``circuit`` and update the DMM readout widgets."""
        self.refresh_nodes(circuit)
        eff_mode = mode or str(self.dmm_mode_combo.currentData() or "dc_v")
        eff_pos = pos_target if pos_target is not None else str(self.dmm_pos_combo.currentData() or "0")
        eff_neg = neg_target if neg_target is not None else str(self.dmm_neg_combo.currentData() or "0")
        reading = measure_dmm(
            circuit,
            mode=eff_mode,
            pos_target=eff_pos,
            neg_target=eff_neg,
            transient_result=self.latest_transient if eff_mode == "ac_v_rms" else None,
        )
        self.latest_dmm_reading = reading
        self.dmm_readout_label.setText(reading.formatted_text)
        if reading.continuity_active:
            self.dmm_continuity_label.setText("CONTINUITY (< 50 Ω)")
            self.dmm_continuity_label.setStyleSheet("color: #3FB950; font-weight: 700;")
        else:
            self.dmm_continuity_label.setText("Continuity (< 50 Ω): Open / Over 50 Ω")
            self.dmm_continuity_label.setStyleSheet("")
        return reading

    def refresh_nodes(self, circuit: Circuit | None = None) -> None:
        """Populate probe node selectors from the supplied or current schematic circuit."""
        if circuit is None and self._circuit_provider is not None:
            try:
                circuit = self._circuit_provider()
            except Exception:
                circuit = None
        if circuit is None:
            return
        nodes = sorted(circuit.non_ground_nodes())

        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            cmb = self.ch_combos[ch_name]
            prev = cmb.currentData()
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("(None)", "")
            for n in nodes:
                cmb.addItem(n, n)
            if prev and prev in nodes:
                cmb.setCurrentIndex(nodes.index(prev) + 1)
            elif ch_idx < len(nodes):
                cmb.setCurrentIndex(ch_idx + 1)
            cmb.blockSignals(False)
            sel = str(cmb.currentData() or "") or None
            self.scope_core.assign_probe(ch_name, sel, enabled=(sel is not None))

        prev_bode = self.bode_node_combo.currentText()
        self.bode_node_combo.clear()
        for n in nodes:
            self.bode_node_combo.addItem(n)
        if prev_bode in nodes:
            self.bode_node_combo.setCurrentText(prev_bode)
        elif nodes:
            self.bode_node_combo.setCurrentIndex(len(nodes) - 1)

        if hasattr(self, "fft_channel_combo"):
            prev_fft = self.fft_channel_combo.currentText()
            self.fft_channel_combo.clear()
            for n in nodes:
                self.fft_channel_combo.addItem(n)
            if prev_fft in nodes:
                self.fft_channel_combo.setCurrentText(prev_fft)
            elif nodes:
                self.fft_channel_combo.setCurrentIndex(len(nodes) - 1)

        if hasattr(self, "dmm_pos_combo"):
            prev_pos = self.dmm_pos_combo.currentData()
            self.dmm_pos_combo.clear()
            for comp in circuit.components:
                if len(comp.nodes) >= 2:
                    self.dmm_pos_combo.addItem(f"{comp.name} (Across {comp.component_type})", comp.name)
            for n in nodes:
                self.dmm_pos_combo.addItem(f"Node: {n}", n)
            if prev_pos:
                idx = self.dmm_pos_combo.findData(prev_pos)
                if idx >= 0:
                    self.dmm_pos_combo.setCurrentIndex(idx)

            prev_neg = self.dmm_neg_combo.currentData()
            self.dmm_neg_combo.clear()
            self.dmm_neg_combo.addItem("0 (GND)", "0")
            for n in nodes:
                self.dmm_neg_combo.addItem(f"Node: {n}", n)
            if prev_neg:
                idx = self.dmm_neg_combo.findData(prev_neg)
                if idx >= 0:
                    self.dmm_neg_combo.setCurrentIndex(idx)

    def _on_channel_assignment_changed(self) -> None:
        for ch_name, cmb in self.ch_combos.items():
            node_name = str(cmb.currentData() or "") or None
            self.scope_core.assign_probe(ch_name, node_name, enabled=(node_name is not None))

    def _on_run_capture_clicked(self) -> None:
        if self._circuit_provider is None:
            return
        circuit = self._circuit_provider()
        self.capture_transient(
            circuit,
            duration_s=self.duration_ms_spin.value() * 1e-3,
            dt_s=self.dt_us_spin.value() * 1e-6,
        )

    def _on_run_ac_clicked(self) -> None:
        if self._circuit_provider is None:
            return
        circuit = self._circuit_provider()
        self.run_ac_analysis(
            circuit,
            f_start=self.bode_f_start_spin.value(),
            f_stop=self.bode_f_stop_spin.value(),
            points_per_decade=self.bode_ppd_spin.value(),
            z0_ohm=self.bode_z0_spin.value(),
        )

    def capture_transient(
        self,
        circuit: Circuit,
        *,
        duration_s: float = 10e-3,
        dt_s: float | None = None,
        channel_nodes: dict[str, str | None] | None = None,
    ) -> ScopeFrame:
        """Run a transient capture into `ScopeCore`, updating waveforms, measurements, and cursors."""
        self.refresh_nodes(circuit)
        if channel_nodes is not None:
            for ch_name, node_name in channel_nodes.items():
                self.scope_core.assign_probe(ch_name, node_name, enabled=(node_name is not None))
                if ch_name in self.ch_combos:
                    idx = self.ch_combos[ch_name].findData(node_name or "")
                    if idx >= 0:
                        self.ch_combos[ch_name].blockSignals(True)
                        self.ch_combos[ch_name].setCurrentIndex(idx)
                        self.ch_combos[ch_name].blockSignals(False)

        edge = self.trig_edge_combo.currentData()
        self.scope_core.set_trigger(
            edge=edge if isinstance(edge, TriggerEdge) else TriggerEdge.RISING,
            level_v=self.trig_level_spin.value(),
            mode=TriggerMode.AUTO,
            auto_timeout_s=min(0.05, 0.5 * max(duration_s, 1e-6)),
        )
        self.scope_core.set_timebase(seconds_per_div=max(duration_s / 10.0, 1e-9))
        self.scope_core.clear()

        solver = MNASolver(circuit)
        res = solver.solve_transient(duration=duration_s, dt=dt_s)
        self.latest_transient = res

        times = res.times
        step_dt = float(times[1] - times[0]) if len(times) > 1 else (dt_s or 1e-6)
        node_names = tuple(sorted(res.node_voltages.keys()))
        data = np.column_stack([res.node_voltages[n] for n in node_names]) if node_names else np.zeros((len(times), 0))
        block = SampleBlock(t0=float(times[0]), dt=step_dt, node_names=node_names, data=data)
        self.scope_core.ingest_block(block)
        frame = self.scope_core.produce_frame()
        self.latest_scope_frame = frame

        active_traces: dict[str, tuple[str, np.ndarray]] = {}
        for row_idx, ch_name in enumerate(CHANNEL_NAMES):
            cfg = self.scope_core.channels[ch_name]
            node = cfg.node_name
            if not cfg.enabled or not node or node not in res.node_voltages:
                self.meas_table.setItem(row_idx, 0, QTableWidgetItem(f"{ch_name} (Off)"))
                for c in range(1, 5):
                    self.meas_table.setItem(row_idx, c, QTableWidgetItem("—"))
                continue

            arr = res.node_voltages[node]
            active_traces[ch_name] = (node, arr)
            meas = measure_channel(times, arr)
            self.meas_table.setItem(row_idx, 0, QTableWidgetItem(f"{ch_name} ({node})"))
            self.meas_table.setItem(row_idx, 1, QTableWidgetItem(f"{meas.vpp:.4g} V"))
            self.meas_table.setItem(row_idx, 2, QTableWidgetItem(f"{meas.vrms:.4g} V"))
            self.meas_table.setItem(row_idx, 3, QTableWidgetItem(f"{meas.vavg:.4g} V"))
            freq_txt = f"{meas.frequency_hz:.4g} Hz" if meas.frequency_hz is not None else "DC / <1 cyc"
            self.meas_table.setItem(row_idx, 4, QTableWidgetItem(freq_txt))

        self._update_cursor_readout()
        self.wave_canvas.set_capture(
            frame,
            times=times,
            traces=active_traces,
            cursor_t1=self.cursor_t1_spin.value() * 1e-3,
            cursor_t2=self.cursor_t2_spin.value() * 1e-3,
        )
        return frame

    def _update_cursor_readout(self) -> None:
        if self.latest_transient is None:
            return
        ch1_node = str(self.ch_combos["CH1"].currentData() or "")
        if not ch1_node or ch1_node not in self.latest_transient.node_voltages:
            return
        t1 = self.cursor_t1_spin.value() * 1e-3
        t2 = self.cursor_t2_spin.value() * 1e-3
        times = self.latest_transient.times
        vals = self.latest_transient.node_voltages[ch1_node]
        cr = compute_cursors(times, vals, t1, t2)
        self.latest_cursor_readout = cr
        freq_str = f" | 1/dt = {cr.inv_dt_hz:.3g} Hz" if cr.inv_dt_hz is not None else ""
        self.cursor_readout_label.setText(
            f"CH1 ({ch1_node}): V1={cr.v1:.4g} V, V2={cr.v2:.4g} V | dV={cr.dv:.4g} V | dt={cr.dt * 1e3:.4g} ms{freq_str}"
        )
        if self.latest_scope_frame is not None:
            active_traces = {
                ch: (str(cmb.currentData()), self.latest_transient.node_voltages[str(cmb.currentData())])
                for ch, cmb in self.ch_combos.items()
                if cmb.currentData() and str(cmb.currentData()) in self.latest_transient.node_voltages
            }
            self.wave_canvas.set_capture(
                self.latest_scope_frame,
                times=times,
                traces=active_traces,
                cursor_t1=t1,
                cursor_t2=t2,
            )

    def run_ac_analysis(
        self,
        circuit: Circuit,
        *,
        probe_node: str | None = None,
        f_start: float = 10.0,
        f_stop: float = 1e5,
        points_per_decade: int = 25,
        z0_ohm: float = 50.0,
    ) -> ACAnalysisResult:
        """Execute AC Bode + Z_in sweep and render the curves + marker metrics."""
        self.refresh_nodes(circuit)
        ac_res = run_ac_sweep(
            circuit,
            f_start=f_start,
            f_stop=f_stop,
            points_per_decade=points_per_decade,
            z0_ohm=z0_ohm,
        )
        self.latest_ac_result = ac_res
        target_node = probe_node or self.bode_node_combo.currentText()
        if target_node and target_node in ac_res.node_phasors:
            self.bode_node_combo.setCurrentText(target_node)
        self._update_bode_cursor_readout()
        return ac_res

    def _on_run_fft_clicked(self) -> None:
        self.run_fft_analysis()

    def run_fft_analysis(
        self,
        times: object | None = None,
        voltages: object | None = None,
        *,
        node: str | None = None,
    ) -> FFTSpectrumResult | None:
        """Compute Hann-windowed FFT spectrum & THD% for explicit data or captured transient node."""
        t_arr = times
        v_arr = voltages
        target = node or (
            self.fft_channel_combo.currentText().strip()
            if hasattr(self, "fft_channel_combo")
            else ""
        )
        if t_arr is None or v_arr is None:
            if self.latest_transient is None and self._circuit_provider is not None:
                self.capture_transient(
                    self._circuit_provider(),
                    duration_s=self.duration_ms_spin.value() * 1e-3,
                    dt_s=self.dt_us_spin.value() * 1e-6,
                )
            if self.latest_transient is None:
                return None
            tr = self.latest_transient
            if not target or target not in tr.node_voltages:
                keys = sorted(k for k in tr.node_voltages if k != "0")
                target = keys[-1] if keys else ""
            if not target or target not in tr.node_voltages:
                return None
            t_arr = tr.times
            v_arr = tr.node_voltages[target]

        fft_res = compute_fft_spectrum(t_arr, v_arr)  # type: ignore[arg-type]
        self.latest_fft_result = fft_res
        self.fft_canvas.set_spectrum(fft_res)
        self.fft_readout_label.setText(
            f"Fundamental: {fft_res.fundamental_freq_hz:.2f} Hz | "
            f"Peak: {fft_res.peak_dbv:.2f} dBV | "
            f"THD: {fft_res.thd_percent:.2f}%"
        )
        return fft_res

    def _update_bode_cursor_readout(self) -> None:
        ac_res = self.latest_ac_result
        if ac_res is None:
            return
        node = self.bode_node_combo.currentText()
        if node not in ac_res.node_phasors:
            return
        freqs, db_arr, deg_arr = ac_res.bode(node)
        f_mark = self.bode_cursor_f_spin.value()
        idx = int(np.argmin(np.abs(freqs - f_mark)))
        f_val = float(freqs[idx])
        db_val = float(db_arr[idx])
        deg_val = float(deg_arr[idx])
        z_val = complex(ac_res.z_in[idx])
        g_val = complex(ac_res.gamma[idx])
        self.bode_readout_label.setText(
            f"Node {node!r} @ {f_val:.4g} Hz: "
            f"|V| = {db_val:.2f} dB, Phase = {deg_val:.2f} deg | "
            f"Z_in = {z_val.real:.3g}{z_val.imag:+.3g}j Ω (|Z|={abs(z_val):.3g} Ω) | "
            f"|Gamma| = {abs(g_val):.4f}"
        )
        self.bode_canvas.set_bode(freqs, db_arr, deg_arr, cursor_f=f_val)

    def on_simulation_frame(self, frame: SimulationFrame) -> None:
        """Ingest live simulation frames when running interactively."""
        if frame.state == SimulationState.STOPPED:
            self._last_live_t = None
            return
        if not self.isVisible() or not frame.node_voltages or frame.dt <= 0.0:
            return
        if self._last_live_t is not None and frame.sim_time <= self._last_live_t:
            return
        self._last_live_t = frame.sim_time
        names = tuple(sorted(frame.node_voltages.keys()))
        row = np.array([[frame.node_voltages[n] for n in names]], dtype=np.float64)
        self.scope_core.ingest_block(
            SampleBlock(t0=frame.sim_time, dt=frame.dt, node_names=names, data=row)
        )
