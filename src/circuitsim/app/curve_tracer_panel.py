"""Interactive I-V Characteristic Curve Tracer dialog (PySide6 UI)."""

from __future__ import annotations

from typing import Callable
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components import Diode, LED, NMOS, NPNBJT, Resistor, ZenerDiode
from circuitsim.components.base import Component
from circuitsim.dialogs.value_parser import parse_engineering_value
from circuitsim.engine.curve_tracer import (
    CurveFamilyResult,
    export_curve_family_csv,
    trace_transistor_family,
    trace_two_terminal_iv,
)

_CURVE_COLORS: tuple[str, ...] = (
    "#58A6FF",
    "#3FB950",
    "#F78166",
    "#D2A8FF",
    "#FFD166",
    "#FF7B72",
)


class _CurvePlotCanvas(QWidget):
    """Multi-curve I-V characteristic plotter with zero-axes and legend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(540, 320)
        self._result: CurveFamilyResult | None = None

    def set_result(self, result: CurveFamilyResult | None) -> None:
        self._result = result
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#11161C"))

        pad_l, pad_r, pad_t, pad_b = 62.0, 20.0, 22.0, 34.0
        plot = QRectF(
            pad_l,
            pad_t,
            max(40.0, rect.width() - pad_l - pad_r),
            max(40.0, rect.height() - pad_t - pad_b),
        )

        painter.setPen(QPen(QColor("#24303D"), 1.0, Qt.PenStyle.DotLine))
        for i in range(11):
            x = plot.left() + plot.width() * (i / 10.0)
            painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))
        for j in range(9):
            y = plot.top() + plot.height() * (j / 8.0)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))

        painter.setPen(QPen(QColor("#3E5063"), 1.2))
        painter.drawRect(plot)

        res = self._result
        if res is None or not res.curves:
            painter.setPen(QColor("#8CA0B3"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Click 'Run Trace' to plot I-V curves.")
            return

        xs_all = [x for c in res.curves for x in c.voltages]
        ys_all = [y for c in res.curves for y in c.currents_ma]
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        if abs(x_max - x_min) < 1e-9:
            x_min -= 1.0
            x_max += 1.0
        if abs(y_max - y_min) < 1e-9:
            y_min -= 1.0
            y_max += 1.0
        y_pad = 0.08 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        def to_px(xv: float, yv: float) -> tuple[float, float]:
            px = plot.left() + ((xv - x_min) / x_span) * plot.width()
            py = plot.bottom() - ((yv - y_min) / y_span) * plot.height()
            return px, py

        # Highlight X=0 and Y=0 axes
        painter.setPen(QPen(QColor("#4B5B6E"), 1.2, Qt.PenStyle.DashLine))
        if x_min <= 0.0 <= x_max:
            zx, _ = to_px(0.0, y_min)
            painter.drawLine(int(zx), int(plot.top()), int(zx), int(plot.bottom()))
        if y_min <= 0.0 <= y_max:
            _, zy = to_px(x_min, 0.0)
            painter.drawLine(int(plot.left()), int(zy), int(plot.right()), int(zy))

        legend_y = int(plot.top() + 14)
        for idx, curve in enumerate(res.curves):
            col = QColor(_CURVE_COLORS[idx % len(_CURVE_COLORS)])
            painter.setPen(QPen(col, 2.0))
            path = QPainterPath()
            for p_idx, (xv, yv) in enumerate(zip(curve.voltages, curve.currents_ma)):
                px, py = to_px(xv, yv)
                if p_idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.drawPath(path)

            # Legend entry
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(int(plot.left() + 10), legend_y + idx * 14, curve.label)

        painter.setPen(QColor("#8CA0B3"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(4, int(plot.top() + 10), f"{y_max:.2g} mA")
        painter.drawText(4, int(plot.bottom()), f"{y_min:.2g} mA")
        painter.drawText(int(plot.left()), int(plot.bottom() + 18), f"{x_min:.2g} V ({res.x_label})")
        painter.drawText(int(plot.right() - 60), int(plot.bottom() + 18), f"{x_max:.2g} V")


class CurveTracerDialog(QDialog):
    """Interactive I-V Curve Tracer modal/tool window."""

    PRESETS: tuple[tuple[str, str], ...] = (
        ("1N4148 Silicon Diode", "diode"),
        ("5.1V Zener Diode", "zener_diode"),
        ("Red LED (2.0V)", "led"),
        ("Resistor 1k Ohm", "resistor"),
        ("2N3904 NPN BJT (BF=100)", "npn_bjt"),
        ("2N7000 NMOS (Vth=1.5V, KP=0.02)", "nmos"),
    )

    def __init__(
        self,
        selected_component_provider: Callable[[], Component | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("I-V Characteristic Curve Tracer")
        self.setObjectName("CurveTracerDialog")
        self.resize(740, 520)
        self._selected_provider = selected_component_provider
        self.latest_result: CurveFamilyResult | None = None

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.device_combo = QComboBox(self)
        for label, key in self.PRESETS:
            self.device_combo.addItem(label, key)
        if selected_component_provider is not None:
            sel = selected_component_provider()
            if sel is not None and sel.component_type in {
                "resistor",
                "diode",
                "zener_diode",
                "led",
                "npn_bjt",
                "pnp_bjt",
                "nmos",
                "pmos",
            }:
                self.device_combo.insertItem(0, f"Selected: {sel.name} ({sel.component_type})", "__selected__")
                self.device_combo.setCurrentIndex(0)
        self.device_combo.currentIndexChanged.connect(self._on_preset_changed)

        range_row = QHBoxLayout()
        self.v_start_spin = QDoubleSpinBox(self)
        self.v_start_spin.setRange(-100.0, 100.0)
        self.v_start_spin.setValue(-6.0)
        self.v_start_spin.setSuffix(" V")
        self.v_stop_spin = QDoubleSpinBox(self)
        self.v_stop_spin.setRange(-100.0, 100.0)
        self.v_stop_spin.setValue(1.5)
        self.v_stop_spin.setSuffix(" V")
        self.points_spin = QSpinBox(self)
        self.points_spin.setRange(5, 500)
        self.points_spin.setValue(50)
        range_row.addWidget(QLabel("Start:"))
        range_row.addWidget(self.v_start_spin)
        range_row.addWidget(QLabel("Stop:"))
        range_row.addWidget(self.v_stop_spin)
        range_row.addWidget(QLabel("Points:"))
        range_row.addWidget(self.points_spin)

        self.control_edit = QLineEdit("10u, 20u, 30u", self)
        form.addRow("Device Under Test:", self.device_combo)
        form.addRow("Main Voltage Sweep:", range_row)
        form.addRow("Step Bias List (I_B or V_GS):", self.control_edit)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Trace", self)
        self.btn_run.clicked.connect(self.run_trace)
        self.btn_export = QPushButton("Export CSV...", self)
        self.btn_export.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_export)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.canvas = _CurvePlotCanvas(self)
        root.addWidget(self.canvas, 1)
        self._on_preset_changed()

    def _build_selected_device(self) -> Component:
        key = str(self.device_combo.currentData() or "diode")
        if key == "__selected__" and self._selected_provider is not None:
            comp = self._selected_provider()
            if comp is not None:
                return comp
        if key == "resistor":
            return Resistor("DUT", "a", "b", resistance=1_000.0)
        if key == "zener_diode":
            return ZenerDiode("DUT", "a", "b", v_z=5.1)
        if key == "led":
            return LED("DUT", "a", "b", forward_voltage=2.0)
        if key == "npn_bjt":
            return NPNBJT("DUT", "b", "c", "e", beta_f=100.0)
        if key == "nmos":
            return NMOS("DUT", "g", "d", "s", v_th=1.5, k_p=0.02)
        return Diode("DUT", "a", "b")

    def _on_preset_changed(self) -> None:
        key = str(self.device_combo.currentData() or "")
        if key == "npn_bjt":
            self.v_start_spin.setValue(0.0)
            self.v_stop_spin.setValue(8.0)
            self.control_edit.setText("10u, 20u, 30u")
            self.control_edit.setEnabled(True)
        elif key == "nmos":
            self.v_start_spin.setValue(0.0)
            self.v_stop_spin.setValue(8.0)
            self.control_edit.setText("2.0, 2.5, 3.0, 3.5")
            self.control_edit.setEnabled(True)
        elif key == "zener_diode":
            self.v_start_spin.setValue(-6.5)
            self.v_stop_spin.setValue(1.2)
            self.control_edit.setEnabled(False)
        else:
            self.v_start_spin.setValue(-2.0)
            self.v_stop_spin.setValue(2.5)
            self.control_edit.setEnabled(False)

    def run_trace(self) -> CurveFamilyResult:
        comp = self._build_selected_device()
        v0 = self.v_start_spin.value()
        v1 = self.v_stop_spin.value()
        pts = self.points_spin.value()
        if comp.component_type in ("npn_bjt", "pnp_bjt", "nmos", "pmos"):
            steps = [
                parse_engineering_value(tok.strip())
                for tok in self.control_edit.text().split(",")
                if tok.strip()
            ]
            if not steps:
                steps = [10e-6, 20e-6, 30e-6] if "bjt" in comp.component_type else [2.5, 3.5]
            res = trace_transistor_family(comp, v0, v1, steps, points=pts)
        else:
            curve = trace_two_terminal_iv(comp, v0, v1, points=pts)
            res = CurveFamilyResult(
                component_type=comp.component_type,
                x_label="V (V)",
                y_label="I (mA)",
                control_label="device",
                curves=[curve],
            )
        self.latest_result = res
        self.canvas.set_result(res)
        return res

    def _on_export_csv(self) -> None:
        if self.latest_result is None:
            self.run_trace()
        if self.latest_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Curve Trace CSV",
            "curve_trace.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            export_curve_family_csv(self.latest_result, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
