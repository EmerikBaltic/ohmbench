"""Parameter Sweep (.STEP) & Monte Carlo Tolerance Analyzer Dialog (H6)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable
import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components.capacitor import Capacitor
from circuitsim.components.ground import Ground
from circuitsim.components.inductor import Inductor
from circuitsim.components.potentiometer import Potentiometer
from circuitsim.components.resistor import Resistor
from circuitsim.dialogs.value_parser import format_engineering_value, parse_engineering_value
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.sweep import (
    MonteCarloResult,
    MonteCarloTolerance,
    ParameterSweepResult,
    run_monte_carlo,
    run_parameter_sweep,
)


class _SweepCurveCanvas(QWidget):
    """2D XY plotter for `.STEP` DC parameter sweep curves."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 230)
        self._x_vals: np.ndarray | None = None
        self._y_vals: np.ndarray | None = None
        self._x_label: str = "Parameter"
        self._y_label: str = "Voltage (V)"
        self._log_x: bool = False

    def set_data(
        self,
        x_vals: np.ndarray | None,
        y_vals: np.ndarray | None,
        *,
        x_label: str = "Parameter",
        y_label: str = "Voltage (V)",
        log_x: bool = False,
    ) -> None:
        self._x_vals = x_vals
        self._y_vals = y_vals
        self._x_label = x_label
        self._y_label = y_label
        self._log_x = log_x
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0d1117"))

        pad_l, pad_r, pad_t, pad_b = 58.0, 18.0, 18.0, 34.0
        plot = QRectF(
            pad_l,
            pad_t,
            max(60.0, float(self.width()) - pad_l - pad_r),
            max(60.0, float(self.height()) - pad_t - pad_b),
        )
        painter.fillRect(plot, QColor("#111722"))
        painter.setPen(QPen(QColor("#2c3440"), 1.0))
        painter.drawRect(plot)

        for i in range(1, 5):
            y_g = plot.top() + plot.height() * (i / 5.0)
            painter.setPen(QPen(QColor("#1e2633"), 1.0, Qt.PenStyle.DotLine))
            painter.drawLine(plot.left(), y_g, plot.right(), y_g)
            x_g = plot.left() + plot.width() * (i / 5.0)
            painter.drawLine(x_g, plot.top(), x_g, plot.bottom())

        if self._x_vals is None or self._y_vals is None or len(self._x_vals) == 0:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Click 'Run Sweep' to plot curve")
            return

        xs_raw = np.asarray(self._x_vals, dtype=np.float64)
        ys = np.asarray(self._y_vals, dtype=np.float64)
        if self._log_x and np.all(xs_raw > 0.0):
            xs = np.log10(xs_raw)
        else:
            xs = xs_raw

        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        if abs(x_max - x_min) < 1e-18:
            x_min -= 1.0
            x_max += 1.0

        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        span_y = y_max - y_min
        if span_y < 1e-9:
            y_min -= 0.5
            y_max += 0.5
        else:
            y_min -= 0.08 * span_y
            y_max += 0.08 * span_y

        path = QPainterPath()
        pts: list[tuple[float, float]] = []
        for xv, yv in zip(xs, ys):
            px = plot.left() + ((float(xv) - x_min) / (x_max - x_min)) * plot.width()
            py = plot.bottom() - ((float(yv) - y_min) / (y_max - y_min)) * plot.height()
            pts.append((px, py))
            if len(pts) == 1:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        painter.setPen(QPen(QColor("#58a6ff"), 2.2))
        painter.drawPath(path)

        painter.setBrush(QColor("#79c0ff"))
        painter.setPen(QPen(QColor("#0d1117"), 1.0))
        for px, py in pts:
            painter.drawEllipse(QRectF(px - 3.0, py - 3.0, 6.0, 6.0))

        painter.setPen(QColor("#c9d1d9"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(2, plot.top() - 6, pad_l - 6, 18), Qt.AlignmentFlag.AlignRight, f"{y_max:.3g} V")
        painter.drawText(QRectF(2, plot.bottom() - 12, pad_l - 6, 18), Qt.AlignmentFlag.AlignRight, f"{y_min:.3g} V")
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 4, plot.width(), 22),
            Qt.AlignmentFlag.AlignCenter,
            f"{self._x_label} -> {self._y_label}",
        )


class _HistogramCanvas(QWidget):
    """Histogram plotter for Monte Carlo output distribution."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 230)
        self._samples: np.ndarray | None = None
        self._nominal: float = 0.0
        self._mean: float = 0.0

    def set_data(self, samples: np.ndarray | None, nominal: float, mean: float) -> None:
        self._samples = samples
        self._nominal = float(nominal)
        self._mean = float(mean)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0d1117"))

        pad_l, pad_r, pad_t, pad_b = 50.0, 18.0, 18.0, 34.0
        plot = QRectF(
            pad_l,
            pad_t,
            max(60.0, float(self.width()) - pad_l - pad_r),
            max(60.0, float(self.height()) - pad_t - pad_b),
        )
        painter.fillRect(plot, QColor("#111722"))
        painter.setPen(QPen(QColor("#2c3440"), 1.0))
        painter.drawRect(plot)

        if self._samples is None or len(self._samples) == 0:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Click 'Run Monte Carlo' to view histogram")
            return

        vals = np.asarray(self._samples, dtype=np.float64)
        n_bins = min(20, max(6, int(round(math.sqrt(len(vals))))))
        v_min, v_max = float(np.min(vals)), float(np.max(vals))
        if abs(v_max - v_min) < 1e-12:
            v_min -= 0.05
            v_max += 0.05
        counts, edges = np.histogram(vals, bins=n_bins, range=(v_min, v_max))
        max_c = max(1, int(np.max(counts)))

        bar_w = plot.width() / n_bins
        painter.setBrush(QColor("#2ea043"))
        painter.setPen(QPen(QColor("#111722"), 1.0))
        for idx, c in enumerate(counts):
            h = (float(c) / float(max_c)) * (plot.height() - 10.0)
            bx = plot.left() + idx * bar_w + 1.0
            by = plot.bottom() - h
            painter.drawRect(QRectF(bx, by, max(2.0, bar_w - 2.0), h))

        # Mean dashed line
        mx = plot.left() + ((self._mean - v_min) / (v_max - v_min)) * plot.width()
        mx = max(plot.left(), min(plot.right(), mx))
        painter.setPen(QPen(QColor("#ffa726"), 1.8, Qt.PenStyle.DashLine))
        painter.drawLine(mx, plot.top(), mx, plot.bottom())

        painter.setPen(QColor("#c9d1d9"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(plot.left(), plot.bottom() + 4, 100, 20), Qt.AlignmentFlag.AlignLeft, f"{v_min:.4g} V")
        painter.drawText(QRectF(plot.right() - 100, plot.bottom() + 4, 100, 20), Qt.AlignmentFlag.AlignRight, f"{v_max:.4g} V")


class SweepAnalysisDialog(QDialog):
    """Parameter Sweep (.STEP) & Monte Carlo Tolerance dialog."""

    def __init__(
        self,
        circuit_provider: Callable[[], Circuit] | None = None,
        parent: QWidget | None = None,
        *,
        circuit: Circuit | None = None,
    ) -> None:
        super().__init__(parent)
        if circuit_provider is None and circuit is not None:
            circuit_provider = lambda: circuit  # noqa: E731
        self._circuit_provider = circuit_provider
        self.last_sweep_result: ParameterSweepResult | None = None
        self.last_mc_result: MonteCarloResult | None = None
        self._comp_params_map: dict[str, list[tuple[str, str]]] = {}

        self.setWindowTitle("Parameter Sweep (.STEP) & Monte Carlo Analyzer")
        self.setMinimumSize(760, 580)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, 1)

        self._build_sweep_tab()
        self._build_mc_tab()
        self.refresh_circuit()

    def _get_circuit(self) -> Circuit:
        if self._circuit_provider is None:
            raise RuntimeError("No circuit provider configured.")
        return self._circuit_provider()

    def _build_sweep_tab(self) -> None:
        tab = QWidget(self.tabs)
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)

        ctrl_box = QGroupBox("Sweep Settings (.STEP)", tab)
        form = QFormLayout(ctrl_box)

        row1 = QHBoxLayout()
        self.component_combo = QComboBox(ctrl_box)
        self.component_combo.currentIndexChanged.connect(self._on_sweep_component_changed)
        row1.addWidget(self.component_combo, 1)
        row1.addWidget(QLabel("Parameter:"))
        self.param_combo = QComboBox(ctrl_box)
        self.param_combo.currentIndexChanged.connect(self._on_sweep_param_changed)
        row1.addWidget(self.param_combo, 1)
        form.addRow("Component:", row1)

        row2 = QHBoxLayout()
        self.start_edit = QLineEdit("500", ctrl_box)
        row2.addWidget(self.start_edit)
        row2.addWidget(QLabel("Stop:"))
        self.stop_edit = QLineEdit("5k", ctrl_box)
        row2.addWidget(self.stop_edit)
        row2.addWidget(QLabel("Points:"))
        self.points_spin = QSpinBox(ctrl_box)
        self.points_spin.setRange(2, 500)
        self.points_spin.setValue(9)
        row2.addWidget(self.points_spin)
        row2.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox(ctrl_box)
        self.scale_combo.addItems(["Linear", "Log"])
        row2.addWidget(self.scale_combo)
        form.addRow("Start:", row2)

        row3 = QHBoxLayout()
        self.target_node_combo = QComboBox(ctrl_box)
        row3.addWidget(self.target_node_combo, 1)
        self.run_sweep_btn = QPushButton("Run Sweep", ctrl_box)
        self.run_sweep_btn.clicked.connect(self.run_parameter_sweep_now)
        row3.addWidget(self.run_sweep_btn)
        self.export_sweep_csv_btn = QPushButton("Export CSV...", ctrl_box)
        self.export_sweep_csv_btn.clicked.connect(self._on_export_sweep_csv)
        row3.addWidget(self.export_sweep_csv_btn)
        form.addRow("Probe Node:", row3)

        lay.addWidget(ctrl_box)

        self.sweep_plot = _SweepCurveCanvas(tab)
        lay.addWidget(self.sweep_plot, 1)

        self.sweep_table = QTableWidget(0, 2, tab)
        self.sweep_table.setHorizontalHeaderLabels(["Parameter Value", "V(Node) (V)"])
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sweep_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sweep_table.setMaximumHeight(165)
        lay.addWidget(self.sweep_table)

        self.tabs.addTab(tab, "Parameter Sweep (.STEP)")

    def _build_mc_tab(self) -> None:
        tab = QWidget(self.tabs)
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)

        ctrl_box = QGroupBox("Monte Carlo Tolerance Settings", tab)
        form = QFormLayout(ctrl_box)

        row1 = QHBoxLayout()
        self.default_tol_spin = QDoubleSpinBox(ctrl_box)
        self.default_tol_spin.setRange(0.01, 50.0)
        self.default_tol_spin.setValue(5.0)
        self.default_tol_spin.setSuffix(" %")
        row1.addWidget(self.default_tol_spin)

        row1.addWidget(QLabel("Trials:"))
        self.trials_spin = QSpinBox(ctrl_box)
        self.trials_spin.setRange(10, 500)
        self.trials_spin.setValue(50)
        row1.addWidget(self.trials_spin)

        row1.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox(ctrl_box)
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        row1.addWidget(self.seed_spin)

        row1.addWidget(QLabel("Distribution:"))
        self.mc_dist_combo = QComboBox(ctrl_box)
        self.mc_dist_combo.addItems(["normal", "uniform"])
        row1.addWidget(self.mc_dist_combo)
        form.addRow("Default Tol:", row1)

        row2 = QHBoxLayout()
        self.mc_target_node_combo = QComboBox(ctrl_box)
        row2.addWidget(self.mc_target_node_combo, 1)
        self.run_mc_btn = QPushButton("Run Monte Carlo", ctrl_box)
        self.run_mc_btn.clicked.connect(self.run_monte_carlo_now)
        row2.addWidget(self.run_mc_btn)
        self.export_mc_csv_btn = QPushButton("Export CSV...", ctrl_box)
        self.export_mc_csv_btn.clicked.connect(self._on_export_mc_csv)
        row2.addWidget(self.export_mc_csv_btn)
        form.addRow("Probe Node:", row2)

        lay.addWidget(ctrl_box)

        self.mc_tol_table = QTableWidget(0, 3, tab)
        self.mc_tol_table.setHorizontalHeaderLabels(["Component", "Parameter", "Tolerance (%)"])
        self.mc_tol_table.verticalHeader().setVisible(False)
        self.mc_tol_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mc_tol_table.setMaximumHeight(130)
        lay.addWidget(self.mc_tol_table)

        self.mc_histogram = _HistogramCanvas(tab)
        lay.addWidget(self.mc_histogram, 1)

        stats_row = QHBoxLayout()
        self.mc_nominal_label = QLabel("Nominal: --", tab)
        self.mc_mean_label = QLabel("Mean: --", tab)
        self.mc_std_label = QLabel("StdDev: --", tab)
        self.mc_min_label = QLabel("Min: --", tab)
        self.mc_max_label = QLabel("Max: --", tab)
        for lbl in (
            self.mc_nominal_label,
            self.mc_mean_label,
            self.mc_std_label,
            self.mc_min_label,
            self.mc_max_label,
        ):
            stats_row.addWidget(lbl)
        lay.addLayout(stats_row)

        self.tabs.addTab(tab, "Monte Carlo Tolerance")

    def refresh_circuit(self) -> None:
        """Populate component, parameter, tolerance, and target-node dropdowns."""
        try:
            circ = self._get_circuit()
        except Exception:
            return

        self._comp_params_map.clear()
        self.component_combo.blockSignals(True)
        prev_comp = self.component_combo.currentData()
        self.component_combo.clear()
        for comp in circ.components:
            if isinstance(comp, Ground):
                continue
            params = [(p.key, p.label) for p in comp.get_editable_parameters()]
            if params:
                self._comp_params_map[comp.name] = params
                self.component_combo.addItem(f"{comp.name} ({comp.component_type})", comp.name)
        if prev_comp:
            idx = self.component_combo.findData(prev_comp)
            if idx >= 0:
                self.component_combo.setCurrentIndex(idx)
        self.component_combo.blockSignals(False)
        self._on_sweep_component_changed()

        nodes = sorted(circ.non_ground_nodes())
        for combo in (self.target_node_combo, self.mc_target_node_combo):
            prev_n = combo.currentText()
            combo.clear()
            combo.addItems(nodes)
            if prev_n in nodes:
                combo.setCurrentText(prev_n)
            elif nodes:
                combo.setCurrentIndex(len(nodes) - 1)

        # Populate per-component default tolerances
        tol_rows: list[tuple[str, str, str]] = []
        for comp in circ.components:
            if isinstance(comp, (Resistor, Potentiometer)):
                tol_rows.append((comp.name, "resistance", "5.0"))
            elif isinstance(comp, Capacitor):
                tol_rows.append((comp.name, "capacitance", "10.0"))
            elif isinstance(comp, Inductor):
                tol_rows.append((comp.name, "inductance", "10.0"))
        self.mc_tol_table.setRowCount(len(tol_rows))
        for r, (cname, pkey, tol_str) in enumerate(tol_rows):
            it0 = QTableWidgetItem(cname)
            it0.setFlags(it0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it1 = QTableWidgetItem(pkey)
            it1.setFlags(it1.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it2 = QTableWidgetItem(tol_str)
            self.mc_tol_table.setItem(r, 0, it0)
            self.mc_tol_table.setItem(r, 1, it1)
            self.mc_tol_table.setItem(r, 2, it2)

    def _on_sweep_component_changed(self) -> None:
        cname = str(self.component_combo.currentData() or "")
        params = self._comp_params_map.get(cname, [])
        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        for key, label in params:
            self.param_combo.addItem(f"{label} ({key})", key)
        self.param_combo.blockSignals(False)
        self._on_sweep_param_changed()

    def _on_sweep_param_changed(self) -> None:
        cname = str(self.component_combo.currentData() or "")
        pkey = str(self.param_combo.currentData() or "")
        if not cname or not pkey:
            return
        try:
            circ = self._get_circuit()
            comp = circ.get_component(cname)
            val = float(getattr(comp, pkey))
        except Exception:
            return
        if val > 0.0:
            self.start_edit.setText(f"{val * 0.5:g}")
            self.stop_edit.setText(f"{val * 2.0:g}")

    def run_parameter_sweep_now(self) -> ParameterSweepResult | None:
        """Execute `.STEP` DC parameter sweep and update plot + table."""
        circ = self._get_circuit()
        cname = str(self.component_combo.currentData() or "")
        pkey = str(self.param_combo.currentData() or "")
        node = self.target_node_combo.currentText().strip()
        if not cname or not pkey or not node:
            return None

        start_v = parse_engineering_value(self.start_edit.text().strip())
        stop_v = parse_engineering_value(self.stop_edit.text().strip())
        n_pts = int(self.points_spin.value())
        is_log = self.scale_combo.currentText().strip().lower() == "log"

        if is_log:
            if start_v <= 0.0 or stop_v <= 0.0:
                raise ValueError("Logarithmic sweep requires positive Start and Stop values.")
            vals = np.geomspace(start_v, stop_v, n_pts)
        else:
            vals = np.linspace(start_v, stop_v, n_pts)

        res = run_parameter_sweep(
            circ,
            component_name=cname,
            parameter=pkey,
            values=vals,
            mode="dc",
            probe_node=node,
        )
        self.last_sweep_result = res

        self.sweep_plot.set_data(
            res.values,
            res.metrics,
            x_label=f"{cname}.{pkey}",
            y_label=f"V({node})",
            log_x=is_log,
        )
        self.sweep_table.setRowCount(len(res.values))
        for idx, (xv, yv) in enumerate(zip(res.values, res.metrics)):
            self.sweep_table.setItem(idx, 0, QTableWidgetItem(f"{float(xv):g}"))
            self.sweep_table.setItem(idx, 1, QTableWidgetItem(f"{float(yv):.6g}"))
        return res

    def run_monte_carlo_now(self) -> MonteCarloResult | None:
        """Execute seeded Monte Carlo trials and update histogram + statistics."""
        circ = self._get_circuit()
        node = self.mc_target_node_combo.currentText().strip()
        if not node:
            return None

        tols: list[MonteCarloTolerance] = []
        dist = self.mc_dist_combo.currentText().strip().lower()
        for r in range(self.mc_tol_table.rowCount()):
            c_it = self.mc_tol_table.item(r, 0)
            p_it = self.mc_tol_table.item(r, 1)
            t_it = self.mc_tol_table.item(r, 2)
            if c_it is None or p_it is None or t_it is None:
                continue
            try:
                pct = float(t_it.text())
            except ValueError:
                pct = float(self.default_tol_spin.value())
            tols.append(
                MonteCarloTolerance(
                    component_name=c_it.text(),
                    parameter=p_it.text(),
                    tolerance_pct=max(0.0, pct),
                    distribution=dist,
                )
            )

        res = run_monte_carlo(
            circ,
            tolerances=tols if tols else None,
            default_tolerance_pct=float(self.default_tol_spin.value()),
            distribution=dist,
            num_trials=int(self.trials_spin.value()),
            seed=int(self.seed_spin.value()),
            mode="dc",
            probe_node=node,
        )
        self.last_mc_result = res

        self.mc_histogram.set_data(res.trial_metrics, res.nominal_metric, res.mean)
        self.mc_nominal_label.setText(f"Nominal: {res.nominal_metric:.4f} V")
        self.mc_mean_label.setText(f"Mean: {res.mean:.4f} V")
        self.mc_std_label.setText(f"StdDev: {format_engineering_value(res.std, 'V')}")
        self.mc_min_label.setText(f"Min: {res.min_val:.4f} V")
        self.mc_max_label.setText(f"Max: {res.max_val:.4f} V")
        return res

    def export_sweep_csv(self, file_path: str | Path) -> Path:
        """Write last `.STEP` sweep table to CSV and return `Path`."""
        if self.last_sweep_result is None:
            raise ValueError("No parameter sweep results available to export.")
        out = Path(file_path)
        res = self.last_sweep_result
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([f"{res.component_name}.{res.parameter}", f"V({res.probe_node or 'node'})"])
            for xv, yv in zip(res.values, res.metrics):
                writer.writerow([f"{float(xv):.9g}", f"{float(yv):.9g}"])
        return out

    def export_monte_carlo_csv(self, file_path: str | Path) -> Path:
        """Write last Monte Carlo trial metrics to CSV and return `Path`."""
        if self.last_mc_result is None:
            raise ValueError("No Monte Carlo results available to export.")
        out = Path(file_path)
        res = self.last_mc_result
        node = self.mc_target_node_combo.currentText().strip() or "node"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["trial", f"V({node})"])
            for idx, val in enumerate(res.trial_metrics):
                writer.writerow([idx + 1, f"{float(val):.9g}"])
        return out

    def _on_export_sweep_csv(self) -> None:
        if self.last_sweep_result is None:
            QMessageBox.information(self, "Export CSV", "Run a parameter sweep first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Sweep CSV", "sweep.csv", "CSV Files (*.csv)")
        if path:
            self.export_sweep_csv(path)

    def _on_export_mc_csv(self) -> None:
        if self.last_mc_result is None:
            QMessageBox.information(self, "Export CSV", "Run Monte Carlo analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Monte Carlo CSV", "monte_carlo.csv", "CSV Files (*.csv)")
        if path:
            self.export_monte_carlo_csv(path)
