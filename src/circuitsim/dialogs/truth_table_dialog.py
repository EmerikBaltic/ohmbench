"""Truth-Table Generator dialog (Simulation > Truth-Table Generator...)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from circuitsim.engine.truth_table import (
    TruthTableResult,
    detect_truth_table_candidates,
    generate_truth_table,
)


class TruthTableDialog(QDialog):
    """Interactive dialog for generating, viewing, and exporting circuit truth tables."""

    def __init__(
        self,
        circuit_provider: Callable[[], object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._circuit_provider = circuit_provider
        self._result: TruthTableResult | None = None
        self._input_checks: list[QCheckBox] = []
        self._output_checks: list[QCheckBox] = []

        self.setWindowTitle("Truth-Table Generator")
        self.resize(620, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        info_lbl = QLabel(
            "Exhaustively simulates all binary input combinations (up to 6 inputs) "
            "across switches/DC sources and reads outputs from Logic Probes, meters, or gates."
        )
        info_lbl.setWordWrap(True)
        root.addWidget(info_lbl)

        # Selection lists
        sel_row = QHBoxLayout()
        self.in_group = QGroupBox("Inputs (max 6)")
        self.in_layout = QVBoxLayout(self.in_group)
        self.out_group = QGroupBox("Outputs")
        self.out_layout = QVBoxLayout(self.out_group)
        sel_row.addWidget(self.in_group, 1)
        sel_row.addWidget(self.out_group, 1)
        root.addLayout(sel_row)

        # Action bar
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Detect I/O")
        self.btn_generate = QPushButton("Generate Truth Table")
        self.btn_generate.setDefault(True)
        self.btn_copy_md = QPushButton("Copy Markdown")
        self.btn_export_csv = QPushButton("Export CSV...")
        self.btn_close = QPushButton("Close")
        for b in (
            self.btn_refresh,
            self.btn_generate,
            self.btn_copy_md,
            self.btn_export_csv,
        ):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        self.btn_refresh.clicked.connect(self.refresh_candidates)
        self.btn_generate.clicked.connect(self.run_generation)
        self.btn_copy_md.clicked.connect(self._copy_markdown)
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_close.clicked.connect(self.close)

        self.table = QTableWidget(self)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

        self.refresh_candidates()

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def refresh_candidates(self) -> None:
        """Scan current schematic for candidate binary inputs and outputs."""
        self._clear_layout(self.in_layout)
        self._clear_layout(self.out_layout)
        self._input_checks.clear()
        self._output_checks.clear()

        try:
            target = self._circuit_provider()
            in_names, out_names = detect_truth_table_candidates(target)
        except Exception as exc:
            self.status_label.setText(f"Could not inspect circuit: {exc}")
            return

        for idx, name in enumerate(in_names):
            cb = QCheckBox(name, self.in_group)
            cb.setChecked(idx < 6)
            self._input_checks.append(cb)
            self.in_layout.addWidget(cb)
        if not in_names:
            self.in_layout.addWidget(QLabel("(No switches or DC sources found)"))
        self.in_layout.addStretch(1)

        for name in out_names:
            cb = QCheckBox(name, self.out_group)
            cb.setChecked(True)
            self._output_checks.append(cb)
            self.out_layout.addWidget(cb)
        if not out_names:
            self.out_layout.addWidget(QLabel("(No Logic Probes or outputs found)"))
        self.out_layout.addStretch(1)

    def run_generation(self) -> TruthTableResult | None:
        """Simulate all selected combinations and populate the table widget."""
        sel_in = [cb.text() for cb in self._input_checks if cb.isChecked()]
        sel_out = [cb.text() for cb in self._output_checks if cb.isChecked()]
        try:
            target = self._circuit_provider()
            res = generate_truth_table(
                target,
                input_names=sel_in or None,
                output_names=sel_out or None,
            )
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
            return None

        self._result = res
        headers = [*res.input_names, *res.output_names]
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(res.rows))

        n_in = len(res.input_names)
        for r_idx, row in enumerate(res.rows):
            for c_idx, bit in enumerate(row.inputs):
                item = QTableWidgetItem(str(bit))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r_idx, c_idx, item)
            for o_idx, val in enumerate(row.outputs):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if val == "1":
                    item.setForeground(QColor("#22C55E"))
                elif val == "0":
                    item.setForeground(QColor("#38BDF8"))
                else:
                    item.setForeground(QColor("#F59E0B"))
                self.table.setItem(r_idx, n_in + o_idx, item)

        self.table.resizeColumnsToContents()
        self.status_label.setText(
            f"Generated {len(res.rows)} rows ({len(res.input_names)} inputs -> {len(res.output_names)} outputs)."
        )
        return res

    def _copy_markdown(self) -> None:
        if self._result is None:
            self.run_generation()
        if self._result is None:
            return
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self._result.to_markdown())
        self.status_label.setText("Copied Markdown truth table to clipboard.")

    def _export_csv(self) -> None:
        if self._result is None:
            self.run_generation()
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Truth Table CSV", "truth_table.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(self._result.to_csv())
            self.status_label.setText(f"Exported CSV to {path}")
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
