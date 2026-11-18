"""Interactive MCU Firmware Python Editor & Live Serial / Pin Monitor Dialog (H5)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components.mcu import MCUBlock, compile_firmware_code


class MCUFirmwareDialog(QDialog):
    """Modal editor and live telemetry inspector for an `MCUBlock`."""

    def __init__(
        self,
        mcu_item_or_model: Any,
        scene: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene = scene
        if isinstance(mcu_item_or_model, MCUBlock):
            self._item = None
            self.mcu: MCUBlock = mcu_item_or_model
        else:
            self._item = mcu_item_or_model
            comp = getattr(mcu_item_or_model, "component", None)
            if not isinstance(comp, MCUBlock):
                raise TypeError(f"Expected MCUBlock or ComponentItem wrapping MCUBlock, got {type(mcu_item_or_model)!r}")
            self.mcu = comp

        self.setWindowTitle(f"{self.mcu.name} — MCU Firmware & Serial Monitor")
        self.setMinimumSize(780, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        header = QLabel(
            f"<b>{self.mcu.name}</b> — Python Firmware (`mcu.adc_read('A0')`, "
            "`mcu.digital_write('D0', True)`, `mcu.log('...')`)"
        )
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter, 1)

        # Left pane: firmware code editor + Verify/Compile bar
        left_widget = QWidget(splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel("Firmware Script (Python):"))
        self.code_edit = QPlainTextEdit(left_widget)
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.code_edit.setFont(mono)
        self.code_edit.setPlainText(self.mcu.code)
        left_layout.addWidget(self.code_edit, 1)

        verify_row = QHBoxLayout()
        self.verify_btn = QPushButton("Verify / Compile", left_widget)
        self.verify_btn.clicked.connect(self.verify_code)
        verify_row.addWidget(self.verify_btn)

        self.status_label = QLabel("Ready", left_widget)
        self.status_label.setWordWrap(True)
        verify_row.addWidget(self.status_label, 1)
        left_layout.addLayout(verify_row)

        # Right pane: 6-row live Pin Table + Read-only Serial Monitor
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(6)

        pin_hdr_row = QHBoxLayout()
        pin_hdr_row.addWidget(QLabel("Live Pin State Table:"))
        pin_hdr_row.addStretch(1)
        self.refresh_btn = QPushButton("Refresh", right_widget)
        self.refresh_btn.clicked.connect(self.refresh_telemetry)
        pin_hdr_row.addWidget(self.refresh_btn)
        right_layout.addLayout(pin_hdr_row)

        self.pin_table = QTableWidget(6, 4, right_widget)
        self.pin_table.setHorizontalHeaderLabels(["Pin", "Mode", "Voltage", "ADC / Logic"])
        self.pin_table.verticalHeader().setVisible(False)
        self.pin_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pin_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.pin_table, 0)

        ser_hdr_row = QHBoxLayout()
        ser_hdr_row.addWidget(QLabel("Serial Monitor (`mcu.log(...)`):"))
        ser_hdr_row.addStretch(1)
        self.clear_log_btn = QPushButton("Clear Log", right_widget)
        self.clear_log_btn.clicked.connect(self.clear_serial_log)
        ser_hdr_row.addWidget(self.clear_log_btn)
        right_layout.addLayout(ser_hdr_row)

        self.serial_monitor_edit = QPlainTextEdit(right_widget)
        self.serial_monitor_edit.setReadOnly(True)
        self.serial_monitor_edit.setFont(mono)
        right_layout.addWidget(self.serial_monitor_edit, 1)

        splitter.setSizes([430, 340])

        # Footer Apply / OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.apply_btn = QPushButton("Apply", self)
        self.apply_btn.clicked.connect(lambda: self.apply_code(close_after=False))
        btn_row.addWidget(self.apply_btn)

        self.ok_btn = QPushButton("OK", self)
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(lambda: self.apply_code(close_after=True))
        btn_row.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        root.addLayout(btn_row)

        self.refresh_telemetry()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self.refresh_telemetry)
        self._poll_timer.start()

    def verify_code(self) -> bool:
        """Compile code in the editor and update `status_label` without crashing."""
        src = self.code_edit.toPlainText()
        try:
            compile_firmware_code(src)
        except SyntaxError as exc:
            line_no = exc.lineno if exc.lineno is not None else "?"
            msg = f"SyntaxError: line {line_no}: {exc.msg}"
            self.status_label.setText(msg)
            self.status_label.setStyleSheet("color: #FF6B6B; font-weight: 600;")
            return False
        except Exception as exc:
            self.status_label.setText(f"Compile Error: {exc}")
            self.status_label.setStyleSheet("color: #FF6B6B; font-weight: 600;")
            return False

        self.status_label.setText("Compiled OK")
        self.status_label.setStyleSheet("color: #51CF66; font-weight: 600;")
        return True

    def refresh_telemetry(self) -> None:
        """Populate the 6-row live Pin Table and Serial Monitor from `self.mcu`."""
        rows = self.mcu.get_pin_states()
        if self.pin_table.rowCount() != len(rows):
            self.pin_table.setRowCount(len(rows))
        for r_idx, (pin, mode, volts, state_str) in enumerate(rows):
            vals = (pin, mode, f"{volts:.3f} V", state_str)
            for c_idx, text in enumerate(vals):
                cell = self.pin_table.item(r_idx, c_idx)
                if cell is None:
                    cell = QTableWidgetItem(text)
                    self.pin_table.setItem(r_idx, c_idx, cell)
                else:
                    cell.setText(text)

        log_text = "\n".join(self.mcu.serial_log)
        if self.serial_monitor_edit.toPlainText() != log_text:
            self.serial_monitor_edit.setPlainText(log_text)
            sb = self.serial_monitor_edit.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def clear_serial_log(self) -> None:
        """Clear `self.mcu.serial_log` and refresh monitor view."""
        self.mcu.serial_log.clear()
        self.serial_monitor_edit.clear()

    def apply_code(self, *, close_after: bool = False) -> bool:
        """Validate, apply firmware code to `self.mcu`, and notify scene."""
        if not self.verify_code():
            return False
        src = self.code_edit.toPlainText()
        self.mcu.set_firmware_code(src)
        if self._item is not None and hasattr(self._item, "update"):
            self._item.update()
        if self._scene is not None and hasattr(self._scene, "circuit_modified"):
            self._scene.circuit_modified.emit()
        if close_after:
            self.accept()
        return True

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)

    def reject(self) -> None:
        self._poll_timer.stop()
        super().reject()

    def accept(self) -> None:
        self._poll_timer.stop()
        super().accept()
