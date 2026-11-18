"""Import 3rd-party SPICE .MODEL / .SUBCKT files or text with explicit pin-order mapping."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components import Diode
from circuitsim.io.spice_import import (
    EncryptedModelError,
    SpiceImportError,
    SpiceLibrary,
    SpiceModelDef,
    SpiceSubcktDef,
    instantiate_diode_model,
    map_subckt_pins,
    parse_spice_file,
    parse_spice_library,
)

_CANONICAL_TERMINAL_CHOICES: tuple[str, ...] = (
    "anode",
    "cathode",
    "a",
    "b",
    "pos",
    "neg",
    "in",
    "out",
    "w",
)


class ImportSPICEDialog(QDialog):
    """Dialog for loading `.MODEL`/`.SUBCKT` libraries with pin-order mapper & encryption guard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import 3rd-Party SPICE Model / Subcircuit")
        self.resize(620, 480)

        self.library: SpiceLibrary | None = None
        self.selected_diode: Diode | None = None
        self.selected_subckt: SpiceSubcktDef | None = None
        self.selected_pin_mapping: dict[str, str] = {}
        self.last_error: str = ""

        root = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_browse = QPushButton("Load SPICE File...")
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_parse = QPushButton("Parse SPICE Text")
        self.btn_parse.clicked.connect(self._on_parse_text)
        top_bar.addWidget(self.btn_browse)
        top_bar.addWidget(self.btn_parse)
        top_bar.addStretch(1)
        root.addLayout(top_bar)

        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setPlaceholderText(
            "Paste .MODEL D1N4148 D(IS=2.52n N=1.752 RS=0.568 BV=100 IBV=100u) or .SUBCKT ..."
        )
        root.addWidget(self.text_edit, 2)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Parsed Entry:"))
        self.entry_combo = QComboBox(self)
        self.entry_combo.currentIndexChanged.connect(self._on_entry_selected)
        sel_row.addWidget(self.entry_combo, 1)
        root.addLayout(sel_row)

        root.addWidget(QLabel("Pin-Order Mapper (.SUBCKT SPICE pin -> Ohmbench Terminal):"))
        self.pin_table = QTableWidget(0, 2, self)
        self.pin_table.setHorizontalHeaderLabels(["SPICE Pin", "Canonical Terminal"])
        self.pin_table.verticalHeader().setVisible(False)
        self.pin_table.setMaximumHeight(130)
        root.addWidget(self.pin_table)

        self.status_label = QLabel("Load a file or paste SPICE directives to inspect.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SPICE Library",
            "",
            "SPICE Files (*.lib *.mod *.cir *.sp *.txt);;All Files (*.*)",
        )
        if not path:
            return
        try:
            lib = parse_spice_file(path)
        except EncryptedModelError as exc:
            self._set_error(str(exc))
            return
        except Exception as exc:
            self._set_error(f"Could not read SPICE file: {exc}")
            return
        from pathlib import Path

        self.text_edit.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
        self._apply_library(lib)

    def _on_parse_text(self) -> None:
        self.parse_text(self.text_edit.toPlainText())

    def parse_text(self, text: str) -> bool:
        """Parse SPICE text into the dialog UI; return True on success, False on error."""
        self.text_edit.blockSignals(True)
        if self.text_edit.toPlainText() != text:
            self.text_edit.setPlainText(text)
        self.text_edit.blockSignals(False)
        try:
            lib = parse_spice_library(text)
        except (EncryptedModelError, SpiceImportError) as exc:
            self._set_error(str(exc))
            return False
        self._apply_library(lib)
        return True

    def _set_error(self, msg: str) -> None:
        self.last_error = msg
        self.library = None
        self.entry_combo.clear()
        self.pin_table.setRowCount(0)
        self.status_label.setStyleSheet("color: #FF6B6B; font-weight: 600;")
        self.status_label.setText(msg)

    def _apply_library(self, lib: SpiceLibrary) -> None:
        self.last_error = ""
        self.library = lib
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        for m in lib.models.values():
            self.entry_combo.addItem(f".MODEL {m.name} ({m.model_type})", ("model", m.name))
        for s in lib.subckts.values():
            self.entry_combo.addItem(f".SUBCKT {s.name} ({', '.join(s.pins)})", ("subckt", s.name))
        self.entry_combo.blockSignals(False)

        self.status_label.setStyleSheet("color: #6CCB5F;")
        self.status_label.setText(
            f"Parsed {len(lib.models)} .MODEL(s) and {len(lib.subckts)} .SUBCKT(s)."
        )
        self._on_entry_selected()

    def _on_entry_selected(self) -> None:
        self.pin_table.setRowCount(0)
        if self.library is None:
            return
        data = self.entry_combo.currentData()
        if not data:
            return
        kind, name = data
        if kind == "model":
            mdef: SpiceModelDef = self.library.models[name]
            self.status_label.setText(
                f"Model {mdef.name} ({mdef.model_type}): "
                + ", ".join(f"{k}={v:.4g}" for k, v in mdef.params.items())
            )
            return

        subckt: SpiceSubcktDef = self.library.subckts[name]
        self.pin_table.setRowCount(len(subckt.pins))
        for row_idx, pin_name in enumerate(subckt.pins):
            item = QTableWidgetItem(pin_name)
            self.pin_table.setItem(row_idx, 0, item)
            cmb = QComboBox(self.pin_table)
            for c in _CANONICAL_TERMINAL_CHOICES:
                cmb.addItem(c, c)
            default_canon = _CANONICAL_TERMINAL_CHOICES[min(row_idx, len(_CANONICAL_TERMINAL_CHOICES) - 1)]
            if pin_name.upper().startswith("K"):
                default_canon = "cathode"
            elif pin_name.upper().startswith("A"):
                default_canon = "anode"
            cmb.setCurrentText(default_canon)
            self.pin_table.setCellWidget(row_idx, 1, cmb)

    def _collect_pin_mapping(self, subckt: SpiceSubcktDef) -> dict[str, str]:
        raw_map: dict[str, str] = {}
        for row_idx, pin in enumerate(subckt.pins):
            w = self.pin_table.cellWidget(row_idx, 1)
            if isinstance(w, QComboBox):
                raw_map[pin] = w.currentText()
        return map_subckt_pins(subckt, raw_map)

    def _on_accept(self) -> None:
        if self.library is None:
            self.parse_text(self.text_edit.toPlainText())
        if self.library is None:
            return
        data = self.entry_combo.currentData()
        if not data:
            return
        kind, name = data
        if kind == "model":
            mdef = self.library.models[name]
            try:
                self.selected_diode = instantiate_diode_model(mdef, mdef.name, "anode", "cathode")
            except SpiceImportError as exc:
                self._set_error(str(exc))
                return
            self.selected_subckt = None
            self.accept()
            return

        subckt = self.library.subckts[name]
        try:
            self.selected_pin_mapping = self._collect_pin_mapping(subckt)
        except SpiceImportError as exc:
            self._set_error(str(exc))
            return
        self.selected_subckt = subckt
        self.selected_diode = None
        self.accept()
