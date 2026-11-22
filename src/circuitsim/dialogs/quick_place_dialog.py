"""Spacebar Quick-Place searchable component picker popup."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from circuitsim.canvas.symbol_renderer import SymbolLibrary

_SEARCH_ALIASES: dict[str, str] = {
    "function_generator": "fg funcgen wave sine square triangle sawtooth generator",
    "oscilloscope": "scope dso 1062 waveform",
    "potentiometer": "pot trim wiper divider",
    "ne555": "555 timer astable monostable",
    "logic_probe": "probe digital high low logic",
    "gate_not": "inverter not",
    "gate_nand": "nand logic gate",
    "gate_and": "and logic gate",
    "gate_or": "or logic gate",
    "gate_nor": "nor logic gate",
    "gate_xor": "xor logic gate",
    "gate_xnor": "xnor logic gate",
    "dc_voltage": "battery vdc supply source",
    "ac_voltage": "vac sine source",
    "spdt_relay": "relay coil switch",
}


class QuickPlaceDialog(QDialog):
    """Compact searchable popup invoked via `Space` to place parts at the cursor."""

    def __init__(
        self,
        symbol_library: SymbolLibrary,
        *,
        user_library: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.symbol_library = symbol_library
        self.user_library = user_library
        self._entries: list[tuple[str, str, str, str]] = []  # (id, display, category, search_blob)
        self._selected_id: str | None = None

        self.setWindowTitle("Quick Place Component")
        self.setMinimumSize(QSize(420, 380))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hint = QLabel("Type to filter built-in or custom parts, then press Enter to place:")
        root.addWidget(hint)

        self.search_input = QLineEdit(self)
        self.search_input.setObjectName("quick_place_search")
        self.search_input.setPlaceholderText("e.g. nand, resistor, fg, 555, probe, zener...")
        self.search_input.installEventFilter(self)
        root.addWidget(self.search_input)

        self.results_list = QListWidget(self)
        self.results_list.setObjectName("quick_place_results")
        self.results_list.setIconSize(QSize(32, 18))
        root.addWidget(self.results_list, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_catalog()
        self._populate_list("")

        self.search_input.textChanged.connect(self._populate_list)
        self.results_list.itemDoubleClicked.connect(lambda _it: self.accept())
        self.search_input.returnPressed.connect(self.accept)
        self.search_input.setFocus()

    def _load_catalog(self) -> None:
        self._entries.clear()
        if self.user_library is not None and hasattr(self.user_library, "list_parts"):
            for entry in self.user_library.list_parts():
                part_id = f"custom:{entry.part_id}"
                blob = " ".join(
                    (
                        entry.name,
                        entry.part_id,
                        entry.base_component_type,
                        entry.description,
                        "my custom parts",
                    )
                ).lower()
                self._entries.append((part_id, entry.name, "My Custom Parts", blob))

        for comp_id, meta in sorted(
            self.symbol_library.metadata_catalog.items(),
            key=lambda kv: (kv[1].display_name.lower(), kv[0]),
        ):
            aliases = _SEARCH_ALIASES.get(comp_id, "")
            blob = f"{meta.display_name} {comp_id} {meta.category} {aliases}".lower()
            self._entries.append((comp_id, meta.display_name, meta.category, blob))

    def _populate_list(self, query: str) -> None:
        needle = query.strip().lower()
        tokens = [tok for tok in needle.split() if tok]
        self.results_list.clear()

        for comp_id, disp_name, category, blob in self._entries:
            if tokens and not all(tok in blob for tok in tokens):
                continue
            item = QListWidgetItem(f"{disp_name}  [{category}]  ({comp_id})")
            item.setData(Qt.ItemDataRole.UserRole, comp_id)
            icon_id = comp_id
            if comp_id.startswith("custom:") and self.user_library is not None:
                entry = self.user_library.get_part(comp_id.removeprefix("custom:"))
                if entry is not None and entry.base_component_type:
                    icon_id = entry.base_component_type
                else:
                    icon_id = "opamp"
            if icon_id in self.symbol_library.metadata_catalog:
                item.setIcon(self.symbol_library.create_palette_icon(icon_id, size=(32, 18)))
            self.results_list.addItem(item)

        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is self.search_input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.results_list.count() > 0:
                cur = self.results_list.currentRow()
                nxt = (
                    min(self.results_list.count() - 1, cur + 1)
                    if key == Qt.Key.Key_Down
                    else max(0, cur - 1)
                )
                self.results_list.setCurrentRow(nxt)
                return True
        return super().eventFilter(watched, event)

    def accept(self) -> None:
        cur = self.results_list.currentItem()
        if cur is not None:
            self._selected_id = str(cur.data(Qt.ItemDataRole.UserRole))
        elif self.results_list.count() > 0:
            self._selected_id = str(self.results_list.item(0).data(Qt.ItemDataRole.UserRole))
        else:
            self._selected_id = None
        super().accept()

    def selected_component_id(self) -> str | None:
        cur = self.results_list.currentItem()
        if self._selected_id is not None:
            return self._selected_id
        if cur is not None:
            return str(cur.data(Qt.ItemDataRole.UserRole))
        return None
