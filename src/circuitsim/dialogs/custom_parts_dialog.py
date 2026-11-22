"""Dialogs for saving components to and managing the persistent `My Custom Parts` library."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components.base import Component
from circuitsim.io.user_library import (
    CustomPartEntry,
    UserLibraryStore,
    create_preset_from_component,
)


class SaveCustomPartDialog(QDialog):
    """Prompt for Name & Description when saving a schematic component as a Custom Part."""

    def __init__(
        self,
        component: Component,
        user_library: UserLibraryStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.component = component
        self.user_library = user_library
        self.saved_entry: CustomPartEntry | None = None

        self.setWindowTitle("Save to My Custom Parts")
        self.setMinimumWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        intro = QLabel(
            f"Save <b>{component.name}</b> ({component.component_type}) as a reusable custom part:"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("custom_part_name_edit")
        self.name_edit.setText(f"{component.name} Custom")
        self.desc_edit = QLineEdit(self)
        self.desc_edit.setObjectName("custom_part_desc_edit")
        self.desc_edit.setPlaceholderText("Optional description or notes")
        form.addRow("Part Name:", self.name_edit)
        form.addRow("Description:", self.desc_edit)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def save_and_accept(self) -> CustomPartEntry | None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a non-empty part name.")
            return None
        entry = create_preset_from_component(
            self.component,
            name=name,
            description=self.desc_edit.text().strip(),
        )
        self.saved_entry = self.user_library.add_or_update_part(entry)
        self.accept()
        return self.saved_entry


class ManageCustomPartsDialog(QDialog):
    """Manage, rename, delete, place, export, and import `My Custom Parts`."""

    def __init__(
        self,
        user_library: UserLibraryStore,
        *,
        on_library_changed: Callable[[], None] | None = None,
        on_place_part: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.user_library = user_library
        self._on_library_changed = on_library_changed
        self._on_place_part = on_place_part

        self.setWindowTitle("Manage Custom Parts Library")
        self.resize(560, 400)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        path_lbl = QLabel(f"Library file: <code>{self.user_library.storage_path}</code>")
        path_lbl.setWordWrap(True)
        root.addWidget(path_lbl)

        self.parts_list = QListWidget(self)
        self.parts_list.setObjectName("custom_parts_list")
        root.addWidget(self.parts_list, 1)

        btn_row = QHBoxLayout()
        self.btn_place = QPushButton("Place on Canvas")
        self.btn_rename = QPushButton("Rename...")
        self.btn_delete = QPushButton("Delete")
        self.btn_export = QPushButton("Export JSON...")
        self.btn_import = QPushButton("Import JSON...")
        self.btn_close = QPushButton("Close")

        for b in (
            self.btn_place,
            self.btn_rename,
            self.btn_delete,
            self.btn_export,
            self.btn_import,
        ):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        self.btn_place.clicked.connect(self.place_selected)
        self.btn_rename.clicked.connect(self._prompt_rename_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_export.clicked.connect(self._prompt_export)
        self.btn_import.clicked.connect(self._prompt_import)
        self.btn_close.clicked.connect(self.close)

        self.refresh_list()

    def refresh_list(self) -> None:
        self.parts_list.clear()
        for entry in self.user_library.list_parts():
            kind_tag = "Macro" if entry.kind == "macro" else f"Preset: {entry.base_component_type}"
            desc_part = f" — {entry.description}" if entry.description else ""
            item = QListWidgetItem(f"{entry.name}  [{kind_tag}]{desc_part}")
            item.setData(Qt.ItemDataRole.UserRole, entry.part_id)
            self.parts_list.addItem(item)
        if self.parts_list.count() > 0:
            self.parts_list.setCurrentRow(0)

    def selected_part_id(self) -> str | None:
        item = self.parts_list.currentItem()
        if item is None:
            return None
        val = item.data(Qt.ItemDataRole.UserRole)
        return str(val) if val else None

    def place_selected(self) -> bool:
        pid = self.selected_part_id()
        if not pid or self._on_place_part is None:
            return False
        self._on_place_part(f"custom:{pid}")
        return True

    def rename_selected(self, new_name: str) -> bool:
        pid = self.selected_part_id()
        if not pid:
            return False
        ok = self.user_library.rename_part(pid, new_name)
        if ok:
            self.refresh_list()
            if self._on_library_changed is not None:
                self._on_library_changed()
        return ok

    def delete_selected(self) -> bool:
        pid = self.selected_part_id()
        if not pid:
            return False
        ok = self.user_library.delete_part(pid)
        if ok:
            self.refresh_list()
            if self._on_library_changed is not None:
                self._on_library_changed()
        return ok

    def export_to_path(self, path: str) -> None:
        self.user_library.export_library(path)

    def import_from_path(self, path: str) -> list[CustomPartEntry]:
        entries = self.user_library.import_library(path)
        self.refresh_list()
        if self._on_library_changed is not None:
            self._on_library_changed()
        return entries

    def _prompt_rename_selected(self) -> None:
        pid = self.selected_part_id()
        if not pid:
            return
        entry = self.user_library.get_part(pid)
        cur_name = entry.name if entry is not None else ""
        new_name, ok = QInputDialog.getText(
            self, "Rename Custom Part", "New part name:", text=cur_name
        )
        if ok and new_name.strip():
            self.rename_selected(new_name.strip())

    def _prompt_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Custom Parts Library", "custom_parts.json", "JSON Files (*.json)"
        )
        if path:
            self.export_to_path(path)

    def _prompt_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Custom Parts Library", "", "JSON Files (*.json)"
        )
        if path:
            try:
                self.import_from_path(path)
            except Exception as exc:
                QMessageBox.warning(self, "Import Failed", str(exc))
