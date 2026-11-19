"""Preferences and programmable keyboard shortcuts dialogs."""

from __future__ import annotations

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from circuitsim.app.settings import (
    DEFAULT_SHORTCUTS,
    SHORTCUT_LABELS,
    AppSettings,
)
from circuitsim.dialogs.value_parser import set_global_number_format


def _norm_seq(seq: str | QKeySequence) -> str:
    ks = seq if isinstance(seq, QKeySequence) else QKeySequence(str(seq).strip())
    return ks.toString(QKeySequence.SequenceFormat.PortableText).strip()


class ShortcutsEditorWidget(QWidget):
    """Editable keyboard shortcut table with duplicate resolution and reset-to-defaults."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings: AppSettings = settings
        self.key_edits: dict[str, QKeySequenceEdit] = {}
        self._syncing: bool = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget(scroll)
        form = QFormLayout(inner)
        form.setSpacing(6)

        cur_shortcuts = settings.shortcuts
        for action_id, default_seq in DEFAULT_SHORTCUTS.items():
            edit = QKeySequenceEdit(QKeySequence(cur_shortcuts.get(action_id, default_seq)), inner)
            edit.setObjectName(f"shortcut_{action_id}")
            edit.keySequenceChanged.connect(
                lambda seq, aid=action_id: self._on_sequence_changed(aid, seq)
            )
            self.key_edits[action_id] = edit
            lbl = SHORTCUT_LABELS.get(action_id, action_id)
            form.addRow(f"{lbl} ({default_seq}):", edit)

        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        self.conflict_label = QLabel("", self)
        self.conflict_label.setWordWrap(True)
        self.conflict_label.setStyleSheet("color: #FFA726; font-size: 9pt;")
        outer.addWidget(self.conflict_label)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self.btn_reset_defaults = QPushButton("Reset to Defaults", self)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        footer.addWidget(self.btn_reset_defaults)
        outer.addLayout(footer)

    def _on_sequence_changed(self, action_id: str, seq: QKeySequence) -> None:
        if self._syncing:
            return
        new_str = _norm_seq(seq)
        if not new_str:
            return
        cleared_names: list[str] = []
        self._syncing = True
        try:
            for other_id, other_edit in self.key_edits.items():
                if other_id == action_id:
                    continue
                if _norm_seq(other_edit.keySequence()).lower() == new_str.lower():
                    other_edit.setKeySequence(QKeySequence())
                    cleared_names.append(SHORTCUT_LABELS.get(other_id, other_id))
        finally:
            self._syncing = False
        if cleared_names:
            target_lbl = SHORTCUT_LABELS.get(action_id, action_id)
            self.conflict_label.setText(
                f"Reassigned '{new_str}' to {target_lbl} (cleared duplicate from {', '.join(cleared_names)})."
            )
        else:
            self.conflict_label.setText("")

    def set_binding(self, action_id: str, sequence: str) -> None:
        """Programmatically set a key binding with duplicate conflict resolution."""
        edit = self.key_edits[action_id]
        seq = QKeySequence(_norm_seq(sequence))
        if _norm_seq(edit.keySequence()).lower() == _norm_seq(seq).lower():
            self._on_sequence_changed(action_id, seq)
        else:
            edit.setKeySequence(seq)

    def get_bindings(self) -> dict[str, str]:
        return {
            action_id: _norm_seq(edit.keySequence())
            for action_id, edit in self.key_edits.items()
        }

    def reset_defaults(self) -> None:
        self._syncing = True
        try:
            for action_id, default_seq in DEFAULT_SHORTCUTS.items():
                self.key_edits[action_id].setKeySequence(QKeySequence(default_seq))
        finally:
            self._syncing = False
        self.conflict_label.setText("Shortcuts restored to defaults.")

    def apply_to_settings(self) -> dict[str, str]:
        bindings = self.get_bindings()
        self.settings.set_shortcuts(bindings)
        return bindings


class ShortcutsDialog(QDialog):
    """Standalone dialog for inspecting, editing, and resetting keyboard shortcuts."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings: AppSettings = settings
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.resize(460, 480)

        layout = QVBoxLayout(self)
        self.editor = ShortcutsEditorWidget(settings, self)
        layout.addWidget(self.editor, stretch=1)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self) -> None:
        self.editor.apply_to_settings()
        self.accept()


class PreferencesDialog(QDialog):
    """Preferences dialog managing number format, meter mode, and programmable shortcuts."""

    def __init__(
        self, settings: AppSettings, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.settings: AppSettings = settings
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.resize(480, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, stretch=1)

        # Tab 0: General preferences
        general_tab = QWidget(self.tabs)
        gen_layout = QVBoxLayout(general_tab)
        gen_layout.setContentsMargins(8, 8, 8, 8)
        gen_layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)

        self.combo_number_format = QComboBox(general_tab)
        self.combo_number_format.addItem("Automatic (System locale)", "auto")
        self.combo_number_format.addItem("Dot: 4.7 kΩ", "dot")
        self.combo_number_format.addItem("Comma: 4,7 kΩ", "comma")

        current_fmt = settings.get_number_format()
        idx = self.combo_number_format.findData(current_fmt)
        if idx >= 0:
            self.combo_number_format.setCurrentIndex(idx)

        form.addRow("Number format:", self.combo_number_format)

        self.combo_meters_mode = QComboBox(general_tab)
        self.combo_meters_mode.setObjectName("combo_meters_mode")
        self.combo_meters_mode.addItem("Ideal", "ideal")
        self.combo_meters_mode.addItem("Realistic", "realistic")

        current_meter_mode = settings.get_new_meters_mode()
        idx_meter = self.combo_meters_mode.findData(current_meter_mode)
        if idx_meter >= 0:
            self.combo_meters_mode.setCurrentIndex(idx_meter)

        form.addRow("New meters are:", self.combo_meters_mode)
        gen_layout.addLayout(form)

        hint = QLabel(
            "Controls display formatting across schematic labels, meter readouts, "
            "tooltips, and dialogs. Both dot and comma are always accepted when typing inputs."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8B949E; font-size: 9pt;")
        gen_layout.addWidget(hint)
        gen_layout.addStretch(1)
        self.tabs.addTab(general_tab, "General")

        # Tab 1: Programmable keyboard shortcuts
        self.shortcuts_editor = ShortcutsEditorWidget(settings, self.tabs)
        self.tabs.addTab(self.shortcuts_editor, "Shortcuts")

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self) -> None:
        fmt = str(self.combo_number_format.currentData())
        self.settings.set_number_format(fmt)
        set_global_number_format(fmt)
        meter_mode = str(self.combo_meters_mode.currentData())
        self.settings.set_new_meters_mode(meter_mode)
        self.shortcuts_editor.apply_to_settings()
        self.accept()
