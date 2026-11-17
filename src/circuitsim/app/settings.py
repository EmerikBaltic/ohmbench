"""Application settings, recent files, and atomic JSON persistence."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from circuitsim.app_info import APP_NAME, APP_USER_MODEL_ID, ORG_NAME

MAX_RECENT_FILES: int = 8
SETTINGS_FILENAME: str = "settings.json"
MIGRATABLE_USER_FILES: tuple[str, ...] = ("settings.json", "crash.log", "recent_files.json")

DEFAULT_SHORTCUTS: dict[str, str] = {
    "rotate": "Ctrl+R",
    "flip": "F",
    "wire_mode": "W",
    "junction": "J",
    "delete": "Del",
    "place_resistor": "R",
    "place_capacitor": "C",
    "place_inductor": "L",
    "place_diode": "D",
    "place_dc_voltage": "V",
    "place_dc_current": "I",
    "place_ground": "G",
    "place_voltmeter": "U",
    "place_ammeter": "A",
}

SHORTCUT_LABELS: dict[str, str] = {
    "rotate": "Rotate 90° CW",
    "flip": "Flip Horizontal",
    "wire_mode": "Toggle Wire Mode",
    "junction": "Toggle Junction Dot",
    "delete": "Delete Selected",
    "place_resistor": "Place Resistor",
    "place_capacitor": "Place Capacitor",
    "place_inductor": "Place Inductor",
    "place_diode": "Place Diode / LED",
    "place_dc_voltage": "Place DC Voltage Source",
    "place_dc_current": "Place DC Current Source",
    "place_ground": "Place Ground",
    "place_voltmeter": "Place Voltmeter",
    "place_ammeter": "Place Ammeter",
}


def get_default_data_dir() -> Path:
    """Return default user application data directory for Ohmbench."""
    override = os.environ.get("OHMBENCH_DATA_DIR") or os.environ.get("CIRCUITSIM_DATA_DIR")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    try:
        from PySide6.QtCore import QStandardPaths

        loc = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        if loc:
            p = Path(loc)
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        p = Path(appdata) / APP_NAME if appdata else Path.home() / f".{APP_NAME.lower()}"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        p = (
            Path(xdg) / APP_NAME.lower()
            if xdg
            else Path.home() / ".local" / "share" / APP_NAME.lower()
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_legacy_data_dirs() -> list[Path]:
    """Return legacy pre-rename user data directories (e.g. AppData\\Roaming\\python)."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "python")
            candidates.append(Path(appdata) / "CircuitSim")
        candidates.append(Path.home() / ".circuitsim")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        candidates.append(base / "python")
        candidates.append(base / "CircuitSim")
    else:
        xdg = Path(os.environ["XDG_DATA_HOME"]) if os.environ.get("XDG_DATA_HOME") else Path.home() / ".local" / "share"
        candidates.append(xdg / "python")
        candidates.append(xdg / "circuitsim")
    return candidates


def migrate_legacy_data_files(
    target_dir: Path,
    *,
    legacy_dirs: Iterable[Path] | None = None,
) -> list[str]:
    """Copy existing user files from legacy directories into target_dir once (never deletes originals)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    copied: list[str] = []
    search_dirs = list(legacy_dirs) if legacy_dirs is not None else get_legacy_data_dirs()
    for src_dir in search_dirs:
        try:
            if not src_dir.exists() or src_dir.resolve() == target_resolved:
                continue
        except OSError:
            continue
        for filename in MIGRATABLE_USER_FILES:
            src_file = src_dir / filename
            dst_file = target_dir / filename
            if src_file.is_file() and not dst_file.exists():
                try:
                    shutil.copy2(src_file, dst_file)
                    copied.append(filename)
                except OSError:
                    pass
    return copied


def set_windows_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Set Windows AppUserModelID for taskbar icon grouping when running on win32."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception:
        return False


def configure_qt_identity() -> None:
    """Configure Qt organization/application identity and platform AppUserModelID."""
    try:
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.setOrganizationName(ORG_NAME)
        QCoreApplication.setApplicationName(APP_NAME)
    except Exception:
        pass
    set_windows_app_user_model_id()


class AppSettings:
    """Manages application settings and recent files with atomic persistence."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir: Path = (
            Path(data_dir) if data_dir is not None else get_default_data_dir()
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path: Path = self.data_dir / SETTINGS_FILENAME
        self._data: dict[str, Any] = {
            "recent_files": [],
            "number_format": "auto",  # auto, dot, comma
            "shortcuts": dict(DEFAULT_SHORTCUTS),
        }
        self._shortcut_listeners: list[Any] = []
        self.load()

    def load(self) -> None:
        """Load settings from JSON file if it exists."""
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                raw_shortcuts = loaded.pop("shortcuts", None)
                self._data.update(loaded)
                merged_sc = dict(DEFAULT_SHORTCUTS)
                if isinstance(raw_shortcuts, dict):
                    for k, v in raw_shortcuts.items():
                        if k in merged_sc:
                            merged_sc[str(k)] = str(v)
                self._data["shortcuts"] = merged_sc
        except Exception:
            # If damaged or unreadable, ignore and keep defaults
            pass

    def save(self) -> None:
        """Atomically write settings to disk."""
        tmp_path = self.file_path.with_suffix(".tmp")
        payload = json.dumps(self._data, indent=2)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.file_path)

    # -------------------------------------------------------------
    # Recent files
    # -------------------------------------------------------------
    def get_recent_files(self) -> list[str]:
        """Return list of up to 8 recent file paths (most recent first)."""
        raw = self._data.get("recent_files", [])
        if not isinstance(raw, list):
            return []
        return [str(p) for p in raw]

    def add_recent_file(self, file_path: str | Path) -> None:
        """Add or promote file_path to the top of the recent files list."""
        resolved = str(Path(file_path).resolve())
        current = self.get_recent_files()
        current = [p for p in current if p != resolved]
        current.insert(0, resolved)
        self._data["recent_files"] = current[:MAX_RECENT_FILES]
        self.save()

    def remove_recent_file(self, file_path: str | Path) -> None:
        """Remove file_path from the recent files list."""
        resolved = str(Path(file_path).resolve())
        current = [p for p in self.get_recent_files() if p != resolved]
        self._data["recent_files"] = current
        self.save()

    def clear_recent_files(self) -> None:
        """Clear all recent files."""
        self._data["recent_files"] = []
        self.save()

    # -------------------------------------------------------------
    # Number format setting (Task 9)
    # -------------------------------------------------------------
    def get_number_format(self) -> str:
        return str(self._data.get("number_format", "auto"))

    def set_number_format(self, fmt: str) -> None:
        if fmt not in ("auto", "dot", "comma"):
            raise ValueError(
                f"Invalid number format {fmt!r}, expected 'auto', 'dot', or 'comma'."
            )
        self._data["number_format"] = fmt
        self.save()

    # -------------------------------------------------------------
    # New meters default mode: Ideal vs Realistic
    # -------------------------------------------------------------
    def get_new_meters_mode(self) -> str:
        """Return 'ideal' (default) or 'realistic' for newly placed meters."""
        return str(self._data.get("new_meters_mode", "ideal"))

    def set_new_meters_mode(self, mode: str) -> None:
        """Set whether newly placed meters default to ideal or realistic."""
        mode_clean = str(mode).strip().lower()
        if mode_clean not in ("ideal", "realistic"):
            raise ValueError(
                f"Invalid meter mode {mode!r}, expected 'ideal' or 'realistic'."
            )
        self._data["new_meters_mode"] = mode_clean
        self.save()

    # -------------------------------------------------------------
    # Programmable keyboard shortcuts (G8)
    # -------------------------------------------------------------
    @property
    def shortcuts(self) -> dict[str, str]:
        """Return live mapping of action_id -> shortcut string."""
        raw = self._data.get("shortcuts")
        if not isinstance(raw, dict):
            raw = dict(DEFAULT_SHORTCUTS)
            self._data["shortcuts"] = raw
        for k, v in DEFAULT_SHORTCUTS.items():
            raw.setdefault(k, v)
        return raw

    @shortcuts.setter
    def shortcuts(self, mapping: dict[str, str]) -> None:
        self.set_shortcuts(mapping)

    def register_shortcuts_listener(self, cb: Any) -> None:
        if cb not in self._shortcut_listeners:
            self._shortcut_listeners.append(cb)

    def unregister_shortcuts_listener(self, cb: Any) -> None:
        if cb in self._shortcut_listeners:
            self._shortcut_listeners.remove(cb)

    def _notify_shortcut_listeners(self) -> None:
        for cb in list(self._shortcut_listeners):
            try:
                cb()
            except Exception:
                pass

    def get_shortcut(self, action_id: str) -> str:
        return str(self.shortcuts.get(action_id, DEFAULT_SHORTCUTS.get(action_id, "")))

    def set_shortcut(self, action_id: str, sequence: str) -> None:
        merged = dict(self.shortcuts)
        merged[str(action_id)] = str(sequence).strip()
        self._data["shortcuts"] = merged
        self.save()
        self._notify_shortcut_listeners()

    def set_shortcuts(self, mapping: dict[str, str]) -> None:
        merged = dict(DEFAULT_SHORTCUTS)
        for k, v in mapping.items():
            merged[str(k)] = str(v).strip()
        self._data["shortcuts"] = merged
        self.save()
        self._notify_shortcut_listeners()

    def reset_shortcuts(self) -> dict[str, str]:
        self._data["shortcuts"] = dict(DEFAULT_SHORTCUTS)
        self.save()
        self._notify_shortcut_listeners()
        return dict(self._data["shortcuts"])
