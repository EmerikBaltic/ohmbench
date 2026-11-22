"""Qt-free persistent user custom parts & subcircuit macro library (`My Custom Parts`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from circuitsim.components.base import Component
from circuitsim.components.custom_macro import CustomMacroBlock, SubcircuitMacroDef
from circuitsim.components.factory import create_default_component_model
from circuitsim.io.circ_serializer import _EXTRA_SETTINGS, _apply_setting

USER_LIBRARY_FORMAT = "ohmbench-user-library"
USER_LIBRARY_SCHEMA_VERSION = 1


def default_user_library_path() -> Path:
    """Return the persistent user library JSON path (honoring `OHMBENCH_USER_LIBRARY_PATH`)."""
    override = os.environ.get("OHMBENCH_USER_LIBRARY_PATH", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ohmbench" / "user_library.json"


def _slugify_part_id(name: str, fallback: str = "part") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or fallback


def _extract_component_params_and_settings(
    comp: Component,
) -> tuple[dict[str, float], dict[str, Any]]:
    ctype = comp.component_type
    params: dict[str, float] = {}
    for p in comp.get_editable_parameters():
        raw = getattr(comp, p.key, None)
        if raw is not None:
            params[p.key] = float(raw)

    settings: dict[str, Any] = {}
    for key, kind in _EXTRA_SETTINGS.get(ctype, {}).items():
        if not hasattr(comp, key):
            continue
        raw = getattr(comp, key)
        if kind == "bool":
            settings[key] = bool(raw)
        elif kind == "str":
            settings[key] = str(getattr(raw, "value", raw))
        elif kind == "float_or_none":
            settings[key] = None if raw is None else float(raw)
        else:
            settings[key] = float(raw)
    return params, settings


def _apply_params_and_settings(
    comp: Component,
    parameters: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    ctype = comp.component_type
    allowed_params = {p.key for p in comp.get_editable_parameters()}
    for key, val in parameters.items():
        if val is None or key not in allowed_params:
            continue
        comp.set_parameter_value(key, float(val))

    allowed_extra = _EXTRA_SETTINGS.get(ctype, {})
    for key, val in settings.items():
        if key not in allowed_extra:
            continue
        _apply_setting(comp, ctype, key, allowed_extra[key], val, f"CustomPart {comp.name}")


def serialize_macro_def(macro_def: SubcircuitMacroDef) -> dict[str, Any]:
    """Serialize a `SubcircuitMacroDef` including internal component models to JSON-ready dict."""
    internal_list: list[dict[str, Any]] = []
    for comp in macro_def.internal_components:
        params, settings = _extract_component_params_and_settings(comp)
        item_dict: dict[str, Any] = {
            "type": comp.component_type,
            "name": comp.name,
            "nodes": list(comp.nodes),
            "label": comp.label,
            "show_value": bool(comp.show_value),
            "parameters": params,
            "settings": settings,
        }
        if isinstance(comp, CustomMacroBlock):
            item_dict["macro_payload"] = serialize_macro_def(comp.macro_def)
        internal_list.append(item_dict)

    return {
        "macro_name": macro_def.macro_name,
        "pins": list(macro_def.pins),
        "internal_components": internal_list,
    }


def deserialize_macro_def(payload: Mapping[str, Any]) -> SubcircuitMacroDef:
    """Reconstruct a `SubcircuitMacroDef` cleanly from its serialized dictionary."""
    macro_name = str(payload.get("macro_name", "CustomMacro")).strip() or "CustomMacro"
    pins = tuple(str(p).strip() for p in payload.get("pins", ()) if str(p).strip())
    if not pins:
        pins = ("IN", "OUT")

    internals: list[Component] = []
    for idx, entry in enumerate(payload.get("internal_components", ())):
        if not isinstance(entry, Mapping):
            continue
        ctype = str(entry.get("type", "")).strip()
        cname = str(entry.get("name", f"U_{idx + 1}")).strip() or f"U_{idx + 1}"
        raw_nodes = tuple(str(n).strip() for n in entry.get("nodes", ()))
        if ctype == "custom_macro" and isinstance(entry.get("macro_payload"), Mapping):
            nested_def = deserialize_macro_def(entry["macro_payload"])
            comp: Component = CustomMacroBlock(
                cname,
                nested_def,
                nodes=raw_nodes or None,
                label=str(entry.get("label", cname)),
                show_value=bool(entry.get("show_value", True)),
            )
        else:
            comp = create_default_component_model(ctype, cname)
            _apply_params_and_settings(
                comp,
                entry.get("parameters", {}) if isinstance(entry.get("parameters"), Mapping) else {},
                entry.get("settings", {}) if isinstance(entry.get("settings"), Mapping) else {},
            )
            if raw_nodes and len(raw_nodes) == len(comp.nodes):
                comp.nodes = raw_nodes
            comp.label = str(entry.get("label", cname))
            comp.show_value = bool(entry.get("show_value", True))
        internals.append(comp)

    return SubcircuitMacroDef(
        macro_name=macro_name,
        pins=pins,
        internal_components=tuple(internals),
    )


@dataclass
class CustomPartEntry:
    """Single entry in the persistent user library (`preset` or reusable `macro`)."""

    part_id: str
    name: str
    kind: str  # "preset" or "macro"
    category: str = "My Custom Parts"
    description: str = ""
    base_component_type: str = ""
    parameters: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    macro_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "kind": self.kind,
            "category": self.category,
            "description": self.description,
            "base_component_type": self.base_component_type,
            "parameters": dict(self.parameters),
            "settings": dict(self.settings),
            "macro_payload": dict(self.macro_payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CustomPartEntry":
        name = str(data.get("name", "Custom Part")).strip() or "Custom Part"
        part_id = str(data.get("part_id", "")).strip() or _slugify_part_id(name)
        kind = str(data.get("kind", "preset")).strip().lower()
        if kind not in ("preset", "macro"):
            kind = "preset"
        params_raw = data.get("parameters", {})
        params = (
            {str(k): float(v) for k, v in params_raw.items() if v is not None}
            if isinstance(params_raw, Mapping)
            else {}
        )
        settings_raw = data.get("settings", {})
        settings = dict(settings_raw) if isinstance(settings_raw, Mapping) else {}
        macro_raw = data.get("macro_payload", {})
        macro_payload = dict(macro_raw) if isinstance(macro_raw, Mapping) else {}
        return cls(
            part_id=part_id,
            name=name,
            kind=kind,
            category=str(data.get("category", "My Custom Parts")) or "My Custom Parts",
            description=str(data.get("description", "")),
            base_component_type=str(data.get("base_component_type", "")),
            parameters=params,
            settings=settings,
            macro_payload=macro_payload,
        )


def create_preset_from_component(
    comp: Component,
    name: str,
    description: str = "",
    *,
    part_id: str | None = None,
) -> CustomPartEntry:
    """Create a `CustomPartEntry` (`kind='preset'`) capturing `comp`'s parameters & settings."""
    if isinstance(comp, CustomMacroBlock):
        return create_macro_part_entry(
            comp.macro_def,
            description=description,
            part_id=part_id,
            name=name,
        )
    clean_name = name.strip() or f"{comp.component_type} preset"
    params, settings = _extract_component_params_and_settings(comp)
    pid = part_id or f"preset_{_slugify_part_id(clean_name)}_{uuid.uuid4().hex[:6]}"
    return CustomPartEntry(
        part_id=pid,
        name=clean_name,
        kind="preset",
        category="My Custom Parts",
        description=description,
        base_component_type=comp.component_type,
        parameters=params,
        settings=settings,
    )


def create_macro_part_entry(
    macro_def: SubcircuitMacroDef,
    description: str = "",
    *,
    part_id: str | None = None,
    name: str | None = None,
) -> CustomPartEntry:
    """Create a `CustomPartEntry` (`kind='macro'`) wrapping a `SubcircuitMacroDef`."""
    clean_name = (name or macro_def.macro_name).strip() or "CustomMacro"
    effective_def = (
        macro_def
        if clean_name == macro_def.macro_name
        else SubcircuitMacroDef(
            macro_name=clean_name,
            pins=macro_def.pins,
            internal_components=macro_def.internal_components,
        )
    )
    pid = part_id or f"macro_{_slugify_part_id(clean_name)}_{uuid.uuid4().hex[:6]}"
    return CustomPartEntry(
        part_id=pid,
        name=clean_name,
        kind="macro",
        category="My Custom Parts",
        description=description,
        base_component_type="custom_macro",
        macro_payload=serialize_macro_def(effective_def),
    )


def instantiate_custom_part(
    entry: CustomPartEntry,
    designator: str,
    node_names: Sequence[str] | None = None,
) -> Component:
    """Instantiate a fresh Qt-free `Component` from a saved `CustomPartEntry`."""
    if entry.kind == "macro":
        macro_def = deserialize_macro_def(entry.macro_payload)
        return CustomMacroBlock(
            designator,
            macro_def,
            nodes=node_names,
            label=designator,
            show_value=True,
        )

    comp = create_default_component_model(entry.base_component_type, designator)
    _apply_params_and_settings(comp, entry.parameters, entry.settings)
    if node_names is not None:
        resolved = tuple(str(n).strip() for n in node_names)
        if len(resolved) != len(comp.nodes):
            raise ValueError(
                f"Custom preset {entry.name!r} expects {len(comp.nodes)} nodes, got {len(resolved)}."
            )
        comp.nodes = resolved
    return comp


class UserLibraryStore:
    """JSON-backed persistent custom parts library with atomic writes and export/import."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self.storage_path: Path = (
            Path(storage_path) if storage_path is not None else default_user_library_path()
        )
        self._parts: dict[str, CustomPartEntry] = {}
        self.reload()

    def reload(self) -> None:
        """Load entries from `self.storage_path` if present on disk."""
        self._parts.clear()
        if not self.storage_path.exists():
            return
        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except Exception:
            return
        parts_list = data.get("parts", []) if isinstance(data, Mapping) else []
        if not isinstance(parts_list, list):
            return
        for raw_entry in parts_list:
            if isinstance(raw_entry, Mapping):
                entry = CustomPartEntry.from_dict(raw_entry)
                self._parts[entry.part_id] = entry

    def _atomic_write(self, path: Path, entries: Sequence[CustomPartEntry]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": USER_LIBRARY_FORMAT,
            "schema_version": USER_LIBRARY_SCHEMA_VERSION,
            "parts": [e.to_dict() for e in entries],
        }
        serialized = json.dumps(payload, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".ohmbench_lib_", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(serialized)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def save(self) -> None:
        self._atomic_write(self.storage_path, list(self._parts.values()))

    def list_parts(self) -> list[CustomPartEntry]:
        return list(self._parts.values())

    def get_part(self, part_id: str) -> CustomPartEntry | None:
        return self._parts.get(part_id)

    def add_or_update_part(self, entry: CustomPartEntry) -> CustomPartEntry:
        self._parts[entry.part_id] = entry
        self.save()
        return entry

    def rename_part(self, part_id: str, new_name: str) -> bool:
        entry = self._parts.get(part_id)
        clean = new_name.strip()
        if entry is None or not clean:
            return False
        entry.name = clean
        if entry.kind == "macro" and entry.macro_payload:
            entry.macro_payload["macro_name"] = clean
        self.save()
        return True

    def delete_part(self, part_id: str) -> bool:
        if part_id not in self._parts:
            return False
        del self._parts[part_id]
        self.save()
        return True

    def export_library(
        self,
        export_path: Path | str,
        part_ids: Sequence[str] | None = None,
    ) -> Path:
        out = Path(export_path)
        if part_ids is None:
            entries = self.list_parts()
        else:
            id_set = set(part_ids)
            entries = [e for e in self._parts.values() if e.part_id in id_set]
        self._atomic_write(out, entries)
        return out

    def import_library(
        self,
        import_path: Path | str,
        *,
        overwrite: bool = True,
    ) -> list[CustomPartEntry]:
        src = Path(import_path)
        data = json.loads(src.read_text(encoding="utf-8"))
        parts_list = data.get("parts", []) if isinstance(data, Mapping) else []
        imported: list[CustomPartEntry] = []
        for raw_entry in parts_list:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = CustomPartEntry.from_dict(raw_entry)
            if not overwrite and entry.part_id in self._parts:
                entry.part_id = f"{entry.part_id}_{uuid.uuid4().hex[:4]}"
            self._parts[entry.part_id] = entry
            imported.append(entry)
        self.save()
        return imported
