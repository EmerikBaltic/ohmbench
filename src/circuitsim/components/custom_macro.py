"""Qt-free Custom Subcircuit Macro Packaging & Hierarchical MNA Netlist Flattening."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

from circuitsim.components.base import Component
from circuitsim.components.ground import Ground


@dataclass(frozen=True)
class SubcircuitMacroDef:
    """Reusable hierarchical subcircuit template with ordered named interface pins."""

    macro_name: str
    pins: tuple[str, ...]
    internal_components: tuple[Component, ...]


def package_subcircuit_macro(
    macro_name: str,
    pins: Sequence[str],
    components: Sequence[Component],
) -> SubcircuitMacroDef:
    """Package a subcircuit of Qt-free ``Component`` models into a reusable ``SubcircuitMacroDef``."""
    clean_name = str(macro_name or "CustomMacro").strip() or "CustomMacro"
    pin_tuple = tuple(str(p).strip() for p in pins if str(p).strip())
    if not pin_tuple:
        raise ValueError("package_subcircuit_macro requires at least one named pin.")
    cloned = tuple(
        copy.deepcopy(c)
        for c in components
        if not isinstance(c, Ground)
    )
    return SubcircuitMacroDef(
        macro_name=clean_name,
        pins=pin_tuple,
        internal_components=cloned,
    )


class CustomMacroBlock(Component):
    """Instantiated custom subcircuit block that flattens cleanly into MNA netlists."""

    def __init__(
        self,
        name: str,
        macro_def: SubcircuitMacroDef,
        nodes: Sequence[str] | None = None,
        *,
        label: str | None = None,
        show_value: bool = True,
    ) -> None:
        resolved_nodes = (
            tuple(str(n).strip() for n in nodes)
            if nodes is not None
            else tuple(f"{name}_{p}" for p in macro_def.pins)
        )
        if len(resolved_nodes) != len(macro_def.pins):
            raise ValueError(
                f"CustomMacroBlock {name!r} expects {len(macro_def.pins)} pin nodes "
                f"{macro_def.pins}, got {len(resolved_nodes)}."
            )
        super().__init__(name=name, nodes=resolved_nodes, label=label, show_value=show_value)
        self.macro_def: SubcircuitMacroDef = macro_def
        self._last_expanded: list[Component] = []

    @property
    def component_type(self) -> str:
        return "custom_macro"

    @property
    def display_value(self) -> str:
        return self.macro_def.macro_name

    def expand_internal_components(self, prefix: str | None = None) -> list[Component]:
        """Clone internal subcircuit components, binding macro pins to instance external nodes."""
        inst_prefix = prefix or self.name
        pin_map = {
            pin_name: ext_node
            for pin_name, ext_node in zip(self.macro_def.pins, self.nodes)
        }
        expanded: list[Component] = []
        for idx, tmpl in enumerate(self.macro_def.internal_components):
            cloned = copy.deepcopy(tmpl)
            cloned.name = f"{inst_prefix}__{tmpl.name}_{idx}"
            mapped_nodes: list[str] = []
            for raw_n in cloned.nodes:
                if raw_n in pin_map:
                    mapped_nodes.append(pin_map[raw_n])
                elif raw_n == "0":
                    mapped_nodes.append("0")
                else:
                    mapped_nodes.append(f"{inst_prefix}__{raw_n}")
            cloned.nodes = tuple(mapped_nodes)
            if isinstance(cloned, CustomMacroBlock):
                expanded.extend(cloned.expand_internal_components(prefix=cloned.name))
            else:
                expanded.append(cloned)
        self._last_expanded = expanded
        return expanded

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_name": self.macro_def.macro_name,
            "pins": list(self.macro_def.pins),
        }
