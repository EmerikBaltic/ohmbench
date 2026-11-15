"""QUndoCommand implementations for all schematic editing actions (Step 2B)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from PySide6.QtCore import QPointF
from PySide6.QtGui import QUndoCommand

from circuitsim.canvas.connectivity import PointPx
from circuitsim.components.logic_gate import LogicGate

if TYPE_CHECKING:
    from circuitsim.canvas.component_item import ComponentItem
    from circuitsim.canvas.scene import CircuitScene
    from circuitsim.canvas.wire_item import WireItem
    from circuitsim.wiring.model import WireGraph


CMD_ID_MOVE_COMPONENTS: int = 1001


class SetWireGraphCommand(QUndoCommand):
    """Undoable command that transitions `CircuitScene.wire_graph` between immutable states."""

    def __init__(
        self,
        scene: CircuitScene,
        old_graph: WireGraph,
        new_graph: WireGraph,
        *,
        preferred_wire_items: dict[str, WireItem] | None = None,
        description: str = "Edit Wires",
    ) -> None:
        super().__init__(description)
        self.scene: CircuitScene = scene
        self.old_graph: WireGraph = old_graph
        self.new_graph: WireGraph = new_graph
        self.preferred_wire_items: dict[str, WireItem] = (
            dict(preferred_wire_items) if preferred_wire_items else {}
        )

    def redo(self) -> None:
        self.scene.set_wire_graph(
            self.new_graph, preferred_wire_items=self.preferred_wire_items
        )

    def undo(self) -> None:
        self.scene.set_wire_graph(self.old_graph)


class AddComponentCommand(QUndoCommand):
    """Undoable command that places a `ComponentItem` onto `CircuitScene`."""

    def __init__(
        self,
        scene: CircuitScene,
        item: ComponentItem,
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        super().__init__(f"Add {item.component.name}")
        self.scene: CircuitScene = scene
        self.item: ComponentItem = item
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        self.scene._insert_component_item(
            self.item, select=True, refresh=self.new_graph is None
        )
        if self.new_graph is not None:
            self.scene.set_wire_graph(self.new_graph)

    def undo(self) -> None:
        if self.old_graph is not None:
            self.scene.set_wire_graph(self.old_graph, refresh=False)
        self.scene._remove_component_item(self.item)


class AddWireCommand(QUndoCommand):
    """Undoable command that adds a `WireItem` (and updates `WireGraph`) onto `CircuitScene`."""

    def __init__(
        self,
        scene: CircuitScene,
        wire: WireItem,
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        super().__init__(f"Add Wire {wire.wire_id}")
        self.scene: CircuitScene = scene
        self.wire: WireItem = wire
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        if self.new_graph is not None:
            self.scene.set_wire_graph(
                self.new_graph,
                preferred_wire_items={self.wire.wire_id: self.wire},
            )
        else:
            self.scene._insert_wire_item(self.wire, select=False)

    def undo(self) -> None:
        if self.old_graph is not None:
            self.scene.set_wire_graph(self.old_graph)
        else:
            self.scene._remove_wire_item(self.wire)


class ToggleJunctionCommand(QUndoCommand):
    """Undoable command that adds or removes an explicit junction dot at `pt`."""

    def __init__(
        self,
        scene: CircuitScene,
        pt: PointPx,
        *,
        adding: bool,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        action = "Add" if adding else "Remove"
        super().__init__(f"{action} Junction at {pt}")
        self.scene: CircuitScene = scene
        self.pt: PointPx = pt
        self.adding: bool = adding
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        if self.new_graph is not None:
            self.scene._set_explicit_junction(self.pt, present=self.adding, refresh=False)
            self.scene.set_wire_graph(self.new_graph)
        else:
            self.scene._set_explicit_junction(self.pt, present=self.adding)

    def undo(self) -> None:
        if self.old_graph is not None:
            self.scene._set_explicit_junction(
                self.pt, present=not self.adding, refresh=False
            )
            self.scene.set_wire_graph(self.old_graph)
        else:
            self.scene._set_explicit_junction(self.pt, present=not self.adding)


class MoveComponentsCommand(QUndoCommand):
    """Undoable command that moves components and re-routes attached wires."""

    def __init__(
        self,
        scene: CircuitScene,
        old_positions: dict[ComponentItem, tuple[float, float]],
        new_positions: dict[ComponentItem, tuple[float, float]],
        old_wire_points: dict[WireItem, tuple[PointPx, ...]],
        new_wire_points: dict[WireItem, tuple[PointPx, ...]],
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
        mergeable: bool = False,
    ) -> None:
        count = len(new_positions)
        label = (
            f"Move {next(iter(new_positions)).component.name}"
            if count == 1
            else f"Move {count} Components"
        )
        super().__init__(label)
        self.scene: CircuitScene = scene
        self.old_positions = dict(old_positions)
        self.new_positions = dict(new_positions)
        self.old_wire_points = dict(old_wire_points)
        self.new_wire_points = dict(new_wire_points)
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph
        self.mergeable: bool = mergeable

    def id(self) -> int:
        return CMD_ID_MOVE_COMPONENTS

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, MoveComponentsCommand):
            return False
        if not self.mergeable or not other.mergeable:
            return False
        if other.scene is not self.scene:
            return False
        if set(other.new_positions.keys()) != set(self.new_positions.keys()):
            return False
        self.new_positions = dict(other.new_positions)
        if other.new_wire_points:
            for wire, pts in other.new_wire_points.items():
                self.new_wire_points[wire] = pts
        if other.new_graph is not None:
            self.new_graph = other.new_graph
        return True

    def redo(self) -> None:
        self.scene._suppress_live_reroute = True
        try:
            for item, (nx, ny) in self.new_positions.items():
                item.setPos(QPointF(nx, ny))
            if self.new_graph is not None:
                self.scene.set_wire_graph(self.new_graph, refresh=False)
            else:
                self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.new_wire_points.items():
                if self.new_graph is None:
                    wire.set_points(pts)
        finally:
            self.scene._suppress_live_reroute = False
        self.scene.refresh_connectivity_visuals()

    def undo(self) -> None:
        self.scene._suppress_live_reroute = True
        try:
            for item, (ox, oy) in self.old_positions.items():
                item.setPos(QPointF(ox, oy))
            if self.old_graph is not None:
                self.scene.set_wire_graph(self.old_graph, refresh=False)
            else:
                self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.old_wire_points.items():
                if self.old_graph is None:
                    wire.set_points(pts)
        finally:
            self.scene._suppress_live_reroute = False
        self.scene.refresh_connectivity_visuals()


class RotateComponentsCommand(QUndoCommand):
    """Undoable command that rotates components by 90° CW and re-routes attached wires."""

    def __init__(
        self,
        scene: CircuitScene,
        items: Sequence[ComponentItem],
        old_rotations: dict[ComponentItem, int],
        new_rotations: dict[ComponentItem, int],
        old_wire_points: dict[WireItem, tuple[PointPx, ...]],
        new_wire_points: dict[WireItem, tuple[PointPx, ...]],
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        label = (
            f"Rotate {items[0].component.name}"
            if len(items) == 1
            else f"Rotate {len(items)} Components"
        )
        super().__init__(label)
        self.scene: CircuitScene = scene
        self.old_rotations = dict(old_rotations)
        self.new_rotations = dict(new_rotations)
        self.old_wire_points = dict(old_wire_points)
        self.new_wire_points = dict(new_wire_points)
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        for item, deg in self.new_rotations.items():
            item.set_rotation_deg(deg)
        if self.new_graph is not None:
            self.scene.set_wire_graph(self.new_graph, refresh=False)
        else:
            self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.new_wire_points.items():
                wire.set_points(pts)
        self.scene.refresh_connectivity_visuals()

    def undo(self) -> None:
        for item, deg in self.old_rotations.items():
            item.set_rotation_deg(deg)
        if self.old_graph is not None:
            self.scene.set_wire_graph(self.old_graph, refresh=False)
        else:
            self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.old_wire_points.items():
                wire.set_points(pts)
        self.scene.refresh_connectivity_visuals()


class FlipComponentsCommand(QUndoCommand):
    """Undoable command that flips components horizontally and re-routes attached wires."""

    def __init__(
        self,
        scene: CircuitScene,
        items: Sequence[ComponentItem],
        old_flips: dict[ComponentItem, bool],
        new_flips: dict[ComponentItem, bool],
        old_wire_points: dict[WireItem, tuple[PointPx, ...]],
        new_wire_points: dict[WireItem, tuple[PointPx, ...]],
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        label = (
            f"Flip {items[0].component.name}"
            if len(items) == 1
            else f"Flip {len(items)} Components"
        )
        super().__init__(label)
        self.scene: CircuitScene = scene
        self.old_flips = dict(old_flips)
        self.new_flips = dict(new_flips)
        self.old_wire_points = dict(old_wire_points)
        self.new_wire_points = dict(new_wire_points)
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        for item, flipped in self.new_flips.items():
            item.set_flipped_h(flipped)
        if self.new_graph is not None:
            self.scene.set_wire_graph(self.new_graph, refresh=False)
        else:
            self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.new_wire_points.items():
                wire.set_points(pts)
        self.scene.refresh_connectivity_visuals()

    def undo(self) -> None:
        for item, flipped in self.old_flips.items():
            item.set_flipped_h(flipped)
        if self.old_graph is not None:
            self.scene.set_wire_graph(self.old_graph, refresh=False)
        else:
            self.scene._sync_graphics_from_wire_graph()
            for wire, pts in self.old_wire_points.items():
                wire.set_points(pts)
        self.scene.refresh_connectivity_visuals()


class DeleteItemsCommand(QUndoCommand):
    """Undoable command that deletes selected components, wires, and junctions."""

    def __init__(
        self,
        scene: CircuitScene,
        components: Sequence[ComponentItem],
        wires: Sequence[WireItem],
        junctions: Sequence[PointPx] = (),
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
        description: str = "Delete Selection",
    ) -> None:
        super().__init__(description)
        self.scene: CircuitScene = scene
        self.components: tuple[ComponentItem, ...] = tuple(components)
        self.wires: tuple[WireItem, ...] = tuple(wires)
        self.junctions: tuple[PointPx, ...] = tuple(junctions)
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        if self.new_graph is not None:
            for comp in self.components:
                self.scene._remove_component_item(comp, refresh=False)
            for jp in self.junctions:
                self.scene._set_explicit_junction(jp, present=False, refresh=False)
            self.scene.set_wire_graph(self.new_graph, refresh=False)
        else:
            for wire in self.wires:
                self.scene._remove_wire_item(wire, refresh=False)
            for comp in self.components:
                self.scene._remove_component_item(comp, refresh=False)
            for jp in self.junctions:
                self.scene._set_explicit_junction(jp, present=False, refresh=False)
        self.scene.refresh_connectivity_visuals()

    def undo(self) -> None:
        for comp in self.components:
            self.scene._insert_component_item(comp, select=False, refresh=False)
        for jp in self.junctions:
            self.scene._set_explicit_junction(jp, present=True, refresh=False)
        if self.old_graph is not None:
            self.scene.set_wire_graph(
                self.old_graph,
                refresh=False,
                preferred_wire_items={w.wire_id: w for w in self.wires},
            )
        else:
            for wire in self.wires:
                self.scene._insert_wire_item(wire, select=False, refresh=False)
        self.scene.refresh_connectivity_visuals()


class PasteItemsCommand(QUndoCommand):
    """Undoable command that pastes copied components, wires, and junctions onto `CircuitScene`."""

    def __init__(
        self,
        scene: CircuitScene,
        components: Sequence[ComponentItem],
        wires: Sequence[WireItem],
        junctions: Sequence[PointPx] = (),
        *,
        old_graph: WireGraph | None = None,
        new_graph: WireGraph | None = None,
    ) -> None:
        super().__init__(
            f"Paste ({len(components)} components, {len(wires)} wires)"
        )
        self.scene: CircuitScene = scene
        self.components: tuple[ComponentItem, ...] = tuple(components)
        self.wires: tuple[WireItem, ...] = tuple(wires)
        self.junctions: tuple[PointPx, ...] = tuple(junctions)
        self.old_graph: WireGraph | None = old_graph
        self.new_graph: WireGraph | None = new_graph

    def redo(self) -> None:
        self.scene.clearSelection()
        for comp in self.components:
            self.scene._insert_component_item(comp, select=True, refresh=False)
        for jp in self.junctions:
            self.scene._set_explicit_junction(jp, present=True, refresh=False)
        if self.new_graph is not None:
            self.scene.set_wire_graph(
                self.new_graph,
                refresh=False,
                preferred_wire_items={w.wire_id: w for w in self.wires},
                select_wire_ids={w.wire_id for w in self.wires},
            )
        else:
            for wire in self.wires:
                self.scene._insert_wire_item(wire, select=True, refresh=False)
        self.scene.refresh_connectivity_visuals()

    def undo(self) -> None:
        for comp in self.components:
            self.scene._remove_component_item(comp, refresh=False)
        for jp in self.junctions:
            self.scene._set_explicit_junction(jp, present=False, refresh=False)
        if self.old_graph is not None:
            self.scene.set_wire_graph(self.old_graph, refresh=False)
        else:
            for wire in self.wires:
                self.scene._remove_wire_item(wire, refresh=False)
        self.scene.refresh_connectivity_visuals()


class EditComponentParametersCommand(QUndoCommand):
    """Undoable command that edits a component's numerical parameters and updates live simulation."""

    def __init__(
        self,
        scene: CircuitScene,
        item: ComponentItem,
        old_params: dict[str, float],
        new_params: dict[str, float],
        *,
        old_show_value: bool | None = None,
        new_show_value: bool | None = None,
    ) -> None:
        super().__init__(f"Edit {item.component.name} Values")
        self.scene: CircuitScene = scene
        self.item: ComponentItem = item
        self.old_params: dict[str, float] = dict(old_params)
        self.new_params: dict[str, float] = dict(new_params)
        self.old_show_value: bool = (
            item.component.show_value
            if old_show_value is None
            else bool(old_show_value)
        )
        self.new_show_value: bool = (
            item.component.show_value
            if new_show_value is None
            else bool(new_show_value)
        )

    def _apply(self, params: dict[str, object], show_val: bool) -> None:
        comp = self.item.component
        was_running = self.item.live_display_value is not None
        if isinstance(comp, LogicGate):
            comp.set_parameters_batch(params)  # type: ignore[arg-type]
            if not was_running and "initial_state" in params:
                comp.reset_state()
        else:
            param_keys = {p.key for p in comp.get_editable_parameters()}
            for k, v in params.items():
                if k in param_keys:
                    comp.set_parameter_value(k, v)
                elif hasattr(comp, k):
                    setattr(comp, k, v)
        comp.show_value = show_val
        self.item.live_display_value = None
        self.item.setToolTip(self.item.get_live_tooltip(is_running=False))
        self.item.update()
        emitted_params: dict[str, object] = {}
        for k, v in params.items():
            if isinstance(v, bool):
                emitted_params[k] = v
            elif isinstance(v, (int, float)):
                emitted_params[k] = float(v)
            else:
                emitted_params[k] = v
        self.scene.component_parameters_edited.emit(comp.name, emitted_params)
        self.scene.selection_info_changed.emit(
            self.item if self.item.isSelected() else None
        )

    def redo(self) -> None:
        self._apply(self.new_params, self.new_show_value)

    def undo(self) -> None:
        self._apply(self.old_params, self.old_show_value)

