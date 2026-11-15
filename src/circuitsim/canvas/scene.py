"""QGraphicsScene for the schematic editor with 20 px grid, wiring, undo/redo, and netlist extraction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from circuitsim.app.settings import AppSettings

from PySide6.QtCore import QLineF, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QUndoStack
from PySide6.QtWidgets import QGraphicsScene

from circuitsim.app.themes import ThemeMode, get_theme_palette
from circuitsim.components.logic_gate import LogicGate
from circuitsim.canvas.commands import (
    AddComponentCommand,
    AddWireCommand,
    DeleteItemsCommand,
    EditComponentParametersCommand,
    FlipComponentsCommand,
    MoveComponentsCommand,
    PasteItemsCommand,
    RotateComponentsCommand,
    SetWireGraphCommand,
    ToggleJunctionCommand,
)
from circuitsim.canvas.component_item import (
    COMPONENT_PREFIXES,
    ComponentItem,
    create_default_component_model,
    rect_intersects_segment,
)
from circuitsim.canvas.connectivity import (
    ConnectivityResult,
    PlacedComponentSpec,
    PointPx,
    WireSpec,
    extract_connectivity,
    find_wires_attached_to_components,
    normalize_point_px,
    point_on_wire,
    reroute_wires_for_updated_terminals,
    route_orthogonal_wire,
    simplify_orthogonal_points,
)
from circuitsim.canvas.symbol_geometry import GRID_PITCH_PX, snap_to_grid
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.canvas.wire_item import JunctionItem, WireItem
from circuitsim.components.base import Component
from circuitsim.components.switch import Switch
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.engine.netlist import Circuit
from circuitsim.engine.simulation_session import (
    SimulationFrame,
    SimulationState,
)
from circuitsim.io.circ_serializer import CircuitDocument, CircuitFileError, ViewState
from circuitsim.wiring.model import (
    Attachment,
    FreeEnd,
    JunctionRef,
    TerminalRef,
    Wire,
    WireGraph,
    WireGraphError,
    snap_px as wire_snap_px,
)
from circuitsim.wiring.router import (
    normalize_dir as wire_normalize_dir,
    route as wire_route,
    simplify_orthogonal_points as wire_simplify_points,
)


@dataclass(frozen=True)
class _ClipboardComponentEntry:
    component_snapshot: Component
    pos_px: tuple[float, float]
    rotation_deg: int
    flipped_h: bool


@dataclass(frozen=True)
class _ClipboardData:
    components: tuple[_ClipboardComponentEntry, ...]
    wires: tuple[tuple[PointPx, ...], ...]
    junctions: tuple[PointPx, ...]


class CircuitScene(QGraphicsScene):
    """Schematic canvas scene managing components, orthogonal wires, junctions, and undo/redo."""

    component_placed = Signal(object)  # Emits placed ComponentItem
    selection_info_changed = Signal(object)  # Emits selected ComponentItem or None
    circuit_modified = Signal()  # Emits whenever components, wires, or junctions change
    switch_toggled_by_user = Signal(str, bool)  # (switch_name, closed)
    component_parameters_edited = Signal(str, object)  # (comp_name, dict[str, float])

    def __init__(
        self,
        symbol_library: SymbolLibrary,
        *,
        theme: ThemeMode = ThemeMode.DARK,
        settings: AppSettings | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_shutting_down: bool = False
        self.symbol_library: SymbolLibrary = symbol_library
        self.theme: ThemeMode = theme
        self.settings: AppSettings | None = settings
        self.show_grid: bool = True
        self.undo_stack: QUndoStack = QUndoStack(self)

        self._counters: dict[str, int] = {}
        self._wire_counter: int = 0
        self._placed_items: list[ComponentItem] = []
        self._wire_items: list[WireItem] = []
        self._wire_graph: WireGraph = WireGraph()
        self._wire_items_by_id: dict[str, WireItem] = {}
        self._junction_items_by_id: dict[str, JunctionItem] = {}
        self._explicit_junctions: set[PointPx] = set()
        self._active_junction_dots: set[PointPx] = set()

        # Ghost component preview during palette drag
        self._ghost_item: ComponentItem | None = None

        # Interactive wire routing preview & terminal hover
        self._hovered_terminal_pt: PointPx | None = None
        self._wire_start_pt: PointPx | None = None
        self._wire_prefer_horiz: bool = True
        self._preview_wire_item: WireItem | None = None

        # Component drag-move tracking for live attached-wire re-routing & QUndoStack
        self._move_tracking_active: bool = False
        self._suppress_live_reroute: bool = False
        self._move_start_positions: dict[ComponentItem, tuple[float, float]] = {}
        self._move_start_terminals: dict[str, dict[str, PointPx]] = {}
        self._move_start_wires: dict[WireItem, tuple[PointPx, ...]] = {}

        # Internal clipboard for Copy / Cut / Paste
        self._clipboard: _ClipboardData | None = None
        self._paste_serial: int = 0

        # Live simulation telemetry cache
        self.latest_node_voltages: dict[str, float] = {}
        self.is_simulation_running: bool = False

        self.setSceneRect(-2000.0, -2000.0, 4000.0, 4000.0)
        self.selectionChanged.connect(self._on_selection_changed)

    @property
    def placed_items(self) -> list[ComponentItem]:
        """Return list of placed `ComponentItem` objects on the canvas."""
        return list(self._placed_items)

    @property
    def placed_wires(self) -> list[WireItem]:
        """Return list of placed `WireItem` objects on the canvas."""
        return list(self._wire_items)

    @property
    def wire_graph(self) -> WireGraph:
        """Return the immutable `WireGraph` backing all scene wires and junctions."""
        return self._wire_graph

    @property
    def junction_items(self) -> list[JunctionItem]:
        """Return list of visible `JunctionItem` objects on the canvas."""
        return list(self._junction_items_by_id.values())

    @property
    def explicit_junctions(self) -> set[PointPx]:
        """Return copy of user-placed explicit junction dot coordinates."""
        return set(self._explicit_junctions)

    @property
    def active_junction_dots(self) -> set[PointPx]:
        """Return copy of all visual junction dots (T-junctions + explicit junctions)."""
        return set(self._active_junction_dots)

    @property
    def is_drawing_wire(self) -> bool:
        """Return True if an interactive wire route is currently in progress."""
        return self._wire_start_pt is not None

    def set_theme(self, theme: ThemeMode) -> None:
        """Switch canvas and all placed items/wires between Dark and Light themes."""
        self.theme = theme
        for item in self._placed_items:
            item.set_theme(theme)
        for wire in self._wire_items:
            wire.set_theme(theme)
        for junc in self._junction_items_by_id.values():
            junc.set_theme(theme)
        if self._ghost_item is not None:
            self. _ghost_item.set_theme(theme)
        if self._preview_wire_item is not None:
            self._preview_wire_item.set_theme(theme)
        self.update()

    def set_show_grid(self, visible: bool) -> None:
        """Toggle visibility of the 20 px background grid."""
        self.show_grid = bool(visible)
        self.update()

    def next_component_name(
        self, component_id: str, extra_taken: set[str] | None = None
    ) -> str:
        """Generate the lowest unused sequential designator for the prefix (e.g. 'R1', 'VM1')."""
        prefix = COMPONENT_PREFIXES.get(component_id, component_id)
        existing_names = {it.component.name for it in self._placed_items}
        if extra_taken:
            existing_names.update(extra_taken)
        count = 1
        while f"{prefix}{count}" in existing_names:
            count += 1
        return f"{prefix}{count}"

    def next_wire_id(self) -> str:
        """Generate the next sequential wire identifier ('W1', 'W2', ...)."""
        while True:
            self._wire_counter += 1
            candidate = f"W{self._wire_counter}"
            if not self._wire_graph.has_wire(candidate):
                return candidate

    def _terminal_pos_map(self) -> dict[tuple[str, str], PointPx]:
        mapping: dict[tuple[str, str], PointPx] = {}
        for item in self._placed_items:
            cname = item.component.name
            for tname, pt in item.terminal_scene_positions_px().items():
                mapping[(cname, tname)] = pt
        return mapping

    def _terminal_dir_map(self) -> dict[tuple[str, str], tuple[int, int] | None]:
        mapping: dict[tuple[str, str], tuple[int, int] | None] = {}
        for item in self._placed_items:
            cname = item.component.name
            cx = int(round(item.pos().x()))
            cy = int(round(item.pos().y()))
            for tname, pt in item.terminal_scene_positions_px().items():
                mapping[(cname, tname)] = wire_normalize_dir(
                    (pt[0] - cx, pt[1] - cy)
                )
        return mapping

    def _component_obstacles_px(self) -> tuple[tuple[int, int, int, int], ...]:
        rects: list[tuple[int, int, int, int]] = []
        for item in self._placed_items:
            r = item.scene_body_rect()
            rects.append(
                (
                    int(round(r.left())),
                    int(round(r.top())),
                    int(round(r.right())),
                    int(round(r.bottom())),
                )
            )
        return tuple(rects)

    def _attachment_outward_dir(
        self,
        att: Attachment,
        dir_map: dict[tuple[str, str], tuple[int, int] | None] | None = None,
    ) -> tuple[int, int] | None:
        if isinstance(att, TerminalRef):
            dmap = dir_map if dir_map is not None else self._terminal_dir_map()
            return dmap.get((att.component, att.terminal))
        return None

    def _attachment_pos_px(
        self,
        att: Attachment,
        graph: WireGraph | None = None,
        term_map: dict[tuple[str, str], PointPx] | None = None,
    ) -> PointPx | None:
        g = graph if graph is not None else self._wire_graph
        if isinstance(att, TerminalRef):
            tmap = term_map if term_map is not None else self._terminal_pos_map()
            return tmap.get((att.component, att.terminal))
        if isinstance(att, JunctionRef):
            if g.has_junction(att.id):
                return g.junction(att.id).pos
            return None
        if isinstance(att, FreeEnd):
            return att.pos
        return None

    def _compute_wire_points_from_graph(
        self,
        wire: Wire,
        graph: WireGraph,
        *,
        term_map: dict[tuple[str, str], PointPx] | None = None,
        dir_map: dict[tuple[str, str], tuple[int, int] | None] | None = None,
        obstacles: tuple[tuple[int, int, int, int], ...] | None = None,
        fallback_points: tuple[PointPx, ...] | None = None,
    ) -> tuple[PointPx, ...]:
        p_a = self._attachment_pos_px(wire.end_a, graph, term_map)
        p_b = self._attachment_pos_px(wire.end_b, graph, term_map)
        if p_a is None or p_b is None:
            return fallback_points or ()
        if p_a == p_b:
            return (p_a,)
        dmap = dir_map if dir_map is not None else self._terminal_dir_map()
        obs = obstacles if obstacles is not None else self._component_obstacles_px()
        dir_a = self._attachment_outward_dir(wire.end_a, dmap)
        dir_b = self._attachment_outward_dir(wire.end_b, dmap)
        if wire.hints:
            waypoints = [p_a, *wire.hints, p_b]
            full_pts: list[PointPx] = [wire_snap_px(waypoints[0])]
            for idx in range(len(waypoints) - 1):
                prev = full_pts[-1]
                nxt = wire_snap_px(waypoints[idx + 1])
                if prev == nxt:
                    continue
                if prev[0] == nxt[0] or prev[1] == nxt[1]:
                    full_pts.append(nxt)
                else:
                    seg = wire_route(
                        prev,
                        nxt,
                        obstacles=obs,
                        previous=fallback_points if idx == 0 else None,
                    )
                    full_pts.extend(seg[1:])
            simp = wire_simplify_points(full_pts)
            return simp if simp else (p_a, p_b)
        return wire_route(
            p_a,
            p_b,
            start_dir=dir_a,
            end_dir=dir_b,
            obstacles=obs,
            previous=fallback_points,
        )

    def _sync_junction_items_from_wire_graph(self) -> None:
        """Create, update, or remove `JunctionItem` objects matching `graph.junction_dot_visible(jid)`."""
        graph = self._wire_graph
        visible_junctions = {
            j.id: j for j in graph.junctions if graph.junction_dot_visible(j.id)
        }
        for old_jid in list(self._junction_items_by_id):
            if old_jid not in visible_junctions:
                jitem = self._junction_items_by_id.pop(old_jid)
                if jitem.scene() is self:
                    self.removeItem(jitem)
        for jid, junc in visible_junctions.items():
            existing = self._junction_items_by_id.get(jid)
            target_pos = QPointF(float(junc.pos[0]), float(junc.pos[1]))
            if existing is not None:
                if existing.pos() != target_pos:
                    existing.setPos(target_pos)
                if existing.scene() is not self:
                    self.addItem(existing)
            else:
                jitem = JunctionItem(
                    target_pos,
                    theme=self.theme,
                    junction_id=jid,
                )
                self._junction_items_by_id[jid] = jitem
                self.addItem(jitem)

    def _sync_graphics_from_wire_graph(
        self,
        *,
        preferred_wire_items: dict[str, WireItem] | None = None,
        select_wire_ids: set[str] | None = None,
    ) -> None:
        graph = self._wire_graph
        term_map = self._terminal_pos_map()
        dir_map = self._terminal_dir_map()
        obstacles = self._component_obstacles_px()
        pref = preferred_wire_items or {}
        want_ids = {w.id for w in graph.wires}

        # Remove graphics items whose wires are no longer in the graph.
        for old_wid in list(self._wire_items_by_id):
            if old_wid not in want_ids:
                old_item = self._wire_items_by_id.pop(old_wid)
                if old_item.scene() is self:
                    self.removeItem(old_item)

        new_items: list[WireItem] = []
        new_by_id: dict[str, WireItem] = {}
        max_w_num = self._wire_counter
        for wire in graph.wires:
            if wire.id.startswith("W") and wire.id[1:].isdigit():
                max_w_num = max(max_w_num, int(wire.id[1:]))
            existing = self._wire_items_by_id.get(wire.id) or pref.get(wire.id)
            fallback = existing.points if existing is not None else None
            pts = self._compute_wire_points_from_graph(
                wire,
                graph,
                term_map=term_map,
                dir_map=dir_map,
                obstacles=obstacles,
                fallback_points=fallback,
            )
            if existing is not None:
                existing.end_a = wire.end_a
                existing.end_b = wire.end_b
                existing.hints = wire.hints
                existing.set_points(pts)
                if existing.scene() is not self:
                    self.addItem(existing)
                item = existing
            else:
                item = WireItem(
                    wire.id,
                    pts,
                    theme=self.theme,
                    end_a=wire.end_a,
                    end_b=wire.end_b,
                    hints=wire.hints,
                )
                self.addItem(item)
            if select_wire_ids and wire.id in select_wire_ids:
                item.setSelected(True)
            new_items.append(item)
            new_by_id[wire.id] = item

        self._wire_counter = max_w_num
        self._wire_items = new_items
        self._wire_items_by_id = new_by_id
        self._sync_junction_items_from_wire_graph()

    def set_wire_graph(
        self,
        graph: WireGraph,
        *,
        refresh: bool = True,
        preferred_wire_items: dict[str, WireItem] | None = None,
        select_wire_ids: set[str] | None = None,
    ) -> None:
        """Replace `self._wire_graph`, synchronize canvas items, and optionally refresh visuals."""
        self._wire_graph = graph
        self._sync_graphics_from_wire_graph(
            preferred_wire_items=preferred_wire_items,
            select_wire_ids=select_wire_ids,
        )
        if refresh:
            self.refresh_connectivity_visuals()

    # ------------------------------------------------------------------
    # Low-level item registration (called by QUndoCommand redo/undo)
    # ------------------------------------------------------------------

    def _insert_component_item(
        self,
        item: ComponentItem,
        *,
        select: bool = True,
        refresh: bool = True,
    ) -> None:
        if item not in self._placed_items:
            self._placed_items.append(item)
        if item.scene() is not self:
            self.addItem(item)
        if select:
            self.clearSelection()
            item.setSelected(True)
        if refresh:
            self.refresh_connectivity_visuals()
            self.component_placed.emit(item)

    def _remove_component_item(
        self,
        item: ComponentItem,
        *,
        refresh: bool = True,
    ) -> None:
        if item in self._placed_items:
            self._placed_items.remove(item)
        if item.scene() is self:
            self.removeItem(item)
        if refresh:
            self.refresh_connectivity_visuals()

    def _insert_wire_item(
        self,
        wire: WireItem,
        *,
        select: bool = False,
        refresh: bool = True,
    ) -> None:
        if wire not in self._wire_items:
            self._wire_items.append(wire)
        if wire.scene() is not self:
            self.addItem(wire)
        if select:
            wire.setSelected(True)
        if refresh:
            self.refresh_connectivity_visuals()

    def _remove_wire_item(
        self,
        wire: WireItem,
        *,
        refresh: bool = True,
    ) -> None:
        if wire in self._wire_items:
            self._wire_items.remove(wire)
        if wire.scene() is self:
            self.removeItem(wire)
        if refresh:
            self.refresh_connectivity_visuals()

    def _set_explicit_junction(
        self,
        pt: PointPx,
        *,
        present: bool,
        refresh: bool = True,
    ) -> None:
        norm = normalize_point_px(pt, snap=True)
        if present:
            self._explicit_junctions.add(norm)
        else:
            self._explicit_junctions.discard(norm)
        if refresh:
            self.refresh_connectivity_visuals()

    def refresh_connectivity_visuals(self) -> None:
        """Recompute active junction dots (T-junctions + explicit junctions) and repaint."""
        res = self.extract_connectivity_result()
        self._active_junction_dots = res.junction_dots
        unconn_set = set(res.unconnected_components)
        has_connected = len(self._placed_items) > len(unconn_set)
        for item in self._placed_items:
            item.set_unconnected(
                has_connected and item.component.name in unconn_set
            )
        self.update_label_positions()
        self.circuit_modified.emit()
        self.update()

    def update_label_positions(self) -> None:
        """Recompute label candidate positions to avoid overlaps deterministically.

        Candidates: 'default', 'below', 'above', 'right', 'left'.
        A candidate is picked if its visual bounding rectangles do not intersect:
            1. Any other component body.
            2. Any already placed label.
            3. Any wire segment.
        If no candidate is free, keeps 'default'.
        """
        items = sorted(self._placed_items, key=lambda it: it.component.name)
        if not items:
            return

        part_bodies = [(it, it.scene_body_rect()) for it in items]

        wire_segments: list[tuple[PointPx, PointPx]] = []
        for w in self._wire_items:
            pts = w.points
            for i in range(len(pts) - 1):
                wire_segments.append((pts[i], pts[i + 1]))

        placed_labels: list[QRectF] = []
        for it in items:
            chosen_pos = "default"
            for cand in ("default", "below", "above", "right", "left"):
                rects = it.candidate_label_rects_scene(cand)

                # 1. Must not intersect any other part body
                collides_body = any(
                    rect.intersects(body)
                    for rect in rects
                    for other_it, body in part_bodies
                    if other_it is not it
                )
                if collides_body:
                    continue

                # 2. Must not intersect any already placed label
                collides_label = any(
                    rect.intersects(lbl)
                    for rect in rects
                    for lbl in placed_labels
                )
                if collides_label:
                    continue

                # 3. Must not intersect any wire segment
                collides_wire = any(
                    rect_intersects_segment(rect, p1, p2)
                    for rect in rects
                    for p1, p2 in wire_segments
                )
                if collides_wire:
                    continue

                chosen_pos = cand
                break

            it.set_label_position(chosen_pos)
            placed_labels.extend(it.candidate_label_rects_scene(chosen_pos))

    # ------------------------------------------------------------------
    # Palette Ghost Preview & Component Placement
    # ------------------------------------------------------------------

    def show_ghost_preview(self, component_id: str, scene_pos: QPointF) -> None:
        """Create or move the snapping ghost preview item for `component_id`."""
        if (
            self._ghost_item is None
            or self._ghost_item.component.component_type != component_id
        ):
            self.hide_ghost_preview()
            temp_model = create_default_component_model(component_id, "PREVIEW")
            self._ghost_item = ComponentItem(
                temp_model,
                self.symbol_library,
                theme=self.theme,
                is_ghost=True,
            )
            self.addItem(self._ghost_item)

        self._ghost_item.set_snapped_pos(scene_pos.x(), scene_pos.y())

    def hide_ghost_preview(self) -> None:
        """Remove the ghost preview item from the scene if active."""
        if self._ghost_item is not None:
            self.removeItem(self._ghost_item)
            self._ghost_item = None

    def place_new_component(
        self,
        component_id: str,
        scene_pos: QPointF,
        *,
        rotation_deg: int = 0,
        flipped_h: bool = False,
    ) -> ComponentItem:
        """Instantiate and place a new component snapped to the grid via `QUndoStack`."""
        name = self.next_component_name(component_id)
        model = create_default_component_model(component_id, name)
        if component_id in ("voltmeter", "ammeter"):
            meter_mode = (
                self.settings.get_new_meters_mode()
                if self.settings is not None
                else "ideal"
            )
            if meter_mode == "ideal":
                model.is_ideal = True
        item = ComponentItem(
            model,
            self.symbol_library,
            theme=self.theme,
            is_ghost=False,
        )
        px = scene_pos[0] if isinstance(scene_pos, (tuple, list)) else scene_pos.x()
        py = scene_pos[1] if isinstance(scene_pos, (tuple, list)) else scene_pos.y()
        sx, sy = snap_to_grid(px, py, GRID_PITCH_PX)
        item.setPos(sx, sy)
        if rotation_deg:
            item.set_rotation_deg(rotation_deg)
        if flipped_h:
            item.set_flipped_h(flipped_h)

        self.undo_stack.push(AddComponentCommand(self, item))
        return item

    # ------------------------------------------------------------------
    # Wiring, Terminal Hit-Testing, Live Wire Preview & Junction Dots
    # ------------------------------------------------------------------

    def all_component_terminals_px(self) -> dict[str, dict[str, PointPx]]:
        """Return `{comp_name: {term_name: (x_px, y_px)}}` for all placed components."""
        return {
            item.component.name: item.terminal_scene_positions_px()
            for item in self._placed_items
        }

    TERMINAL_SNAP_RADIUS_PX: float = 10.0  # Snap radius of half a grid cell (10 px)

    def find_terminal_near(
        self,
        scene_pos: QPointF | tuple[float, float],
        *,
        tolerance_px: float = 10.0,
    ) -> tuple[ComponentItem, str, PointPx] | None:
        """Find the closest component terminal within `tolerance_px` (default 10 px) of `scene_pos`."""
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])
        best_dist = tolerance_px
        best_match: tuple[ComponentItem, str, PointPx] | None = None

        for item in self._placed_items:
            for tname, pt in item.terminal_scene_positions_px().items():
                dist = math.hypot(sx - pt[0], sy - pt[1])
                if dist <= best_dist:
                    best_dist = dist
                    best_match = (item, tname, pt)
        return best_match

    def find_wire_point_near(
        self,
        scene_pos: QPointF | tuple[float, float],
        *,
        tolerance_px: float = 10.0,
    ) -> PointPx | None:
        """Return the snapped grid point on an existing wire if within `tolerance_px`."""
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])
        snapped = normalize_point_px((sx, sy), snap=True)
        if math.hypot(sx - snapped[0], sy - snapped[1]) > tolerance_px:
            return None
        for wire in self._wire_items:
            if point_on_wire(snapped, wire.points):
                return snapped
        return None

    def update_hover_highlight(self, scene_pos: QPointF) -> PointPx | None:
        """Update the highlighted terminal (or wire snap target) under `scene_pos`."""
        term_hit = self.find_terminal_near(scene_pos)
        new_hover: PointPx | None = None
        if term_hit is not None:
            new_hover = term_hit[2]
        elif self.is_drawing_wire:
            wire_hit = self.find_wire_point_near(scene_pos)
            if wire_hit is not None:
                new_hover = wire_hit

        if new_hover != self._hovered_terminal_pt:
            self._hovered_terminal_pt = new_hover
            self.update()
        return self._hovered_terminal_pt

    def start_wire_preview(
        self, start_pos: QPointF | tuple[float, float]
    ) -> PointPx:
        """Begin interactive right-angle wire drawing from `start_pos`."""
        term_hit = self.find_terminal_near(start_pos)
        if term_hit is not None:
            start_pt = term_hit[2]
        else:
            start_pt = normalize_point_px(
                (
                    start_pos.x()
                    if isinstance(start_pos, QPointF)
                    else float(start_pos[0]),
                    start_pos.y()
                    if isinstance(start_pos, QPointF)
                    else float(start_pos[1]),
                ),
                snap=True,
            )
        self._wire_start_pt = start_pt
        self._wire_prefer_horiz = True
        if self._preview_wire_item is None:
            self._preview_wire_item = WireItem(
                "PREVIEW_WIRE",
                (start_pt, start_pt),
                theme=self.theme,
                is_preview=True,
            )
            self.addItem(self._preview_wire_item)
        else:
            self._preview_wire_item.set_points((start_pt, start_pt))
        self.update()
        return start_pt

    def update_wire_preview(
        self, cursor_pos: QPointF | tuple[float, float]
    ) -> tuple[PointPx, ...]:
        """Update the live right-angle preview wire from `_wire_start_pt` to `cursor_pos`."""
        if self._wire_start_pt is None:
            return ()

        cx = cursor_pos.x() if isinstance(cursor_pos, QPointF) else float(cursor_pos[0])
        cy = cursor_pos.y() if isinstance(cursor_pos, QPointF) else float(cursor_pos[1])
        term_hit = self.find_terminal_near((cx, cy))
        end_pt = (
            term_hit[2]
            if term_hit is not None
            else normalize_point_px((cx, cy), snap=True)
        )

        # Infer initial horizontal/vertical preference when leaving the start point
        dx = abs(end_pt[0] - self._wire_start_pt[0])
        dy = abs(end_pt[1] - self._wire_start_pt[1])
        if dx == 0 and dy > 0:
            self._wire_prefer_horiz = False
        elif dy == 0 and dx > 0:
            self._wire_prefer_horiz = True

        pts = route_orthogonal_wire(
            self._wire_start_pt,
            end_pt,
            prefer_horizontal_first=self._wire_prefer_horiz,
        )
        if self._preview_wire_item is not None:
            self._preview_wire_item.set_points(pts)
        self.update_hover_highlight(QPointF(float(cx), float(cy)))
        self.update()
        return pts

    def finish_wire_preview(
        self, end_pos: QPointF | tuple[float, float]
    ) -> WireItem | None:
        """Commit the previewed wire onto the canvas via `QUndoStack` and clear preview."""
        if self._wire_start_pt is None:
            return None
        pts = self.update_wire_preview(end_pos)
        self.cancel_wire_preview()
        if len(pts) < 2:
            return None
        return self.add_wire(pts)

    def cancel_wire_preview(self) -> None:
        """Cancel any active wire preview without creating a wire."""
        self._wire_start_pt = None
        if self._preview_wire_item is not None:
            self.removeItem(self._preview_wire_item)
            self._preview_wire_item = None
        self.update()

    def _terminals_at_point(self, pt: PointPx) -> list[TerminalRef]:
        refs: list[TerminalRef] = []
        for item in self._placed_items:
            cname = item.component.name
            for tname, tpt in item.terminal_scene_positions_px().items():
                if tpt == pt:
                    refs.append(TerminalRef(cname, tname))
        return refs

    def _find_junction_at(
        self, pt: PointPx, graph: WireGraph, *, tolerance_px: float = 0.0
    ) -> JunctionRef | None:
        if tolerance_px <= 0.0:
            for j in graph.junctions:
                if j.pos == pt:
                    return JunctionRef(j.id)
            return None
        best_id: str | None = None
        best_dist = tolerance_px
        for j in graph.junctions:
            dist = math.hypot(j.pos[0] - pt[0], j.pos[1] - pt[1])
            if dist <= best_dist:
                best_dist = dist
                best_id = j.id
        return JunctionRef(best_id) if best_id is not None else None

    def _find_wire_interior_at(
        self, pt: PointPx, graph: WireGraph
    ) -> str | None:
        from circuitsim.canvas.connectivity import point_strictly_inside_wire

        term_map = self._terminal_pos_map()
        for w in graph.wires:
            item = self._wire_items_by_id.get(w.id)
            if item is not None:
                if item.scene() is not self or len(item.points) < 2:
                    continue
                pts = item.points
            else:
                pts = self._compute_wire_points_from_graph(w, graph, term_map=term_map)
            if len(pts) >= 2 and point_strictly_inside_wire(pt, pts):
                return w.id
        return None

    def _bond_overlapping_terminals(
        self, graph: WireGraph, pt: PointPx, primary: TerminalRef
    ) -> WireGraph:
        for other in self._terminals_at_point(pt):
            if other != primary:
                if not any(
                    (w.end_a == primary and w.end_b == other)
                    or (w.end_a == other and w.end_b == primary)
                    for w in graph.wires
                ):
                    wid = self._next_unused_wire_id(graph)
                    graph = graph.add_wire(primary, other, wire_id=wid)
        return graph

    def _next_unused_wire_id(self, graph: WireGraph) -> str:
        while True:
            self._wire_counter += 1
            candidate = f"W{self._wire_counter}"
            if not graph.has_wire(candidate):
                return candidate

    def _find_duplicate_graph_wire(
        self, graph: WireGraph, end_a: Attachment, end_b: Attachment
    ) -> Wire | None:
        for w in graph.wires:
            if (w.end_a == end_a and w.end_b == end_b) or (
                w.end_a == end_b and w.end_b == end_a
            ):
                return w
        return None

    def add_wire(
        self,
        points_or_end_a: Sequence[tuple[float, float] | PointPx] | Attachment,
        end_b: Attachment | None = None,
        *,
        hints: Sequence[tuple[float, float] | PointPx] = (),
        wire_id: str | None = None,
    ) -> WireItem | None:
        """Add an orthogonal wire via `QUndoStack`.

        Supports either:
        - Attachment API: `scene.add_wire(end_a, end_b, hints=(), wire_id=None)`
        - Coordinate sequence API: `scene.add_wire([(x1, y1), ...])`
        """
        if isinstance(points_or_end_a, (TerminalRef, JunctionRef, FreeEnd)):
            if end_b is None:
                raise TypeError("add_wire(end_a, end_b) requires end_b when end_a is an Attachment")
            att_a = points_or_end_a
            att_b = end_b
            if att_a == att_b:
                return None
            dup = self._find_duplicate_graph_wire(self._wire_graph, att_a, att_b)
            if dup is not None:
                return self._wire_items_by_id.get(dup.id)
            norm_hints = tuple(wire_snap_px(h) for h in hints)
            wid = wire_id if wire_id is not None else self.next_wire_id()
            try:
                new_graph = self._wire_graph.add_wire(
                    att_a, att_b, hints=norm_hints, wire_id=wid
                )
            except WireGraphError:
                return None
            wire_obj = new_graph.wire(wid)
            pts = self._compute_wire_points_from_graph(wire_obj, new_graph)
            wire_item = WireItem(
                wid,
                pts,
                theme=self.theme,
                end_a=att_a,
                end_b=att_b,
                hints=norm_hints,
            )
            self.undo_stack.push(
                AddWireCommand(
                    self, wire_item, old_graph=self._wire_graph, new_graph=new_graph
                )
            )
            return wire_item

        points = points_or_end_a
        if len(points) < 2:
            return None

        # Snap endpoints to nearest terminal if within snap radius (half a grid cell = 10 px)
        raw_pts = [normalize_point_px(p, snap=False) for p in points]
        term_start = self.find_terminal_near(
            raw_pts[0], tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
        )
        if term_start is not None:
            raw_pts[0] = term_start[2]
        term_end = self.find_terminal_near(
            raw_pts[-1], tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
        )
        if term_end is not None:
            raw_pts[-1] = term_end[2]

        pts = simplify_orthogonal_points(raw_pts, snap=True)
        if len(pts) < 2:
            return None

        # Rule 3: Ignore exact duplicate wire polylines (forward or reversed)
        rev_pts = tuple(reversed(pts))
        for existing in self._wire_items:
            if existing.points == pts or existing.points == rev_pts:
                return existing

        working_graph = self._wire_graph

        if term_start is not None:
            att_a: Attachment = TerminalRef(term_start[0].component.name, term_start[1])
        else:
            j_a = self._find_junction_at(pts[0], working_graph)
            if j_a is not None:
                att_a = j_a
            else:
                split_wid = self._find_wire_interior_at(pts[0], working_graph)
                if split_wid is not None:
                    split_res = working_graph.split_wire_at(split_wid, pts[0])
                    working_graph = split_res.graph
                    att_a = JunctionRef(split_res.junction_id)
                else:
                    att_a = FreeEnd(*pts[0])

        if term_end is not None:
            att_b: Attachment = TerminalRef(term_end[0].component.name, term_end[1])
        else:
            j_b = self._find_junction_at(pts[-1], working_graph)
            if j_b is not None:
                att_b = j_b
            else:
                split_wid = self._find_wire_interior_at(pts[-1], working_graph)
                if split_wid is not None:
                    split_res = working_graph.split_wire_at(split_wid, pts[-1])
                    working_graph = split_res.graph
                    att_b = JunctionRef(split_res.junction_id)
                else:
                    att_b = FreeEnd(*pts[-1])

        if att_a == att_b:
            return None

        dup = self._find_duplicate_graph_wire(working_graph, att_a, att_b)
        if dup is not None:
            if working_graph != self._wire_graph:
                self.undo_stack.push(
                    SetWireGraphCommand(
                        self,
                        self._wire_graph,
                        working_graph,
                        description=f"Split Wire {dup.id}",
                    )
                )
            return self._wire_items_by_id.get(dup.id)

        norm_hints = tuple(pts[1:-1])
        wid = wire_id if wire_id is not None else self._next_unused_wire_id(working_graph)
        try:
            new_graph = working_graph.add_wire(
                att_a, att_b, hints=norm_hints, wire_id=wid
            )
        except WireGraphError:
            return None

        wire = WireItem(
            wid,
            pts,
            theme=self.theme,
            end_a=att_a,
            end_b=att_b,
            hints=norm_hints,
        )
        self.undo_stack.push(
            AddWireCommand(
                self, wire, old_graph=self._wire_graph, new_graph=new_graph
            )
        )
        return wire

    def add_routed_wire(
        self,
        start_px: tuple[float, float] | PointPx,
        end_px: tuple[float, float] | PointPx,
        *,
        prefer_horizontal_first: bool = True,
    ) -> WireItem | None:
        """Route a right-angle wire between `start_px` and `end_px` and add it via `QUndoStack`."""
        pts = route_orthogonal_wire(
            start_px,
            end_px,
            prefer_horizontal_first=prefer_horizontal_first,
        )
        return self.add_wire(pts)

    def toggle_junction_at(
        self, pos: QPointF | tuple[float, float] | PointPx
    ) -> bool:
        """Toggle an explicit junction dot at the snapped grid position via `QUndoStack`."""
        raw = (
            (pos.x(), pos.y())
            if isinstance(pos, QPointF)
            else (float(pos[0]), float(pos[1]))
        )
        pt = normalize_point_px(raw, snap=True)
        adding = pt not in self._explicit_junctions
        self.undo_stack.push(ToggleJunctionCommand(self, pt, adding=adding))
        return adding

    # ------------------------------------------------------------------
    # Component Move Tracking, Rotate, Flip, Delete, Copy/Cut/Paste
    # ------------------------------------------------------------------

    def begin_component_move_tracking(self) -> None:
        """Snapshot component positions, terminal positions, and wire polylines before drag."""
        self._move_tracking_active = True
        self._move_start_positions = {
            item: (item.pos().x(), item.pos().y())
            for item in self._placed_items
        }
        self._move_start_terminals = self.all_component_terminals_px()
        self._move_start_wires = {
            wire: wire.points for wire in self._wire_items
        }

    def on_component_position_changed(self, item: ComponentItem) -> None:
        """Re-route wires attached to moved components in real time during mouse drag."""
        _ = item
        if self._suppress_live_reroute or not self._move_tracking_active:
            return
        self._sync_graphics_from_wire_graph()
        self.refresh_connectivity_visuals()

    def commit_component_move_tracking(self) -> bool:
        """Finalize a mouse drag-move and push `MoveComponentsCommand` if anything moved."""
        if not self._move_tracking_active:
            return False
        self._move_tracking_active = False

        moved_old: dict[ComponentItem, tuple[float, float]] = {}
        moved_new: dict[ComponentItem, tuple[float, float]] = {}
        for item, old_pos in self._move_start_positions.items():
            cur_pos = (item.pos().x(), item.pos().y())
            if cur_pos != old_pos:
                moved_old[item] = old_pos
                moved_new[item] = cur_pos

        if not moved_new:
            return False

        old_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        new_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        for wire, old_pts in self._move_start_wires.items():
            if wire.points != old_pts:
                old_wires[wire] = old_pts
                new_wires[wire] = wire.points

        self.undo_stack.push(
            MoveComponentsCommand(
                self,
                moved_old,
                moved_new,
                old_wires,
                new_wires,
                old_graph=self._wire_graph,
                new_graph=self._wire_graph,
            )
        )
        return True

    def move_components_by(
        self,
        items: Sequence[ComponentItem],
        dx_px: float,
        dy_px: float,
        *,
        merge: bool = True,
    ) -> bool:
        """Programmatically move `items` by `(dx_px, dy_px)` and re-route attached wires via `QUndoStack`."""
        target_items = [it for it in items if it in self._placed_items]
        if not target_items:
            return False

        old_wire_map = {w: w.points for w in self._wire_items}
        old_positions: dict[ComponentItem, tuple[float, float]] = {}
        new_positions: dict[ComponentItem, tuple[float, float]] = {}

        self._suppress_live_reroute = True
        try:
            for item in target_items:
                ox, oy = item.pos().x(), item.pos().y()
                nx, ny = snap_to_grid(ox + dx_px, oy + dy_px, GRID_PITCH_PX)
                if (nx, ny) != (ox, oy):
                    old_positions[item] = (ox, oy)
                    new_positions[item] = (nx, ny)
                    item.setPos(QPointF(nx, ny))
        finally:
            self._suppress_live_reroute = False

        if not new_positions:
            return False

        self._sync_graphics_from_wire_graph()
        old_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        new_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        for w in self._wire_items:
            old_pts = old_wire_map.get(w, w.points)
            if w.points != old_pts:
                old_wires[w] = old_pts
                new_wires[w] = w.points

        self.undo_stack.push(
            MoveComponentsCommand(
                self,
                old_positions,
                new_positions,
                old_wires,
                new_wires,
                old_graph=self._wire_graph,
                new_graph=self._wire_graph,
                mergeable=merge,
            )
        )
        return True

    def rotate_selected(self) -> bool:
        """Rotate selected `ComponentItem`s by 90° CW and re-route attached wires via `QUndoStack`."""
        selected = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        if not selected:
            return False

        old_wire_map = {w: w.points for w in self._wire_items}
        old_rotations = {it: it.rotation_deg for it in selected}
        new_rotations = {it: (it.rotation_deg + 90) % 360 for it in selected}

        for it, deg in new_rotations.items():
            it.set_rotation_deg(deg)

        self._sync_graphics_from_wire_graph()
        old_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        new_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        for w in self._wire_items:
            old_pts = old_wire_map.get(w, w.points)
            if w.points != old_pts:
                old_wires[w] = old_pts
                new_wires[w] = w.points

        self.undo_stack.push(
            RotateComponentsCommand(
                self,
                selected,
                old_rotations,
                new_rotations,
                old_wires,
                new_wires,
                old_graph=self._wire_graph,
                new_graph=self._wire_graph,
            )
        )
        return True

    def flip_selected(self) -> bool:
        """Flip selected `ComponentItem`s horizontally and re-route attached wires via `QUndoStack`."""
        selected = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        if not selected:
            return False

        old_wire_map = {w: w.points for w in self._wire_items}
        old_flips = {it: it.flipped_h for it in selected}
        new_flips = {it: not it.flipped_h for it in selected}

        for it, flipped in new_flips.items():
            it.set_flipped_h(flipped)

        self._sync_graphics_from_wire_graph()
        old_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        new_wires: dict[WireItem, tuple[PointPx, ...]] = {}
        for w in self._wire_items:
            old_pts = old_wire_map.get(w, w.points)
            if w.points != old_pts:
                old_wires[w] = old_pts
                new_wires[w] = w.points

        self.undo_stack.push(
            FlipComponentsCommand(
                self,
                selected,
                old_flips,
                new_flips,
                old_wires,
                new_wires,
                old_graph=self._wire_graph,
                new_graph=self._wire_graph,
            )
        )
        return True

    def delete_selected(self, *, description: str = "Delete Selection") -> bool:
        """Delete selected components and wires (plus all wires attached to deleted components)."""
        selected_comps = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        selected_wires = [
            it for it in self.selectedItems() if isinstance(it, WireItem)
        ]
        if not selected_comps and not selected_wires:
            return False

        old_graph = self._wire_graph
        new_graph = old_graph
        for comp in selected_comps:
            new_graph = new_graph.remove_component(comp.component.name)
        for w in selected_wires:
            if new_graph.has_wire(w.wire_id):
                new_graph = new_graph.remove_wire(w.wire_id)

        remaining_ids = {w.id for w in new_graph.wires}
        wires_to_delete = [w for w in self._wire_items if w.wire_id not in remaining_ids]

        self.undo_stack.push(
            DeleteItemsCommand(
                self,
                selected_comps,
                wires_to_delete,
                old_graph=old_graph,
                new_graph=new_graph,
                description=description,
            )
        )
        return True

    def copy_selected(self) -> bool:
        """Copy selected components and wires to the internal clipboard."""
        selected_comps = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        selected_wires = [
            it for it in self.selectedItems() if isinstance(it, WireItem)
        ]
        if not selected_comps and not selected_wires:
            return False

        # Also include wires whose both endpoints sit on terminals of selected_comps
        selected_term_pts: set[PointPx] = set()
        for comp in selected_comps:
            selected_term_pts.update(comp.terminal_scene_positions_px().values())

        wires_to_copy: list[tuple[PointPx, ...]] = []
        seen_ids: set[str] = set()
        for w in self._wire_items:
            pts = w.points
            both_on_selected = (
                len(pts) >= 2
                and pts[0] in selected_term_pts
                and pts[-1] in selected_term_pts
            )
            if w in selected_wires or both_on_selected:
                if w.wire_id not in seen_ids:
                    wires_to_copy.append(pts)
                    seen_ids.add(w.wire_id)

        comp_entries = tuple(
            _ClipboardComponentEntry(
                component_snapshot=copy.deepcopy(it.component),
                pos_px=(it.pos().x(), it.pos().y()),
                rotation_deg=it.rotation_deg,
                flipped_h=it.flipped_h,
            )
            for it in selected_comps
        )
        self._clipboard = _ClipboardData(
            components=comp_entries,
            wires=tuple(wires_to_copy),
            junctions=(),
        )
        self._paste_serial = 0
        return True

    def cut_selected(self) -> bool:
        """Copy selected items to clipboard and delete them via `QUndoStack`."""
        if not self.copy_selected():
            return False
        return self.delete_selected(description="Cut Selection")

    def paste_clipboard(
        self, offset_px: tuple[float, float] | None = None
    ) -> bool:
        """Paste clipboard contents onto the canvas with a grid-aligned offset via `QUndoStack`."""
        if self._clipboard is None:
            return False

        self._paste_serial += 1
        if offset_px is None:
            step = float(GRID_PITCH_PX * 2 * self._paste_serial)
            dx, dy = step, step
        else:
            dx, dy = snap_to_grid(offset_px[0], offset_px[1], GRID_PITCH_PX)

        new_comps: list[ComponentItem] = []
        assigned_names: set[str] = set()
        for entry in self._clipboard.components:
            cloned = copy.deepcopy(entry.component_snapshot)
            new_name = self.next_component_name(
                cloned.component_type, extra_taken=assigned_names
            )
            assigned_names.add(new_name)
            cloned.name = new_name
            cloned.label = new_name
            item = ComponentItem(
                cloned,
                self.symbol_library,
                theme=self.theme,
                is_ghost=False,
            )
            nx, ny = snap_to_grid(
                entry.pos_px[0] + dx, entry.pos_px[1] + dy, GRID_PITCH_PX
            )
            item.setPos(QPointF(nx, ny))
            item.set_rotation_deg(entry.rotation_deg)
            item.set_flipped_h(entry.flipped_h)
            new_comps.append(item)

        new_wires: list[WireItem] = []
        id_x = int(round(dx))
        id_y = int(round(dy))
        working_graph = self._wire_graph
        pasted_term_map: dict[PointPx, TerminalRef] = {}
        for comp_item in new_comps:
            cname = comp_item.component.name
            for tname, tpt in comp_item.terminal_scene_positions_px().items():
                pasted_term_map.setdefault(tpt, TerminalRef(cname, tname))

        for pts in self._clipboard.wires:
            shifted_pts = tuple((px + id_x, py + id_y) for px, py in pts)
            wid = self._next_unused_wire_id(working_graph)
            att_a: Attachment = pasted_term_map.get(shifted_pts[0], FreeEnd(*shifted_pts[0]))
            att_b: Attachment = pasted_term_map.get(shifted_pts[-1], FreeEnd(*shifted_pts[-1]))
            hints_pts = shifted_pts[1:-1]
            if att_a != att_b:
                working_graph = working_graph.add_wire(
                    att_a, att_b, hints=hints_pts, wire_id=wid
                )
            new_wires.append(
                WireItem(
                    wid,
                    shifted_pts,
                    theme=self.theme,
                    end_a=att_a,
                    end_b=att_b,
                    hints=hints_pts,
                )
            )

        self.undo_stack.push(
            PasteItemsCommand(
                self,
                new_comps,
                new_wires,
                old_graph=self._wire_graph,
                new_graph=working_graph,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Netlist Extraction (CircuitScene.build_circuit)
    # ------------------------------------------------------------------

    def extract_connectivity_result(self) -> ConnectivityResult:
        """Run the pure-Python Union-Find connectivity extractor on the current scene."""
        placed_specs = [item.to_placed_spec() for item in self._placed_items]
        wire_specs = [wire.to_wire_spec() for wire in self._wire_items]
        return extract_connectivity(
            placed_specs,
            wire_specs,
            self._explicit_junctions,
            symbol_catalog=self.symbol_library.metadata_catalog,
        )

    def build_circuit(self) -> Circuit:
        """Build and return a solver-ready `Circuit` from the schematic's components and wires."""
        return self.extract_connectivity_result().circuit

    def toggle_switch_item(self, item: ComponentItem) -> bool:
        """Toggle a `Switch` `ComponentItem` with lever animation and notify listeners."""
        if not isinstance(item.component, Switch):
            return False
        new_closed = not item.component.closed
        item.animate_switch_state(new_closed)
        self.switch_toggled_by_user.emit(item.component.name, new_closed)
        self.selection_info_changed.emit(item if item.isSelected() else None)
        return new_closed

    def edit_component_parameters(
        self,
        item: ComponentItem,
        new_params: dict[str, object],
        *,
        show_value: bool | None = None,
    ) -> bool:
        """Apply parameter changes to `item` via `QUndoStack` (`EditComponentParametersCommand`)."""
        comp = item.component
        old_params: dict[str, object] = {}
        parsed_new: dict[str, object] = {}
        param_keys = {p.key for p in comp.get_editable_parameters()}
        if isinstance(comp, LogicGate):
            old_params = {
                k: (comp.get_parameter_value(k) if k in param_keys else getattr(comp, k))
                for k in new_params
                if k in param_keys or hasattr(comp, k)
            }
            if "family" not in old_params and any(
                k in new_params for k in ("v_il", "v_ih", "v_ol", "v_oh")
            ):
                old_params["family"] = comp.family
            parsed_new = comp.set_parameters_batch(new_params)  # type: ignore[arg-type]
            if "family" in old_params and "family" not in parsed_new:
                parsed_new["family"] = comp.family
            comp.set_parameters_batch(old_params)  # type: ignore[arg-type]
        else:
            for k, raw_val in new_params.items():
                if k in param_keys:
                    old_params[k] = comp.get_parameter_value(k)
                    parsed_new[k] = comp.set_parameter_value(k, raw_val)  # type: ignore[arg-type]
                    comp.set_parameter_value(k, old_params[k])  # type: ignore[arg-type]
                elif hasattr(comp, k):
                    old_params[k] = getattr(comp, k)
                    parsed_new[k] = raw_val

        old_show = comp.show_value
        new_show = old_show if show_value is None else bool(show_value)
        if old_params == parsed_new and old_show == new_show:
            return False

        self.undo_stack.push(
            EditComponentParametersCommand(
                self,
                item,
                old_params,  # type: ignore[arg-type]
                parsed_new,  # type: ignore[arg-type]
                old_show_value=old_show,
                new_show_value=new_show,
            )
        )
        return True

    def clear_diagnostics(self) -> None:
        """Clear all problem highlighting on components and wires."""
        for item in self._placed_items:
            item.set_diagnostic(None, "", code="")
        for wire in self._wire_items:
            wire.set_diagnostic(None, "")

    def apply_simulation_frame(self, frame: SimulationFrame) -> None:
        """Apply telemetry and problem diagnostics from a `SimulationFrame` to canvas items."""
        if frame.state in (SimulationState.STOPPED, SimulationState.ERROR):
            self.latest_node_voltages = {}
            self.is_simulation_running = False
            for item in self._placed_items:
                item.reset_live_telemetry()
        else:
            self.latest_node_voltages = dict(frame.node_voltages)
            self.is_simulation_running = True
            for item in self._placed_items:
                telem = frame.components.get(item.component.name)
                if telem is not None:
                    item.apply_telemetry(telem)

        # Map diagnostics to components and wires (error takes precedence over warning)
        comp_diag: dict[str, tuple[str, str, str]] = {}
        wire_diag: dict[str, tuple[str, str]] = {}
        unconnected_names: set[str] = set()
        for diag in frame.diagnostics:
            if diag.code == "unconnected_component":
                unconnected_names.update(diag.component_names)
            for cname in diag.component_names:
                prev = comp_diag.get(cname)
                if prev is None or prev[0] != "error":
                    comp_diag[cname] = (diag.severity, diag.message, diag.code)
            for wid in diag.wire_ids:
                prev_w = wire_diag.get(wid)
                if prev_w is None or prev_w[0] != "error":
                    wire_diag[wid] = (diag.severity, diag.message)

        for item in self._placed_items:
            cname = item.component.name
            item.set_unconnected(cname in unconnected_names)
            d = comp_diag.get(cname)
            if d is not None:
                item.set_diagnostic(d[0], d[1], code=d[2])
            else:
                item.set_diagnostic(None, "", code="")

        for wire in self._wire_items:
            dw = wire_diag.get(wire.wire_id)
            if dw is not None:
                wire.set_diagnostic(dw[0], dw[1])
            else:
                wire.set_diagnostic(None, "")

        selected = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        if selected:
            self.selection_info_changed.emit(selected[0])

    def get_component_tooltip(self, item: ComponentItem) -> str:
        """Return the live tooltip for a component item based on current simulation state."""
        return item.get_live_tooltip(is_running=self.is_simulation_running)

    def get_terminal_tooltip(self, item: ComponentItem, terminal_name: str) -> str:
        """Return the live tooltip for a component terminal (terminal name, net name, node voltage)."""
        conn = self.extract_connectivity_result()
        net_name = conn.component_nodes.get(item.component.name, {}).get(terminal_name, "--")
        lines = [
            f"Terminal: {terminal_name}",
            f"Net: {net_name}",
        ]
        if self.is_simulation_running and self.latest_node_voltages and net_name in self.latest_node_voltages:
            v = self.latest_node_voltages[net_name]
            lines.append(f"Node voltage: {format_engineering_value(v, 'V')}")
        else:
            lines.append("Node voltage: --")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save / Load (.circ) support
    # ------------------------------------------------------------------

    def to_document(self, view: ViewState | None = None) -> CircuitDocument:
        """Snapshot the schematic as a Qt-free `CircuitDocument` ready to be saved."""
        return CircuitDocument(
            placed=[item.to_placed_spec() for item in self._placed_items],
            wires=[wire.to_wire_spec() for wire in self._wire_items],
            junctions=sorted(self._explicit_junctions),
            view=view if view is not None else ViewState(theme=self.theme.value),
        )

    def load_document(self, doc: CircuitDocument) -> None:
        """Replace the whole schematic with `doc`, all-or-nothing.

        Step 1 builds every new item without touching the scene. If anything
        goes wrong there, `CircuitFileError` is raised and the current circuit
        is left exactly as it was. Step 2 (the swap) only runs after step 1
        succeeded and contains no code that can reasonably fail.
        """
        try:
            new_items: list[ComponentItem] = []
            for spec in doc.placed:
                item = ComponentItem(
                    spec.component,
                    self.symbol_library,
                    theme=self.theme,
                    is_ghost=False,
                )
                item.setPos(float(spec.center_px[0]), float(spec.center_px[1]))
                item.set_rotation_deg(spec.rotation_deg)
                item.set_flipped_h(spec.flipped_h)
                new_items.append(item)
            new_wires = [
                WireItem(w.wire_id, tuple(w.points), theme=self.theme)
                for w in doc.wires
            ]
            new_junctions = {
                normalize_point_px(p, snap=True) for p in doc.junctions
            }
            # Dry run of netlist extraction: proves the circuit can be analysed.
            extract_connectivity(
                [it.to_placed_spec() for it in new_items],
                [w.to_wire_spec() for w in new_wires],
                new_junctions,
                symbol_catalog=self.symbol_library.metadata_catalog,
            )
        except CircuitFileError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure must keep old scene
            raise CircuitFileError(
                f"The circuit in this file could not be built ({exc}). "
                "Your current circuit was not changed."
            ) from exc

        # ---- point of no return: swap old for new ----
        self.hide_ghost_preview()
        self.cancel_wire_preview()
        for jitem in list(self._junction_items_by_id.values()):
            if jitem.scene() is self:
                self.removeItem(jitem)
        self._junction_items_by_id.clear()
        for wire in list(self._wire_items):
            self.removeItem(wire)
        for item in list(self._placed_items):
            self.removeItem(item)
        self._wire_items = []
        self._wire_items_by_id = {}
        self._placed_items = []
        self._counters.clear()
        self._explicit_junctions = new_junctions
        for item in new_items:
            self._placed_items.append(item)
            self.addItem(item)

        # Build WireGraph corresponding to the loaded document while preserving WireItems
        loaded_term_map: dict[PointPx, TerminalRef] = {}
        for comp_item in new_items:
            cname = comp_item.component.name
            for tname, tpt in comp_item.terminal_scene_positions_px().items():
                loaded_term_map.setdefault(tpt, TerminalRef(cname, tname))

        loaded_graph = WireGraph()
        for jp in sorted(new_junctions):
            loaded_graph = loaded_graph.add_junction(jp)
        jpos_to_ref: dict[PointPx, JunctionRef] = {
            j.pos: JunctionRef(j.id) for j in loaded_graph.junctions
        }

        for wire in new_wires:
            pts = wire.points
            if len(pts) < 2:
                continue
            att_a: Attachment = loaded_term_map.get(
                pts[0], jpos_to_ref.get(pts[0], FreeEnd(*pts[0]))
            )
            att_b: Attachment = loaded_term_map.get(
                pts[-1], jpos_to_ref.get(pts[-1], FreeEnd(*pts[-1]))
            )
            hints_pts = pts[1:-1]
            wire.end_a = att_a
            wire.end_b = att_b
            wire.hints = hints_pts
            if att_a != att_b and not loaded_graph.has_wire(wire.wire_id):
                loaded_graph = loaded_graph.add_wire(
                    att_a, att_b, hints=hints_pts, wire_id=wire.wire_id
                )
            self._wire_items.append(wire)
            self._wire_items_by_id[wire.wire_id] = wire
            self.addItem(wire)
        self._wire_graph = loaded_graph
        self._sync_junction_items_from_wire_graph()

        # Keep future wire ids unique: continue after the highest loaded "W<n>".
        numbers = [
            int(w.wire_id[1:])
            for w in new_wires
            if w.wire_id.startswith("W") and w.wire_id[1:].isdigit()
        ]
        self._wire_counter = max(numbers, default=0)
        self.undo_stack.clear()  # a freshly loaded file has no undo history
        self.clearSelection()
        self.refresh_connectivity_visuals()
        self.selection_info_changed.emit(None)
        self.update()

    def clear_schematic(self) -> None:
        """Remove all placed components, wires, junctions, and clear the undo stack."""
        self.hide_ghost_preview()
        self.cancel_wire_preview()
        for jitem in list(self._junction_items_by_id.values()):
            if jitem.scene() is self:
                self.removeItem(jitem)
        self._junction_items_by_id.clear()
        for wire in list(self._wire_items):
            self.removeItem(wire)
        for item in list(self._placed_items):
            self.removeItem(item)
        self._wire_items.clear()
        self._wire_items_by_id.clear()
        self._wire_graph = WireGraph()
        self._placed_items.clear()
        self._explicit_junctions.clear()
        self._active_junction_dots.clear()
        self._counters.clear()
        self._wire_counter = 0
        self.undo_stack.clear()
        self.selection_info_changed.emit(None)
        self.circuit_modified.emit()
        self.update()

    def _on_selection_changed(self) -> None:
        if self._is_shutting_down:
            return
        selected = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        self.selection_info_changed.emit(selected[0] if selected else None)

    def shutdown_scene(self) -> None:
        """Disconnect internal signals and clear selection before scene destruction."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        try:
            self.selectionChanged.disconnect(self._on_selection_changed)
        except (RuntimeError, TypeError):
            pass
        try:
            self.clearSelection()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Background Grid & Foreground Junction / Terminal Highlight Drawing
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw the themed canvas background and 20 px minor / 100 px major grid."""
        palette = get_theme_palette(self.theme)
        painter.fillRect(rect, QColor(palette.canvas_bg))

        if not self.show_grid:
            return

        minor_step = GRID_PITCH_PX
        major_step = GRID_PITCH_PX * 5

        left = int(math.floor(rect.left() / minor_step) * minor_step)
        right = int(math.ceil(rect.right() / minor_step) * minor_step)
        top = int(math.floor(rect.top() / minor_step) * minor_step)
        bottom = int(math.ceil(rect.bottom() / minor_step) * minor_step)

        minor_lines: list[QLineF] = []
        major_lines: list[QLineF] = []

        for x in range(left, right + minor_step, minor_step):
            line = QLineF(float(x), rect.top(), float(x), rect.bottom())
            if x % major_step == 0:
                major_lines.append(line)
            else:
                minor_lines.append(line)

        for y in range(top, bottom + minor_step, minor_step):
            line = QLineF(rect.left(), float(y), rect.right(), float(y))
            if y % major_step == 0:
                major_lines.append(line)
            else:
                minor_lines.append(line)

        painter.save()
        painter.setPen(QPen(QColor(palette.grid_minor), 1.0))
        painter.drawLines(minor_lines)

        painter.setPen(QPen(QColor(palette.grid_major), 1.0))
        painter.drawLines(major_lines)

        # Subtle origin cross at (0, 0)
        origin_pen = QPen(QColor(palette.grid_origin), 1.2)
        painter.setPen(origin_pen)
        painter.drawLine(QLineF(-10.0, 0.0, 10.0, 0.0))
        painter.drawLine(QLineF(0.0, -10.0, 0.0, 10.0))
        painter.restore()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw filled junction dots (T-junctions + explicit junctions) and terminal hover rings."""
        _ = rect
        try:
            palette = get_theme_palette(self.theme)
            painter.save()

            # 1. Draw any remaining active junction dots not already rendered by a JunctionItem
            rendered_by_items = {
                (int(round(j.pos().x())), int(round(j.pos().y())))
                for j in self._junction_items_by_id.values()
            }
            junc_color = QColor(palette.symbol_stroke)
            painter.setPen(QPen(junc_color, 1.2))
            painter.setBrush(QBrush(junc_color))
            for jx, jy in self._active_junction_dots:
                if (jx, jy) in rendered_by_items:
                    continue
                painter.drawEllipse(QPointF(float(jx), float(jy)), 4.2, 4.2)

            # 2. Draw glowing highlight ring around hovered terminal or wire snap target
            if self._hovered_terminal_pt is not None:
                hx, hy = self._hovered_terminal_pt
                hover_col = QColor(palette.terminal_hover)
                fill_col = QColor(palette.terminal_hover)
                fill_col.setAlpha(65)
                painter.setPen(QPen(hover_col, 2.0))
                painter.setBrush(QBrush(fill_col))
                painter.drawEllipse(QPointF(float(hx), float(hy)), 7.0, 7.0)

            painter.restore()
        except Exception as exc:
            if not getattr(self, "_foreground_paint_error_logged", False):
                self._foreground_paint_error_logged = True
                try:
                    from circuitsim.app.crash_handler import write_crash_log

                    write_crash_log(type(exc), exc, exc.__traceback__, count=1)
                except Exception:
                    pass
            try:
                painter.save()
                painter.setPen(QPen(QColor("#FF5252"), 1.5, Qt.PenStyle.DashLine))
                for pt in self._active_junction_dots:
                    r = QRectF(float(pt[0]) - 6.0, float(pt[1]) - 6.0, 12.0, 12.0)
                    painter.drawRect(r)
                painter.restore()
            except Exception:
                pass
