"""QGraphicsScene for the schematic editor with 20 px grid, wiring, undo/redo, and netlist extraction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from circuitsim.app.settings import AppSettings

from PySide6.QtCore import QLineF, QObject, QPointF, QRectF, Qt, QTimer, Signal
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
    MoveJunctionCommand,
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
    normalize_point_px,
    point_on_wire,
    simplify_orthogonal_points,
)
from circuitsim.canvas.symbol_geometry import (
    GRID_PITCH_PX,
    snap_to_grid,
    transform_terminal_position,
)
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
    CopiedSubgraph,
    FreeEnd,
    JunctionRef,
    TerminalRef,
    Wire,
    WireGraph,
    WireGraphError,
    extract_subgraph,
    paste_subgraph,
    snap_px as wire_snap_px,
)
from circuitsim.wiring.netlist import build_circuit as wiring_build_circuit
from circuitsim.wiring.overlay import (
    compute_current_dot_velocity,
    estimate_wire_currents,
    format_power_probe_text,
    map_voltage_to_rgb,
)
from circuitsim.wiring.router import (
    normalize_dir as wire_normalize_dir,
    route as wire_route,
    simplify_orthogonal_points as wire_simplify_points,
    slide_segment_hints as wire_slide_segment_hints,
)
from circuitsim.wiring.serializer import CircuitDocumentV2, migrate_v1_to_v2


@dataclass(frozen=True)
class _ClipboardComponentEntry:
    component_snapshot: Component
    pos_px: tuple[float, float]
    rotation_deg: int
    flipped_h: bool


@dataclass(frozen=True)
class _ClipboardData:
    components: tuple[_ClipboardComponentEntry, ...]
    subgraph: CopiedSubgraph
    explicit_junctions: tuple[PointPx, ...] = ()


class CircuitScene(QGraphicsScene):
    """Schematic canvas scene managing components, orthogonal wires, junctions, and undo/redo."""

    component_placed = Signal(object)  # Emits placed ComponentItem
    selection_info_changed = Signal(object)  # Emits selected ComponentItem or None
    circuit_modified = Signal()  # Emits whenever components, wires, or junctions change
    switch_toggled_by_user = Signal(str, bool)  # (switch_name, closed)
    component_parameters_edited = Signal(str, object)  # (comp_name, dict[str, float])
    status_notice = Signal(str)  # Non-modal status/feedback note for the user

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
        self._pin_to_pin_bonds: list[tuple[ComponentItem, str, ComponentItem, str]] = []
        self._explicit_junctions: set[PointPx] = set()
        self._active_junction_dots: set[PointPx] = set()

        # Ghost component preview during palette drag
        self._ghost_item: ComponentItem | None = None

        # Interactive wire routing preview & terminal hover
        self._hovered_terminal_pt: PointPx | None = None
        self._wire_start_pt: PointPx | None = None
        self._wire_start_att: Attachment | None = None
        self._wire_end_att: Attachment | None = None
        self._wire_hints: list[PointPx] = []
        self._wire_preview_previous: tuple[PointPx, ...] | None = None
        self._wire_prefer_horiz: bool = True
        self._preview_wire_item: WireItem | None = None
        self.last_status_notice: str = ""

        # Interactive wire segment dragging state
        self._seg_drag_old_graph: WireGraph | None = None
        self._seg_drag_wire_id: str | None = None
        self._seg_drag_index: int = -1
        self._seg_drag_orient: str = ""
        self._seg_drag_orig_pts: tuple[PointPx, ...] = ()
        self._seg_drag_orig_hints: tuple[PointPx, ...] = ()

        # Interactive wire endpoint editing state
        self._end_drag_old_graph: WireGraph | None = None
        self._end_drag_wire_id: str | None = None
        self._end_drag_which: str = ""
        self._end_drag_orig_att: Attachment | None = None
        self._end_drag_orig_pts: tuple[PointPx, ...] = ()
        self._hovered_wire_end_pt: PointPx | None = None
        self._dragging_junction_id: str | None = None
        self._dragging_junction_start_pos: PointPx | None = None
        self._dragging_junction_start_graph: WireGraph | None = None
        self._dragging_junction_explicit: bool = False

        # Component drag-move tracking for live attached-wire re-routing & QUndoStack
        self._move_tracking_active: bool = False
        self._suppress_live_reroute: bool = False
        self._suppress_selection_info: bool = False
        self._move_start_positions: dict[ComponentItem, tuple[float, float]] = {}
        self._move_start_graph: WireGraph = self._wire_graph
        self._move_start_wire_points: dict[str, tuple[PointPx, ...]] = {}

        # Internal clipboard for Copy / Cut / Paste
        self._clipboard: _ClipboardData | None = None
        self._paste_serial: int = 0

        # Live simulation telemetry cache & visual intuition overlays (F3)
        self.latest_node_voltages: dict[str, float] = {}
        self.is_simulation_running: bool = False
        self._last_sim_frame: SimulationFrame | None = None
        self.voltage_colors_enabled: bool = False
        self.current_dots_enabled: bool = False
        self.last_power_probe_text: str = ""
        self._dot_anim_timer = QTimer(self)
        self._dot_anim_timer.setInterval(50)
        self._dot_anim_timer.timeout.connect(self.step_current_dot_animation)

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

    def _terminal_pos_map(
        self, *, extra_items: Sequence[ComponentItem] = ()
    ) -> dict[tuple[str, str], PointPx]:
        mapping: dict[tuple[str, str], PointPx] = {}
        for item in (*self._placed_items, *extra_items):
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

    def _component_obstacles_px(
        self, *, exclude_names: set[str] | frozenset[str] = frozenset()
    ) -> tuple[tuple[int, int, int, int], ...]:
        rects: list[tuple[int, int, int, int]] = []
        for item in self._placed_items:
            if item.component.name in exclude_names:
                continue
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
        """Create, update, or remove `JunctionItem` objects matching visible or explicit junctions."""
        graph = self._wire_graph
        graph_junc_positions = {j.pos for j in graph.junctions}
        self._explicit_junctions.intersection_update(graph_junc_positions)
        visible_junctions = {
            j.id: j
            for j in graph.junctions
            if graph.junction_dot_visible(j.id) or j.pos in self._explicit_junctions
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
        frozen_polylines: dict[str, tuple[PointPx, ...]] | None = None,
    ) -> None:
        graph = self._wire_graph
        term_map = self._terminal_pos_map()
        dir_map = self._terminal_dir_map()
        obstacles = self._component_obstacles_px()
        pref = preferred_wire_items or {}
        frozen = frozen_polylines or {}
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
            if wire.id in frozen and len(frozen[wire.id]) >= 2:
                pts = frozen[wire.id]
            else:
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
        if self._move_tracking_active:
            return
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

    def show_ghost_preview(
        self, component_id: str, scene_pos: QPointF, *, rotation_deg: int = 0
    ) -> None:
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

        if rotation_deg % 360 != self._ghost_item.rotation_deg:
            self._ghost_item.set_rotation_deg(rotation_deg % 360)
        self._ghost_item.set_snapped_pos(scene_pos.x(), scene_pos.y())

    def hide_ghost_preview(self) -> None:
        """Remove the ghost preview item from the scene if active."""
        if self._ghost_item is not None:
            self.removeItem(self._ghost_item)
            self._ghost_item = None

    def _component_body_rect_at(
        self,
        component_id: str,
        cx: float,
        cy: float,
        rotation_deg: int = 0,
    ) -> QRectF:
        meta = self.symbol_library.get_metadata(component_id)
        raw_w, raw_h = meta.pixel_size
        # Inset terminal lead extensions (>= 80 px axis) so touching pins at corners do not collide
        inset_x = 22.0 if raw_w >= 80 else 2.0
        inset_y = 22.0 if raw_h >= 80 else 2.0
        rw = max(12.0, float(raw_w) - 2.0 * inset_x)
        rh = max(12.0, float(raw_h) - 2.0 * inset_y)
        if rotation_deg in (90, 270):
            rw, rh = rh, rw
        return QRectF(float(cx) - rw / 2.0, float(cy) - rh / 2.0, rw, rh)

    def _overlaps_any_component_body(
        self,
        component_id: str,
        cx: float,
        cy: float,
        rotation_deg: int = 0,
        *,
        exclude: set[ComponentItem] | None = None,
    ) -> bool:
        cand = self._component_body_rect_at(component_id, cx, cy, rotation_deg)
        excl = exclude or set()
        for existing in self._placed_items:
            if existing in excl:
                continue
            other = self._component_body_rect_at(
                existing.component.component_type,
                existing.pos().x(),
                existing.pos().y(),
                existing.rotation_deg,
            )
            if cand.intersects(other):
                return True
        return False

    def _find_nearest_free_grid_pos(
        self,
        component_id: str,
        cx: float,
        cy: float,
        rotation_deg: int = 0,
        *,
        exclude: set[ComponentItem] | None = None,
    ) -> tuple[float, float]:
        sx, sy = snap_to_grid(cx, cy, GRID_PITCH_PX)
        if component_id == "voltmeter" or not self._overlaps_any_component_body(
            component_id, sx, sy, rotation_deg, exclude=exclude
        ):
            return (sx, sy)
        best: tuple[float, float] | None = None
        best_dist2 = float("inf")
        # Search grid offsets outward (step=20 px) preferring +(60, 0) / +(0, 60) nearest cells
        for ring in range(1, 20):
            for dx_g in range(-ring, ring + 1):
                for dy_g in range(-ring, ring + 1):
                    if max(abs(dx_g), abs(dy_g)) != ring:
                        continue
                    tx = sx + dx_g * GRID_PITCH_PX
                    ty = sy + dy_g * GRID_PITCH_PX
                    d2 = (tx - sx) ** 2 + (ty - sy) ** 2
                    if d2 >= best_dist2:
                        continue
                    if not self._overlaps_any_component_body(
                        component_id, tx, ty, rotation_deg, exclude=exclude
                    ):
                        best = (tx, ty)
                        best_dist2 = d2
            if best is not None and ring >= 5:
                break
        return best if best is not None else (sx, sy + 60.0)

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
        sx, sy = self._find_nearest_free_grid_pos(component_id, px, py, rotation_deg)
        item.setPos(sx, sy)
        if rotation_deg:
            item.set_rotation_deg(rotation_deg)
        if flipped_h:
            item.set_flipped_h(flipped_h)

        self._record_pin_to_pin_bonds_for(item, self._placed_items)
        old_graph = self._wire_graph
        overlay = (
            self._find_overlaid_2terminal_target(item, (sx, sy), self._placed_items)
            if component_id == "voltmeter"
            else None
        )
        if overlay is not None:
            new_graph = self._connect_voltmeter_parallel_with(
                item, overlay[0], overlay[1], overlay[2], old_graph
            )
        else:
            new_graph = self._apply_component_split_on_drop(item, old_graph)
            new_graph = self._auto_connect_graph_endpoints(
                new_graph, extra_items=(item,)
            )
        if new_graph != old_graph:
            self.undo_stack.push(
                AddComponentCommand(
                    self,
                    item,
                    old_graph=old_graph,
                    new_graph=new_graph,
                )
            )
        else:
            self.undo_stack.push(AddComponentCommand(self, item))
        return item

    def _find_overlaid_2terminal_target(
        self,
        vm_item: ComponentItem,
        center_xy: tuple[float, float],
        candidates: Sequence[ComponentItem],
        *,
        tolerance_px: float = 10.0,
    ) -> tuple[ComponentItem, str, str] | None:
        """Return `(target, term_for_pos, term_for_neg)` if `vm_item` overlays a 2-terminal part."""
        if vm_item.component.component_type != "voltmeter":
            return None
        vm_terms = {
            t.name: normalize_point_px(
                transform_terminal_position(
                    t.grid_pos,
                    center_xy,
                    rotation_deg=vm_item.rotation_deg,
                    flipped_h=vm_item.flipped_h,
                    grid_px=GRID_PITCH_PX,
                ),
                snap=True,
            )
            for t in vm_item.metadata.terminals
        }
        p_pos = vm_terms.get("pos")
        p_neg = vm_terms.get("neg")
        if p_pos is None or p_neg is None:
            return None

        best: tuple[float, ComponentItem, str, str] | None = None
        for other in candidates:
            if other is vm_item or other.component.component_type == "voltmeter":
                continue
            if len(other.metadata.terminals) != 2:
                continue
            o_terms = other.terminal_scene_positions_px()
            t0, t1 = (t.name for t in other.metadata.terminals)
            o0, o1 = o_terms[t0], o_terms[t1]
            # Try straight pairing (pos->t0, neg->t1) vs swapped (pos->t1, neg->t0)
            for cand_pos, cand_neg in ((t0, t1), (t1, t0)):
                op, on = o_terms[cand_pos], o_terms[cand_neg]
                dp = math.hypot(p_pos[0] - op[0], p_pos[1] - op[1])
                dn = math.hypot(p_neg[0] - on[0], p_neg[1] - on[1])
                if dp <= tolerance_px and dn <= tolerance_px:
                    score = dp + dn
                    if best is None or score < best[0]:
                        best = (score, other, cand_pos, cand_neg)
        return (best[1], best[2], best[3]) if best is not None else None

    def _connect_voltmeter_parallel_with(
        self,
        vm_item: ComponentItem,
        target_item: ComponentItem,
        term_for_pos: str,
        term_for_neg: str,
        graph: WireGraph,
    ) -> WireGraph:
        """Wire `vm_item` (`pos`, `neg`) in parallel across `target_item`."""
        vm_name = vm_item.component.name
        t_name = target_item.component.name
        working = graph
        for vm_t, tgt_t in (("pos", term_for_pos), ("neg", term_for_neg)):
            a = TerminalRef(vm_name, vm_t)
            b = TerminalRef(t_name, tgt_t)
            if self._find_duplicate_graph_wire(working, a, b) is None:
                wid = self._next_unused_wire_id(working)
                working = working.add_wire(a, b, wire_id=wid)
        return working

    def _record_pin_to_pin_bonds_for(
        self,
        item: ComponentItem,
        against: Sequence[ComponentItem],
    ) -> None:
        """Record initial co-located pin contacts created at placement or legacy load."""
        item_terms = item.terminal_scene_positions_px()
        for other in against:
            if other is item:
                continue
            other_terms = other.terminal_scene_positions_px()
            for t_a, pt_a in item_terms.items():
                for t_b, pt_b in other_terms.items():
                    if pt_a == pt_b:
                        self._pin_to_pin_bonds.append((item, t_a, other, t_b))

    # ------------------------------------------------------------------
    # Wiring, Terminal Hit-Testing, Live Wire Preview & Junction Dots
    # ------------------------------------------------------------------

    def all_component_terminals_px(self) -> dict[str, dict[str, PointPx]]:
        """Return `{comp_name: {term_name: (x_px, y_px)}}` for all placed components."""
        return {
            item.component.name: item.terminal_scene_positions_px()
            for item in self._placed_items
        }

    TERMINAL_SNAP_RADIUS_PX: float = 10.0  # Endpoint-drag terminal snap radius (10 px)
    AUTO_CONNECT_TOLERANCE_PX: float = 12.5  # Zoom-independent scene tolerance (>= 12.0 px)

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

    def _emit_status_notice(self, message: str) -> None:
        self.last_status_notice = message
        self.status_notice.emit(message)

    def _resolve_wire_attachment(
        self,
        pos: QPointF | tuple[float, float],
        *,
        graph: WireGraph | None = None,
        tolerance_px: float | None = None,
    ) -> tuple[Attachment, PointPx]:
        """Resolve cursor `pos` to `(TerminalRef | JunctionRef | FreeEnd, point_px)`."""
        g = graph if graph is not None else self._wire_graph
        tol = self.AUTO_CONNECT_TOLERANCE_PX if tolerance_px is None else float(tolerance_px)
        sx = pos.x() if isinstance(pos, QPointF) else float(pos[0])
        sy = pos.y() if isinstance(pos, QPointF) else float(pos[1])
        term_hit = self.find_terminal_near((sx, sy), tolerance_px=tol)
        if term_hit is not None:
            return (TerminalRef(term_hit[0].component.name, term_hit[1]), term_hit[2])
        j_ref = self._find_junction_at((sx, sy), g, tolerance_px=tol)
        if j_ref is not None:
            return (j_ref, g.junction(j_ref.id).pos)
        snapped = wire_snap_px((sx, sy))
        return (FreeEnd(*snapped), snapped)

    def is_snap_target_at(self, pos: QPointF | tuple[float, float]) -> bool:
        """Return True if `pos` is within auto-connect tolerance of a terminal, junction, wire, or FreeEnd."""
        tol = self.AUTO_CONNECT_TOLERANCE_PX
        if self.find_terminal_near(pos, tolerance_px=tol) is not None:
            return True
        if self._find_junction_at(pos, self._wire_graph, tolerance_px=tol) is not None:
            return True
        sx = pos.x() if isinstance(pos, QPointF) else float(pos[0])
        sy = pos.y() if isinstance(pos, QPointF) else float(pos[1])
        if self._nearest_point_on_wire_segments((sx, sy), self._wire_graph, tol) is not None:
            return True
        for w in self._wire_graph.wires:
            for att in (w.end_a, w.end_b):
                if isinstance(att, FreeEnd) and math.hypot(att.x - sx, att.y - sy) <= tol:
                    return True
        return False

    def add_wire_preview_waypoint(
        self, pos: QPointF | tuple[float, float]
    ) -> bool:
        """Append a grid-snapped intermediate waypoint to the active preview wire.

        Returns False if `pos` snaps to the start/previous waypoint (caller can finish).
        """
        if self._wire_start_pt is None or self._wire_start_att is None:
            return False
        sx = pos.x() if isinstance(pos, QPointF) else float(pos[0])
        sy = pos.y() if isinstance(pos, QPointF) else float(pos[1])
        snapped = wire_snap_px((sx, sy))
        last_anchor = (
            self._wire_hints[-1] if self._wire_hints else self._wire_start_pt
        )
        if snapped == last_anchor:
            return False
        self._wire_hints.append(snapped)
        self.update_wire_preview((sx, sy))
        return True

    def pop_wire_preview_waypoint(
        self, cursor_pos: QPointF | tuple[float, float] | None = None
    ) -> bool:
        """Remove the last preview waypoint, or cancel preview when no waypoints remain."""
        if not self.is_drawing_wire:
            return False
        if self._wire_hints:
            self._wire_hints.pop()
            target = (
                cursor_pos
                if cursor_pos is not None
                else (
                    self._wire_hints[-1]
                    if self._wire_hints
                    else self._wire_start_pt
                )
            )
            if target is not None:
                self.update_wire_preview(target)
            return True
        self.cancel_wire_preview()
        return False

    def _point_hits_existing_wire(
        self,
        pt: PointPx,
        graph: WireGraph,
        *,
        exclude_wire_id: str | None = None,
    ) -> bool:
        """Return True if `pt` lies anywhere on an existing wire polyline in `graph`."""
        term_map = self._terminal_pos_map()
        for w in graph.wires:
            if exclude_wire_id is not None and w.id == exclude_wire_id:
                continue
            item = self._wire_items_by_id.get(w.id)
            if item is not None and item.scene() is self and len(item.points) >= 2:
                pts = item.points
            else:
                pts = self._compute_wire_points_from_graph(w, graph, term_map=term_map)
            if len(pts) >= 2 and point_on_wire(pt, pts):
                return True
        return False

    @property
    def hovered_wire_end_pt(self) -> PointPx | None:
        """Return the hovered wire endpoint snap coordinate, if any."""
        return self._hovered_wire_end_pt

    def update_hover_highlight(
        self, scene_pos: QPointF, *, graph: WireGraph | None = None
    ) -> PointPx | None:
        """Update the highlighted terminal, junction, or wire endpoint under `scene_pos`."""
        g = graph if graph is not None else self._wire_graph
        term_hit = self.find_terminal_near(scene_pos)
        new_hover: PointPx | None = None
        if term_hit is not None:
            new_hover = term_hit[2]
        else:
            j_ref = self._find_junction_at(
                scene_pos,
                g,
                tolerance_px=self.TERMINAL_SNAP_RADIUS_PX,
            )
            if j_ref is not None:
                new_hover = g.junction(j_ref.id).pos

        new_end_hover: PointPx | None = None
        if (
            not self.is_drawing_wire
            and not self.is_dragging_wire_endpoint
            and not self.is_dragging_wire_segment
        ):
            end_hit = self.find_wire_endpoint_near(
                scene_pos, tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
            )
            if end_hit is not None:
                new_end_hover = end_hit[2]

        if (
            new_hover != self._hovered_terminal_pt
            or new_end_hover != self._hovered_wire_end_pt
        ):
            self._hovered_terminal_pt = new_hover
            self._hovered_wire_end_pt = new_end_hover
            self.update()
        return self._hovered_terminal_pt

    def start_wire_preview(
        self, start_pos: QPointF | tuple[float, float]
    ) -> PointPx:
        """Begin interactive orthogonal wire drawing from `start_pos` using `WireGraph`."""
        att_a, start_pt = self._resolve_wire_attachment(start_pos)
        self._wire_start_att = att_a
        self._wire_end_att = None
        self._wire_start_pt = start_pt
        self._wire_hints = []
        self._wire_preview_previous = None
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
        """Update the live preview wire using `circuitsim.wiring.router.route`."""
        if self._wire_start_pt is None or self._wire_start_att is None:
            return ()

        cx = cursor_pos.x() if isinstance(cursor_pos, QPointF) else float(cursor_pos[0])
        cy = cursor_pos.y() if isinstance(cursor_pos, QPointF) else float(cursor_pos[1])
        att_b, _end_pt = self._resolve_wire_attachment((cx, cy))
        self._wire_end_att = att_b

        preview_wire = Wire(
            id="PREVIEW_WIRE",
            end_a=self._wire_start_att,
            end_b=att_b,
            hints=tuple(self._wire_hints),
        )
        pts = self._compute_wire_points_from_graph(
            preview_wire,
            self._wire_graph,
            fallback_points=self._wire_preview_previous,
        )
        if self._preview_wire_item is not None:
            if len(pts) >= 2:
                self._wire_preview_previous = pts
                self._preview_wire_item.set_points(pts)
            else:
                self._wire_preview_previous = None
                self._preview_wire_item.set_points(
                    (self._wire_start_pt, self._wire_start_pt)
                )
        self.update_hover_highlight(QPointF(float(cx), float(cy)))
        self.update()
        return pts

    def finish_wire_preview(
        self, end_pos: QPointF | tuple[float, float]
    ) -> WireItem | None:
        """Commit the previewed wire onto `WireGraph` via `QUndoStack` and clear preview."""
        if self._wire_start_pt is None or self._wire_start_att is None:
            return None
        pts = self.update_wire_preview(end_pos)
        att_a = self._wire_start_att
        att_b = self._wire_end_att
        start_pt = self._wire_start_pt
        hints_list = list(self._wire_hints)
        self.cancel_wire_preview()
        if att_b is None:
            return None
        if att_a == att_b:
            if isinstance(att_a, TerminalRef):
                self._emit_status_notice("Cannot connect a terminal to itself")
            elif isinstance(att_a, JunctionRef):
                self._emit_status_notice("Cannot connect a junction to itself")
            else:
                self._emit_status_notice("Zero-length wire ignored")
            return None
        if len(pts) < 2:
            self._emit_status_notice("Zero-length wire ignored")
            return None
        if self._find_duplicate_graph_wire(self._wire_graph, att_a, att_b) is not None:
            self._emit_status_notice("Wire already exists between these points")
            return None

        end_pt = pts[-1]
        while hints_list and hints_list[0] == start_pt:
            hints_list.pop(0)
        while hints_list and hints_list[-1] == end_pt:
            hints_list.pop()
        hints = tuple(hints_list)

        wid = self._next_unused_wire_id(self._wire_graph)
        try:
            base_graph = self._wire_graph.add_wire(
                att_a, att_b, hints=hints, wire_id=wid
            )
        except WireGraphError:
            return None
        auto_graph = self._auto_connect_graph_endpoints(base_graph)
        if auto_graph.has_wire(wid):
            w_obj = auto_graph.wire(wid)
            pts_final = self._compute_wire_points_from_graph(w_obj, auto_graph)
            wire = WireItem(
                wid,
                pts_final,
                theme=self.theme,
                end_a=w_obj.end_a,
                end_b=w_obj.end_b,
                hints=w_obj.hints,
            )
            self.undo_stack.push(
                AddWireCommand(
                    self, wire, old_graph=self._wire_graph, new_graph=auto_graph
                )
            )
            return wire
        if auto_graph != self._wire_graph:
            self.undo_stack.push(
                SetWireGraphCommand(
                    self,
                    self._wire_graph,
                    auto_graph,
                    description="Auto-connect Wire",
                )
            )
        return None

    def cancel_wire_preview(self) -> None:
        """Cancel any active wire preview without creating a wire."""
        self._wire_start_pt = None
        self._wire_start_att = None
        self._wire_end_att = None
        self._wire_hints = []
        self._wire_preview_previous = None
        if self._preview_wire_item is not None:
            self.removeItem(self._preview_wire_item)
            self._preview_wire_item = None
        self.update()

    # ------------------------------------------------------------------
    # Interactive wire segment dragging (updates Wire.hints, 1 undo step)
    # ------------------------------------------------------------------

    @property
    def is_dragging_wire_segment(self) -> bool:
        """Return whether a wire segment drag is currently in progress."""
        return self._seg_drag_wire_id is not None

    @property
    def active_wire_segment_orientation(self) -> str:
        """Return `'H'`, `'V'`, or `''` for the segment currently being dragged."""
        return self._seg_drag_orient

    def find_wire_segment_near(
        self,
        scene_pos: QPointF | tuple[float, float],
        *,
        tolerance_px: float = 8.0,
        endpoint_guard_px: float = 10.0,
    ) -> tuple[WireItem, int, str] | None:
        """Find the nearest movable wire segment within `tolerance_px` of `scene_pos`.

        Returns `(wire_item, seg_index, orientation)` where `orientation` is
        `'H'` (horizontal segment that slides vertically) or `'V'` (vertical
        segment that slides horizontally), or `None` if `scene_pos` is near a
        terminal, junction, or wire free end, or not within `tolerance_px`.
        """
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])

        if (
            self.find_terminal_near((sx, sy), tolerance_px=endpoint_guard_px)
            is not None
        ):
            return None
        if (
            self._find_junction_at(
                (sx, sy), self._wire_graph, tolerance_px=endpoint_guard_px
            )
            is not None
        ):
            return None

        best: tuple[WireItem, int, str] | None = None
        best_key: tuple[float, float, int] | None = None

        for wire in self._wire_items:
            if wire.scene() is not self:
                continue
            pts = wire.points
            if len(pts) < 2:
                continue
            # Keep free endpoints reserved for wire-end handle dragging.
            if (
                math.hypot(sx - pts[0][0], sy - pts[0][1]) < endpoint_guard_px
                or math.hypot(sx - pts[-1][0], sy - pts[-1][1]) < endpoint_guard_px
            ):
                continue

            for idx in range(len(pts) - 1):
                p1, p2 = pts[idx], pts[idx + 1]
                if p1 == p2:
                    continue
                if p1[1] == p2[1]:
                    orient = "H"
                    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
                    dx = max(x_min - sx, 0.0, sx - x_max)
                    dy = abs(sy - p1[1])
                    dist = math.hypot(dx, dy)
                    perp_dist = dy
                    seg_len = float(x_max - x_min)
                elif p1[0] == p2[0]:
                    orient = "V"
                    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
                    dx = abs(sx - p1[0])
                    dy = max(y_min - sy, 0.0, sy - y_max)
                    dist = math.hypot(dx, dy)
                    perp_dist = dx
                    seg_len = float(y_max - y_min)
                else:
                    continue

                if dist > tolerance_px:
                    continue
                cand_key = (round(dist, 6), round(perp_dist, 6), -int(seg_len))
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best = (wire, idx, orient)

        return best

    def begin_wire_segment_drag(
        self, wire_item: WireItem, seg_index: int, orientation: str
    ) -> bool:
        """Start interactively sliding segment `seg_index` of `wire_item`."""
        if (
            orientation not in ("H", "V")
            or not self._wire_graph.has_wire(wire_item.wire_id)
            or not (0 <= seg_index <= len(wire_item.points) - 2)
        ):
            return False
        wire_obj = self._wire_graph.wire(wire_item.wire_id)
        self._seg_drag_old_graph = self._wire_graph
        self._seg_drag_wire_id = wire_item.wire_id
        self._seg_drag_index = seg_index
        self._seg_drag_orient = orientation
        self._seg_drag_orig_pts = tuple(wire_item.points)
        self._seg_drag_orig_hints = wire_obj.hints
        return True

    def _compute_slid_hints_for_pos(self, scene_pos: QPointF | tuple[float, float]) -> tuple[PointPx, ...]:
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])
        target_raw = sy if self._seg_drag_orient == "H" else sx
        snapped_coord = int(round(target_raw / GRID_PITCH_PX)) * GRID_PITCH_PX
        p_seg = self._seg_drag_orig_pts[self._seg_drag_index]
        orig_coord = p_seg[1] if self._seg_drag_orient == "H" else p_seg[0]
        if snapped_coord == orig_coord:
            return self._seg_drag_orig_hints
        return wire_slide_segment_hints(
            self._seg_drag_orig_pts,
            self._seg_drag_index,
            self._seg_drag_orient,
            snapped_coord,
        )

    def update_wire_segment_drag(
        self, scene_pos: QPointF | tuple[float, float]
    ) -> tuple[PointPx, ...]:
        """Update the live preview of the dragged wire segment via `route()`."""
        wid = self._seg_drag_wire_id
        old_graph = self._seg_drag_old_graph
        if wid is None or old_graph is None or not old_graph.has_wire(wid):
            return ()
        hints = self._compute_slid_hints_for_pos(scene_pos)
        preview_graph = old_graph.set_wire_hints(wid, hints)
        item = self._wire_items_by_id.get(wid)
        pref = {wid: item} if item is not None else None
        self.set_wire_graph(preview_graph, refresh=False, preferred_wire_items=pref)
        self.update()
        updated_item = self._wire_items_by_id.get(wid)
        return updated_item.points if updated_item is not None else ()

    def commit_wire_segment_drag(self) -> bool:
        """Finish wire segment drag, pushing 1 undo step if geometry/hints changed."""
        wid = self._seg_drag_wire_id
        old_graph = self._seg_drag_old_graph
        new_graph = self._wire_graph
        orig_pts = self._seg_drag_orig_pts
        self._clear_wire_segment_drag_state()
        if wid is None or old_graph is None:
            return False

        item = self._wire_items_by_id.get(wid)
        new_pts = item.points if item is not None else ()
        if new_graph == old_graph or new_pts == orig_pts:
            pref = {wid: item} if item is not None else None
            self.set_wire_graph(old_graph, refresh=False, preferred_wire_items=pref)
            return False

        pref = {wid: item} if item is not None else None
        # Restore old_graph briefly so QUndoStack.push() executes redo() cleanly once.
        self.set_wire_graph(old_graph, refresh=False, preferred_wire_items=pref)
        self.undo_stack.push(
            SetWireGraphCommand(
                self,
                old_graph,
                new_graph,
                preferred_wire_items=pref,
                description=f"Move Wire {wid} Segment",
            )
        )
        return True

    def cancel_wire_segment_drag(self) -> None:
        """Cancel an active wire segment drag and restore the previous wire geometry."""
        wid = self._seg_drag_wire_id
        old_graph = self._seg_drag_old_graph
        self._clear_wire_segment_drag_state()
        if old_graph is not None:
            item = self._wire_items_by_id.get(wid) if wid is not None else None
            pref = {wid: item} if wid is not None and item is not None else None
            self.set_wire_graph(old_graph, refresh=True, preferred_wire_items=pref)

    def _clear_wire_segment_drag_state(self) -> None:
        self._seg_drag_old_graph = None
        self._seg_drag_wire_id = None
        self._seg_drag_index = -1
        self._seg_drag_orient = ""
        self._seg_drag_orig_pts = ()
        self._seg_drag_orig_hints = ()

    # ------------------------------------------------------------------
    # Interactive wire endpoint editing (Acceptance S3: detach/reattach)
    # ------------------------------------------------------------------

    @property
    def is_dragging_wire_endpoint(self) -> bool:
        """Return whether an existing wire's endpoint is currently being dragged."""
        return self._end_drag_wire_id is not None

    def find_wire_endpoint_near(
        self,
        scene_pos: QPointF | tuple[float, float],
        *,
        tolerance_px: float = 10.0,
    ) -> tuple[WireItem, str, PointPx] | None:
        """Find the nearest wire endpoint `(wire_item, 'end_a'|'end_b', pt)` within `tolerance_px`.

        Prioritizes selected wires first, then `FreeEnd` handles, then closest distance.
        """
        sx = scene_pos.x() if isinstance(scene_pos, QPointF) else float(scene_pos[0])
        sy = scene_pos.y() if isinstance(scene_pos, QPointF) else float(scene_pos[1])

        best: tuple[WireItem, str, PointPx] | None = None
        best_key: tuple[int, int, float] | None = None
        for wire in self._wire_items:
            if wire.scene() is not self or len(wire.points) < 2:
                continue
            if not self._wire_graph.has_wire(wire.wire_id):
                continue
            w_obj = self._wire_graph.wire(wire.wire_id)
            candidates = (
                ("end_a", w_obj.end_a, wire.points[0]),
                ("end_b", w_obj.end_b, wire.points[-1]),
            )
            sel_rank = 0 if wire.isSelected() else 1
            for which, att, pt in candidates:
                dist = math.hypot(sx - pt[0], sy - pt[1])
                if dist > tolerance_px:
                    continue
                free_rank = 0 if isinstance(att, FreeEnd) else 1
                key = (sel_rank, free_rank, round(dist, 6))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (wire, which, pt)
        return best

    def begin_wire_endpoint_drag(self, wire_item: WireItem, which_end: str) -> bool:
        """Start interactively detaching and dragging `which_end` of `wire_item`."""
        if (
            which_end not in ("end_a", "end_b")
            or not self._wire_graph.has_wire(wire_item.wire_id)
        ):
            return False
        w_obj = self._wire_graph.wire(wire_item.wire_id)
        self._end_drag_old_graph = self._wire_graph
        self._end_drag_wire_id = wire_item.wire_id
        self._end_drag_which = which_end
        self._end_drag_orig_att = (
            w_obj.end_a if which_end == "end_a" else w_obj.end_b
        )
        self._end_drag_orig_pts = tuple(wire_item.points)
        self._hovered_wire_end_pt = (
            wire_item.points[0] if which_end == "end_a" else wire_item.points[-1]
        )
        return True

    def update_wire_endpoint_drag(
        self, scene_pos: QPointF | tuple[float, float]
    ) -> tuple[PointPx, ...]:
        """Update the live routed preview while dragging a wire endpoint."""
        wid = self._end_drag_wire_id
        which = self._end_drag_which
        old_graph = self._end_drag_old_graph
        orig_att = self._end_drag_orig_att
        if wid is None or old_graph is None or not old_graph.has_wire(wid):
            return ()

        target_att, target_pt = self._resolve_wire_attachment(
            scene_pos, graph=old_graph, tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
        )
        old_wire = old_graph.wire(wid)
        other_att = old_wire.end_b if which == "end_a" else old_wire.end_a
        item = self._wire_items_by_id.get(wid)
        pref = {wid: item} if item is not None else None

        if target_att == orig_att:
            self.set_wire_graph(old_graph, refresh=False, preferred_wire_items=pref)
            self.update_hover_highlight(
                QPointF(float(target_pt[0]), float(target_pt[1])), graph=old_graph
            )
            self.update()
            updated = self._wire_items_by_id.get(wid)
            return updated.points if updated is not None else ()

        try:
            preview_graph = old_graph.reconnect_wire_end(
                wid, which, target_att, hints=()
            )
            self.set_wire_graph(
                preview_graph, refresh=False, preferred_wire_items=pref
            )
            self.update_hover_highlight(
                QPointF(float(target_pt[0]), float(target_pt[1])), graph=old_graph
            )
            self.update()
            updated = self._wire_items_by_id.get(wid)
            return updated.points if updated is not None else ()
        except WireGraphError:
            self.set_wire_graph(old_graph, refresh=False, preferred_wire_items=pref)
            item = self._wire_items_by_id.get(wid)
            p_fixed = self._attachment_pos_px(other_att, old_graph)
            if item is not None and p_fixed is not None:
                dmap = self._terminal_dir_map()
                dir_fixed = self._attachment_outward_dir(other_att, dmap)
                dir_target = self._attachment_outward_dir(target_att, dmap)
                p_a_prev, p_b_prev = (
                    (target_pt, p_fixed) if which == "end_a" else (p_fixed, target_pt)
                )
                d_a_prev, d_b_prev = (
                    (dir_target, dir_fixed)
                    if which == "end_a"
                    else (dir_fixed, dir_target)
                )
                pts = (
                    (p_a_prev,)
                    if p_a_prev == p_b_prev
                    else wire_route(
                        p_a_prev,
                        p_b_prev,
                        start_dir=d_a_prev,
                        end_dir=d_b_prev,
                        obstacles=self._component_obstacles_px(),
                    )
                )
                item.end_a = target_att if which == "end_a" else other_att
                item.end_b = target_att if which == "end_b" else other_att
                item.hints = ()
                item.set_points(pts)
            else:
                pts = ()
            self.update_hover_highlight(
                QPointF(float(target_pt[0]), float(target_pt[1])), graph=old_graph
            )
            self.update()
            return pts

    def commit_wire_endpoint_drag(
        self, scene_pos: QPointF | tuple[float, float]
    ) -> bool:
        """Commit wire endpoint reattachment in 1 undo step, or reject with 0 steps."""
        wid = self._end_drag_wire_id
        which = self._end_drag_which
        old_graph = self._end_drag_old_graph
        orig_att = self._end_drag_orig_att
        self._clear_wire_endpoint_drag_state()
        if wid is None or old_graph is None or not old_graph.has_wire(wid):
            return False

        old_wire = old_graph.wire(wid)
        other_att = old_wire.end_b if which == "end_a" else old_wire.end_a
        target_att, _target_pt = self._resolve_wire_attachment(
            scene_pos, graph=old_graph, tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
        )
        item = self._wire_items_by_id.get(wid)
        pref = {wid: item} if item is not None else None

        # Restore old_graph cleanly before deciding commit vs rejection.
        self.set_wire_graph(old_graph, refresh=False, preferred_wire_items=pref)

        if target_att == orig_att:
            self.refresh_connectivity_visuals()
            return False

        if target_att == other_att:
            if isinstance(target_att, TerminalRef):
                self._emit_status_notice("Cannot connect a terminal to itself")
            elif isinstance(target_att, JunctionRef):
                self._emit_status_notice("Cannot connect a junction to itself")
            else:
                self._emit_status_notice("Zero-length wire ignored")
            self.refresh_connectivity_visuals()
            return False

        new_a = target_att if which == "end_a" else old_wire.end_a
        new_b = target_att if which == "end_b" else old_wire.end_b
        for other_w in old_graph.wires:
            if other_w.id == wid:
                continue
            if (other_w.end_a == new_a and other_w.end_b == new_b) or (
                other_w.end_a == new_b and other_w.end_b == new_a
            ):
                self._emit_status_notice("Wire already exists between these points")
                self.refresh_connectivity_visuals()
                return False

        try:
            new_graph = old_graph.reconnect_wire_end(wid, which, target_att, hints=())
        except WireGraphError as exc:
            self._emit_status_notice(str(exc))
            self.refresh_connectivity_visuals()
            return False

        if isinstance(target_att, FreeEnd):
            new_graph = self._auto_connect_graph_endpoints(new_graph)
        self.undo_stack.push(
            SetWireGraphCommand(
                self,
                old_graph,
                new_graph,
                preferred_wire_items=pref,
                description=f"Edit Wire {wid} End",
            )
        )
        return True

    def cancel_wire_endpoint_drag(self) -> None:
        """Cancel an active wire endpoint drag and restore the original graph."""
        wid = self._end_drag_wire_id
        old_graph = self._end_drag_old_graph
        self._clear_wire_endpoint_drag_state()
        if old_graph is not None:
            item = self._wire_items_by_id.get(wid) if wid is not None else None
            pref = {wid: item} if wid is not None and item is not None else None
            self.set_wire_graph(old_graph, refresh=True, preferred_wire_items=pref)

    def _clear_wire_endpoint_drag_state(self) -> None:
        self._end_drag_old_graph = None
        self._end_drag_wire_id = None
        self._end_drag_which = ""
        self._end_drag_orig_att = None
        self._end_drag_orig_pts = ()
        self._hovered_wire_end_pt = None

    @property
    def is_dragging_junction(self) -> bool:
        """Return whether a junction dot is currently being dragged."""
        return self._dragging_junction_id is not None

    def find_junction_item_near(
        self,
        scene_pos: QPointF | tuple[float, float],
        *,
        tolerance_px: float | None = None,
    ) -> JunctionItem | None:
        """Return the closest visible `JunctionItem` within `tolerance_px`, if any."""
        tol = (
            self.TERMINAL_SNAP_RADIUS_PX
            if tolerance_px is None
            else float(tolerance_px)
        )
        px = scene_pos[0] if isinstance(scene_pos, (tuple, list)) else scene_pos.x()
        py = scene_pos[1] if isinstance(scene_pos, (tuple, list)) else scene_pos.y()
        best_item: JunctionItem | None = None
        best_dist = tol
        for jitem in self._junction_items_by_id.values():
            if jitem.scene() is not self:
                continue
            dist = math.hypot(jitem.pos().x() - px, jitem.pos().y() - py)
            if dist <= best_dist:
                best_dist = dist
                best_item = jitem
        return best_item

    def begin_junction_drag(self, junction_id: str) -> bool:
        """Begin interactively dragging `junction_id` to a new grid position."""
        if not self._wire_graph.has_junction(junction_id):
            return False
        junc = self._wire_graph.junction(junction_id)
        self._dragging_junction_id = junction_id
        self._dragging_junction_start_pos = junc.pos
        self._dragging_junction_start_graph = self._wire_graph
        self._dragging_junction_explicit = junc.pos in self._explicit_junctions
        self._hovered_wire_end_pt = None
        return True

    def update_junction_drag(
        self, scene_pos: QPointF | tuple[float, float]
    ) -> PointPx | None:
        """Live-preview moving the active junction to snapped `scene_pos` and rerouting wires."""
        jid = self._dragging_junction_id
        old_graph = self._dragging_junction_start_graph
        if jid is None or old_graph is None or not old_graph.has_junction(jid):
            return None
        raw = (
            (scene_pos.x(), scene_pos.y())
            if isinstance(scene_pos, QPointF)
            else (float(scene_pos[0]), float(scene_pos[1]))
        )
        new_pos = wire_snap_px(raw)
        if self._dragging_junction_explicit:
            old_j = (
                self._wire_graph.junction(jid)
                if self._wire_graph.has_junction(jid)
                else None
            )
            if old_j is not None:
                self._explicit_junctions.discard(old_j.pos)
            self._explicit_junctions.add(new_pos)
        new_graph = old_graph.move_junction(jid, new_pos)
        self.set_wire_graph(new_graph, refresh=True)
        return new_pos

    def commit_junction_drag(
        self, scene_pos: QPointF | tuple[float, float] | None = None
    ) -> bool:
        """Finalize junction drag and push `MoveJunctionCommand` if its position changed."""
        jid = self._dragging_junction_id
        start_pos = self._dragging_junction_start_pos
        old_graph = self._dragging_junction_start_graph
        was_explicit = self._dragging_junction_explicit
        self._clear_junction_drag_state()
        if (
            jid is None
            or start_pos is None
            or old_graph is None
            or not old_graph.has_junction(jid)
        ):
            return False
        if scene_pos is not None:
            raw = (
                (scene_pos.x(), scene_pos.y())
                if isinstance(scene_pos, QPointF)
                else (float(scene_pos[0]), float(scene_pos[1]))
            )
            new_pos = wire_snap_px(raw)
        elif self._wire_graph.has_junction(jid):
            new_pos = self._wire_graph.junction(jid).pos
        else:
            new_pos = start_pos

        if new_pos == start_pos:
            if was_explicit:
                self._explicit_junctions.add(start_pos)
            self.set_wire_graph(old_graph, refresh=True)
            return False

        new_graph = old_graph.move_junction(jid, new_pos)
        if was_explicit:
            self._explicit_junctions.discard(new_pos)
            self._explicit_junctions.add(start_pos)
        self.undo_stack.push(
            MoveJunctionCommand(
                self,
                jid,
                start_pos,
                new_pos,
                old_graph=old_graph,
                new_graph=new_graph,
            )
        )
        return True

    def cancel_junction_drag(self) -> None:
        """Cancel an active junction drag and restore the original graph (0 undo steps)."""
        start_pos = self._dragging_junction_start_pos
        old_graph = self._dragging_junction_start_graph
        was_explicit = self._dragging_junction_explicit
        self._clear_junction_drag_state()
        if old_graph is not None:
            if was_explicit and start_pos is not None:
                self._explicit_junctions.add(start_pos)
            self.set_wire_graph(old_graph, refresh=True)

    def _clear_junction_drag_state(self) -> None:
        self._dragging_junction_id = None
        self._dragging_junction_start_pos = None
        self._dragging_junction_start_graph = None
        self._dragging_junction_explicit = False

    def _terminals_at_point(self, pt: PointPx) -> list[TerminalRef]:
        refs: list[TerminalRef] = []
        for item in self._placed_items:
            cname = item.component.name
            for tname, tpt in item.terminal_scene_positions_px().items():
                if tpt == pt:
                    refs.append(TerminalRef(cname, tname))
        return refs

    def _find_junction_at(
        self,
        pt: QPointF | tuple[float, float],
        graph: WireGraph,
        *,
        tolerance_px: float = 0.0,
    ) -> JunctionRef | None:
        px = pt.x() if isinstance(pt, QPointF) else float(pt[0])
        py = pt.y() if isinstance(pt, QPointF) else float(pt[1])
        if tolerance_px <= 0.0:
            ix = int(round(px))
            iy = int(round(py))
            for j in graph.junctions:
                if j.pos == (ix, iy):
                    return JunctionRef(j.id)
            return None
        best_id: str | None = None
        best_dist = tolerance_px
        for j in graph.junctions:
            dist = math.hypot(j.pos[0] - px, j.pos[1] - py)
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

    def _wire_polyline_in_graph(
        self,
        wire: Wire,
        graph: WireGraph,
        term_map: dict[tuple[str, str], PointPx],
    ) -> tuple[PointPx, ...]:
        item = self._wire_items_by_id.get(wire.id)
        if (
            item is not None
            and item.scene() is self
            and len(item.points) >= 2
            and self._wire_graph.has_wire(wire.id)
            and self._wire_graph.wire(wire.id) == wire
        ):
            return item.points
        return self._compute_wire_points_from_graph(wire, graph, term_map=term_map)

    @staticmethod
    def _arc_distance_along_polyline(
        pt: PointPx, pts: Sequence[PointPx]
    ) -> float | None:
        from circuitsim.canvas.connectivity import point_on_orthogonal_segment

        dist = 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if point_on_orthogonal_segment(pt, a, b):
                return dist + float(abs(pt[0] - a[0]) + abs(pt[1] - a[1]))
            dist += float(abs(b[0] - a[0]) + abs(b[1] - a[1]))
        return None

    def _hints_before_point(
        self, pt: PointPx, wire: Wire, pts: Sequence[PointPx]
    ) -> int:
        if not wire.hints:
            return 0
        d_pt = self._arc_distance_along_polyline(pt, pts)
        if d_pt is None:
            return 0
        count = 0
        for h in wire.hints:
            d_h = self._arc_distance_along_polyline(h, pts)
            if d_h is not None and d_h < d_pt:
                count += 1
            else:
                break
        return count

    def _drop_candidate_wire_polyline(
        self,
        wire: Wire,
        graph: WireGraph,
        term_map: dict[tuple[str, str], PointPx],
        obstacles: tuple[tuple[int, int, int, int], ...],
        pre_drop_polylines: dict[str, tuple[PointPx, ...]] | None,
    ) -> tuple[PointPx, ...]:
        """Return `wire`'s routed polyline ignoring the dropped component's own obstacle."""
        if pre_drop_polylines is not None and wire.id in pre_drop_polylines:
            cached = pre_drop_polylines[wire.id]
            p_a = self._attachment_pos_px(wire.end_a, graph, term_map)
            p_b = self._attachment_pos_px(wire.end_b, graph, term_map)
            if len(cached) >= 2 and p_a == cached[0] and p_b == cached[-1]:
                return cached
        return self._compute_wire_points_from_graph(
            wire, graph, term_map=term_map, obstacles=obstacles
        )

    def _apply_component_split_on_drop(
        self,
        item: ComponentItem,
        graph: WireGraph,
        *,
        pre_drop_polylines: dict[str, tuple[PointPx, ...]] | None = None,
    ) -> WireGraph:
        """Split a wire inline or tap a junction when an unconnected `item` is dropped onto wires."""
        from circuitsim.canvas.connectivity import point_strictly_inside_wire

        cname = item.component.name
        terms = item.terminal_scene_positions_px()
        if not terms:
            return graph
        if any(graph.attached_wires(TerminalRef(cname, t)) for t in terms):
            return graph

        term_map = self._terminal_pos_map()
        obstacles = self._component_obstacles_px(exclude_names={cname})

        # Case A: 2-terminal component dropped inline onto the interior/bend of one wire.
        if len(terms) == 2:
            (t1, p1), (t2, p2) = list(terms.items())
            for w in graph.wires:
                pts = self._drop_candidate_wire_polyline(
                    w, graph, term_map, obstacles, pre_drop_polylines
                )
                if len(pts) < 2:
                    continue
                if point_strictly_inside_wire(p1, pts) and point_strictly_inside_wire(p2, pts):
                    d1 = self._arc_distance_along_polyline(p1, pts)
                    d2 = self._arc_distance_along_polyline(p2, pts)
                    if d1 is not None and d2 is not None and d1 != d2:
                        if d1 < d2:
                            first_t, d_first = t1, d1
                            second_t, d_second = t2, d2
                        else:
                            first_t, d_first = t2, d2
                            second_t, d_second = t1, d1
                        hint_source = list(w.hints) if w.hints else list(pts[1:-1])
                        hints_a = [
                            h
                            for h in hint_source
                            if (dh := self._arc_distance_along_polyline(h, pts)) is not None
                            and dh < d_first
                        ]
                        hints_b = [
                            h
                            for h in hint_source
                            if (dh := self._arc_distance_along_polyline(h, pts)) is not None
                            and dh > d_second
                        ]
                        return graph.insert_component_inline(
                            w.id,
                            TerminalRef(cname, first_t),
                            TerminalRef(cname, second_t),
                            hints_a=hints_a,
                            hints_b=hints_b,
                        )

        # Case B: Single-terminal (or non-inline multi-terminal) tap onto wire interior(s).
        working = graph
        tapped_wire_ids: set[str] = set()
        for t_name, pt in sorted(terms.items()):
            j_exact = self._find_junction_at(pt, working, tolerance_px=0.5)
            if j_exact is not None:
                if len(terms) == 2 and tapped_wire_ids:
                    continue
                working = working.add_wire(j_exact, TerminalRef(cname, t_name))
                continue
            for w in working.wires:
                if len(terms) == 2 and tapped_wire_ids and w.id in tapped_wire_ids:
                    continue
                pts = self._drop_candidate_wire_polyline(
                    w, working, term_map, obstacles, pre_drop_polylines
                )
                if len(pts) >= 2 and point_strictly_inside_wire(pt, pts):
                    hb = self._hints_before_point(pt, w, pts)
                    split_res = working.split_wire_at(w.id, pt, hints_before=hb)
                    tapped_wire_ids.add(w.id)
                    tapped_wire_ids.add(split_res.second_wire_id)
                    working = split_res.graph.add_wire(
                        JunctionRef(split_res.junction_id), TerminalRef(cname, t_name)
                    )
                    break
        return working

    def _auto_connect_graph_endpoints(
        self,
        graph: WireGraph,
        *,
        tolerance_px: float | None = None,
        extra_items: Sequence[ComponentItem] = (),
    ) -> WireGraph:
        """Deterministically upgrade touching `FreeEnd`s in `graph` to explicit refs/junctions."""
        from circuitsim.canvas.connectivity import point_strictly_inside_wire

        tol = self.AUTO_CONNECT_TOLERANCE_PX if tolerance_px is None else float(tolerance_px)
        term_map = self._terminal_pos_map(extra_items=extra_items)
        working = graph

        def _reconnect_or_drop_dup(
            g: WireGraph, wid: str, which: str, att: Attachment, hints: tuple[PointPx, ...]
        ) -> WireGraph:
            try:
                return g.reconnect_wire_end(wid, which, att, hints=hints)
            except WireGraphError:
                return g.remove_wire(wid)

        for _ in range(max(16, len(working.wires) * 4)):
            changed = False
            for w in working.wires:
                for which_end, att, other_att in (
                    ("end_a", w.end_a, w.end_b),
                    ("end_b", w.end_b, w.end_a),
                ):
                    if not isinstance(att, FreeEnd):
                        continue
                    fx, fy = float(att.x), float(att.y)

                    # 1. Nearest component terminal within tolerance
                    best_term: TerminalRef | None = None
                    best_term_dist = tol
                    for (cname, tname), (tx, ty) in term_map.items():
                        cand_ref = TerminalRef(cname, tname)
                        if cand_ref == other_att:
                            continue
                        dist = math.hypot(tx - fx, ty - fy)
                        if dist <= best_term_dist:
                            best_term_dist = dist
                            best_term = cand_ref
                    if best_term is not None:
                        working = _reconnect_or_drop_dup(
                            working, w.id, which_end, best_term, w.hints
                        )
                        changed = True
                        break

                    # 2. Nearest existing junction within tolerance
                    best_jref: JunctionRef | None = None
                    best_jdist = tol
                    for junc in working.junctions:
                        jref = JunctionRef(junc.id)
                        if jref == other_att:
                            continue
                        dist = math.hypot(junc.pos[0] - fx, junc.pos[1] - fy)
                        if dist <= best_jdist:
                            best_jdist = dist
                            best_jref = jref
                    if best_jref is not None:
                        working = _reconnect_or_drop_dup(
                            working, w.id, which_end, best_jref, w.hints
                        )
                        changed = True
                        break

                    # 3. Interior segment OR corner/bend of another wire
                    best_split: tuple[float, Wire, tuple[PointPx, ...], PointPx] | None = None
                    for w_other in working.wires:
                        if w_other.id == w.id:
                            continue
                        if other_att in (w_other.end_a, w_other.end_b):
                            continue
                        pts_other = (
                            self._wire_polyline_in_graph(w_other, working, term_map)
                            if not extra_items
                            else self._compute_wire_points_from_graph(
                                w_other, working, term_map=term_map
                            )
                        )
                        if len(pts_other) < 2:
                            continue
                        for idx_seg in range(len(pts_other) - 1):
                            ax, ay = pts_other[idx_seg]
                            bx, by = pts_other[idx_seg + 1]
                            if ax == bx:
                                lo, hi = min(ay, by), max(ay, by)
                                cy = min(max(fy, float(lo)), float(hi))
                                cx = float(ax)
                            elif ay == by:
                                lo, hi = min(ax, bx), max(ax, bx)
                                cx = min(max(fx, float(lo)), float(hi))
                                cy = float(ay)
                            else:
                                continue
                            dist = math.hypot(cx - fx, cy - fy)
                            if dist <= tol:
                                cand_pt = wire_snap_px((cx, cy))
                                if point_strictly_inside_wire(cand_pt, pts_other):
                                    if best_split is None or dist < best_split[0]:
                                        best_split = (dist, w_other, pts_other, cand_pt)
                    if best_split is not None:
                        _dist, w_other, pts_other, split_pt = best_split
                        hb = self._hints_before_point(split_pt, w_other, pts_other)
                        split_res = working.split_wire_at(
                            w_other.id, split_pt, hints_before=hb
                        )
                        jref = JunctionRef(split_res.junction_id)
                        working = _reconnect_or_drop_dup(
                            split_res.graph, w.id, which_end, jref, w.hints
                        )
                        changed = True
                        break

                    # 4. Another wire's FreeEnd within tolerance
                    best_free: tuple[float, Wire, str, PointPx] | None = None
                    for w_other in working.wires:
                        if w_other.id == w.id:
                            continue
                        for o_which, o_att in (
                            ("end_a", w_other.end_a),
                            ("end_b", w_other.end_b),
                        ):
                            if not isinstance(o_att, FreeEnd):
                                continue
                            dist = math.hypot(o_att.x - fx, o_att.y - fy)
                            if dist <= tol:
                                if best_free is None or dist < best_free[0]:
                                    best_free = (dist, w_other, o_which, o_att.pos)
                    if best_free is not None:
                        _dist, w_other, o_which, jpos = best_free
                        jid = working.new_junction_id()
                        jref = JunctionRef(jid)
                        g1 = working.add_junction(jpos, jid)
                        g2 = _reconnect_or_drop_dup(g1, w.id, which_end, jref, w.hints)
                        working = _reconnect_or_drop_dup(
                            g2, w_other.id, o_which, jref, w_other.hints
                        )
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break
        return working

    def _auto_connect_touching_endpoints(
        self, *, tolerance_px: float | None = None
    ) -> bool:
        """Apply `_auto_connect_graph_endpoints` directly onto the current scene graph."""
        old_graph = self._wire_graph
        new_graph = self._auto_connect_graph_endpoints(
            old_graph, tolerance_px=tolerance_px
        )
        if new_graph == old_graph:
            return False
        self.set_wire_graph(new_graph)
        return True

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

        pts = self._compute_wire_points_from_graph(new_graph.wire(wid), new_graph)
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
        pts = wire_route(
            wire_snap_px((float(start_px[0]), float(start_px[1]))),
            wire_snap_px((float(end_px[0]), float(end_px[1]))),
            start_dir=(1, 0) if prefer_horizontal_first else (0, 1),
        )
        return self.add_wire(pts)

    def _nearest_point_on_wire_segments(
        self, raw: tuple[float, float], graph: WireGraph, max_dist_px: float
    ) -> PointPx | None:
        rx, ry = raw
        term_map = self._terminal_pos_map()
        best_dist = max_dist_px
        best_pt: PointPx | None = None
        for w in graph.wires:
            pts = self._wire_polyline_in_graph(w, graph, term_map)
            for i in range(len(pts) - 1):
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                if ax == bx:
                    lo, hi = min(ay, by), max(ay, by)
                    cy = min(max(ry, float(lo)), float(hi))
                    cx = float(ax)
                elif ay == by:
                    lo, hi = min(ax, bx), max(ax, bx)
                    cx = min(max(rx, float(lo)), float(hi))
                    cy = float(ay)
                else:
                    continue
                dist = math.hypot(rx - cx, ry - cy)
                if dist <= best_dist:
                    cand = wire_snap_px((cx, cy))
                    if cand != pts[0] and cand != pts[-1]:
                        best_dist = dist
                        best_pt = cand
        return best_pt

    def toggle_junction_at(
        self, pos: QPointF | tuple[float, float] | PointPx
    ) -> bool:
        """Toggle an explicit junction on a wire at the snapped cursor via `QUndoStack`."""
        from circuitsim.canvas.connectivity import point_strictly_inside_wire

        raw = (
            (pos.x(), pos.y())
            if isinstance(pos, QPointF)
            else (float(pos[0]), float(pos[1]))
        )
        old_graph = self._wire_graph
        pt = wire_snap_px(raw)

        # Check existing junction at snapped point (or within snap radius of cursor).
        j_ref = self._find_junction_at(pt, old_graph, tolerance_px=0.5)
        if j_ref is None:
            j_ref = self._find_junction_at(
                raw, old_graph, tolerance_px=self.TERMINAL_SNAP_RADIUS_PX
            )
        if j_ref is not None:
            junc = old_graph.junction(j_ref.id)
            if (
                old_graph.junction_degree(junc.id) <= 2
                or junc.pos in self._explicit_junctions
            ):
                new_graph = old_graph.collapse_junction(junc.id)
                self.undo_stack.push(
                    ToggleJunctionCommand(
                        self,
                        junc.pos,
                        adding=False,
                        old_graph=old_graph,
                        new_graph=new_graph,
                    )
                )
                return False
            return False

        # Find all wires whose interior passes strictly through snapped `pt`.
        term_map = self._terminal_pos_map()
        matching: list[tuple[Wire, tuple[PointPx, ...]]] = []
        for w in old_graph.wires:
            pts = self._wire_polyline_in_graph(w, old_graph, term_map)
            if len(pts) >= 2 and point_strictly_inside_wire(pt, pts):
                matching.append((w, pts))

        if not matching:
            projected = self._nearest_point_on_wire_segments(
                raw, old_graph, self.TERMINAL_SNAP_RADIUS_PX
            )
            if projected is not None:
                pt = projected
                for w in old_graph.wires:
                    pts = self._wire_polyline_in_graph(w, old_graph, term_map)
                    if len(pts) >= 2 and point_strictly_inside_wire(pt, pts):
                        matching.append((w, pts))

        if not matching:
            return False

        working = old_graph
        shared_jid: str | None = None
        for w, pts in matching:
            hb = self._hints_before_point(pt, w, pts)
            split_res = working.split_wire_at(
                w.id,
                pt,
                junction_id=shared_jid,
                hints_before=hb,
            )
            working = split_res.graph
            shared_jid = split_res.junction_id

        self.undo_stack.push(
            ToggleJunctionCommand(
                self,
                pt,
                adding=True,
                old_graph=old_graph,
                new_graph=working,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Component Move Tracking, Rotate, Flip, Delete, Copy/Cut/Paste
    # ------------------------------------------------------------------

    def _update_graph_for_transformed_components(
        self,
        graph: WireGraph,
        comp_names: set[str],
        *,
        uniform_delta: tuple[int, int] | None = None,
    ) -> WireGraph:
        """Clear stale hints on wires incident to transformed components (or translate rigid moves)."""
        if not comp_names:
            return graph
        new_wires: list[Wire] = []
        changed = False
        for w in graph.wires:
            if not w.hints:
                new_wires.append(w)
                continue
            a_hit = isinstance(w.end_a, TerminalRef) and w.end_a.component in comp_names
            b_hit = isinstance(w.end_b, TerminalRef) and w.end_b.component in comp_names
            if uniform_delta is not None and a_hit and b_hit:
                dx_i, dy_i = uniform_delta
                shifted_hints = tuple((hx + dx_i, hy + dy_i) for hx, hy in w.hints)
                if shifted_hints != w.hints:
                    changed = True
                new_wires.append(Wire(w.id, w.end_a, w.end_b, shifted_hints))
            elif a_hit or b_hit:
                changed = True
                new_wires.append(Wire(w.id, w.end_a, w.end_b, ()))
            else:
                new_wires.append(w)
        if not changed:
            return graph
        return WireGraph(wires=tuple(new_wires), junctions=graph.junctions)

    def begin_component_move_tracking(self) -> None:
        """Snapshot component positions and `WireGraph` before drag."""
        self._move_tracking_active = True
        self._move_start_positions = {
            item: (item.pos().x(), item.pos().y())
            for item in self._placed_items
        }
        self._move_start_graph = self._wire_graph
        self._move_start_wire_points = {
            wid: witem.points for wid, witem in self._wire_items_by_id.items()
        }

    def _moved_graph_from_drag_start(
        self,
        moved_items: Sequence[tuple[ComponentItem, tuple[float, float], tuple[float, float]]],
    ) -> WireGraph:
        if not moved_items:
            return self._move_start_graph
        deltas = {
            (
                int(round(new_pos[0] - old_pos[0])),
                int(round(new_pos[1] - old_pos[1])),
            )
            for _item, old_pos, new_pos in moved_items
        }
        uniform_delta = next(iter(deltas)) if len(deltas) == 1 else None
        moved_names = {item.component.name for item, _o, _n in moved_items}
        return self._update_graph_for_transformed_components(
            self._move_start_graph,
            moved_names,
            uniform_delta=uniform_delta,
        )

    def on_component_position_changed(self, item: ComponentItem) -> None:
        """Update visual wire polylines during mouse drag WITHOUT mutating `wire_graph` (G7)."""
        _ = item
        if self._suppress_live_reroute or not self._move_tracking_active:
            return
        moved_items = [
            (comp, old_pos, (comp.pos().x(), comp.pos().y()))
            for comp, old_pos in self._move_start_positions.items()
            if (comp.pos().x(), comp.pos().y()) != old_pos
        ]
        moved_names = {comp.component.name for comp, _o, _n in moved_items}
        preview_graph = self._moved_graph_from_drag_start(moved_items)
        tmap = self._terminal_pos_map()
        dmap = self._terminal_dir_map()
        obs = self._component_obstacles_px()
        for w in preview_graph.wires:
            witem = self._wire_items_by_id.get(w.id)
            if witem is None:
                continue
            attached_to_moved = (
                (isinstance(w.end_a, TerminalRef) and w.end_a.component in moved_names)
                or (isinstance(w.end_b, TerminalRef) and w.end_b.component in moved_names)
            )
            if not attached_to_moved and w.id in self._move_start_wire_points:
                witem.set_points(self._move_start_wire_points[w.id])
            else:
                pts = self._compute_wire_points_from_graph(
                    w,
                    preview_graph,
                    term_map=tmap,
                    dir_map=dmap,
                    obstacles=obs,
                    fallback_points=witem.points or None,
                )
                witem.set_points(pts)
        self.refresh_connectivity_visuals()

    def commit_component_move_tracking(self) -> bool:
        """Finalize a mouse drag-move and push `MoveComponentsCommand` if anything moved."""
        if not self._move_tracking_active:
            return False
        self._move_tracking_active = False

        moved_old: dict[ComponentItem, tuple[float, float]] = {}
        moved_new: dict[ComponentItem, tuple[float, float]] = {}
        moved_items: list[tuple[ComponentItem, tuple[float, float], tuple[float, float]]] = []
        for item, old_pos in self._move_start_positions.items():
            cur_pos = (item.pos().x(), item.pos().y())
            if cur_pos != old_pos:
                moved_old[item] = old_pos
                moved_new[item] = cur_pos
                moved_items.append((item, old_pos, cur_pos))

        pre_drop_polylines = dict(self._move_start_wire_points)
        self._move_start_wire_points = {}

        if not moved_new:
            self._wire_graph = self._move_start_graph
            self._sync_graphics_from_wire_graph(frozen_polylines=pre_drop_polylines)
            self.update_label_positions()
            return False

        # G4: Prevent non-voltmeter single/batch drag drops from landing overlapped on another body
        excl_moved = set(moved_new.keys())
        self._suppress_live_reroute = True
        try:
            for item in list(moved_new.keys()):
                if item.component.component_type == "voltmeter":
                    continue
                cur_x, cur_y = moved_new[item]
                free_x, free_y = self._find_nearest_free_grid_pos(
                    item.component.component_type,
                    cur_x,
                    cur_y,
                    item.rotation_deg,
                    exclude=excl_moved,
                )
                if (free_x, free_y) != (cur_x, cur_y):
                    item.setPos(free_x, free_y)
                    moved_new[item] = (free_x, free_y)
            moved_items = [
                (item, moved_old[item], moved_new[item])
                for item in moved_new
            ]
        finally:
            self._suppress_live_reroute = False

        target_graph = self._moved_graph_from_drag_start(moved_items)
        if len(moved_items) == 1:
            moved_item, old_xy, new_xy = moved_items[0]
            if moved_item.component.component_type == "voltmeter":
                vm_name = moved_item.component.name
                has_wires = bool(
                    target_graph.attached_wires(TerminalRef(vm_name, "pos"))
                    or target_graph.attached_wires(TerminalRef(vm_name, "neg"))
                )
                candidates = [c for c in self._placed_items if c is not moved_item]
                overlay_target = None
                if not has_wires:
                    overlay_target = self._find_overlaid_2terminal_target(
                        moved_item, old_xy, candidates
                    )
                    if overlay_target is None:
                        overlay_target = self._find_overlaid_2terminal_target(
                            moved_item, new_xy, candidates
                        )
                if overlay_target is not None:
                    target_graph = self._connect_voltmeter_parallel_with(
                        moved_item,
                        overlay_target[0],
                        overlay_target[1],
                        overlay_target[2],
                        target_graph,
                    )
                else:
                    target_graph = self._apply_component_split_on_drop(
                        moved_item,
                        target_graph,
                        pre_drop_polylines=pre_drop_polylines,
                    )
            else:
                target_graph = self._apply_component_split_on_drop(
                    moved_item,
                    target_graph,
                    pre_drop_polylines=pre_drop_polylines,
                )
        target_graph = self._auto_connect_graph_endpoints(target_graph)
        self._wire_graph = target_graph
        self._sync_graphics_from_wire_graph()

        self.undo_stack.push(
            MoveComponentsCommand(
                self,
                moved_old,
                moved_new,
                old_graph=self._move_start_graph,
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

        old_graph = self._wire_graph
        deltas = {
            (
                int(round(new_positions[it][0] - old_positions[it][0])),
                int(round(new_positions[it][1] - old_positions[it][1])),
            )
            for it in new_positions
        }
        uniform_delta = next(iter(deltas)) if len(deltas) == 1 else None
        new_graph = self._update_graph_for_transformed_components(
            old_graph,
            {it.component.name for it in new_positions},
            uniform_delta=uniform_delta,
        )
        self._wire_graph = new_graph
        self._sync_graphics_from_wire_graph()

        self.undo_stack.push(
            MoveComponentsCommand(
                self,
                old_positions,
                new_positions,
                old_graph=old_graph,
                new_graph=new_graph,
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

        old_rotations = {it: it.rotation_deg for it in selected}
        new_rotations = {it: (it.rotation_deg + 90) % 360 for it in selected}

        for it, deg in new_rotations.items():
            it.set_rotation_deg(deg)

        old_graph = self._wire_graph
        new_graph = self._update_graph_for_transformed_components(
            old_graph, {it.component.name for it in selected}
        )
        self._wire_graph = new_graph
        self._sync_graphics_from_wire_graph()

        self.undo_stack.push(
            RotateComponentsCommand(
                self,
                selected,
                old_rotations,
                new_rotations,
                old_graph=old_graph,
                new_graph=new_graph,
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

        old_flips = {it: it.flipped_h for it in selected}
        new_flips = {it: not it.flipped_h for it in selected}

        for it, flipped in new_flips.items():
            it.set_flipped_h(flipped)

        old_graph = self._wire_graph
        new_graph = self._update_graph_for_transformed_components(
            old_graph, {it.component.name for it in selected}
        )
        self._wire_graph = new_graph
        self._sync_graphics_from_wire_graph()

        self.undo_stack.push(
            FlipComponentsCommand(
                self,
                selected,
                old_flips,
                new_flips,
                old_graph=old_graph,
                new_graph=new_graph,
            )
        )
        return True

    def select_all(self) -> None:
        """Select all placed components, wires, and visible junctions atomically."""
        prev = self._suppress_selection_info
        self._suppress_selection_info = True
        try:
            for item in self._placed_items:
                item.setSelected(True)
            for wire in self._wire_items:
                wire.setSelected(True)
            for jitem in self._junction_items_by_id.values():
                jitem.setSelected(True)
        finally:
            self._suppress_selection_info = prev
        self._on_selection_changed()

    def delete_selected(
        self,
        *,
        description: str = "Delete Selection",
        detach_unselected_wires: bool = False,
    ) -> bool:
        """Delete selected components, wires, and junctions.

        When `detach_unselected_wires` is True (UI delete actions), unselected wires
        attached to a deleted component are detached at that end into `FreeEnd` rather
        than deleted. Programmatic callers default to `detach_unselected_wires=False`,
        which removes wires attached to deleted components.
        """
        selected_comps = [
            it for it in self.selectedItems() if isinstance(it, ComponentItem)
        ]
        selected_wires = [
            it for it in self.selectedItems() if isinstance(it, WireItem)
        ]
        selected_junctions = [
            it for it in self.selectedItems() if isinstance(it, JunctionItem)
        ]
        if not selected_comps and not selected_wires and not selected_junctions:
            return False

        old_graph = self._wire_graph
        new_graph = old_graph
        deleted_explicit_junctions: list[PointPx] = []

        if detach_unselected_wires:
            for w in selected_wires:
                if new_graph.has_wire(w.wire_id):
                    new_graph = new_graph.remove_wire(w.wire_id)
            for jitem in selected_junctions:
                if jitem.junction_id and new_graph.has_junction(jitem.junction_id):
                    jpos = new_graph.junction(jitem.junction_id).pos
                    if jpos in self._explicit_junctions:
                        deleted_explicit_junctions.append(jpos)
                    new_graph = new_graph.remove_junction(jitem.junction_id)
            for comp in selected_comps:
                term_map = {
                    (comp.component.name, tname): pos
                    for tname, pos in comp.terminal_scene_positions_px().items()
                }
                new_graph = new_graph.detach_component(comp.component.name, term_map)
        else:
            for comp in selected_comps:
                new_graph = new_graph.remove_component(comp.component.name)
            for w in selected_wires:
                if new_graph.has_wire(w.wire_id):
                    new_graph = new_graph.remove_wire(w.wire_id)
            for jitem in selected_junctions:
                if jitem.junction_id and new_graph.has_junction(jitem.junction_id):
                    jpos = new_graph.junction(jitem.junction_id).pos
                    if jpos in self._explicit_junctions:
                        deleted_explicit_junctions.append(jpos)
                    new_graph = new_graph.remove_junction(jitem.junction_id)

        remaining_ids = {w.id for w in new_graph.wires}
        wires_to_delete = [w for w in self._wire_items if w.wire_id not in remaining_ids]

        self.undo_stack.push(
            DeleteItemsCommand(
                self,
                selected_comps,
                wires_to_delete,
                deleted_explicit_junctions,
                old_graph=old_graph,
                new_graph=new_graph,
                description=description,
            )
        )
        return True

    def copy_selected(self) -> bool:
        """Copy selected components, wires, and junctions to the internal clipboard."""
        placed_order = {id(it): idx for idx, it in enumerate(self._placed_items)}
        selected_comps = sorted(
            (it for it in self.selectedItems() if isinstance(it, ComponentItem)),
            key=lambda it: placed_order.get(id(it), 0),
        )
        selected_wires = [
            it for it in self.selectedItems() if isinstance(it, WireItem)
        ]
        selected_junctions = [
            it for it in self.selectedItems() if isinstance(it, JunctionItem)
        ]
        if not selected_comps and not selected_wires and not selected_junctions:
            return False

        comp_names = [it.component.name for it in selected_comps]
        wire_ids = [w.wire_id for w in selected_wires]
        junc_ids = [j.junction_id for j in selected_junctions if j.junction_id]
        subgraph = extract_subgraph(
            self._wire_graph,
            comp_names,
            wire_ids,
            junc_ids,
            self._terminal_pos_map(),
        )
        explicit_junctions = tuple(
            j.pos for j in subgraph.junctions if j.pos in self._explicit_junctions
        )

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
            subgraph=subgraph,
            explicit_junctions=explicit_junctions,
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
        if offset_px is None or isinstance(offset_px, bool):
            step = float(GRID_PITCH_PX * 2 * self._paste_serial)
            dx, dy = step, step
        else:
            dx, dy = snap_to_grid(offset_px[0], offset_px[1], GRID_PITCH_PX)

        new_comps: list[ComponentItem] = []
        component_name_map: dict[str, str] = {}
        assigned_names: set[str] = set()
        for entry in self._clipboard.components:
            cloned = copy.deepcopy(entry.component_snapshot)
            old_name = cloned.name
            new_name = self.next_component_name(
                cloned.component_type, extra_taken=assigned_names
            )
            assigned_names.add(new_name)
            component_name_map[old_name] = new_name
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

        id_x = int(round(dx))
        id_y = int(round(dy))
        working_graph, new_wire_ids, new_junction_ids = paste_subgraph(
            self._wire_graph,
            self._clipboard.subgraph,
            component_name_map,
            (id_x, id_y),
        )
        combined_term_map = self._terminal_pos_map()
        combined_dir_map = self._terminal_dir_map()
        combined_obstacles = list(self._component_obstacles_px())
        for comp_item in new_comps:
            cname = comp_item.component.name
            cx = int(round(comp_item.pos().x()))
            cy = int(round(comp_item.pos().y()))
            for tname, tpt in comp_item.terminal_scene_positions_px().items():
                combined_term_map[(cname, tname)] = tpt
                combined_dir_map[(cname, tname)] = wire_normalize_dir(
                    (tpt[0] - cx, tpt[1] - cy)
                )
            rect = comp_item.scene_body_rect()
            combined_obstacles.append(
                (
                    int(round(rect.left())),
                    int(round(rect.top())),
                    int(round(rect.right())),
                    int(round(rect.bottom())),
                )
            )
        obstacles_tuple = tuple(combined_obstacles)

        new_wires: list[WireItem] = []
        for wid in new_wire_ids:
            wire_obj = working_graph.wire(wid)
            pts = self._compute_wire_points_from_graph(
                wire_obj,
                working_graph,
                term_map=combined_term_map,
                dir_map=combined_dir_map,
                obstacles=obstacles_tuple,
            )
            new_wires.append(
                WireItem(
                    wid,
                    pts,
                    theme=self.theme,
                    end_a=wire_obj.end_a,
                    end_b=wire_obj.end_b,
                    hints=wire_obj.hints,
                )
            )

        explicit_pts = tuple(
            (jx + id_x, jy + id_y) for jx, jy in self._clipboard.explicit_junctions
        )

        self.undo_stack.push(
            PasteItemsCommand(
                self,
                new_comps,
                new_wires,
                junctions=explicit_pts,
                old_graph=self._wire_graph,
                new_graph=working_graph,
                select_junction_ids=set(new_junction_ids),
            )
        )
        return True

    # ------------------------------------------------------------------
    # Netlist Extraction (CircuitScene.build_circuit)
    # ------------------------------------------------------------------

    def extract_connectivity_result(self) -> ConnectivityResult:
        """Build the electrical `Circuit` and telemetry maps from `self._wire_graph`."""
        components = [item.component for item in self._placed_items]
        effective_graph = self._wire_graph

        # Preserve pin-to-pin contact recorded when co-located components were placed/loaded.
        placed_by_id = {id(item): item for item in self._placed_items}
        term_pos_by_id = {
            id(item): item.terminal_scene_positions_px()
            for item in self._placed_items
        }
        existing_pairs: set[frozenset[Attachment]] = {
            frozenset((w.end_a, w.end_b)) for w in effective_graph.wires
        }
        bond_idx = 0
        for item_a, term_a, item_b, term_b in self._pin_to_pin_bonds:
            if (
                placed_by_id.get(id(item_a)) is not item_a
                or placed_by_id.get(id(item_b)) is not item_b
            ):
                continue
            pt_a = term_pos_by_id[id(item_a)].get(term_a)
            pt_b = term_pos_by_id[id(item_b)].get(term_b)
            if pt_a is None or pt_a != pt_b:
                continue
            ref_a = TerminalRef(item_a.component.name, term_a)
            ref_b = TerminalRef(item_b.component.name, term_b)
            if ref_a == ref_b:
                continue
            pair = frozenset((ref_a, ref_b))
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            while True:
                bond_idx += 1
                bid = f"__pin_bond_{bond_idx}"
                if not effective_graph.has_wire(bid):
                    break
            effective_graph = effective_graph.add_wire(
                ref_a, ref_b, wire_id=bid
            )

        wnet = wiring_build_circuit(
            effective_graph,
            components,
            symbol_catalog=self.symbol_library.metadata_catalog,
        )

        wire_to_node = {
            wid: node
            for wid, node in wnet.wire_to_node.items()
            if not wid.startswith("__pin_bond_")
        }
        open_wire_ids = [
            w.id
            for w in self._wire_graph.wires
            if isinstance(w.end_a, FreeEnd) or isinstance(w.end_b, FreeEnd)
        ]

        point_to_node: dict[PointPx, str] = {}
        for item in self._placed_items:
            cname = item.component.name
            node_map = wnet.component_nodes.get(cname, {})
            for tname, pt in item.terminal_scene_positions_px().items():
                if tname in node_map:
                    point_to_node[pt] = node_map[tname]
        for j in self._wire_graph.junctions:
            if j.id in wnet.junction_to_node:
                point_to_node[j.pos] = wnet.junction_to_node[j.id]
        for w_item in self._wire_items:
            if w_item.wire_id in wire_to_node and w_item.points:
                wnode = wire_to_node[w_item.wire_id]
                point_to_node.setdefault(w_item.points[0], wnode)
                point_to_node.setdefault(w_item.points[-1], wnode)

        junction_dots = {
            j.pos
            for j in self._wire_graph.junctions
            if self._wire_graph.junction_dot_visible(j.id)
        } | set(self._explicit_junctions)
        valid_wires = tuple(
            (w.wire_id, w.points)
            for w in self._wire_items
            if len(w.points) >= 2
        )

        return ConnectivityResult(
            circuit=wnet.circuit,
            component_nodes=wnet.component_nodes,
            wire_to_node=wire_to_node,
            point_to_node=point_to_node,
            junction_dots=junction_dots,
            valid_wires=valid_wires,
            unconnected_components=wnet.unconnected_components,
            open_wire_ids=open_wire_ids,
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

    def set_switch_item_state(self, item: ComponentItem, closed: bool) -> bool:
        """Explicitly set a `Switch` `ComponentItem` open/closed (used by momentary switches)."""
        if not isinstance(item.component, Switch):
            return False
        new_closed = bool(closed)
        if item.component.closed != new_closed:
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
            self._last_sim_frame = None
            for item in self._placed_items:
                item.reset_live_telemetry()
        else:
            self.latest_node_voltages = dict(frame.node_voltages)
            self.is_simulation_running = True
            self._last_sim_frame = frame
            for item in self._placed_items:
                telem = frame.components.get(item.component.name)
                if telem is not None:
                    item.apply_telemetry(telem)

        self._refresh_wire_overlays()
        self._sync_dot_anim_timer()

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

    def set_voltage_colors_enabled(self, enabled: bool) -> None:
        """Enable or disable live voltage color-coding on wires."""
        val = bool(enabled)
        if val != self.voltage_colors_enabled:
            self.voltage_colors_enabled = val
            self._refresh_wire_overlays()
            self.update()

    def set_current_dots_enabled(self, enabled: bool) -> None:
        """Enable or disable animated current flow dots on wires."""
        val = bool(enabled)
        if val != self.current_dots_enabled:
            self.current_dots_enabled = val
            self._refresh_wire_overlays()
            self._sync_dot_anim_timer()
            self.update()

    def _sync_dot_anim_timer(self) -> None:
        should_run = (
            self.current_dots_enabled
            and self.is_simulation_running
            and not self._is_shutting_down
        )
        if should_run and not self._dot_anim_timer.isActive():
            self._dot_anim_timer.start()
        elif not should_run and self._dot_anim_timer.isActive():
            self._dot_anim_timer.stop()

    def step_current_dot_animation(self, step_scale: float = 0.12) -> None:
        """Advance animated current dot phase on all placed wires."""
        if not self.current_dots_enabled or not self.is_simulation_running:
            return
        for wire in self._wire_items:
            wire.advance_dot_phase(step_scale)

    def _refresh_wire_overlays(self) -> None:
        if not self.is_simulation_running or self._last_sim_frame is None:
            for wire in self._wire_items:
                wire.set_overlay_state(
                    voltage_color_enabled=self.voltage_colors_enabled,
                    current_dots_enabled=self.current_dots_enabled,
                    clear_overlay=True,
                )
            return

        if not self.voltage_colors_enabled and not self.current_dots_enabled:
            for wire in self._wire_items:
                wire.set_overlay_state(
                    voltage_color_enabled=False,
                    current_dots_enabled=False,
                    clear_overlay=True,
                )
            return

        conn = self.extract_connectivity_result()
        term_names = {
            it.component.name: tuple(t.name for t in it.metadata.terminals)
            for it in self._placed_items
        }
        branch_currents = {
            cname: telem.current
            for cname, telem in self._last_sim_frame.components.items()
        }
        wire_currents = (
            estimate_wire_currents(self._wire_graph, term_names, branch_currents)
            if self.current_dots_enabled
            else {}
        )
        for wire in self._wire_items:
            net_name = conn.wire_to_node.get(wire.wire_id)
            rgb = None
            if self.voltage_colors_enabled and net_name is not None and net_name in self.latest_node_voltages:
                rgb = map_voltage_to_rgb(self.latest_node_voltages[net_name])
            vel = (
                compute_current_dot_velocity(wire_currents.get(wire.wire_id, 0.0))
                if self.current_dots_enabled
                else 0.0
            )
            wire.set_overlay_state(
                voltage_color_enabled=self.voltage_colors_enabled,
                current_dots_enabled=self.current_dots_enabled,
                overlay_rgb=rgb,
                dot_velocity=vel,
            )

    def get_alt_power_probe_text(self, item: ComponentItem) -> str:
        """Compute and format instantaneous `P = V_drop * I_branch` power reading for Alt-hover."""
        comp = item.component
        is_src = comp.component_type in ("dc_voltage", "ac_voltage", "dc_current", "wav_voltage")
        p_w = float(getattr(comp, "power", comp.voltage * comp.current))
        text = format_power_probe_text(
            comp.name,
            p_w,
            voltage_v=float(comp.voltage),
            current_a=float(comp.current),
            is_running=self.is_simulation_running,
            is_source=is_src,
        )
        self.last_power_probe_text = text
        return text

    def query_alt_power_probe_at(
        self, scene_pos: QPointF | tuple[float, float]
    ) -> str | None:
        """Return formatted Alt-hover power probe text for the component at `scene_pos`, if any."""
        pt = (
            scene_pos
            if isinstance(scene_pos, QPointF)
            else QPointF(float(scene_pos[0]), float(scene_pos[1]))
        )
        for item in reversed(self._placed_items):
            local_pt = item.mapFromScene(pt)
            if item.shape().contains(local_pt) or item.boundingRect().contains(local_pt):
                return self.get_alt_power_probe_text(item)
        return None

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

    def to_document_v2(self, view: ViewState | None = None) -> CircuitDocumentV2:
        """Snapshot the schematic as a Qt-free Schema v2 `CircuitDocumentV2` ready to be saved."""
        export_graph = self._wire_graph
        placed_by_id = {id(item): item for item in self._placed_items}
        term_pos_by_id = {
            id(item): item.terminal_scene_positions_px()
            for item in self._placed_items
        }
        existing_pairs: set[frozenset[Attachment]] = {
            frozenset((w.end_a, w.end_b)) for w in export_graph.wires
        }
        for item_a, term_a, item_b, term_b in self._pin_to_pin_bonds:
            if (
                placed_by_id.get(id(item_a)) is not item_a
                or placed_by_id.get(id(item_b)) is not item_b
            ):
                continue
            pt_a = term_pos_by_id[id(item_a)].get(term_a)
            pt_b = term_pos_by_id[id(item_b)].get(term_b)
            if pt_a is None or pt_a != pt_b:
                continue
            ref_a = TerminalRef(item_a.component.name, term_a)
            ref_b = TerminalRef(item_b.component.name, term_b)
            if ref_a == ref_b:
                continue
            pair = frozenset((ref_a, ref_b))
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            wid = export_graph.new_wire_id()
            export_graph = export_graph.add_wire(ref_a, ref_b, wire_id=wid)
        return CircuitDocumentV2(
            placed=[item.to_placed_spec() for item in self._placed_items],
            graph=export_graph,
            view=view if view is not None else ViewState(theme=self.theme.value),
        )

    def to_document(self, view: ViewState | None = None) -> CircuitDocument:
        """Snapshot the schematic as a Qt-free `CircuitDocument` ready to be saved."""
        return CircuitDocument(
            placed=[item.to_placed_spec() for item in self._placed_items],
            wires=[wire.to_wire_spec() for wire in self._wire_items],
            junctions=sorted(self._explicit_junctions),
            view=view if view is not None else ViewState(theme=self.theme.value),
        )

    def load_document(self, doc: CircuitDocument | CircuitDocumentV2) -> None:
        """Replace the whole schematic with `doc`, all-or-nothing.

        Accepts both Schema v2 (`CircuitDocumentV2`) and legacy Schema v1
        (`CircuitDocument`, auto-migrated via `migrate_v1_to_v2`).
        Step 1 builds every new item without touching the scene. If anything
        goes wrong there, `CircuitFileError` is raised and the current circuit
        is left exactly as it was. Step 2 (the swap) only runs after step 1
        succeeded and contains no code that can reasonably fail.
        """
        is_v1 = not isinstance(doc, CircuitDocumentV2)
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
            doc_v2 = migrate_v1_to_v2(doc) if is_v1 else doc
            # Dry run of netlist extraction: proves the circuit can be analysed.
            doc_v2.build_circuit(
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
        self._pin_to_pin_bonds.clear()
        self._counters.clear()
        self._explicit_junctions = set()
        for idx, item in enumerate(new_items):
            self._placed_items.append(item)
            self.addItem(item)
            if is_v1:
                self._record_pin_to_pin_bonds_for(item, new_items[:idx])

        self._wire_counter = 0
        self._wire_graph = doc_v2.graph
        self._sync_graphics_from_wire_graph()
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
        self._pin_to_pin_bonds.clear()
        self._explicit_junctions.clear()
        self._active_junction_dots.clear()
        self._counters.clear()
        self._wire_counter = 0
        self.undo_stack.clear()
        self.selection_info_changed.emit(None)
        self.circuit_modified.emit()
        self.update()

    def _on_selection_changed(self) -> None:
        if self._is_shutting_down or self._suppress_selection_info:
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
            self._dot_anim_timer.stop()
        except RuntimeError:
            pass
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

            # 2. Draw subtle endpoint handles on selected wires or hovered wire endpoint
            handle_pts: set[PointPx] = set()
            if not self.is_drawing_wire:
                for witem in self._wire_items:
                    if (
                        witem is not self._preview_wire_item
                        and witem.scene() is self
                        and witem.isSelected()
                        and len(witem.points) >= 2
                    ):
                        handle_pts.add(witem.points[0])
                        handle_pts.add(witem.points[-1])
                if self._hovered_wire_end_pt is not None:
                    handle_pts.add(self._hovered_wire_end_pt)
            if handle_pts:
                h_col = QColor(palette.terminal_hover)
                h_fill = QColor(palette.terminal_hover)
                h_fill.setAlpha(45)
                painter.setPen(QPen(h_col, 1.5))
                painter.setBrush(QBrush(h_fill))
                for hx, hy in handle_pts:
                    painter.drawEllipse(QPointF(float(hx), float(hy)), 4.5, 4.5)

            # 3. Draw glowing highlight ring around hovered terminal or wire snap target
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
