"""Canvas QGraphicsObject for schematic components, upright labels, switch animation, and bulb glow."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneHoverEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from circuitsim.app.themes import ThemeMode, ThemePalette, get_theme_palette
from circuitsim.canvas.symbol_geometry import logic_gate_marks
from circuitsim.components.logic_gate import LogicGate
from circuitsim.canvas.connectivity import (
    PlacedComponentSpec,
    PointPx,
    normalize_point_px,
)
from circuitsim.canvas.symbol_geometry import (
    GRID_PITCH_PX,
    SymbolMetadata,
    led_marks,
    snap_to_grid,
    transform_terminal_position,
)
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.components import (
    ACVoltageSource,
    Ammeter,
    BJTBase,
    Capacitor,
    Comparator,
    Component,
    Counter4Bit,
    DCCurrentSource,
    DCVoltageSource,
    DFlipFlop,
    DigitalSequentialBlock,
    Diode,
    FunctionGenerator,
    Fuse,
    Ground,
    Inductor,
    JKFlipFlop,
    LED,
    Lightbulb,
    MOSFETBase,
    NE555,
    OpAmp,
    Oscilloscope,
    Potentiometer,
    Resistor,
    SPDTRelay,
    SevenSegDisplay,
    Switch,
    Transformer,
    Voltmeter,
    ZenerDiode,
)
from circuitsim.dialogs.value_parser import format_engineering_value
from circuitsim.engine.simulation_session import ComponentTelemetry

COMPONENT_PREFIXES: dict[str, str] = {
    "resistor": "R",
    "potentiometer": "POT",
    "capacitor": "C",
    "inductor": "L",
    "dc_voltage": "V",
    "ac_voltage": "Vac",
    "dc_current": "I",
    "ground": "GND",
    "switch": "SW",
    "spdt_relay": "K",
    "fuse": "F",
    "transformer": "T",
    "diode": "D",
    "zener_diode": "DZ",
    "npn_bjt": "Q",
    "pnp_bjt": "Q",
    "nmos": "M",
    "pmos": "M",
    "opamp": "U",
    "comparator": "U",
    "ne555": "U",
    "d_flipflop": "U",
    "jk_flipflop": "U",
    "counter_4bit": "U",
    "seven_seg_display": "DS",
    "gate_not": "U",
    "gate_and": "U",
    "gate_or": "U",
    "gate_nand": "U",
    "gate_nor": "U",
    "gate_xor": "U",
    "gate_xnor": "U",
    "lightbulb": "B",
    "voltmeter": "VM",
    "ammeter": "A",
    "mcu": "U",
    "oscilloscope": "OSC",
    "function_generator": "FG",
}


# Re-exported from the Qt-free factory so the .circ loader can use it too.
from circuitsim.components.factory import create_default_component_model  # noqa: E402,F401


def format_component_value_summary(comp: Component) -> str:
    """Return a concise engineering-notation summary string for canvas display."""
    if isinstance(comp, Resistor):
        return format_engineering_value(comp.resistance, "Ω")
    if isinstance(comp, Potentiometer):
        return f"{format_engineering_value(comp.resistance, 'Ω')} ({comp.alpha * 100.0:.0f}%)"
    if isinstance(comp, Capacitor):
        return format_engineering_value(comp.capacitance, "F")
    if isinstance(comp, Inductor):
        return format_engineering_value(comp.inductance, "H")
    if isinstance(comp, DCVoltageSource):
        return format_engineering_value(comp.dc_voltage, "V")
    if isinstance(comp, FunctionGenerator):
        return comp.display_value
    if isinstance(comp, ACVoltageSource):
        v_str = format_engineering_value(comp.amplitude, "V")
        f_str = format_engineering_value(comp.frequency, "Hz")
        return f"{v_str} {f_str}"
    if isinstance(comp, DCCurrentSource):
        return format_engineering_value(comp.dc_current, "A")
    if isinstance(comp, Switch):
        if comp.arcing and not comp.closed:
            return "ARC!"
        return "ON" if comp.closed else "OFF"
    if isinstance(comp, Fuse):
        return comp.display_value
    if isinstance(comp, SPDTRelay):
        return comp.display_value
    if isinstance(comp, Transformer):
        return comp.display_value
    if isinstance(comp, Lightbulb):
        if comp.burned_out:
            return "BURNED OUT"
        return f"{comp.rated_power:g}W / {comp.rated_voltage:g}V"
    if isinstance(comp, ZenerDiode):
        return f"{comp.v_z:g} V"
    if isinstance(comp, Diode):
        return comp.model_name
    if isinstance(comp, BJTBase):
        return f"β={comp.beta_f:g}"
    if isinstance(comp, MOSFETBase):
        return f"Vth={comp.v_th:g}V"
    if isinstance(comp, Comparator):
        return comp.display_value
    if isinstance(comp, OpAmp):
        return f"A={comp.a_ol:.0e}"
    if isinstance(comp, NE555):
        return comp.display_value
    if isinstance(comp, DigitalSequentialBlock):
        return comp.display_value
    if isinstance(comp, Oscilloscope):
        return comp.display_value
    if isinstance(comp, (Voltmeter, Ammeter)):
        return comp.format_reading()
    if isinstance(comp, LogicGate):
        family_labels = {
            "5V_CMOS": "5V CMOS",
            "5V_TTL": "5V TTL",
            "3V3_LVCMOS": "3.3V LVCMOS",
            "Custom": "Custom",
        }
        return family_labels.get(comp.family, str(comp.family))
    return ""


def rect_intersects_segment(
    rect: QRectF,
    p1: tuple[float, float] | QPointF,
    p2: tuple[float, float] | QPointF,
) -> bool:
    """Check if an orthogonal or general line segment intersects or enters a rectangle."""
    rx1 = rect.left()
    ry1 = rect.top()
    rx2 = rect.right()
    ry2 = rect.bottom()

    x1 = p1.x() if isinstance(p1, QPointF) else p1[0]
    y1 = p1.y() if isinstance(p1, QPointF) else p1[1]
    x2 = p2.x() if isinstance(p2, QPointF) else p2[0]
    y2 = p2.y() if isinstance(p2, QPointF) else p2[1]

    # Either endpoint is inside the rectangle
    if (rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2) or (rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2):
        return True

    # Orthogonal horizontal segment
    if abs(y1 - y2) < 1e-6:
        seg_y = y1
        if ry1 <= seg_y <= ry2:
            seg_min_x = min(x1, x2)
            seg_max_x = max(x1, x2)
            if max(seg_min_x, rx1) <= min(seg_max_x, rx2):
                return True
        return False

    # Orthogonal vertical segment
    if abs(x1 - x2) < 1e-6:
        seg_x = x1
        if rx1 <= seg_x <= rx2:
            seg_min_y = min(y1, y2)
            seg_max_y = max(y1, y2)
            if max(seg_min_y, ry1) <= min(seg_max_y, ry2):
                return True
        return False

    # Bounding box test for general line
    if max(x1, x2) < rx1 or min(x1, x2) > rx2 or max(y1, y2) < ry1 or min(y1, y2) > ry2:
        return False
    return True


class ComponentItem(QGraphicsObject):
    """Visual schematic item representing an electrical component on the canvas.

    Supports:
        - Grid snapping (20 px pitch)
        - 90-degree rotations and horizontal flip with upright, readable text labels
        - SVG rendering (if file exists in `resources/symbols/iec/`) or procedural IEC fallback
        - Animated switch lever, real-power lightbulb glow + over-bright warning, and
          live meter readouts (`-- V` / `-- A` until simulation produces a reading)
        - Problem diagnostic highlighting (`error` in red, `warning` in amber)
    """

    OPEN_LEVER_ANGLE_DEG: float = 26.0
    CLOSED_LEVER_ANGLE_DEG: float = 0.0

    def __init__(
        self,
        component: Component,
        symbol_library: SymbolLibrary,
        *,
        theme: ThemeMode = ThemeMode.DARK,
        is_ghost: bool = False,
        rotation_deg: int = 0,
        flipped_h: bool = False,
    ) -> None:
        super().__init__()
        self.component: Component = component
        self.symbol_library: SymbolLibrary = symbol_library
        self.metadata: SymbolMetadata = symbol_library.get_metadata(
            component.component_type
        )
        self.theme: ThemeMode = theme
        self.is_ghost: bool = is_ghost
        self._rotation_deg: int = 0
        self._flipped_h: bool = False
        if rotation_deg:
            self.set_rotation_deg(rotation_deg)
        if flipped_h:
            self.set_flipped_h(flipped_h)

        # Live simulation visual states
        self.live_display_value: str | None = None
        self.diagnostic_severity: str | None = None  # "error", "warning", or None
        self.diagnostic_message: str = ""
        self.diagnostic_code: str = ""
        self.stress_level: str = "normal"  # "normal", "warning", "overload"
        self.stress_reason: str = ""
        self.stress_utilization_pct: float = 0.0
        self.stress_warnings_enabled: bool = True
        self._pinout_overlay_enabled: bool = False
        self.is_unconnected: bool = False
        self.label_position: str = "default"
        self._cached_bounds_key: tuple[object, ...] | None = None
        self._cached_bounding_rect: QRectF | None = None
        self._bounding_rect_recompute_count: int = 0

        # Switch lever animation state
        self._lever_angle_deg: float = (
            self.CLOSED_LEVER_ANGLE_DEG
            if (isinstance(component, Switch) and component.closed)
            else self.OPEN_LEVER_ANGLE_DEG
        )
        self._lever_anim: QVariantAnimation | None = None

        # Blown fuse POP! starburst animation state
        self._fuse_pop_progress: float = 0.0
        self._fuse_pop_anim: QVariantAnimation | None = None

        self.setToolTip(self.get_live_tooltip(is_running=False))
        self.setAcceptHoverEvents(True)
        self._is_hovered: bool = False

        if not is_ghost:
            self.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if isinstance(component, Switch)
                else Qt.CursorShape.OpenHandCursor
            )
            self.setZValue(20.0)
        else:
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.setOpacity(0.65)
            self.setZValue(100.0)

    @property
    def symbol_state(self) -> str | None:
        """Return active symbol state name (e.g. 'open' or 'closed' for Switch)."""
        if isinstance(self.component, Switch):
            return "closed" if self.component.closed else "open"
        if isinstance(self.component, Fuse):
            return "blown" if self.component.is_blown else "intact"
        if isinstance(self.component, SPDTRelay):
            return "closed" if self.component.energized else "open"
        if isinstance(self.component, Lightbulb):
            if self.component.burned_out:
                return "burned_out"
            if self.component.brightness > 0.05:
                return "lit"
            return "normal"
        if isinstance(self.component, LED):
            if self.component.burned_out:
                return "burned_out"
            if getattr(self.component, "brightness", 0.0) > 0.0:
                return "lit"
            return "normal"
        return None

    @property
    def lever_angle_deg(self) -> float:
        return self._lever_angle_deg

    def effective_text_angle_deg(self) -> float:
        """Return the scene rotation angle of text labels (always 0° upright)."""
        return 0.0

    def set_theme(self, theme: ThemeMode) -> None:
        """Update item theme and trigger a repaint."""
        self.theme = theme
        self.update()

    def set_unconnected(self, unconnected: bool) -> None:
        """Dim or restore the component visually when it has no wires connected."""
        if self.is_unconnected == bool(unconnected):
            return
        self.is_unconnected = bool(unconnected)
        if not self.is_ghost:
            self.setOpacity(0.45 if self.is_unconnected else 1.0)
        self.update()

    def get_live_tooltip(self, is_running: bool = False) -> str:
        """Return formatted live tooltip text for this component."""
        if isinstance(self.component, FunctionGenerator):
            fg = self.component
            z_txt = "Hi-Z" if fg.output_impedance < 1e-6 else format_engineering_value(fg.output_impedance, "Ω")
            lines = [
                f"{fg.name} (Function Generator)",
                "Double-click to open Function Generator",
                f"Waveform: {fg.waveform.value.title()} | {'ON' if fg.output_enabled else 'OFF'}",
                f"Freq: {format_engineering_value(fg.frequency, 'Hz')} | Amp: {format_engineering_value(fg.amplitude, 'V')} | Zout: {z_txt}",
            ]
            return "\n".join(lines)
        if isinstance(self.component, Oscilloscope):
            osc = self.component
            lines = [
                f"{osc.name} (DSO-1062D)",
                "Double-click to open DSO",
            ]
            if is_running:
                lines.append(
                    f"CH1: {format_engineering_value(osc.v_ch1, 'V')} | CH2: {format_engineering_value(osc.v_ch2, 'V')}"
                )
            return "\n".join(lines)
        if isinstance(self.component, LED):
            led = self.component
            lines: list[str] = []
            if self.stress_level in ("overload", "warning") and self.stress_reason:
                lines.append(f"{self.stress_level.upper()}: {self.stress_reason}")
            if self.diagnostic_message:
                lines.append(f"Problem: {self.diagnostic_message}")
            lines.append(f"Name: {led.name}")
            lines.append(f"Colour: {led.color_name}")
            if is_running:
                lines.append(f"Voltage: {format_engineering_value(abs(led.voltage), 'V')}")
                lines.append(f"Current: {format_engineering_value(abs(led.current), 'A')}")
            else:
                lines.append("Voltage: --")
                lines.append("Current: --")
            if led.burned_out:
                st = "Burned out"
            elif not is_running:
                st = "Off"
            elif getattr(led, "is_over_current", False):
                st = "Over-current"
            elif getattr(led, "brightness", 0.0) > 0.0 or abs(led.current) >= 1e-5:
                st = "On"
            elif led.voltage < -0.1:
                st = "Reverse-biased"
            else:
                st = "Off"
            lines.append(f"State: {st}")
            return "\n".join(lines)

        lines: list[str] = []
        if self.stress_level in ("overload", "warning") and self.stress_reason:
            lines.append(f"{self.stress_level.upper()}: {self.stress_reason}")
        if self.diagnostic_message:
            lines.append(f"Problem: {self.diagnostic_message}")
        lines.append(f"Name: {self.component.name}")
        val_str = format_component_value_summary(self.component)
        lines.append(f"Value: {val_str if val_str else '--'}")
        if is_running:
            lines.append(f"Voltage: {format_engineering_value(abs(self.component.voltage), 'V')}")
            lines.append(f"Current: {format_engineering_value(abs(self.component.current), 'A')}")
            is_src = isinstance(
                self.component,
                (DCVoltageSource, ACVoltageSource, DCCurrentSource),
            )
            if is_src:
                p_val = self.component.power
                p_str = format_engineering_value(p_val, "W")
                if p_val < -1e-12:
                    lines.append(f"Power: {p_str} (delivering)")
                elif p_val > 1e-12:
                    lines.append(f"Power: {p_str} (absorbing)")
                else:
                    lines.append("Power: 0 W")
                lines.append(
                    "Positive power is absorbed; sources indicate delivered power."
                )
            else:
                lines.append(f"Power: {format_engineering_value(self.component.power, 'W')}")
        else:
            lines.append("Voltage: --")
            lines.append("Current: --")
            lines.append("Power: --")
        if (
            isinstance(self.component, Voltmeter)
            and getattr(self.component, "loading_error_message", None)
        ):
            lines.append(self.component.loading_error_message)
        if isinstance(self.component, LogicGate):
            g = self.component
            lines.append(f"Operation: {g.operation.upper()} | Family: {g.family}")
            lines.append(
                f"Delay: t_pd={format_engineering_value(g.t_pd, 's')}, "
                f"t_edge={format_engineering_value(g.t_edge, 's')}"
            )
            lines.append(
                f"Thresholds: V_IL={g.v_il:.3g} V, V_IH={g.v_ih:.3g} V | "
                f"Rails: V_OL={g.v_ol:.3g} V, V_OH={g.v_oh:.3g} V"
            )
            if is_running:
                lvl = getattr(self, "logic_level", "") or ("H" if g.output_level else "L")
                lines.append(f"Output Level: {lvl}")
            else:
                lines.append(f"Initial Output: {'H' if g.initial_state else 'L'}")
        return "\n".join(lines)

    def set_diagnostic(
        self, severity: str | None, message: str = "", *, code: str = ""
    ) -> None:
        """Set or clear canvas problem highlighting on this component."""
        if (
            self.diagnostic_severity == severity
            and self.diagnostic_message == message
            and self.diagnostic_code == code
        ):
            return
        self.diagnostic_severity = severity
        self.diagnostic_message = message
        self.diagnostic_code = code
        self.setToolTip(self.get_live_tooltip(is_running=self.live_display_value is not None))
        self.update()

    def effective_highlight_color(self) -> str | None:
        """Return the active problem-highlight hex color ('#FF5252' for error, '#FFA726' for warning), or None."""
        if self.diagnostic_severity == "error":
            return "#FF5252"
        if self.diagnostic_severity == "warning":
            return "#FFA726"
        return None

    def apply_telemetry(self, telemetry: ComponentTelemetry) -> None:
        """Update live operating state from a `SimulationFrame` telemetry snapshot."""
        self.component.voltage = telemetry.voltage
        self.component.current = telemetry.current
        self.component.power = telemetry.power
        self.live_display_value = telemetry.display_value

        if isinstance(self.component, Lightbulb):
            self.component.burned_out = telemetry.burned_out
        elif isinstance(self.component, LED):
            self.component.burned_out = telemetry.burned_out
            self.component.brightness = telemetry.brightness
            self.component.is_over_current = telemetry.is_over_current
        elif isinstance(self.component, Switch):
            if self.component.closed != telemetry.switch_closed:
                self.animate_switch_state(telemetry.switch_closed)
            self.component.arcing = telemetry.switch_arcing
        elif isinstance(self.component, (Voltmeter, Ammeter)):
            self.component.has_reading = not telemetry.display_value.startswith("--")
            if isinstance(self.component, Voltmeter):
                self.component.loading_error_message = telemetry.loading_error_message
        elif isinstance(self.component, LogicGate):
            self.logic_level = telemetry.logic_level
            if telemetry.logic_level in ("H", "L"):
                self.component._output_level = telemetry.logic_level == "H"
        elif isinstance(self.component, SevenSegDisplay):
            self.component.decoded_hex_char = telemetry.decoded_hex_char
            self.component.active_segments = telemetry.active_segments
            self.component.value_nibble = telemetry.value_nibble
        elif isinstance(self.component, (DFlipFlop, JKFlipFlop)):
            self.component.state_q = telemetry.state_q
        elif isinstance(self.component, Counter4Bit):
            self.component.count = telemetry.count
        elif isinstance(self.component, Fuse):
            was_blown = self.component.is_blown
            self.component.is_blown = telemetry.is_blown
            if not was_blown and telemetry.is_blown and hasattr(self, "trigger_fuse_pop_animation"):
                self.trigger_fuse_pop_animation()
        elif isinstance(self.component, SPDTRelay):
            self.component.energized = telemetry.energized
        elif isinstance(self.component, NE555):
            self.component.output_high = telemetry.output_high

        self.setToolTip(self.get_live_tooltip(is_running=True))
        self.update()

    def set_stress_state(
        self,
        level: str | None,
        reason: str = "",
        utilization_pct: float = 0.0,
        *,
        warnings_enabled: bool | None = None,
    ) -> None:
        """Update component stress/overload badge state (F4)."""
        norm_level = (level or "normal").lower()
        changed = (
            self.stress_level != norm_level
            or self.stress_reason != reason
            or abs(self.stress_utilization_pct - utilization_pct) > 1e-9
            or (warnings_enabled is not None and self.stress_warnings_enabled != bool(warnings_enabled))
        )
        self.stress_level = norm_level
        self.stress_reason = reason
        self.stress_utilization_pct = float(utilization_pct)
        if warnings_enabled is not None:
            self.stress_warnings_enabled = bool(warnings_enabled)
        if isinstance(self.component, Fuse):
            if norm_level == "overload" and "blown" in reason.lower() and not self.component.is_blown:
                self.component.is_blown = True
                self.trigger_fuse_pop_animation()
        if changed:
            self.setToolTip(
                self.get_live_tooltip(
                    is_running=(self.live_display_value is not None or norm_level != "normal")
                )
            )
            self.update()

    def reset_live_telemetry(self) -> None:
        """Clear live telemetry and restore initial component state (when simulation stops)."""
        self.live_display_value = None
        self.logic_level = ""
        self.stress_level = "normal"
        self.stress_reason = ""
        self.stress_utilization_pct = 0.0
        self.component.reset_state()
        if isinstance(self.component, Switch):
            self._lever_angle_deg = (
                self.CLOSED_LEVER_ANGLE_DEG
                if self.component.closed
                else self.OPEN_LEVER_ANGLE_DEG
            )
        if self._fuse_pop_anim is not None:
            self._fuse_pop_anim.stop()
        self._fuse_pop_progress = 0.0
        self.setToolTip(self.get_live_tooltip(is_running=False))
        self.setAcceptHoverEvents(True)
        self._is_hovered: bool = False
        self.update()

    def animate_switch_state(self, closed: bool, *, duration_ms: int = 110) -> None:
        """Smoothly animate the switch lever between open (`26°`) and closed (`0°`)."""
        if not isinstance(self.component, Switch):
            return
        self.component.set_closed(closed)
        target_angle = (
            self.CLOSED_LEVER_ANGLE_DEG if closed else self.OPEN_LEVER_ANGLE_DEG
        )
        if duration_ms <= 0:
            self._lever_angle_deg = target_angle
            self.update()
            return

        if self._lever_anim is None:
            self._lever_anim = QVariantAnimation(self)
            self._lever_anim.valueChanged.connect(self._on_lever_anim_step)

        self._lever_anim.stop()
        self._lever_anim.setDuration(duration_ms)
        self._lever_anim.setStartValue(float(self._lever_angle_deg))
        self._lever_anim.setEndValue(float(target_angle))
        self._lever_anim.start()
        # Also set immediately in case no event loop spins in a synchronous test
        self._lever_angle_deg = target_angle
        self.update()

    def _on_lever_anim_step(self, value: object) -> None:
        if isinstance(value, (int, float)):
            self._lever_angle_deg = float(value)
            self.update()

    def trigger_fuse_pop_animation(self, *, duration_ms: int = 400) -> None:
        """Trigger a ~400 ms expanding starburst flash when a Fuse blows."""
        self._fuse_pop_progress = 1.0
        if self._fuse_pop_anim is None:
            self._fuse_pop_anim = QVariantAnimation(self)
            self._fuse_pop_anim.valueChanged.connect(self._on_fuse_pop_step)
        self._fuse_pop_anim.stop()
        self._fuse_pop_anim.setDuration(max(50, int(duration_ms)))
        self._fuse_pop_anim.setStartValue(1.0)
        self._fuse_pop_anim.setEndValue(0.0)
        self._fuse_pop_anim.start()
        self.update()

    def _on_fuse_pop_step(self, value: object) -> None:
        if isinstance(value, (int, float)):
            self._fuse_pop_progress = max(0.0, min(1.0, float(value)))
            self.update()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self._is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def set_snapped_pos(self, x_px: float, y_px: float) -> None:
        """Snap `(x_px, y_px)` to the 20 px grid and set the item's scene position."""
        sx, sy = snap_to_grid(x_px, y_px, GRID_PITCH_PX)
        self.setPos(sx, sy)

    @property
    def rotation_deg(self) -> int:
        return self._rotation_deg

    @rotation_deg.setter
    def rotation_deg(self, deg: int) -> None:
        self.set_rotation_deg(deg)

    @property
    def flipped_h(self) -> bool:
        return self._flipped_h

    @flipped_h.setter
    def flipped_h(self, flipped: bool) -> None:
        self.set_flipped_h(flipped)

    def rotate_90_cw(self) -> None:
        """Rotate component 90 degrees clockwise around its center `(0, 0)`."""
        self.set_rotation_deg((self._rotation_deg + 90) % 360)

    def rotate_90_ccw(self) -> None:
        """Rotate component 90 degrees counter-clockwise around its center `(0, 0)`."""
        self.set_rotation_deg((self._rotation_deg - 90) % 360)

    def set_rotation_deg(self, rotation_deg: int) -> None:
        """Set explicit rotation angle in `{0, 90, 180, 270}`."""
        target = int(rotation_deg) % 360
        if self._rotation_deg != target:
            self._rotation_deg = target
            self._apply_local_transform()

    def flip_horizontal(self) -> None:
        """Mirror the component horizontally across its local Y axis."""
        self.set_flipped_h(not self._flipped_h)

    def set_flipped_h(self, flipped: bool) -> None:
        """Set explicit horizontal mirror state."""
        target = bool(flipped)
        if self._flipped_h != target:
            self._flipped_h = target
            self._apply_local_transform()

    def _apply_local_transform(self) -> None:
        # Keep QGraphicsObject transform as identity and apply rotation/flip inside
        # paint() only for the symbol geometry so text labels always remain upright!
        self.prepareGeometryChange()
        self.update()

    def terminal_scene_positions(self) -> dict[str, tuple[float, float]]:
        """Return scene pixel coordinates `(x, y)` for every terminal of this item."""
        cx, cy = self.pos().x(), self.pos().y()
        result: dict[str, tuple[float, float]] = {}
        for term in self.metadata.terminals:
            result[term.name] = transform_terminal_position(
                term.grid_pos,
                (cx, cy),
                rotation_deg=self.rotation_deg,
                flipped_h=self.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
        return result

    def terminal_scene_positions_px(self) -> dict[str, PointPx]:
        """Return integer snapped scene pixel coordinates `(x, y)` for every terminal."""
        return {
            tname: normalize_point_px(pos, snap=True)
            for tname, pos in self.terminal_scene_positions().items()
        }

    def to_placed_spec(self) -> PlacedComponentSpec:
        """Build a pure-Python `PlacedComponentSpec` snapshot for netlist extraction."""
        return PlacedComponentSpec(
            component=self.component,
            center_px=(self.pos().x(), self.pos().y()),
            rotation_deg=self.rotation_deg,
            flipped_h=self.flipped_h,
            terminals_px=self.terminal_scene_positions_px(),
        )

    def selection_outline_rect_local(self) -> QRectF:
        """Return the selection/problem outline rectangle in upright local coordinates."""
        w_px, h_px = self.metadata.pixel_size
        if self.rotation_deg in (90, 270):
            return QRectF(
                -float(h_px) / 2.0 - 3.0,
                -float(w_px) / 2.0 - 3.0,
                float(h_px) + 6.0,
                float(w_px) + 6.0,
            )
        return QRectF(
            -float(w_px) / 2.0 - 3.0,
            -float(h_px) / 2.0 - 3.0,
            float(w_px) + 6.0,
            float(h_px) + 6.0,
        )

    def set_label_position(self, pos: str) -> None:
        """Set the label position ('default', 'below', 'above', 'right', 'left')."""
        if self.label_position != pos:
            self.prepareGeometryChange()
            self.label_position = pos
            self.update()

    def _get_label_and_value_rects_for_position(
        self, pos: str
    ) -> tuple[QRectF, QRectF, Qt.AlignmentFlag]:
        w_px, h_px = self.metadata.pixel_size
        is_vert = self.rotation_deg in (90, 270)
        eff_w = float(h_px if is_vert else w_px)
        eff_h = float(w_px if is_vert else h_px)

        if pos == "default":
            if is_vert:
                right_x = float(h_px) / 2.0 + 7.0
                lbl_rect = QRectF(right_x, -18.0, 96.0, 17.0)
                val_rect = QRectF(right_x, 1.0, 110.0, 17.0)
                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                return lbl_rect, val_rect, align

            if self.component.component_type == "ground":
                lbl_x = 0.0
                lbl_y = -float(h_px) / 2.0 - 2.0 if self.rotation_deg == 180 else float(h_px) / 2.0 + 2.0
                lbl_rect = QRectF(lbl_x - 50.0, lbl_y - 9.0, 100.0, 18.0)
                val_rect = QRectF(-60.0, 0.0, 120.0, 18.0)
                return lbl_rect, val_rect, Qt.AlignmentFlag.AlignCenter

            lbl_x = self.metadata.label_offset[0] * GRID_PITCH_PX
            lbl_y = min(
                self.metadata.label_offset[1] * GRID_PITCH_PX,
                -float(h_px) / 2.0 - 14.0,
            )
            val_x = self.metadata.value_offset[0] * GRID_PITCH_PX
            val_y = max(
                self.metadata.value_offset[1] * GRID_PITCH_PX,
                float(h_px) / 2.0 + 14.0,
            )
            lbl_rect = QRectF(lbl_x - 50.0, lbl_y - 9.0, 100.0, 18.0)
            val_rect = QRectF(val_x - 60.0, val_y - 9.0, 120.0, 18.0)
            return lbl_rect, val_rect, Qt.AlignmentFlag.AlignCenter

        if pos == "below":
            y_top = eff_h / 2.0 + 6.0
            lbl_rect = QRectF(-50.0, y_top, 100.0, 16.0)
            val_rect = QRectF(-60.0, y_top + 17.0, 120.0, 16.0)
            return lbl_rect, val_rect, Qt.AlignmentFlag.AlignCenter

        if pos == "above":
            y_base = -eff_h / 2.0 - 6.0
            lbl_rect = QRectF(-50.0, y_base - 33.0, 100.0, 16.0)
            val_rect = QRectF(-60.0, y_base - 16.0, 120.0, 16.0)
            return lbl_rect, val_rect, Qt.AlignmentFlag.AlignCenter

        if pos == "right":
            x_start = eff_w / 2.0 + 8.0
            lbl_rect = QRectF(x_start, -18.0, 96.0, 17.0)
            val_rect = QRectF(x_start, 1.0, 110.0, 17.0)
            return lbl_rect, val_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        if pos == "left":
            x_end = -eff_w / 2.0 - 8.0
            lbl_rect = QRectF(x_end - 96.0, -18.0, 96.0, 17.0)
            val_rect = QRectF(x_end - 110.0, 1.0, 110.0, 17.0)
            return lbl_rect, val_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        return self._get_label_and_value_rects_for_position("default")

    def _upright_label_and_value_rects(
        self,
    ) -> tuple[QRectF, QRectF, Qt.AlignmentFlag]:
        """Compute upright (unrotated, unmirrored) label and value rectangles clear of the selection box."""
        return self._get_label_and_value_rects_for_position(self.label_position)

    def candidate_label_rects_scene(
        self, pos: str | None = None
    ) -> list[QRectF]:
        """Compute the tight visual bounding rectangles of this item's labels in scene coordinates."""
        target_pos = pos if pos is not None else self.label_position
        lbl_rect, val_rect, text_align = self._get_label_and_value_rects_for_position(target_pos)

        font = QFont("Segoe UI", 8)
        font.setBold(True)
        fm = QFontMetrics(font)

        lbl_box = fm.boundingRect(lbl_rect.toRect(), int(text_align), self.component.label)
        actual_lbl = QRectF(lbl_box).adjusted(-2.0, -1.0, 2.0, 1.0).translated(self.pos())
        rects = [actual_lbl]

        if self.component.show_value and self.component.component_type != "ground":
            if isinstance(self.component, (Voltmeter, Ammeter)):
                font.setBold(True)
            else:
                font.setBold(False)
            fm_val = QFontMetrics(font)
            val_text = (
                self.live_display_value
                if self.live_display_value is not None
                else format_component_value_summary(self.component)
            )
            val_box = fm_val.boundingRect(val_rect.toRect(), int(text_align), val_text)
            actual_val = QRectF(val_box).adjusted(-3.0, -1.0, 3.0, 1.0).translated(self.pos())
            rects.append(actual_val)

        return rects

    def scene_body_rect(self) -> QRectF:
        """Return the rotated/flipped body rectangle in scene coordinates (excluding labels)."""
        w_px, h_px = self.metadata.pixel_size
        is_vert = self.rotation_deg in (90, 270)
        bw = float(h_px if is_vert else w_px)
        bh = float(w_px if is_vert else h_px)
        return QRectF(-bw / 2.0, -bh / 2.0, bw, bh).translated(self.pos())

    def shape(self) -> QPainterPath:
        """Return precision hit-test shape matching the rotated/flipped symbol body and terminals."""
        w_px, h_px = self.metadata.pixel_size
        body_rect = QRectF(
            -float(w_px) / 2.0 - 2.0,
            -float(h_px) / 2.0 - 2.0,
            float(w_px) + 4.0,
            float(h_px) + 4.0,
        )

        t = QTransform()
        if self.flipped_h:
            t.scale(-1.0, 1.0)
        if self.rotation_deg:
            t.rotate(float(self.rotation_deg))

        raw_path = QPainterPath()
        raw_path.setFillRule(Qt.FillRule.WindingFill)
        raw_path.addRect(body_rect)
        path = t.map(raw_path)
        path.setFillRule(Qt.FillRule.WindingFill)

        for term in self.metadata.terminals:
            tx, ty = transform_terminal_position(
                term.grid_pos,
                (0.0, 0.0),
                rotation_deg=self.rotation_deg,
                flipped_h=self.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
            term_path = QPainterPath()
            term_path.addEllipse(QPointF(tx, ty), 6.0, 6.0)
            path = path.united(term_path)

        path.setFillRule(Qt.FillRule.WindingFill)
        return path

    def invalidate_geometry_cache(self) -> None:
        """Invalidate cached `boundingRect()` so Qt updates spatial bounds cleanly."""
        self.prepareGeometryChange()
        self._cached_bounds_key = None
        self._cached_bounding_rect = None

    def set_pinout_overlay_enabled(self, enabled: bool) -> None:
        """Toggle physical package & pinout markings without mutating wiring topology."""
        val = bool(enabled)
        if val != self._pinout_overlay_enabled:
            self._pinout_overlay_enabled = val
            self.invalidate_geometry_cache()
            self.update()

    @staticmethod
    def default_pin_labels(comp: Component) -> dict[str, str]:
        """Return short, always-visible terminal pin labels for multi-pin (>=3) components."""
        ctype = comp.component_type
        if ctype == "seven_seg_display":
            return {"D0": "D0", "D1": "D1", "D2": "D2", "D3": "D3"}
        if ctype == "ne555":
            return {
                "vcc": "VCC",
                "gnd": "GND",
                "trig": "TRIG",
                "thr": "THR",
                "dis": "DIS",
                "out": "OUT",
                "reset": "RST",
                "ctrl": "CTRL",
            }
        if ctype == "d_flipflop":
            return {"D": "D", "CLK": "CLK", "RST": "RST", "Q": "Q", "Q_BAR": "Q̄"}
        if ctype == "jk_flipflop":
            return {"J": "J", "K": "K", "CLK": "CLK", "RST": "RST", "Q": "Q", "Q_BAR": "Q̄"}
        if ctype == "counter_4bit":
            return {"CLK": "CLK", "RST": "RST", "Q0": "Q0", "Q1": "Q1", "Q2": "Q2", "Q3": "Q3"}
        if ctype == "spdt_relay":
            return {"coil_pos": "COIL+", "coil_neg": "COIL-", "com": "COM", "no": "NO", "nc": "NC"}
        if ctype == "mcu":
            return {"A0": "A0", "A1": "A1", "D0": "D0", "D1": "D1", "TX": "TX", "RX": "RX"}
        if ctype in ("opamp", "comparator"):
            return {"in_pos": "+", "in_neg": "-", "out": "OUT"}
        if ctype in ("npn_bjt", "pnp_bjt"):
            return {"B": "B", "C": "C", "E": "E"}
        if ctype in ("nmos", "pmos"):
            return {"G": "G", "D": "D", "S": "S"}
        if ctype == "potentiometer":
            return {"a": "A", "w": "W", "b": "B"}
        if ctype == "transformer":
            return {"p1": "P1", "p2": "P2", "s1": "S1", "s2": "S2"}
        if ctype == "oscilloscope":
            return {"CH1": "CH1", "CH2": "CH2", "EXT_TRIG": "EXT_TRIG", "GND": "GND"}
        if ctype == "function_generator":
            return {"out": "OUT", "gnd": "GND"}
        if ctype.startswith("gate_") and ctype != "gate_not":
            return {"a": "A", "b": "B", "out": "Y", "in0": "A", "in1": "B"}
        return {}

    def default_pin_label_rects(self) -> list[tuple[str, str, QRectF]]:
        """Return `[(terminal_name, label_text, tight_upright_rect)]` clear of all 5 px terminal dots."""
        if len(self.metadata.terminals) < 3 and self.component.component_type not in (
            "function_generator",
        ):
            return []
        pin_map = self.default_pin_labels(self.component)
        if not pin_map:
            return []

        pin_font = QFont("Segoe UI", 7)
        pin_font.setBold(True)
        fm = QFontMetrics(pin_font)
        ctype = self.component.component_type

        term_centers: list[tuple[float, float]] = [
            transform_terminal_position(
                t.grid_pos,
                (0.0, 0.0),
                rotation_deg=self.rotation_deg,
                flipped_h=self.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
            for t in self.metadata.terminals
        ]

        results: list[tuple[str, str, QRectF]] = []
        for term in self.metadata.terminals:
            tag = pin_map.get(term.name)
            if not tag:
                continue
            lx0 = term.grid_pos[0] * GRID_PITCH_PX
            ly0 = term.grid_pos[1] * GRID_PITCH_PX
            if ctype == "transformer":
                lx = -24.0 if lx0 < 0 else 24.0
                ly = -33.0 if ly0 < 0 else 33.0
            elif ctype == "seven_seg_display":
                lx = -27.0
                ly = ly0 * 0.9
            elif ctype == "oscilloscope":
                lx = -44.0 if lx0 < 0 else 44.0
                ly = ly0
            elif ctype == "function_generator":
                lx = 32.0
                ly = ly0
            elif ctype == "spdt_relay":
                if lx0 < 0:
                    lx = -22.0
                    ly = -33.0 if ly0 < 0 else 33.0
                else:
                    lx = 28.0
                    ly = ly0 * 0.85
            elif ctype in ("ne555", "d_flipflop", "jk_flipflop", "counter_4bit", "mcu"):
                if abs(lx0) <= 5.0:
                    lx = 0.0
                    ly = -32.0 if ly0 < 0 else 32.0
                else:
                    lx = -23.0 if lx0 < 0 else 23.0
                    ly = ly0 * 0.9
            elif ctype in ("opamp", "comparator"):
                if abs(ly0) < 5.0:
                    lx = 24.0
                    ly = -14.0
                else:
                    lx = -25.0
                    ly = -33.0 if ly0 < 0 else 33.0
            elif ctype in ("npn_bjt", "pnp_bjt", "nmos", "pmos"):
                if lx0 < 0:
                    lx, ly = -26.0, -13.0
                else:
                    lx = 22.0
                    ly = -33.0 if ly0 < 0 else 33.0
            elif ctype == "potentiometer":
                if abs(lx0) < 5.0:
                    lx, ly = 13.0, -27.0
                else:
                    lx = -26.0 if lx0 < 0 else 26.0
                    ly = -14.0
            elif ctype.startswith("gate_"):
                if lx0 < 0:
                    lx = -26.0
                    ly = -32.0 if ly0 < 0 else 32.0
                else:
                    lx, ly = 26.0, -13.0
            else:
                lx = lx0 * 0.55
                ly = ly0 - 13.0 if abs(ly0) < 5.0 else ly0 * 0.72

            ix, iy = transform_terminal_position(
                (lx / GRID_PITCH_PX, ly / GRID_PITCH_PX),
                (0.0, 0.0),
                rotation_deg=self.rotation_deg,
                flipped_h=self.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
            tb = fm.boundingRect(tag)
            w = max(float(tb.width()) + 2.0, 10.0)
            h = max(float(tb.height()) + 2.0, 10.0)
            rect = QRectF(ix - w / 2.0, iy - h / 2.0, w, h)

            # Guarantee zero overlap with any 5.0 px terminal dot circle (+1 px safety margin)
            min_clear = 6.0
            for _ in range(4):
                nudged = False
                for tx, ty in term_centers:
                    cx = max(rect.left(), min(tx, rect.right()))
                    cy = max(rect.top(), min(ty, rect.bottom()))
                    dx = cx - tx
                    dy = cy - ty
                    dist = math.hypot(dx, dy)
                    if dist < min_clear:
                        if dist > 1e-6:
                            ux, uy = dx / dist, dy / dist
                        else:
                            ux = -1.0 if tx > 0 else (1.0 if tx < 0 else 0.0)
                            uy = -1.0 if ty > 0 else (1.0 if ty < 0 else -1.0)
                            norm = math.hypot(ux, uy) or 1.0
                            ux, uy = ux / norm, uy / norm
                        push = min_clear - dist + 0.5
                        rect.translate(ux * push, uy * push)
                        nudged = True
                if not nudged:
                    break

            results.append((term.name, tag, rect))
        return results

    @staticmethod
    def package_and_pinout_labels(comp: Component) -> tuple[str, dict[str, str]]:
        """Return `(package_name, {terminal_name: physical_pin_label})` for `comp`."""
        ctype = comp.component_type
        if ctype == "diode":
            return ("DO-35", {"anode": "Anode", "cathode": "Cathode"})
        if ctype == "zener_diode":
            return ("DO-41", {"anode": "Anode", "cathode": "Cathode"})
        if ctype in ("npn_bjt", "pnp_bjt"):
            return ("TO-92: 1-E, 2-B, 3-C", {"B": "2:B", "C": "3:C", "E": "1:E"})
        if ctype in ("nmos", "pmos"):
            return ("TO-92: 1-S, 2-G, 3-D", {"G": "2:G", "D": "3:D", "S": "1:S"})
        if ctype == "opamp":
            return ("DIP-8", {"in_pos": "3:+IN", "in_neg": "2:-IN", "out": "1:OUT"})
        if ctype == "comparator":
            return ("DIP-8", {"in_pos": "3:+IN", "in_neg": "2:-IN", "out": "1:OUT"})
        if ctype == "ne555":
            return (
                "DIP-8",
                {
                    "vcc": "8:VCC",
                    "gnd": "1:GND",
                    "trig": "2:TRIG",
                    "thr": "6:THR",
                    "dis": "7:DIS",
                    "out": "3:OUT",
                    "reset": "4:RST",
                },
            )
        if ctype == "led":
            return ("LED-5mm", {"anode": "A(+)", "cathode": "K(-)"})
        if ctype == "capacitor":
            if getattr(comp, "polarized", False):
                return ("ELEC-RAD", {"a": "+", "b": "-"})
            return ("CER-0805", {"a": "1", "b": "2"})
        if ctype == "resistor":
            return ("AXIAL-1/4W", {"a": "1", "b": "2"})
        if ctype == "inductor":
            return ("RADIAL-L", {"a": "1", "b": "2"})
        if ctype == "potentiometer":
            return ("TRIM-3P", {"a": "1:CW", "w": "2:W", "b": "3:CCW"})
        if ctype in ("dc_voltage", "ac_voltage", "wav_voltage", "dc_current", "voltmeter", "ammeter"):
            return (ctype.upper(), {"pos": "+", "neg": "-"})
        if ctype == "mcu":
            return ("DIP-MCU", {"A0": "A0", "A1": "A1", "D0": "D0", "D1": "D1", "TX": "TX", "RX": "RX"})
        if ctype == "d_flipflop":
            return ("74HC74", {"D": "D", "CLK": "CLK", "RST": "RST", "Q": "Q", "Q_BAR": "Q̄"})
        if ctype == "jk_flipflop":
            return ("74LS73", {"J": "J", "K": "K", "CLK": "CLK", "RST": "RST", "Q": "Q", "Q_BAR": "Q̄"})
        if ctype == "counter_4bit":
            return ("74HC161", {"CLK": "CLK", "RST": "RST", "Q0": "Q0", "Q1": "Q1", "Q2": "Q2", "Q3": "Q3"})
        if ctype == "seven_seg_display":
            return ("7SEG-HEX", {"D0": "D0", "D1": "D1", "D2": "D2", "D3": "D3"})
        if ctype == "switch":
            return ("SPST", {"a": "1", "b": "2"})
        if ctype == "fuse":
            return ("FUSE-5x20", {"a": "1", "b": "2"})
        if ctype == "transformer":
            return ("XFMR-4P", {"p1": "P1", "p2": "P2", "s1": "S1", "s2": "S2"})
        if ctype == "spdt_relay":
            return ("RELAY-5P", {"coil_pos": "+COIL", "coil_neg": "-COIL", "com": "COM", "no": "NO", "nc": "NC"})
        if ctype == "oscilloscope":
            return ("DSO-1062D", {"CH1": "CH1", "CH2": "CH2", "EXT_TRIG": "TRIG", "GND": "GND"})
        if ctype == "function_generator":
            return ("FG-BENCH", {"out": "OUT", "gnd": "GND"})
        if ctype == "lightbulb":
            return ("LAMP", {"a": "1", "b": "2"})
        if ctype == "ground":
            return ("GND", {"gnd": "0V"})
        if ctype.startswith("gate_"):
            return ("SOIC-14", {"in": "in", "a": "a", "b": "b", "out": "out"})
        return (ctype.upper(), {})

    def boundingRect(self) -> QRectF:
        """Return cached tight bounding rectangle enclosing body, terminals, and labels."""
        val_text = ""
        if self.component.show_value and self.component.component_type != "ground":
            val_text = (
                self.live_display_value
                if self.live_display_value is not None
                else format_component_value_summary(self.component)
            ) or ""
        is_over_bulb = (
            isinstance(self.component, Lightbulb)
            and self.component.is_over_bright
        )
        is_over_led = (
            self.component.component_type == "led"
            and getattr(self.component, "is_over_current", False)
        )
        bulb_lit = (
            isinstance(self.component, Lightbulb)
            and not self.is_ghost
            and getattr(self.component, "brightness", 0.0) > 0.0
        )
        pinout_vis = bool(getattr(self, "_pinout_overlay_enabled", False))
        cache_key: tuple[object, ...] = (
            self.rotation_deg,
            self.flipped_h,
            self.is_ghost,
            self.component.label or "",
            bool(self.component.show_value),
            val_text,
            bool(self.isSelected()),
            bool(self._is_hovered),
            self.diagnostic_severity,
            self.stress_level if self.stress_warnings_enabled else "normal",
            is_over_bulb,
            is_over_led,
            bulb_lit,
            pinout_vis,
        )
        if self._cached_bounds_key == cache_key and self._cached_bounding_rect is not None:
            return self._cached_bounding_rect

        self._bounding_rect_recompute_count += 1
        w_px, h_px = self.metadata.pixel_size
        t = QTransform()
        if self.flipped_h:
            t.scale(-1.0, 1.0)
        if self.rotation_deg:
            t.rotate(float(self.rotation_deg))
        body = QRectF(-w_px / 2.0 - 4.0, -h_px / 2.0 - 4.0, w_px + 8.0, h_px + 8.0)
        rect = t.mapRect(body)

        for term in self.metadata.terminals:
            tx, ty = transform_terminal_position(
                term.grid_pos,
                (0.0, 0.0),
                rotation_deg=self.rotation_deg,
                flipped_h=self.flipped_h,
                grid_px=GRID_PITCH_PX,
            )
            rect = rect.united(QRectF(tx - 6.0, ty - 6.0, 12.0, 12.0))

        if not self.is_ghost:
            lbl_rect, val_rect, text_align = self._upright_label_and_value_rects()
            font = QFont("Segoe UI", 8)
            font.setBold(True)
            fm = QFontMetrics(font)

            lbl_text = self.component.label or ""
            if lbl_text.strip():
                l_box = fm.boundingRect(lbl_rect.toRect(), int(text_align), lbl_text)
                rect = rect.united(QRectF(l_box).adjusted(-2.0, -1.0, 2.0, 1.0))

            if val_text.strip():
                font_v = QFont("Segoe UI", 8)
                font_v.setBold(isinstance(self.component, (Voltmeter, Ammeter)))
                fm_v = QFontMetrics(font_v)
                v_box = fm_v.boundingRect(val_rect.toRect(), int(text_align), val_text)
                rect = rect.united(QRectF(v_box).adjusted(-3.0, -1.0, 3.0, 1.0))

            if self.component.component_type == "led" and (self.isSelected() or self._is_hovered):
                marks = led_marks(self.rotation_deg, self.flipped_h, (0.0, 0.0))
                ax, ay = marks["anode_tag"]
                kx, ky = marks["cathode_tag"]
                rect = rect.united(QRectF(ax - 8.0, ay - 6.0, 16.0, 12.0))
                rect = rect.united(QRectF(kx - 8.0, ky - 6.0, 16.0, 12.0))

            if (
                len(self.metadata.terminals) >= 3
                or self.component.component_type in ("function_generator",)
            ) and not self._pinout_overlay_enabled:
                for _tname, _tag, pin_rect in self.default_pin_label_rects():
                    rect = rect.united(pin_rect.adjusted(-1.0, -1.0, 1.0, 1.0))

            if isinstance(self.component, LogicGate) and (
                self.isSelected() or self._is_hovered
            ):
                gate_info = logic_gate_marks(
                    self.component.component_type,
                    (0.0, 0.0),
                    rotation_deg=self.rotation_deg,
                    flipped_h=self.flipped_h,
                )
                for _pin_name, (tx, ty) in gate_info["pin_tags"]:  # type: ignore[misc]
                    rect = rect.united(QRectF(tx - 15.0, ty - 7.0, 30.0, 14.0))

            if self.diagnostic_severity in ("error", "warning") or is_over_bulb or is_over_led:
                rect = rect.united(QRectF(11.0, -25.0, 14.0, 14.0))

        if bulb_lit:
            rect = rect.united(QRectF(-36.0, -36.0, 72.0, 72.0))

        if pinout_vis:
            rect = rect.adjusted(-16.0, -16.0, 16.0, 16.0)

        self._cached_bounds_key = cache_key
        self._cached_bounding_rect = rect
        return rect

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        """Snap item position to the 20 px grid whenever dragged on the canvas."""
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and isinstance(value, QPointF)
        ):
            sx, sy = snap_to_grid(value.x(), value.y(), GRID_PITCH_PX)
            return QPointF(sx, sy)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            sc = self.scene()
            if sc is not None and hasattr(sc, "on_component_position_changed"):
                sc.on_component_position_changed(self)
        return super().itemChange(change, value)

    def _paint_lightbulb_glow(self, painter: QPainter) -> None:
        if not isinstance(self.component, Lightbulb) or self.is_ghost:
            return
        bulb = self.component
        if bulb.burned_out or bulb.brightness <= 0.005:
            return

        clamped = bulb.clamped_brightness
        radius = 16.0 + 18.0 * clamped
        grad = QRadialGradient(QPointF(0.0, 0.0), radius)

        if bulb.is_over_bright:
            # Over-bright (> 100% rated power): intense white core with fiery orange/red outer ring
            grad.setColorAt(0.0, QColor(255, 255, 240, 235))
            grad.setColorAt(0.45, QColor(255, 210, 60, 190))
            grad.setColorAt(0.8, QColor(255, 90, 30, 120))
            grad.setColorAt(1.0, QColor(255, 40, 0, 0))
        else:
            alpha_core = int(60 + 165 * clamped)
            alpha_mid = int(30 + 110 * clamped)
            grad.setColorAt(0.0, QColor(255, 248, 190, alpha_core))
            grad.setColorAt(0.55, QColor(255, 213, 79, alpha_mid))
            grad.setColorAt(1.0, QColor(255, 180, 0, 0))

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(0.0, 0.0), radius, radius)
        painter.restore()

    def _paint_led_glow(self, painter: QPainter) -> None:
        if not isinstance(self.component, LED) or self.is_ghost:
            return
        led = self.component
        if led.burned_out or led.brightness <= 0.0 or abs(led.current) < 1e-5:
            return

        vis = min(2.0, float(led.brightness))
        radius = 16.0 + 16.0 * min(1.0, vis)
        grad = QRadialGradient(QPointF(0.0, 0.0), radius)

        base_color = QColor(led.color)
        r, g, b = base_color.red(), base_color.green(), base_color.blue()

        if led.is_over_current:
            grad.setColorAt(0.0, QColor(255, 255, 255, 230))
            grad.setColorAt(0.35, QColor(r, g, b, 230))
            grad.setColorAt(0.7, QColor(r, g, b, 160))
            grad.setColorAt(1.0, QColor(r, g, b, 0))
        else:
            alpha_core = int(min(220, 50 + 150 * min(1.0, vis)))
            alpha_mid = int(min(140, 25 + 90 * min(1.0, vis)))
            grad.setColorAt(0.0, QColor(r, g, b, alpha_core))
            grad.setColorAt(0.55, QColor(r, g, b, alpha_mid))
            grad.setColorAt(1.0, QColor(r, g, b, 0))

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(0.0, 0.0), radius, radius)
        painter.restore()

    def _paint_seven_seg_leds(self, painter: QPainter, disp: SevenSegDisplay) -> None:
        """Render the 7 glowing LED bars (a..g) directly on the 7-segment body."""
        seg_lines = {
            "a": ((-7.0, -24.0), (15.0, -24.0)),
            "b": ((16.0, -22.0), (16.0, -2.0)),
            "c": ((16.0, 2.0), (16.0, 22.0)),
            "d": ((-7.0, 24.0), (15.0, 24.0)),
            "e": ((-8.0, 2.0), (-8.0, 22.0)),
            "f": ((-8.0, -22.0), (-8.0, -2.0)),
            "g": ((-7.0, 0.0), (15.0, 0.0)),
        }
        active = set(disp.active_segments)
        painter.save()
        for seg_name, (p1, p2) in seg_lines.items():
            is_on = seg_name in active
            col = QColor("#FF3B30") if is_on else QColor(110, 120, 130, 75)
            pen = QPen(col, 3.6 if is_on else 2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
        painter.restore()

    def _paint_oscilloscope_preview(self, painter: QPainter, osc: Oscilloscope) -> None:
        """Render miniature yellow/cyan waveforms inside the canvas DSO LCD."""
        lcd = QRectF(-14.0, -34.0, 54.0, 68.0)
        painter.save()
        painter.setClipRect(lcd)
        for samples, hex_col, y_center, amp in (
            (osc.mini_preview_ch1, "#FACC15", lcd.top() + lcd.height() * 0.35, 16.0),
            (osc.mini_preview_ch2, "#22D3EE", lcd.top() + lcd.height() * 0.70, 14.0),
        ):
            pts: list[QPointF] = []
            if len(samples) >= 2:
                n = len(samples)
                for idx, val in enumerate(samples):
                    x = lcd.left() + 2.0 + (idx / (n - 1)) * (lcd.width() - 4.0)
                    clamped = max(-1.0, min(1.0, float(val)))
                    y = y_center - clamped * amp
                    pts.append(QPointF(x, y))
            else:
                # Idle flat trace indicator on miniature LCD
                pts.append(QPointF(lcd.left() + 2.0, y_center))
                pts.append(QPointF(lcd.right() - 2.0, y_center))
            painter.setPen(QPen(QColor(hex_col), 1.4))
            painter.drawPolyline(QPolygonF(pts))
        painter.restore()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        try:
            self._paint_impl(painter, option, widget)
        except Exception as exc:
            self._handle_paint_error(painter, exc)

    def _handle_paint_error(self, painter: QPainter, exc: Exception) -> None:
        """Handle paint failure: log once per item and draw red dashed rectangle with '!'."""
        if not getattr(self, "_paint_error_logged", False):
            self._paint_error_logged = True
            try:
                from circuitsim.app.crash_handler import write_crash_log

                write_crash_log(type(exc), exc, exc.__traceback__, count=1)
            except Exception:
                pass

        try:
            painter.save()
            red_col = QColor("#FF5252")
            pen = QPen(red_col, 2.0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 82, 82, 35)))
            rect = self.boundingRect().adjusted(2.0, 2.0, -2.0, -2.0)
            painter.drawRoundedRect(rect, 4.0, 4.0)

            font = QFont("Segoe UI", 12)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(red_col)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "!")
            painter.restore()
        except Exception:
            pass

    def _plate_intersects_scene_wire(self, local_plate: QRectF) -> bool:
        """Return True if any non-preview wire segment in the scene crosses `local_plate`."""
        sc = self.scene()
        if sc is None:
            return False
        scene_rect = QRectF(
            local_plate.x() + self.pos().x(),
            local_plate.y() + self.pos().y(),
            local_plate.width(),
            local_plate.height(),
        ).adjusted(-2.5, -2.5, 2.5, 2.5)
        for item in sc.items(scene_rect):
            pts = getattr(item, "points", None)
            if not pts or len(pts) < 2 or not hasattr(item, "wire_id"):
                continue
            for idx in range(len(pts) - 1):
                if rect_intersects_segment(scene_rect, pts[idx], pts[idx + 1]):
                    return True
        return False

    def _paint_impl(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        _ = (option, widget)
        palette: ThemePalette = get_theme_palette(self.theme)
        w_px, h_px = self.metadata.pixel_size
        body_rect = QRectF(-w_px / 2.0, -h_px / 2.0, float(w_px), float(h_px))

        # 1. Lightbulb & LED glow (centered at 0, 0)
        self._paint_lightbulb_glow(painter)
        self._paint_led_glow(painter)

        # 2. Rotated/flipped symbol and selection/problem frame
        painter.save()
        if self.flipped_h:
            painter.scale(-1.0, 1.0)
        if self.rotation_deg:
            painter.rotate(float(self.rotation_deg))

        if self.is_ghost:
            pen = QPen(QColor(palette.ghost_tint), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(body_rect.adjusted(-3, -3, 3, 3), 4.0, 4.0)
        elif self.diagnostic_severity == "error":
            err_col = QColor("#FF5252")
            fill_col = QColor(255, 82, 82, 32)
            painter.setPen(QPen(err_col, 2.0, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(body_rect.adjusted(-3, -3, 3, 3), 5.0, 5.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        elif self.diagnostic_severity == "warning":
            warn_col = QColor("#FFA726")
            fill_col = QColor(255, 167, 38, 28)
            painter.setPen(QPen(warn_col, 1.8, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(body_rect.adjusted(-3, -3, 3, 3), 5.0, 5.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        elif self.isSelected():
            pen = QPen(QColor(palette.selection_accent), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(body_rect.adjusted(-3, -3, 3, 3), 4.0, 4.0)

        stroke_col = palette.ghost_tint if self.is_ghost else palette.symbol_stroke
        if self.diagnostic_severity == "error" and not self.is_ghost:
            stroke_col = "#FF6B6B"
        elif self.diagnostic_severity == "warning" and not self.is_ghost:
            stroke_col = "#FFA726"

        lever_arg = (
            self._lever_angle_deg
            if isinstance(self.component, Switch)
            else None
        )
        wiper_arg = (
            self.component.alpha
            if isinstance(self.component, Potentiometer)
            else None
        )
        self.symbol_library.paint_symbol(
            painter,
            self.component.component_type,
            state=self.symbol_state,
            stroke_color=stroke_col,
            show_terminals=True,
            terminal_color=palette.terminal_dot,
            lever_angle_deg=lever_arg,
            wiper_alpha=wiper_arg,
            rotation_deg=self.rotation_deg,
            flipped_h=self.flipped_h,
        )
        if isinstance(self.component, SevenSegDisplay) and not self.is_ghost:
            self._paint_seven_seg_leds(painter, self.component)
        if isinstance(self.component, Oscilloscope) and not self.is_ghost:
            self._paint_oscilloscope_preview(painter, self.component)
        if isinstance(self.component, Fuse) and self._fuse_pop_progress > 0.0 and not self.is_ghost:
            p = self._fuse_pop_progress
            alpha = int(round(235 * p))
            radius_outer = 10.0 + (1.0 - p) * 22.0
            radius_inner = 4.0 + (1.0 - p) * 8.0
            painter.save()
            spark_pen = QPen(QColor(255, 214, 0, alpha), 2.0)
            spark_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(spark_pen)
            for k in range(8):
                ang = k * (math.pi / 4.0)
                painter.drawLine(
                    QPointF(math.cos(ang) * radius_inner, math.sin(ang) * radius_inner),
                    QPointF(math.cos(ang) * radius_outer, math.sin(ang) * radius_outer),
                )
            painter.setPen(QPen(QColor(248, 81, 73, int(round(200 * p))), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0.0, 0.0), radius_outer * 0.72, radius_outer * 0.72)
            painter.restore()
        # Pin-0 / positive-branch-current reference marker (Pin 0 -> Pin 1) inside body bounds
        if not self.is_ghost and self.component.component_type in (
            "resistor",
            "capacitor",
            "inductor",
        ):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(palette.symbol_label)))
            painter.drawEllipse(QPointF(-26.0, -7.0), 1.8, 1.8)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if (
            not self.is_ghost
            and (self.isSelected() or self._is_hovered)
            and self.component.component_type
            in (
                "resistor",
                "capacitor",
                "inductor",
                "dc_voltage",
                "ac_voltage",
                "dc_current",
            )
        ):
            arr_col = QColor(palette.selection_accent)
            painter.setPen(QPen(arr_col, 1.4))
            painter.drawLine(QPointF(-10.0, -14.0), QPointF(10.0, -14.0))
            painter.drawLine(QPointF(6.0, -17.0), QPointF(10.0, -14.0))
            painter.drawLine(QPointF(6.0, -11.0), QPointF(10.0, -14.0))
        painter.restore()

        # 3. Upright component label, value/meter readout, and warning badges (never rotated/flipped)
        if not self.is_ghost:
            painter.save()
            lbl_rect, val_rect, text_align = self._upright_label_and_value_rects()

            font = QFont("Segoe UI", 8)
            font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()

            # Background plate for name label so non-wire areas remain legible without cutting wires
            lbl_text = self.component.label or ""
            if lbl_text.strip():
                lbl_box = fm.boundingRect(lbl_rect.toRect(), int(text_align), lbl_text)
                lbl_plate = QRectF(lbl_box).adjusted(-2.0, -1.0, 2.0, 1.0)
                if not self._plate_intersects_scene_wire(lbl_plate):
                    painter.fillRect(lbl_plate, QColor(palette.canvas_bg))
                painter.setPen(QColor(palette.symbol_text))
                painter.drawText(lbl_rect, text_align, lbl_text)

            if (
                self.component.show_value
                and self.component.component_type != "ground"
            ):
                font.setBold(isinstance(self.component, (Voltmeter, Ammeter)))
                painter.setFont(font)
                fm_val = painter.fontMetrics()
                val_text = (
                    self.live_display_value
                    if self.live_display_value is not None
                    else format_component_value_summary(self.component)
                )
                if val_text and val_text.strip():
                    # Background plate for value readout; omitted if a wire passes through so wire stays visible
                    val_box = fm_val.boundingRect(val_rect.toRect(), int(text_align), val_text)
                    val_plate = QRectF(val_box).adjusted(-3.0, -1.0, 3.0, 1.0)
                    if not self._plate_intersects_scene_wire(val_plate):
                        painter.fillRect(val_plate, QColor(palette.canvas_bg))
                    painter.drawText(val_rect, text_align, val_text)

            # Over-bright, over-current, stress/smoke overload, or problem badge in upright coordinates
            is_over_bulb = (
                isinstance(self.component, Lightbulb)
                and self.component.is_over_bright
            )
            is_over_led = (
                isinstance(self.component, LED)
                and getattr(self.component, "is_over_current", False)
            )
            stress_overload = self.stress_warnings_enabled and self.stress_level == "overload"
            stress_warning = self.stress_warnings_enabled and self.stress_level == "warning"
            if (
                self.diagnostic_severity in ("error", "warning")
                or is_over_bulb
                or is_over_led
                or stress_overload
                or stress_warning
            ):
                if self.diagnostic_severity == "error" or stress_overload:
                    badge_col = QColor("#FF5252")
                else:
                    badge_col = QColor("#FFA726")
                if stress_overload:
                    # Subtle smoke wisps rising inside upright body region
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(255, 82, 82, 70)))
                    painter.drawEllipse(QPointF(16.0, -24.0), 4.5, 4.5)
                    painter.setBrush(QBrush(QColor(180, 180, 180, 75)))
                    painter.drawEllipse(QPointF(21.0, -29.0), 4.0, 4.0)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(badge_col))
                badge_center = QPointF(18.0, -18.0)
                painter.drawEllipse(badge_center, 7.0, 7.0)
                painter.setPen(QColor("#0D1117"))
                b_font = QFont("Segoe UI", 8)
                b_font.setBold(True)
                painter.setFont(b_font)
                painter.drawText(
                    QRectF(11.0, -25.0, 14.0, 14.0),
                    Qt.AlignmentFlag.AlignCenter,
                    "!",
                )

            # LED upright 'A' and 'K' tags next to terminals when selected or hovered
            if self.component.component_type == "led" and (self.isSelected() or self._is_hovered):
                marks = led_marks(self.rotation_deg, self.flipped_h, (0.0, 0.0))
                tag_font = QFont("Segoe UI", 7)
                tag_font.setBold(True)
                painter.setFont(tag_font)
                tag_col = QColor(palette.symbol_label)
                painter.setPen(tag_col)
                ax, ay = marks["anode_tag"]
                kx, ky = marks["cathode_tag"]
                painter.drawText(QRectF(ax - 8.0, ay - 6.0, 16.0, 12.0), Qt.AlignmentFlag.AlignCenter, "A")
                painter.drawText(QRectF(kx - 8.0, ky - 6.0, 16.0, 12.0), Qt.AlignmentFlag.AlignCenter, "K")

            # Logic gate upright pin tags ('in'/'out' or 'a'/'b'/'out') when selected or hovered
            if isinstance(self.component, LogicGate) and (
                self.isSelected() or self._is_hovered
            ):
                gate_info = logic_gate_marks(
                    self.component.component_type,
                    (0.0, 0.0),
                    rotation_deg=self.rotation_deg,
                    flipped_h=self.flipped_h,
                )
                tag_font = QFont("Segoe UI", 7)
                tag_font.setBold(True)
                painter.setFont(tag_font)
                painter.setPen(QColor(palette.symbol_label))
                for pin_name, (tx, ty) in gate_info["pin_tags"]:  # type: ignore[misc]
                    painter.drawText(
                        QRectF(tx - 15.0, ty - 7.0, 30.0, 14.0),
                        Qt.AlignmentFlag.AlignCenter,
                        pin_name,
                    )

            if (
                len(self.metadata.terminals) >= 3
                or self.component.component_type in ("function_generator",)
            ) and not self._pinout_overlay_enabled:
                pin_rects = self.default_pin_label_rects()
                if pin_rects:
                    pin_font = QFont("Segoe UI", 7)
                    pin_font.setBold(True)
                    painter.setFont(pin_font)
                    painter.setPen(
                        QColor("#CBD5E1") if self.theme == ThemeMode.DARK else QColor("#334155")
                    )
                    for _tname, tag, pin_rect in pin_rects:
                        painter.drawText(
                            pin_rect.adjusted(-4.0, -2.0, 4.0, 2.0),
                            Qt.AlignmentFlag.AlignCenter,
                            tag,
                        )

            if self._pinout_overlay_enabled:
                pkg_name, pin_map = self.package_and_pinout_labels(self.component)
                pin_font = QFont("Segoe UI", 7)
                pin_font.setBold(True)
                painter.setFont(pin_font)
                painter.setPen(QColor("#58A6FF") if self.theme == ThemeMode.DARK else QColor("#0969DA"))
                for term in self.metadata.terminals:
                    tag = pin_map.get(term.name, term.name)
                    tx, ty = transform_terminal_position(
                        term.grid_pos,
                        (0.0, 0.0),
                        rotation_deg=self.rotation_deg,
                        flipped_h=self.flipped_h,
                        grid_px=GRID_PITCH_PX,
                    )
                    # Position neatly just inside body away from external terminal wire
                    ix = tx * 0.68
                    iy = ty - 9.0 if abs(ty) < 5.0 else ty * 0.72
                    painter.drawText(
                        QRectF(ix - 16.0, iy - 7.0, 32.0, 14.0),
                        Qt.AlignmentFlag.AlignCenter,
                        tag,
                    )
                if pkg_name and self.component.component_type != "ground":
                    pkg_font = QFont("Segoe UI", 6)
                    painter.setFont(pkg_font)
                    painter.drawText(
                        QRectF(-28.0, 16.0, 56.0, 12.0),
                        Qt.AlignmentFlag.AlignCenter,
                        pkg_name,
                    )

            painter.restore()
