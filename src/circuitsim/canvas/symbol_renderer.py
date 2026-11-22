"""Qt symbol renderer with SVG loading, theme recolouring, and IEC procedural fallback."""

from __future__ import annotations

from pathlib import Path
import math
from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtSvg import QSvgRenderer

from circuitsim.app.themes import ThemeMode, get_theme_palette
from circuitsim.canvas.symbol_geometry import (
    DEFAULT_IEC_SYMBOLS_DIR,
    SymbolMetadata,
    load_all_iec_metadata,
    polarity_mark_positions,
    prepare_svg_for_theme,
)


class SymbolLibrary:
    """Manages IEC symbol metadata, SVG renderers per theme/state, and procedural fallback.

    Behaviour:
        1. Loads all `.json` metadata files from `src/circuitsim/resources/symbols/iec/`.
        2. When rendering a component in a given `state` (e.g. 'open'/'closed') and
           `theme` ('dark'/'light'), checks if the corresponding `.svg` file exists.
        3. If the `.svg` file exists, validates and recolours it via
           `prepare_svg_for_theme`. If valid, renders it with `QSvgRenderer`.
        4. If the `.svg` file is missing or invalid, logs a readable warning (once per
           file) and draws the built-in procedural IEC symbol with `QPainter`.
    """

    def __init__(self, symbols_dir: Path = DEFAULT_IEC_SYMBOLS_DIR) -> None:
        self.symbols_dir: Path = symbols_dir
        self._metadata: dict[str, SymbolMetadata] = load_all_iec_metadata(
            symbols_dir, include_extensions=True
        )
        self._svg_cache: dict[tuple[str, str, str], QSvgRenderer | None] = {}
        self.warnings: list[str] = []
        self._seen_warnings: set[str] = set()

    @property
    def metadata_catalog(self) -> dict[str, SymbolMetadata]:
        """Return mapping from component_id to `SymbolMetadata`."""
        return dict(self._metadata)

    def get_metadata(self, component_id: str) -> SymbolMetadata:
        """Return `SymbolMetadata` for `component_id`."""
        if component_id not in self._metadata:
            raise KeyError(f"Unknown symbol component_id {component_id!r}.")
        return self._metadata[component_id]

    def clear_cache(self) -> None:
        """Clear cached SVG renderers so modified SVG files are reloaded from disk."""
        self._svg_cache.clear()

    def _record_warning(self, message: str) -> None:
        if message not in self._seen_warnings:
            self._seen_warnings.add(message)
            self.warnings.append(message)

    def get_svg_renderer(
        self,
        component_id: str,
        *,
        state: str | None = None,
        stroke_color: str = "#E6EDF3",
        flat_color: str | None = None,
    ) -> QSvgRenderer | None:
        """Return a themed `QSvgRenderer` if a valid SVG file exists, or `None` for fallback."""
        meta = self.get_metadata(component_id)
        svg_path = meta.svg_path_for_state(state)
        cache_key = (
            component_id,
            state or "",
            stroke_color.upper(),
            (flat_color or "").upper(),
        )
        if cache_key in self._svg_cache:
            return self._svg_cache[cache_key]

        if not svg_path.exists():
            # Normal when user has not created a custom SVG yet; use procedural IEC fallback
            self._svg_cache[cache_key] = None
            return None

        prepared = prepare_svg_for_theme(
            svg_path, meta, stroke_color=stroke_color, flat_color=flat_color
        )
        if not prepared.is_valid or prepared.svg_bytes is None:
            if prepared.warning:
                self._record_warning(prepared.warning)
            self._svg_cache[cache_key] = None
            return None

        renderer = QSvgRenderer(QByteArray(prepared.svg_bytes))
        if not renderer.isValid():
            self._record_warning(
                f"Qt QSvgRenderer could not load {svg_path.name}; using procedural IEC fallback."
            )
            self._svg_cache[cache_key] = None
            return None

        self._svg_cache[cache_key] = renderer
        return renderer

    def paint_symbol(
        self,
        painter: QPainter,
        component_id: str,
        *,
        state: str | None = None,
        stroke_color: str = "#E6EDF3",
        polarity_color: str | None = None,
        show_terminals: bool = True,
        terminal_color: str = "#58A6FF",
        lever_angle_deg: float | None = None,
        wiper_alpha: float | None = None,
        rotation_deg: int = 0,
        flipped_h: bool = False,
    ) -> bool:
        """Paint the symbol centered at `(0, 0)` in `painter` coordinates.

        Returns:
            True if rendered from an SVG file on disk, False if drawn by the
            built-in procedural IEC fallback.
        """
        meta = self.get_metadata(component_id)
        w_px, h_px = meta.pixel_size
        target_rect = QRectF(-w_px / 2.0, -h_px / 2.0, float(w_px), float(h_px))

        svg_renderer = self.get_svg_renderer(
            component_id,
            state=state,
            stroke_color=stroke_color,
            flat_color=polarity_color if component_id == "led" else None,
        )
        use_procedural_gate = component_id.startswith("gate_") and (
            (rotation_deg % 360) != 0 or flipped_h
        )
        used_svg = False
        if (
            svg_renderer is not None
            and lever_angle_deg is None
            and wiper_alpha is None
            and state != "burned_out"
            and not use_procedural_gate
        ):
            svg_renderer.render(painter, target_rect)
            used_svg = True
        else:
            self._paint_procedural_iec(
                painter,
                component_id,
                state=state,
                stroke_color=stroke_color,
                polarity_color=polarity_color,
                lever_angle_deg=lever_angle_deg,
                wiper_alpha=wiper_alpha,
                rotation_deg=rotation_deg,
                flipped_h=flipped_h,
            )

        if show_terminals:
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            osc_term_colors = {
                "CH1": "#FACC15",
                "CH2": "#22D3EE",
                "EXT_TRIG": "#94A3B8",
                "GND": "#475569",
            }
            dot_r = 3.5 if component_id == "oscilloscope" else 3.0
            for term in meta.terminals:
                dot_col = (
                    osc_term_colors.get(term.name, terminal_color)
                    if component_id == "oscilloscope"
                    else terminal_color
                )
                painter.setBrush(QColor(dot_col))
                tx, ty = term.local_px
                painter.drawEllipse(QPointF(tx, ty), dot_r, dot_r)
            painter.restore()

        return used_svg

    def create_palette_icon(
        self,
        component_id: str,
        *,
        theme: ThemeMode | str = ThemeMode.DARK,
        size: tuple[int, int] = (64, 36),
    ) -> QIcon:
        """Render a crisp `QIcon` of the symbol for the Component Palette."""
        palette = get_theme_palette(theme)
        pixmap = QPixmap(size[0], size[1])
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(size[0] / 2.0, size[1] / 2.0)

        meta = self.get_metadata(component_id)
        w_px, h_px = meta.pixel_size
        scale = min((size[0] - 8.0) / w_px, (size[1] - 6.0) / h_px)
        painter.scale(scale, scale)

        self.paint_symbol(
            painter,
            component_id,
            state="open" if component_id == "switch" else None,
            stroke_color=palette.symbol_stroke,
            show_terminals=True,
            terminal_color=palette.terminal_dot,
        )
        painter.end()
        return QIcon(pixmap)

    def _paint_procedural_iec(
        self,
        painter: QPainter,
        component_id: str,
        *,
        state: str | None = None,
        stroke_color: str = "#E6EDF3",
        polarity_color: str | None = None,
        lever_angle_deg: float | None = None,
        wiper_alpha: float | None = None,
        rotation_deg: int = 0,
        flipped_h: bool = False,
    ) -> None:
        """Built-in procedural IEC 60617 symbol drawing (stroke-width 2, no fill)."""
        painter.save()
        pen = QPen(QColor(stroke_color), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if component_id == "resistor":
            # IEC rectangular resistor: leads (-40.. -18) and (18..40), box 36x14
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-18.0, 0.0))
            painter.drawRect(QRectF(-18.0, -7.0, 36.0, 14.0))
            painter.drawLine(QPointF(18.0, 0.0), QPointF(40.0, 0.0))

        elif component_id == "potentiometer":
            # IEC resistor body with sliding wiper arrow on top lead (0, -40)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-18.0, 0.0))
            painter.drawRect(QRectF(-18.0, -7.0, 36.0, 14.0))
            painter.drawLine(QPointF(18.0, 0.0), QPointF(40.0, 0.0))
            alpha_clamped = 0.5 if wiper_alpha is None else max(0.0, min(1.0, float(wiper_alpha)))
            wx = -14.0 + 28.0 * (1.0 - alpha_clamped)
            # Note: at alpha=0 (wiper near B on right), wx=+14; at alpha=1 (near A on left), wx=-14
            painter.drawLine(QPointF(0.0, -40.0), QPointF(0.0, -18.0))
            painter.drawLine(QPointF(0.0, -18.0), QPointF(wx, -18.0))
            painter.drawLine(QPointF(wx, -18.0), QPointF(wx, -7.0))
            painter.drawLine(QPointF(wx - 4.0, -12.0), QPointF(wx, -7.0))
            painter.drawLine(QPointF(wx + 4.0, -12.0), QPointF(wx, -7.0))

        elif component_id == "capacitor":
            # Parallel plates at x = -4 and x = +4
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-4.0, 0.0))
            painter.drawLine(QPointF(-4.0, -12.0), QPointF(-4.0, 12.0))
            painter.drawLine(QPointF(4.0, -12.0), QPointF(4.0, 12.0))
            painter.drawLine(QPointF(4.0, 0.0), QPointF(40.0, 0.0))

        elif component_id == "inductor":
            # 4 semi-circular humps from x = -20 to x = +20 (each width 10, radius 5)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-20.0, 0.0))
            path = QPainterPath(QPointF(-20.0, 0.0))
            for i in range(4):
                x0 = -20.0 + i * 10.0
                path.arcTo(QRectF(x0, -6.0, 10.0, 12.0), 180.0, -180.0)
            painter.drawPath(path)
            painter.drawLine(QPointF(20.0, 0.0), QPointF(40.0, 0.0))

        elif component_id == "dc_voltage":
            # Circle (r = 15) and leads
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))

            marks = polarity_mark_positions(component_id, rotation_deg=rotation_deg, flipped_h=flipped_h)
            px, py = marks["plus"]
            mx, my = marks["minus"]

            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)

            pen_pol = QPen(QColor(polarity_color or stroke_color), 2.0)
            pen_pol.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_pol)

            # '+' sign centered at (px, py)
            painter.drawLine(QPointF(px - 3.0, py), QPointF(px + 3.0, py))
            painter.drawLine(QPointF(px, py - 3.0), QPointF(px, py + 3.0))
            # '-' sign centered at (mx, my)
            painter.drawLine(QPointF(mx - 3.0, my), QPointF(mx + 3.0, my))
            painter.restore()

        elif component_id == "ac_voltage":
            # Circle (r = 15) with sine wave inside
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))
            sine = QPainterPath(QPointF(-9.0, 0.0))
            sine.cubicTo(QPointF(-4.5, -10.0), QPointF(0.0, -10.0), QPointF(0.0, 0.0))
            sine.cubicTo(QPointF(0.0, 10.0), QPointF(4.5, 10.0), QPointF(9.0, 0.0))
            painter.drawPath(sine)

        elif component_id == "wav_voltage":
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))
            for bx, bh in ((-7.0, 4.0), (-3.5, 9.0), (0.0, 6.0), (3.5, 10.0), (7.0, 5.0)):
                painter.drawLine(QPointF(bx, -bh), QPointF(bx, bh))

        elif component_id == "dc_current":
            # Circle (r = 15) with arrow pointing toward 'pos' terminal
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))

            marks = polarity_mark_positions(component_id, rotation_deg=rotation_deg, flipped_h=flipped_h)
            tx, ty = marks["arrow_tip"]
            ax, ay = marks["arrow_tail"]

            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)

            pen_pol = QPen(QColor(polarity_color or stroke_color), 2.0)
            pen_pol.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_pol)

            painter.drawLine(QPointF(ax, ay), QPointF(tx, ty))
            dx = tx - ax
            dy = ty - ay
            length = math.hypot(dx, dy)
            if length > 0.0:
                ux = dx / length
                uy = dy / length
                barb1 = QPointF(tx - 6.0 * ux - 4.5 * uy, ty - 6.0 * uy + 4.5 * ux)
                barb2 = QPointF(tx - 6.0 * ux + 4.5 * uy, ty - 6.0 * uy - 4.5 * ux)
                painter.drawLine(QPointF(tx, ty), barb1)
                painter.drawLine(QPointF(tx, ty), barb2)
            painter.restore()

        elif component_id == "ground":
            # Terminal at (0, -20), vertical stem to (0, 0), three horizontal earth bars
            painter.drawLine(QPointF(0.0, -20.0), QPointF(0.0, 0.0))
            painter.drawLine(QPointF(-12.0, 0.0), QPointF(12.0, 0.0))
            painter.drawLine(QPointF(-8.0, 5.0), QPointF(8.0, 5.0))
            painter.drawLine(QPointF(-4.0, 10.0), QPointF(4.0, 10.0))

        elif component_id == "switch":
            # Contacts at (-16, 0) and (16, 0)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-16.0, 0.0))
            painter.drawLine(QPointF(16.0, 0.0), QPointF(40.0, 0.0))
            painter.drawEllipse(QPointF(-16.0, 0.0), 2.5, 2.5)
            painter.drawEllipse(QPointF(16.0, 0.0), 2.5, 2.5)
            angle = (
                lever_angle_deg
                if lever_angle_deg is not None
                else (0.0 if state == "closed" else 26.0)
            )
            rad = math.radians(max(0.0, min(45.0, float(angle))))
            tip_x = -13.5 + 27.0 * math.cos(rad)
            tip_y = -27.0 * math.sin(rad)
            painter.drawLine(QPointF(-13.5, 0.0), QPointF(tip_x, tip_y))

        elif component_id == "lightbulb":
            # IEC lamp symbol: circle (r = 15) with diagonal cross ('X') inside
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            if state == "burned_out":
                painter.save()
                painter.setBrush(QBrush(QColor(45, 45, 45)))
                painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
                painter.restore()
            else:
                painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))
            d = 15.0 * 0.70710678
            painter.drawLine(QPointF(-d, -d), QPointF(d, d))
            painter.drawLine(QPointF(-d, d), QPointF(d, -d))

        elif component_id == "led":
            # IEC LED symbol with cathode flat
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-14.0, 0.0))

            if state == "burned_out":
                painter.save()
                dark_brush = QBrush(QColor(45, 45, 45))
                painter.setBrush(dark_brush)
                case_fill = QPainterPath()
                case_fill.moveTo(10.0, 10.4)
                case_fill.arcTo(QRectF(-21.0, -17.0, 34.0, 34.0), 320.0, 260.0)
                case_fill.closeSubpath()
                painter.drawPath(case_fill)
                painter.restore()

            poly = QPolygonF([QPointF(-14.0, -9.0), QPointF(-14.0, 9.0), QPointF(4.0, 0.0)])
            painter.drawPolygon(poly)
            painter.drawLine(QPointF(4.0, -11.0), QPointF(4.0, 11.0))
            painter.drawLine(QPointF(4.0, 0.0), QPointF(40.0, 0.0))

            case_path = QPainterPath()
            case_path.moveTo(10.0, 10.4)
            case_path.arcTo(QRectF(-21.0, -17.0, 34.0, 34.0), 320.0, 260.0)
            painter.drawPath(case_path)

            painter.save()
            if polarity_color:
                painter.setPen(QPen(QColor(polarity_color), 2.0))
            painter.drawLine(QPointF(10.0, -10.4), QPointF(10.0, 10.4))
            painter.restore()

            if state == "burned_out":
                # Dark body with cross across the LED body
                d = 10.0
                painter.drawLine(QPointF(-d, -d), QPointF(d, d))
                painter.drawLine(QPointF(-d, d), QPointF(d, -d))
            else:
                painter.drawLine(QPointF(1.0, -9.0), QPointF(10.0, -16.0))
                painter.drawLine(QPointF(5.0, -16.0), QPointF(10.0, -16.0))
                painter.drawLine(QPointF(10.0, -11.0), QPointF(10.0, -16.0))
                painter.drawLine(QPointF(7.0, -7.0), QPointF(16.0, -14.0))
                painter.drawLine(QPointF(11.0, -14.0), QPointF(16.0, -14.0))
                painter.drawLine(QPointF(16.0, -9.0), QPointF(16.0, -14.0))

        elif component_id == "diode":
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-10.0, 0.0))
            poly = QPolygonF([QPointF(-10.0, -10.0), QPointF(-10.0, 10.0), QPointF(8.0, 0.0)])
            painter.drawPolygon(poly)
            painter.drawLine(QPointF(8.0, -11.0), QPointF(8.0, 11.0))
            painter.drawLine(QPointF(8.0, 0.0), QPointF(40.0, 0.0))

        elif component_id == "zener_diode":
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-10.0, 0.0))
            poly = QPolygonF([QPointF(-10.0, -10.0), QPointF(-10.0, 10.0), QPointF(8.0, 0.0)])
            painter.drawPolygon(poly)
            z_bar = QPainterPath(QPointF(3.0, -13.0))
            z_bar.lineTo(QPointF(8.0, -10.0))
            z_bar.lineTo(QPointF(8.0, 10.0))
            z_bar.lineTo(QPointF(13.0, 13.0))
            painter.drawPath(z_bar)
            painter.drawLine(QPointF(8.0, 0.0), QPointF(40.0, 0.0))

        elif component_id in ("npn_bjt", "pnp_bjt"):
            painter.drawEllipse(QPointF(2.0, 0.0), 22.0, 22.0)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-8.0, 0.0))
            painter.drawLine(QPointF(-8.0, -14.0), QPointF(-8.0, 14.0))
            # Collector leg to (40, -40)
            painter.drawLine(QPointF(-8.0, -6.0), QPointF(16.0, -22.0))
            painter.drawLine(QPointF(16.0, -22.0), QPointF(16.0, -40.0))
            painter.drawLine(QPointF(16.0, -40.0), QPointF(40.0, -40.0))
            # Emitter leg to (40, 40)
            painter.drawLine(QPointF(-8.0, 6.0), QPointF(16.0, 22.0))
            painter.drawLine(QPointF(16.0, 22.0), QPointF(16.0, 40.0))
            painter.drawLine(QPointF(16.0, 40.0), QPointF(40.0, 40.0))
            # Arrow on emitter leg
            dx, dy = 24.0, 16.0
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            px, py = -uy, ux
            if component_id == "npn_bjt":
                tip = QPointF(14.0, 20.7)
                ax, ay = -ux, -uy
            else:
                tip = QPointF(-4.0, 8.7)
                ax, ay = ux, uy
            painter.drawLine(tip, QPointF(tip.x() + 7.0 * ax + 4.5 * px, tip.y() + 7.0 * ay + 4.5 * py))
            painter.drawLine(tip, QPointF(tip.x() + 7.0 * ax - 4.5 * px, tip.y() + 7.0 * ay - 4.5 * py))

        elif component_id in ("nmos", "pmos"):
            painter.drawEllipse(QPointF(2.0, 0.0), 22.0, 22.0)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-10.0, 0.0))
            painter.drawLine(QPointF(-10.0, -14.0), QPointF(-10.0, 14.0))
            # Broken enhancement channel segments
            for cy in (-12.0, 0.0, 12.0):
                painter.drawLine(QPointF(-3.0, cy - 3.5), QPointF(-3.0, cy + 3.5))
            # Drain lead -> (40, -40)
            painter.drawLine(QPointF(-3.0, -12.0), QPointF(16.0, -12.0))
            painter.drawLine(QPointF(16.0, -12.0), QPointF(16.0, -40.0))
            painter.drawLine(QPointF(16.0, -40.0), QPointF(40.0, -40.0))
            # Source lead -> (40, 40)
            painter.drawLine(QPointF(-3.0, 12.0), QPointF(16.0, 12.0))
            painter.drawLine(QPointF(16.0, 12.0), QPointF(16.0, 40.0))
            painter.drawLine(QPointF(16.0, 40.0), QPointF(40.0, 40.0))
            # Substrate tie arrow (points left into channel for NMOS, right for PMOS)
            painter.drawLine(QPointF(-3.0, 0.0), QPointF(16.0, 0.0))
            painter.drawLine(QPointF(16.0, 0.0), QPointF(16.0, 12.0))
            if component_id == "nmos":
                painter.drawLine(QPointF(-1.0, 0.0), QPointF(5.0, -3.5))
                painter.drawLine(QPointF(-1.0, 0.0), QPointF(5.0, 3.5))
            else:
                painter.drawLine(QPointF(10.0, 0.0), QPointF(4.0, -3.5))
                painter.drawLine(QPointF(10.0, 0.0), QPointF(4.0, 3.5))

        elif component_id in ("opamp", "comparator"):
            painter.drawLine(QPointF(-40.0, -20.0), QPointF(-20.0, -20.0))
            painter.drawLine(QPointF(-40.0, 20.0), QPointF(-20.0, 20.0))
            painter.drawLine(QPointF(22.0, 0.0), QPointF(40.0, 0.0))
            tri = QPolygonF([QPointF(-20.0, -32.0), QPointF(-20.0, 32.0), QPointF(22.0, 0.0)])
            painter.drawPolygon(tri)
            # '+' on in_pos (-20, -20) and '-' on in_neg (-20, 20)
            painter.drawLine(QPointF(-15.0, -14.0), QPointF(-9.0, -14.0))
            painter.drawLine(QPointF(-12.0, -17.0), QPointF(-12.0, -11.0))
            painter.drawLine(QPointF(-15.0, 14.0), QPointF(-9.0, 14.0))
            if component_id == "comparator":
                # Step glyph inside comparator triangle
                painter.drawLine(QPointF(-6.0, 5.0), QPointF(0.0, 5.0))
                painter.drawLine(QPointF(0.0, 5.0), QPointF(0.0, -5.0))
                painter.drawLine(QPointF(0.0, -5.0), QPointF(6.0, -5.0))

        elif component_id in ("voltmeter", "ammeter"):
            # Circle (r = 15) and leads
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-15.0, 0.0))
            painter.drawEllipse(QPointF(0.0, 0.0), 15.0, 15.0)
            painter.drawLine(QPointF(15.0, 0.0), QPointF(40.0, 0.0))

            # The letter ('V' or 'A') and '+' polarity mark drawn upright
            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)

            if component_id == "voltmeter":
                painter.drawLine(QPointF(-5.0, -6.0), QPointF(0.0, 6.0))
                painter.drawLine(QPointF(0.0, 6.0), QPointF(5.0, -6.0))
            else:
                painter.drawLine(QPointF(-5.0, 6.0), QPointF(0.0, -6.0))
                painter.drawLine(QPointF(0.0, -6.0), QPointF(5.0, 6.0))
                painter.drawLine(QPointF(-3.2, 1.5), QPointF(3.2, 1.5))

            marks = polarity_mark_positions(component_id, rotation_deg=rotation_deg, flipped_h=flipped_h)
            if "plus" in marks:
                px, py = marks["plus"]
                pen_pol = QPen(QColor(polarity_color or stroke_color), 1.8)
                pen_pol.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen_pol)
                painter.drawLine(QPointF(px - 2.5, py), QPointF(px + 2.5, py))
                painter.drawLine(QPointF(px, py - 2.5), QPointF(px, py + 2.5))

            painter.restore()

        elif component_id.startswith("gate_"):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if component_id == "gate_not":
                body_cx = -2.0
                painter.drawLine(QPointF(-40.0, 0.0), QPointF(-16.0, 0.0))
                painter.drawRect(QRectF(-16.0, -14.0, 28.0, 28.0))
                painter.drawEllipse(QPointF(16.0, 0.0), 4.0, 4.0)
                painter.drawLine(QPointF(20.0, 0.0), QPointF(40.0, 0.0))
            else:
                inverting = component_id in {"gate_nand", "gate_nor", "gate_xnor"}
                body_cx = -2.0 if inverting else 0.0
                box_w = 32.0 if inverting else 36.0
                painter.drawLine(QPointF(-40.0, -20.0), QPointF(-18.0, -20.0))
                painter.drawLine(QPointF(-40.0, 20.0), QPointF(-18.0, 20.0))
                painter.drawRect(QRectF(-18.0, -28.0, box_w, 56.0))
                if inverting:
                    painter.drawEllipse(QPointF(18.0, 0.0), 4.0, 4.0)
                    painter.drawLine(QPointF(22.0, 0.0), QPointF(40.0, 0.0))
                else:
                    painter.drawLine(QPointF(18.0, 0.0), QPointF(40.0, 0.0))

            painter.save()
            painter.translate(body_cx, 0.0)
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)

            if component_id == "gate_not":
                one = QPainterPath(QPointF(-3.0, -3.5))
                one.lineTo(QPointF(0.0, -6.0))
                one.lineTo(QPointF(0.0, 6.0))
                painter.drawPath(one)
                painter.drawLine(QPointF(-3.0, 6.0), QPointF(3.0, 6.0))
            elif component_id in {"gate_and", "gate_nand"}:
                amp = QPainterPath(QPointF(5.0, 6.5))
                amp.lineTo(QPointF(-2.5, -1.5))
                amp.cubicTo(
                    QPointF(-4.5, -3.5), QPointF(-3.5, -6.5), QPointF(-0.5, -6.5)
                )
                amp.cubicTo(QPointF(2.0, -6.5), QPointF(3.0, -3.5), QPointF(0.5, -1.5))
                amp.lineTo(QPointF(-3.5, 1.5))
                amp.cubicTo(
                    QPointF(-5.5, 3.0), QPointF(-5.0, 6.5), QPointF(-1.0, 6.5)
                )
                amp.cubicTo(QPointF(2.0, 6.5), QPointF(4.0, 4.5), QPointF(5.0, 1.5))
                painter.drawPath(amp)
            elif component_id in {"gate_or", "gate_nor"}:
                geq = QPainterPath(QPointF(-7.5, -6.0))
                geq.lineTo(QPointF(-1.5, -2.25))
                geq.lineTo(QPointF(-7.5, 1.5))
                painter.drawPath(geq)
                painter.drawLine(QPointF(-7.5, 5.5), QPointF(-1.5, 5.5))
                one = QPainterPath(QPointF(2.0, -3.5))
                one.lineTo(QPointF(4.5, -6.0))
                one.lineTo(QPointF(4.5, 6.0))
                painter.drawPath(one)
                painter.drawLine(QPointF(2.0, 6.0), QPointF(7.0, 6.0))
            else:
                painter.drawLine(QPointF(-7.5, -2.5), QPointF(-1.5, -2.5))
                painter.drawLine(QPointF(-7.5, 2.5), QPointF(-1.5, 2.5))
                one = QPainterPath(QPointF(2.0, -3.5))
                one.lineTo(QPointF(4.5, -6.0))
                one.lineTo(QPointF(4.5, 6.0))
                painter.drawPath(one)
                painter.drawLine(QPointF(2.0, 6.0), QPointF(7.0, 6.0))
            painter.restore()

        elif component_id == "mcu":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for py in (-20.0, 0.0, 20.0):
                painter.drawLine(QPointF(-60.0, py), QPointF(-36.0, py))
                painter.drawLine(QPointF(36.0, py), QPointF(60.0, py))
            painter.drawRect(QRectF(-36.0, -34.0, 72.0, 68.0))
            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(QRectF(-34.0, -14.0, 68.0, 28.0), Qt.AlignmentFlag.AlignCenter, "MCU")
            painter.restore()

        elif component_id == "ne555":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(-40.0, -44.0, 80.0, 88.0))
            # Left leads: trig (-40), thr (0), reset (+40)
            for py in (-40.0, 0.0, 40.0):
                painter.drawLine(QPointF(-60.0, py), QPointF(-40.0, py))
            # Right leads: out (-20), dis (+20)
            for py in (-20.0, 20.0):
                painter.drawLine(QPointF(40.0, py), QPointF(60.0, py))
            # Top/bottom leads: vcc (0, -60), gnd (0, 60)
            painter.drawLine(QPointF(0.0, -60.0), QPointF(0.0, -44.0))
            painter.drawLine(QPointF(0.0, 44.0), QPointF(0.0, 60.0))
            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(QRectF(-36.0, -14.0, 72.0, 28.0), Qt.AlignmentFlag.AlignCenter, "555")
            painter.restore()

        elif component_id == "transformer":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Primary leads & winding arcs on left (x=-40..-10)
            painter.drawLine(QPointF(-40.0, -20.0), QPointF(-16.0, -20.0))
            painter.drawLine(QPointF(-40.0, 20.0), QPointF(-16.0, 20.0))
            for wy in (-20.0, -7.0, 6.0):
                painter.drawArc(QRectF(-22.0, wy, 12.0, 13.0), -90 * 16, 180 * 16)
            # Core bars
            painter.drawLine(QPointF(-3.0, -24.0), QPointF(-3.0, 24.0))
            painter.drawLine(QPointF(3.0, -24.0), QPointF(3.0, 24.0))
            # Secondary leads & winding arcs on right (x=+10..+40)
            painter.drawLine(QPointF(16.0, -20.0), QPointF(40.0, -20.0))
            painter.drawLine(QPointF(16.0, 20.0), QPointF(40.0, 20.0))
            for wy in (-20.0, -7.0, 6.0):
                painter.drawArc(QRectF(10.0, wy, 12.0, 13.0), 90 * 16, 180 * 16)
            # Phase polarity dots near P1 and S1 terminals
            painter.save()
            painter.setBrush(QColor(stroke_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(-24.0, -14.0), 2.3, 2.3)
            painter.drawEllipse(QPointF(24.0, -14.0), 2.3, 2.3)
            painter.restore()

        elif component_id == "spdt_relay":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(-44.0, -44.0, 88.0, 88.0))
            # Coil left leads + coil box
            painter.drawLine(QPointF(-60.0, -20.0), QPointF(-34.0, -20.0))
            painter.drawLine(QPointF(-60.0, 20.0), QPointF(-34.0, 20.0))
            painter.drawRect(QRectF(-34.0, -20.0, 18.0, 40.0))
            painter.drawLine(QPointF(-34.0, 12.0), QPointF(-16.0, -12.0))
            # Switch right leads: NO (60,-40), COM (60,0), NC (60,40)
            painter.drawLine(QPointF(26.0, -40.0), QPointF(60.0, -40.0))
            painter.drawLine(QPointF(26.0, 40.0), QPointF(60.0, 40.0))
            painter.drawLine(QPointF(6.0, 0.0), QPointF(60.0, 0.0))
            # Dashed coupling + contact arm
            target_y = -40.0 if state == "closed" else 40.0
            painter.drawLine(QPointF(6.0, 0.0), QPointF(26.0, target_y))

        elif component_id == "fuse":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-18.0, 0.0))
            painter.drawLine(QPointF(18.0, 0.0), QPointF(40.0, 0.0))
            if state == "blown":
                painter.save()
                painter.setBrush(QBrush(QColor(248, 81, 73, 38)))
                painter.setPen(QPen(QColor("#F85149"), 1.8))
                painter.drawRect(QRectF(-18.0, -8.0, 36.0, 16.0))
                # Two separated broken filament stubs with an open gap in the middle
                painter.drawLine(QPointF(-18.0, 0.0), QPointF(-8.0, -4.5))
                painter.drawLine(QPointF(8.0, 4.5), QPointF(18.0, 0.0))
                painter.setBrush(QBrush(QColor("#F85149")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(-8.0, -4.5), 1.6, 1.6)
                painter.drawEllipse(QPointF(8.0, 4.5), 1.6, 1.6)
                painter.restore()
            else:
                painter.drawRect(QRectF(-18.0, -8.0, 36.0, 16.0))
                painter.drawLine(QPointF(-18.0, 0.0), QPointF(18.0, 0.0))

        elif component_id in ("d_flipflop", "jk_flipflop", "counter_4bit"):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if component_id == "d_flipflop":
                painter.drawRect(QRectF(-40.0, -52.0, 80.0, 104.0))
                for py in (-40.0, 0.0, 40.0):
                    painter.drawLine(QPointF(-60.0, py), QPointF(-40.0, py))
                for py in (-20.0, 20.0):
                    painter.drawLine(QPointF(40.0, py), QPointF(60.0, py))
                lbl = "D-FF"
            elif component_id == "jk_flipflop":
                painter.drawRect(QRectF(-40.0, -72.0, 80.0, 144.0))
                for py in (-60.0, -20.0, 20.0, 60.0):
                    painter.drawLine(QPointF(-60.0, py), QPointF(-40.0, py))
                for py in (-40.0, 40.0):
                    painter.drawLine(QPointF(40.0, py), QPointF(60.0, py))
                lbl = "JK-FF"
            else:
                painter.drawRect(QRectF(-40.0, -72.0, 80.0, 144.0))
                for py in (-20.0, 20.0):
                    painter.drawLine(QPointF(-60.0, py), QPointF(-40.0, py))
                for py in (-60.0, -20.0, 20.0, 60.0):
                    painter.drawLine(QPointF(40.0, py), QPointF(60.0, py))
                lbl = "CNT4"
            painter.save()
            if rotation_deg:
                painter.rotate(-float(rotation_deg))
            if flipped_h:
                painter.scale(-1.0, 1.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(QRectF(-34.0, -14.0, 68.0, 28.0), Qt.AlignmentFlag.AlignCenter, lbl)
            painter.restore()

        elif component_id == "seven_seg_display":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(-36.0, -70.0, 76.0, 140.0))
            for py in (-60.0, -20.0, 20.0, 60.0):
                painter.drawLine(QPointF(-60.0, py), QPointF(-36.0, py))
            # Unpowered bezel outline; live glowing LED bars painted by ComponentItem
            painter.drawRect(QRectF(-18.0, -34.0, 44.0, 68.0))

        elif component_id == "logic_probe":
            # Probe tip lead (-40 -> -22) + rounded pill indicator bezel
            painter.drawLine(QPointF(-40.0, 0.0), QPointF(-22.0, 0.0))
            painter.drawRoundedRect(QRectF(-22.0, -14.0, 56.0, 28.0), 12.0, 12.0)

        elif component_id == "function_generator":
            # Benchtop chassis + right-side OUT / GND leads clear of pin labels
            painter.drawLine(QPointF(48.0, -20.0), QPointF(60.0, -20.0))
            painter.drawLine(QPointF(48.0, 20.0), QPointF(60.0, 20.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(-56.0, -34.0, 104.0, 68.0), 5.0, 5.0)
            painter.save()
            painter.setPen(QPen(QColor("#334155"), 1.2))
            painter.setBrush(QColor("#0F172A"))
            lcd_rect = QRectF(-48.0, -22.0, 56.0, 44.0)
            painter.drawRoundedRect(lcd_rect, 3.0, 3.0)
            # Mini green sine wave icon inside LCD
            painter.setPen(QPen(QColor("#4ADE80"), 1.6))
            painter.drawLine(QPointF(-41.0, 0.0), QPointF(-35.0, -10.0))
            painter.drawLine(QPointF(-35.0, -10.0), QPointF(-27.0, 10.0))
            painter.drawLine(QPointF(-27.0, 10.0), QPointF(-21.0, 0.0))
            painter.setPen(QColor("#22D3EE"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(QRectF(-19.0, -12.0, 24.0, 24.0), Qt.AlignmentFlag.AlignCenter, "FG")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#22C55E"))
            painter.drawEllipse(QPointF(16.0, 0.0), 3.0, 3.0)
            painter.restore()

        elif component_id == "oscilloscope":
            bnc_colors = {
                -40.0: "#FACC15",
                -20.0: "#22D3EE",
                20.0: "#94A3B8",
                40.0: "#475569",
            }
            for py, hex_col in bnc_colors.items():
                painter.drawLine(QPointF(-80.0, py), QPointF(-68.0, py))
                painter.save()
                painter.setPen(QPen(QColor(hex_col), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(-70.0, py), 3.0, 3.0)
                painter.restore()
            # Outer transparent frame keeps left [-68, -18] pin labels high-contrast in both themes
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(-68.0, -52.0, 142.0, 104.0), 5.0, 5.0)
            # Dark miniature LCD & right front-panel knob bay
            painter.save()
            painter.setPen(QPen(QColor("#334155"), 1.2))
            painter.setBrush(QColor("#0F172A"))
            lcd_rect = QRectF(-16.0, -38.0, 58.0, 76.0)
            painter.drawRoundedRect(lcd_rect, 3.0, 3.0)
            painter.setPen(QPen(QColor(71, 85, 105, 130), 0.7))
            for gx in range(1, 4):
                x_g = lcd_rect.left() + gx * lcd_rect.width() / 4.0
                painter.drawLine(QPointF(x_g, lcd_rect.top()), QPointF(x_g, lcd_rect.bottom()))
            for gy in range(1, 3):
                y_g = lcd_rect.top() + gy * lcd_rect.height() / 3.0
                painter.drawLine(QPointF(lcd_rect.left(), y_g), QPointF(lcd_rect.right(), y_g))
            painter.setPen(QPen(QColor("#94A3B8"), 1.2))
            painter.setBrush(QColor("#334155"))
            for ky in (-24.0, -4.0, 16.0, 34.0):
                painter.drawEllipse(QPointF(58.0, ky), 5.0, 5.0)
            painter.restore()

        else:
            painter.drawRect(QRectF(-20.0, -10.0, 40.0, 20.0))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                QRectF(-20.0, -10.0, 40.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                component_id[:4],
            )

        painter.restore()
