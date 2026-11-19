"""Dark and Light visual themes and color tokens for CircuitSim."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThemeMode(str, Enum):
    """Supported application visual themes."""

    DARK = "dark"
    LIGHT = "light"


@dataclass(frozen=True)
class ThemePalette:
    """Color tokens used by the canvas, symbol renderer, and dock widgets."""

    mode: ThemeMode
    window_bg: str
    panel_bg: str
    canvas_bg: str
    grid_minor: str
    grid_major: str
    grid_origin: str
    symbol_stroke: str
    symbol_text: str
    symbol_label: str
    terminal_dot: str
    terminal_hover: str
    selection_accent: str
    ghost_tint: str
    border_color: str


DARK_PALETTE = ThemePalette(
    mode=ThemeMode.DARK,
    window_bg="#161B22",
    panel_bg="#1E242C",
    canvas_bg="#0D1117",
    grid_minor="#21262D",
    grid_major="#30363D",
    grid_origin="#388BFD",
    symbol_stroke="#E6EDF3",
    symbol_text="#C9D1D9",
    symbol_label="#8B949E",
    terminal_dot="#58A6FF",
    terminal_hover="#FF7B72",
    selection_accent="#388BFD",
    ghost_tint="#58A6FF",
    border_color="#30363D",
)

LIGHT_PALETTE = ThemePalette(
    mode=ThemeMode.LIGHT,
    window_bg="#E8ECF1",
    panel_bg="#F4F6F9",
    canvas_bg="#F7F8FA",
    grid_minor="#E1E6ED",
    grid_major="#C8D1DC",
    grid_origin="#2563EB",
    symbol_stroke="#111827",
    symbol_text="#1E293B",
    symbol_label="#475569",
    terminal_dot="#2563EB",
    terminal_hover="#DC2626",
    selection_accent="#2563EB",
    ghost_tint="#3B82F6",
    border_color="#B8C2CE",
)


def get_theme_palette(mode: ThemeMode | str) -> ThemePalette:
    """Return the `ThemePalette` for `'dark'` or `'light'`."""
    theme = ThemeMode(mode) if not isinstance(mode, ThemeMode) else mode
    return DARK_PALETTE if theme == ThemeMode.DARK else LIGHT_PALETTE


def build_stylesheet(palette: ThemePalette) -> str:
    """Build a Qt Style Sheet (QSS) for the main window and dock panels."""
    return f"""
    QMainWindow, QWidget {{
        background-color: {palette.window_bg};
        color: {palette.symbol_stroke};
        font-family: "Segoe UI", sans-serif;
        font-size: 10pt;
    }}
    QMenuBar {{
        background-color: {palette.panel_bg};
        border-bottom: 1px solid {palette.border_color};
    }}
    QMenuBar::item:selected {{
        background-color: {palette.selection_accent};
        color: #FFFFFF;
    }}
    QMenu {{
        background-color: {palette.panel_bg};
        border: 1px solid {palette.border_color};
    }}
    QMenu::item:selected {{
        background-color: {palette.selection_accent};
        color: #FFFFFF;
    }}
    QToolBar {{
        background-color: {palette.panel_bg};
        border-bottom: 1px solid {palette.border_color};
        spacing: 3px;
        padding: 2px 4px;
    }}
    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 3px 6px;
    }}
    QToolButton:hover {{
        background-color: {'#DCE6F5' if palette.mode == ThemeMode.LIGHT else palette.panel_bg};
        border: 1px solid {palette.selection_accent};
    }}
    QToolButton:checked {{
        background-color: {palette.selection_accent};
        border: 1px solid {palette.selection_accent};
        color: #FFFFFF;
    }}
    QToolButton:checked:hover {{
        background-color: {palette.ghost_tint};
        border: 1px solid {palette.selection_accent};
        color: #FFFFFF;
    }}
    QDockWidget {{
        titlebar-close-icon: url(none);
        font-weight: 600;
    }}
    QDockWidget::title {{
        background-color: {'#DCE2EA' if palette.mode == ThemeMode.LIGHT else palette.panel_bg};
        color: {palette.symbol_text};
        padding: 6px 10px;
        border: 1px solid {palette.border_color};
    }}
    QLineEdit, QTreeWidget, QListWidget, QTextEdit {{
        background-color: {'#FFFFFF' if palette.mode == ThemeMode.LIGHT else palette.panel_bg};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        padding: 4px;
    }}
    QPushButton {{
        background-color: {'#FFFFFF' if palette.mode == ThemeMode.LIGHT else palette.panel_bg};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: {'#DCE6F5' if palette.mode == ThemeMode.LIGHT else palette.border_color};
        border: 1px solid {palette.selection_accent};
    }}
    QPushButton:hover {{
        border: 1px solid {palette.selection_accent};
    }}
    QScrollBar:vertical {{
        background-color: {palette.panel_bg};
        width: 10px;
        margin: 0px;
        border: 1px solid {palette.border_color};
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {palette.symbol_label};
        min-height: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {palette.selection_accent};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}
    QStatusBar {{
        background-color: {palette.panel_bg};
        border-top: 1px solid {palette.border_color};
    }}
    """
