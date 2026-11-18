"""Phase H Workbench Expansion tests (H0-H10) — offscreen Qt + Qt-free physics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt

from circuitsim.app.palette_dock import ComponentPaletteWidget
from circuitsim.app.themes import ThemeMode, build_stylesheet, get_theme_palette
from circuitsim.canvas.scene import CircuitScene
from circuitsim.canvas.symbol_renderer import SymbolLibrary
from circuitsim.wiring.model import TerminalRef


def test_h0_phase_h_scaffolding(qapp) -> None:
    """Verify offscreen Qt scaffolding and empty CircuitScene setup."""
    assert qapp is not None
    scene = CircuitScene(SymbolLibrary())
    try:
        assert scene.placed_items == []
        assert scene.placed_wires == []
        assert scene.wire_graph.wires == ()
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_h1_parallel_voltmeter_inductor_straight_2point_wires(qapp) -> None:
    """H1: Stacked horizontal VM1 above L1 in parallel connects with straight 2-point vertical wires."""
    scene = CircuitScene(SymbolLibrary())
    try:
        l1 = scene.place_new_component("inductor", QPointF(0.0, 0.0))
        vm1 = scene.place_new_component("voltmeter", QPointF(0.0, -120.0))

        l1_terms = l1.terminal_scene_positions_px()
        vm1_terms = vm1.terminal_scene_positions_px()
        x_left = l1_terms["a"][0]
        x_right = l1_terms["b"][0]
        assert vm1_terms["pos"][0] == x_left
        assert vm1_terms["neg"][0] == x_right

        w_left = scene.add_wire(TerminalRef("VM1", "pos"), TerminalRef("L1", "a"))
        w_right = scene.add_wire(TerminalRef("VM1", "neg"), TerminalRef("L1", "b"))
        assert w_left is not None
        assert w_right is not None
        assert len(w_left.points) == 2
        assert len(w_right.points) == 2
        assert all(px == x_left for px, _py in w_left.points)
        assert all(px == x_right for px, _py in w_right.points)
        assert w_left.points == (vm1_terms["pos"], l1_terms["a"])
        assert w_right.points == (vm1_terms["neg"], l1_terms["b"])
    finally:
        scene.shutdown_scene()
        qapp.processEvents()


def test_h2_palette_vertical_scrollbar_search_and_expand_collapse(qapp) -> None:
    """H2: Component palette has visible vertical scrollbar, live search filter, and Expand/Collapse All."""
    palette = ComponentPaletteWidget(SymbolLibrary(), theme=ThemeMode.DARK)
    try:
        # Verify themes.py QSS includes explicit vertical scrollbar rules in both Dark & Light themes
        for mode in (ThemeMode.DARK, ThemeMode.LIGHT):
            qss = build_stylesheet(get_theme_palette(mode))
            assert "QScrollBar:vertical" in qss
            assert "QScrollBar::handle:vertical" in qss

        palette.resize(240, 300)
        palette.show()
        palette.expand_all()
        qapp.processEvents()

        vbar = palette.tree.verticalScrollBar()
        assert palette.tree.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert vbar.maximum() > 0
        assert vbar.isVisible()

        # Collapse All then filter for "mcu" -> Mixed-Signal auto-expands, others hidden
        palette.btn_collapse_all.click()
        qapp.processEvents()
        palette.search_input.setText("mcu")
        qapp.processEvents()

        visible_cats: list[str] = []
        mcu_visible = False
        for i in range(palette.tree.topLevelItemCount()):
            cat = palette.tree.topLevelItem(i)
            assert cat is not None
            if not cat.isHidden():
                visible_cats.append(cat.text(0))
                assert cat.isExpanded()
                for j in range(cat.childCount()):
                    ch = cat.child(j)
                    if ch is not None and not ch.isHidden() and ch.text(0) == "MCU Block":
                        mcu_visible = True
        assert mcu_visible
        assert visible_cats == ["Mixed-Signal"]

        # Clearing search restores all categories; Expand All works cleanly
        palette.search_input.clear()
        palette.btn_expand_all.click()
        qapp.processEvents()
        hidden_cats = [
            palette.tree.topLevelItem(i).text(0)
            for i in range(palette.tree.topLevelItemCount())
            if palette.tree.topLevelItem(i).isHidden()
        ]
        assert hidden_cats == []
    finally:
        palette.close()
        qapp.processEvents()
