"""Tests for Item F9: Share URL codec (`ohmbench://c/...`), KiCad exporter, & publication SVG/PNG."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import numpy as np
import pytest

from circuitsim.app.themes import ThemeMode
from circuitsim.io import (
    ShareUrlError,
    decode_circuit_share_url,
    encode_circuit_share_url,
    export_to_kicad_sch,
)
from circuitsim.templates import build_test_jig
from circuitsim.wiring.model import FreeEnd


def _count_sexpr_balance(text: str) -> None:
    depth = 0
    in_quotes = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_quotes:
            escape = True
            continue
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if in_quotes:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            assert depth >= 0, "Unmatched closing ')' in KiCad S-expression"
    assert not in_quotes, "Unterminated quoted string in KiCad S-expression"
    assert depth == 0, f"Unclosed '(' remaining in KiCad S-expression (depth={depth})"


def test_f9_share_and_kicad_modules_are_qt_free() -> None:
    """Verify `share_codec.py` and `kicad_export.py` contain no Qt imports."""
    root = Path(__file__).resolve().parent.parent / "src" / "circuitsim" / "io"
    for fname in ("share_codec.py", "kicad_export.py"):
        tree = ast.parse((root / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f9_share_url_lossless_round_trip_and_corruption_guards(qapp, tmp_path: Path) -> None:
    """Verify `ohmbench://c/...` round-trips CircuitDocumentV2 + MainWindow copy/load cleanly."""
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.engine import MNASolver

    doc = build_test_jig("diode_clipper")
    url = encode_circuit_share_url(doc)
    assert url.startswith("ohmbench://c/")
    assert url.isascii()

    decoded = decode_circuit_share_url(url)
    assert len(decoded.placed) == len(doc.placed)
    assert len(decoded.graph.wires) == len(doc.graph.wires)
    assert len(decoded.graph.junctions) == len(doc.graph.junctions)

    t1 = MNASolver(doc.build_circuit().circuit).solve_transient(duration=1e-3, dt=1e-5)
    t2 = MNASolver(decoded.build_circuit().circuit).solve_transient(duration=1e-3, dt=1e-5)
    assert np.allclose(t1.component_voltages["D1"], t2.component_voltages["D1"], atol=1e-12)

    for bad_url in (
        "https://example.com/c/abc",
        "ohmbench://c/",
        "ohmbench://c/!!!not-base64!!!",
        "ohmbench://c/YWJjZGVm",
    ):
        with pytest.raises(ShareUrlError):
            decode_circuit_share_url(bad_url)

    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        win.load_test_jig("potentiometer_divider", prompt_discard=False)
        copied = win.copy_share_url()
        assert copied.startswith("ohmbench://c/")
        win._on_new_circuit_confirmed() if hasattr(win, "_on_new_circuit_confirmed") else win.circuit_scene.clear_schematic()
        assert len(win.circuit_scene.placed_items) == 0
        win.load_share_url(copied)
        assert len(win.circuit_scene.placed_items) == 4
    finally:
        win.close()


def test_f9_kicad_sch_export_valid_sexpr(tmp_path: Path) -> None:
    """Verify `.kicad_sch` exporter emits balanced KiCad S-expressions and mapped symbols/wires."""
    doc = build_test_jig("diode_clipper")
    out_path = tmp_path / "clipper.kicad_sch"
    sexpr = export_to_kicad_sch(doc, out_path, title="Symmetric Diode Clipper")
    assert out_path.exists()
    assert sexpr.startswith("(kicad_sch")
    _count_sexpr_balance(sexpr)

    assert '(version 20231120)' in sexpr
    assert '(generator "ohmbench")' in sexpr
    assert '(lib_id "Device:R")' in sexpr
    assert '(lib_id "Device:D")' in sexpr
    assert '(lib_id "Simulation_SPICE:VAC")' in sexpr
    assert '(wire (pts ' in sexpr
    assert '(junction (at ' in sexpr
    # Grid conversion: R1 at x=200 px -> 200 * 2.54 / 20.0 = 25.4 mm
    assert "(at 25.4 " in sexpr


def test_f9_publication_export_light_bg_no_chrome_from_dark_theme(qapp, tmp_path: Path) -> None:
    """Publication PNG/SVG forces white bg & hides grid/selection/overlays/FreeEnd even in Dark mode."""
    from PySide6.QtGui import QImage
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings
    from circuitsim.canvas.export import export_publication_png, export_publication_svg

    win = MainWindow(initial_theme=ThemeMode.DARK, settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        win.load_test_jig("rc_lowpass", prompt_discard=False)
        scene = win.circuit_scene
        scene.set_theme(ThemeMode.DARK)
        scene.set_voltage_colors_enabled(True)
        scene.set_current_dots_enabled(True)
        scene.set_stress_warnings_enabled(True)

        # Select an item and add a temporary FreeEnd wire to verify chrome suppression
        first_item = scene.placed_items[0]
        first_item.setSelected(True)
        scene.set_wire_graph(
            scene.wire_graph.add_wire(FreeEnd(40, 40), FreeEnd(80, 40))
        )

        png_path = tmp_path / "pub_figure.png"
        svg_path = tmp_path / "pub_figure.svg"
        assert export_publication_png(scene, png_path, scale=2.0) is True
        assert export_publication_svg(scene, svg_path) is True

        # Verify pure white background & dark high-contrast strokes in PNG
        img = QImage(str(png_path))
        assert not img.isNull()
        assert img.width() > 200 and img.height() > 150
        corner = img.pixelColor(2, 2)
        assert (corner.red(), corner.green(), corner.blue()) == (255, 255, 255)

        # Confirm dark high-contrast strokes exist somewhere in the central canvas
        has_dark_stroke = False
        for y in range(0, img.height(), 4):
            for x in range(0, img.width(), 4):
                px = img.pixelColor(x, y)
                if px.red() < 80 and px.green() < 80 and px.blue() < 80:
                    has_dark_stroke = True
                    break
            if has_dark_stroke:
                break
        assert has_dark_stroke

        # Verify valid SVG with light fill
        svg_text = svg_path.read_text(encoding="utf-8")
        assert "<svg" in svg_text and "#ffffff" in svg_text.lower()

        # Verify interactive Dark mode, overlays, and selection were restored cleanly
        assert scene.theme == ThemeMode.DARK
        assert scene.voltage_colors_enabled is True
        assert scene.current_dots_enabled is True
        assert scene.stress_warnings_enabled is True
        assert first_item.isSelected() is True
    finally:
        win.close()
