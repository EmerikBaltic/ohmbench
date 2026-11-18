"""Tests for Item F12: Toggleable physical package/pinout overlay & documentation accuracy."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter

from circuitsim.app.main_window import MainWindow
from circuitsim.app.settings import AppSettings
from circuitsim.canvas.component_item import ComponentItem
from circuitsim.templates import build_test_jig


def test_f12_pinout_overlay_toggles_cleanly_without_mutating_wire_graph(
    qapp, tmp_path: Path
) -> None:
    """Pinout overlay renders distinct pixels and leaves `WireGraph` & node names intact."""
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        win.load_test_jig("diode_clipper", prompt_discard=False)
        scene = win.circuit_scene

        assert win.act_pinout_labels.isChecked() is False
        assert scene.pinout_labels_enabled is False

        # Place an MCUBlock as well to test multi-pin physical pinout labels
        mcu_item = scene.place_new_component("mcu", QPointF(540.0, 200.0))
        pkg, pins = ComponentItem.package_and_pinout_labels(mcu_item.component)
        assert pkg == "DIP-MCU"
        assert set(pins.keys()) == {"A0", "A1", "D0", "D1", "TX", "RX"}

        before_graph = scene.wire_graph
        before_net = scene.extract_connectivity_result().circuit

        # Snapshot pixels before vs after enabling overlay
        def render_bytes() -> bytes:
            img = QImage(320, 240, QImage.Format.Format_ARGB32)
            img.fill(0xFFFFFFFF)
            p = QPainter(img)
            scene.render(p, QRectF(0, 0, 320, 240), QRectF(120, 80, 320, 240))
            p.end()
            return bytes(img.bits())

        raw_off = render_bytes()
        win.act_pinout_labels.trigger()
        assert win.act_pinout_labels.isChecked() is True
        assert scene.pinout_labels_enabled is True
        assert all(it._pinout_overlay_enabled is True for it in scene.placed_items)

        raw_on = render_bytes()
        assert raw_off != raw_on

        # Graph & solver node topology must be strictly unchanged
        assert scene.wire_graph == before_graph
        assert [c.name for c in scene.extract_connectivity_result().circuit.components] == [
            c.name for c in before_net.components
        ]

        # Toggle off cleanly
        win.act_pinout_labels.trigger()
        assert scene.pinout_labels_enabled is False
        assert all(it._pinout_overlay_enabled is False for it in scene.placed_items)
    finally:
        win.circuit_scene.undo_stack.setClean()
        win._switch_modified = False
        win.close()


def test_f12_docs_reference_phase_f_modules_accurately() -> None:
    """Verify `AGENT_INSTRUCTIONS.md` and `PROGRESS.md` list real Phase F modules."""
    root = Path(__file__).resolve().parent.parent
    agent_doc = (root / "AGENT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    prog_doc = (root / "PROGRESS.md").read_text(encoding="utf-8")

    for mod in (
        "ac_analysis.py",
        "sweep.py",
        "diode.py",
        "mcu.py",
        "wav_signal.py",
        "spice_import.py",
        "share_codec.py",
        "kicad_export.py",
        "instruments_panel.py",
    ):
        assert mod in agent_doc, f"{mod} missing from AGENT_INSTRUCTIONS.md"
        assert mod in prog_doc, f"{mod} missing from PROGRESS.md"
