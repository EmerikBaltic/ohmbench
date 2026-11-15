"""Offscreen Qt debug renderer exporting sample `ScopeFrame` waveforms as PNGs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)

from circuitsim.components import (
    ACVoltageSource,
    Capacitor,
    DCVoltageSource,
    Ground,
    Resistor,
    Switch,
    WaveformType,
)
from circuitsim.engine import Circuit, SimulationSession
from circuitsim.scope import (
    CHANNEL_COLOR_HEX,
    ScopeCore,
    ScopeFrame,
    TriggerEdge,
)

_LOADED_FONT_FAMILY: str | None = None


def _ensure_qt_app() -> QGuiApplication:
    global _LOADED_FONT_FAMILY
    app = QGuiApplication.instance()
    if app is None or not isinstance(app, QGuiApplication):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QGuiApplication([sys.argv[0]])

    if _LOADED_FONT_FAMILY is None:
        if QFontDatabase.families():
            _LOADED_FONT_FAMILY = "Consolas"
        else:
            for candidate in (
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ):
                if os.path.exists(candidate):
                    fid = QFontDatabase.addApplicationFont(candidate)
                    fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                    if fams:
                        _LOADED_FONT_FAMILY = fams[0]
                        break
            if _LOADED_FONT_FAMILY is None:
                _LOADED_FONT_FAMILY = app.font().family()
    return app


def _ui_font(point_size: int, *, bold: bool = False) -> QFont:
    fam = _LOADED_FONT_FAMILY or "Consolas"
    font = QFont(fam, point_size)
    if bold:
        font.setBold(True)
    return font


def _format_eng_time(seconds: float) -> str:
    mag = abs(seconds)
    if mag < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if mag < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if mag < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def render_scope_frame_to_qimage(
    frame: ScopeFrame,
    *,
    title: str = "Ohmbench Scope Core",
    fft_mode: bool = False,
    max_fft_hz: float = 10_000.0,
    width: int = 960,
    height: int = 540,
) -> QImage:
    """Render a `ScopeFrame` onto an offscreen `QImage` using dark 10x8 oscilloscope styling."""
    _ensure_qt_app()
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#0D1117"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin_l = 64
    margin_r = 24
    margin_t = 44
    margin_b = 76
    plot_rect = QRectF(
        margin_l,
        margin_t,
        width - margin_l - margin_r,
        height - margin_t - margin_b,
    )

    # Header bar
    painter.setPen(QColor("#E6EDF3"))
    painter.setFont(_ui_font(11, bold=True))
    status_badge = frame.status.value.upper()
    tb_label = f"FFT Spectrum (Hann)" if fft_mode else f"Timebase: {_format_eng_time(frame.timebase.seconds_per_div)}/div"
    painter.drawText(
        QRectF(margin_l, 8, plot_rect.width(), 28),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        f"{title}   |   {status_badge}   |   {tb_label}",
    )

    # Plot background
    painter.fillRect(plot_rect, QColor("#0B0F14"))

    # 10x8 graticule
    n_x = frame.timebase.horizontal_divisions
    n_y = 8
    grid_pen = QPen(QColor("#1F2937"), 1)
    painter.setPen(grid_pen)
    for ix in range(n_x + 1):
        x = plot_rect.left() + (ix / n_x) * plot_rect.width()
        painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
    for iy in range(n_y + 1):
        y = plot_rect.top() + (iy / n_y) * plot_rect.height()
        painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

    # Center crosshairs
    axis_pen = QPen(QColor("#374151"), 1.2)
    painter.setPen(axis_pen)
    cx = plot_rect.center().x()
    cy = plot_rect.center().y()
    painter.drawLine(QPointF(plot_rect.left(), cy), QPointF(plot_rect.right(), cy))
    painter.drawLine(QPointF(cx, plot_rect.top()), QPointF(cx, plot_rect.bottom()))

    painter.save()
    painter.setClipRect(plot_rect)

    painter.setFont(_ui_font(9))

    if fft_mode and frame.fft is not None and len(frame.fft.freqs_hz) > 1:
        fft = frame.fft
        freqs = fft.freqs_hz
        dbv = fft.magnitude_dbv
        mask = freqs <= max_fft_hz
        f_vis = freqs[mask]
        d_vis = np.clip(dbv[mask], -80.0, 20.0) if len(freqs[mask]) else dbv
        db_min, db_max = -80.0, 20.0
        path = QPainterPath()
        for k in range(len(f_vis)):
            px = plot_rect.left() + (float(f_vis[k]) / max_fft_hz) * plot_rect.width()
            py = plot_rect.bottom() - ((float(d_vis[k]) - db_min) / (db_max - db_min)) * plot_rect.height()
            if k == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        painter.setPen(QPen(QColor("#A78BFA"), 2.0))
        painter.drawPath(path)

        if fft.fundamental_freq_hz is not None and fft.fundamental_dbv is not None:
            fx = plot_rect.left() + (fft.fundamental_freq_hz / max_fft_hz) * plot_rect.width()
            fy = plot_rect.bottom() - ((max(db_min, min(db_max, fft.fundamental_dbv)) - db_min) / (db_max - db_min)) * plot_rect.height()
            painter.setPen(QPen(QColor("#FFD600"), 1.5))
            painter.drawEllipse(QPointF(fx, fy), 5.0, 5.0)
    else:
        # Time-domain traces mapped from display_div_y in [-4, +4] div onto plot_rect
        traces = frame.traces or {}
        t_rel = frame.t_rel
        t_left = float(t_rel[0]) if len(t_rel) else -5.0 * frame.timebase.seconds_per_div
        t_span = max(1e-18, float(t_rel[-1] - t_rel[0])) if len(t_rel) > 1 else frame.timebase.window_duration_s

        for ch_name in sorted(traces.keys()):
            tr = traces[ch_name]
            color = QColor(tr.config.color_hex or CHANNEL_COLOR_HEX.get(ch_name, "#FFD600"))
            y_div = tr.display_div_y

            # Peak-detect vertical envelope when active
            if tr.peak_detect:
                env_color = QColor(color)
                env_color.setAlpha(75)
                painter.setPen(QPen(env_color, 1.0))
                y_div_min = (tr.volts_min / tr.config.volts_per_div) + tr.config.offset_div
                y_div_max = (tr.volts_max / tr.config.volts_per_div) + tr.config.offset_div
                for k in range(len(t_rel)):
                    px = plot_rect.left() + ((float(t_rel[k]) - t_left) / t_span) * plot_rect.width()
                    py1 = cy - (float(y_div_min[k]) / (n_y * 0.5)) * (plot_rect.height() * 0.5)
                    py2 = cy - (float(y_div_max[k]) / (n_y * 0.5)) * (plot_rect.height() * 0.5)
                    if abs(py1 - py2) > 0.75:
                        painter.drawLine(QPointF(px, py1), QPointF(px, py2))

            path = QPainterPath()
            for k in range(len(t_rel)):
                px = plot_rect.left() + ((float(t_rel[k]) - t_left) / t_span) * plot_rect.width()
                py = cy - (float(y_div[k]) / (n_y * 0.5)) * (plot_rect.height() * 0.5)
                if k == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.setPen(QPen(color, 2.0))
            painter.drawPath(path)

        # Trigger time & level markers
        if frame.trigger_time_s is not None:
            trig_x = plot_rect.left() + ((0.0 - t_left) / t_span) * plot_rect.width()
            painter.setPen(QPen(QColor("#F97316"), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(trig_x, plot_rect.top()), QPointF(trig_x, plot_rect.bottom()))
        if frame.trigger.source in traces:
            t_ch = traces[frame.trigger.source].config
            trig_div = (frame.trigger.level_v / t_ch.volts_per_div) + t_ch.offset_div
            trig_y = cy - (trig_div / (n_y * 0.5)) * (plot_rect.height() * 0.5)
            painter.setPen(QPen(QColor("#F97316"), 1.2, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(plot_rect.left(), trig_y), QPointF(plot_rect.right(), trig_y))

    painter.restore()

    # Frame border
    painter.setPen(QPen(QColor("#4B5563"), 1.4))
    painter.drawRect(plot_rect)

    # Bottom measurement readout strip
    painter.setFont(_ui_font(9))
    y_bot = plot_rect.bottom() + 10
    if fft_mode and frame.fft is not None:
        fft = frame.fft
        f_txt = f"{fft.fundamental_freq_hz:.1f} Hz" if fft.fundamental_freq_hz is not None else "--"
        m_txt = f"{fft.fundamental_dbv:.2f} dBV" if fft.fundamental_dbv is not None else "--"
        painter.setPen(QColor("#A78BFA"))
        painter.drawText(
            QRectF(margin_l, y_bot, plot_rect.width(), 52),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"FFT Window: Periodic Hann   |   Peak: {f_txt} @ {m_txt}   |   RBW: {fft.rbw_hz:.2f} Hz   |   0-{max_fft_hz/1000:.0f} kHz",
        )
    else:
        traces = frame.traces or {}
        lines: list[str] = []
        for ch_name in sorted(traces.keys()):
            tr = traces[ch_name]
            m = tr.measurements
            freq_str = f"{m.frequency_hz:.1f} Hz" if m.frequency_hz is not None else "--"
            tr_str = f"tr={_format_eng_time(m.rise_time_s)}" if m.rise_time_s is not None else "tr=--"
            lines.append(
                f"{ch_name} ({tr.config.volts_per_div:g}V/div): Vpp={m.vpp:.2f}V  Vrms={m.vrms:.2f}V  f={freq_str}  {tr_str}"
            )
        if frame.phase_lag_ch1_ch2_deg is not None:
            lines.append(f"Phase lag (CH2 behind CH1): {frame.phase_lag_ch1_ch2_deg:+.2f} deg")
        painter.setPen(QColor("#D1D5DB"))
        painter.drawText(
            QRectF(margin_l, y_bot, plot_rect.width(), 58),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "\n".join(lines) if lines else "No active traces",
        )

    painter.end()
    return image


# Import numpy here (already loaded cleanly below QtGui)
import numpy as np


def _build_demo_frames() -> list[tuple[str, str, ScopeFrame, bool]]:
    # 1. Sine low-pass filter at fc = 1 kHz (+45 deg lag)
    c1 = Circuit()
    c1.add(ACVoltageSource("V1", "in", "0", amplitude=5.0, frequency=1000.0))
    c1.add(Resistor("R1", "in", "out", resistance=1000.0))
    c1.add(Capacitor("C1", "out", "0", capacitance=1.0 / (2.0 * math.pi * 1000.0 * 1000.0)))
    c1.add(Ground("G1", "0"))
    s1 = SimulationSession(c1)
    sc1 = ScopeCore()
    sc1.assign_probe("CH1", "in", enabled=True)
    sc1.assign_probe("CH2", "out", enabled=True)
    sc1.set_channel("CH1", volts_per_div=2.0)
    sc1.set_channel("CH2", volts_per_div=2.0)
    sc1.bind_session(s1)
    sc1.set_timebase(500e-6)
    s1.step_chunk(5500)
    f_sine = sc1.produce_frame()

    # 2. Square wave into RC integrator (5 V 1 kHz square -> R=1k, C=220n)
    c2 = Circuit()
    c2.add(ACVoltageSource("V1", "in", "0", waveform=WaveformType.SQUARE, amplitude=5.0, frequency=1000.0))
    c2.add(Resistor("R1", "in", "out", resistance=1000.0))
    c2.add(Capacitor("C1", "out", "0", capacitance=220e-9))
    c2.add(Ground("G1", "0"))
    s2 = SimulationSession(c2, default_dt=5e-6)
    sc2 = ScopeCore()
    sc2.assign_probe("CH1", "in", enabled=True)
    sc2.assign_probe("CH2", "out", enabled=True)
    sc2.set_channel("CH1", volts_per_div=2.0)
    sc2.set_channel("CH2", volts_per_div=2.0)
    sc2.set_timebase(200e-6)
    sc2.bind_session(s2)
    s2.step_chunk(3000)
    f_square = sc2.produce_frame()

    # 3. Single-shot RC step response (tau = 1 ms, tr ≈ 2.20 ms)
    c3 = Circuit()
    c3.add(DCVoltageSource("V1", "src", "0", voltage=5.0))
    c3.add(Switch("SW1", "src", "in", closed=False))
    c3.add(Resistor("R1", "in", "out", resistance=1000.0))
    c3.add(Capacitor("C1", "out", "0", capacitance=1e-6))
    c3.add(Resistor("R_bleed", "out", "0", resistance=1e6))
    c3.add(Ground("G1", "0"))
    s3 = SimulationSession(c3, default_dt=10e-6)
    sc3 = ScopeCore()
    sc3.assign_probe("CH1", "out", enabled=True)
    sc3.set_channel("CH1", volts_per_div=1.0, offset_div=-2.5)
    sc3.set_timebase(1e-3)
    sc3.set_trigger(source="CH1", level_v=2.0, edge=TriggerEdge.RISING, pre_trigger_fraction=0.2)
    sc3.bind_session(s3)
    s3.step_chunk(200)
    sc3.clear()
    sc3.single()
    s3.toggle_switch("SW1")
    s3.step_chunk(1000)
    f_step = sc3.produce_frame()

    # 4. FFT of 1 kHz 5 V square wave (showing 1 kHz, 3 kHz, 5 kHz, 7 kHz harmonics)
    c4 = Circuit()
    c4.add(ACVoltageSource("V1", "in", "0", waveform=WaveformType.SQUARE, amplitude=5.0, frequency=1000.0))
    c4.add(Resistor("R1", "in", "0", resistance=1000.0))
    c4.add(Ground("G1", "0"))
    s4 = SimulationSession(c4, default_dt=10e-6)
    sc4 = ScopeCore()
    sc4.assign_probe("CH1", "in", enabled=True)
    sc4.set_fft("CH1", window="hann")
    sc4.set_timebase(5e-3)
    sc4.bind_session(s4)
    s4.step_chunk(6000)
    f_fft = sc4.produce_frame()

    return [
        ("scope_sine.png", "Demo 1: 1 kHz RC Low-Pass Filter (+45 deg lag)", f_sine, False),
        ("scope_square.png", "Demo 2: 1 kHz Square Wave -> RC Integrator", f_square, False),
        ("scope_rc_step.png", "Demo 3: Single-Shot RC Step Response (tau = 1 ms)", f_step, False),
        ("scope_fft.png", "Demo 4: 1 kHz Square Wave Periodic-Hann FFT", f_fft, True),
    ]


def export_demo_pngs(output_dir: str | Path) -> list[Path]:
    """Run 4 headless circuit captures and export PNG waveform snapshots to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for filename, title, frame, is_fft in _build_demo_frames():
        img = render_scope_frame_to_qimage(frame, title=title, fft_mode=is_fft)
        dest = out / filename
        if not img.save(str(dest), "PNG"):
            raise RuntimeError(f"Failed to save demo PNG: {dest}")
        saved.append(dest)
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export headless Ohmbench scope demo PNGs.")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("docs/design"),
        help="Destination directory for scope_*.png snapshots",
    )
    args = parser.parse_args(argv)
    paths = export_demo_pngs(args.export)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
