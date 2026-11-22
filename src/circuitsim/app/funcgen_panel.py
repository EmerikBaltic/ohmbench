"""Modeless Benchtop Function Generator front-panel dialog (`FG1` / FunctionGenerator)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from circuitsim.components.function_generator import FunctionGenerator
from circuitsim.components.sources import WaveformType
from circuitsim.dialogs.value_parser import format_engineering_value

if TYPE_CHECKING:
    from circuitsim.canvas.component_item import ComponentItem


class FunctionGeneratorDialog(QDialog):
    """Interactive benchtop panel for configuring a `FunctionGenerator` live."""

    parameters_changed = Signal(str, dict)

    def __init__(
        self,
        fg_or_item: FunctionGenerator | "ComponentItem",
        *,
        parameter_changed_callback: Callable[[str, dict[str, object]], None] | None = None,
        closed_callback: Callable[["FunctionGeneratorDialog"], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if hasattr(fg_or_item, "component"):
            self._item: "ComponentItem | None" = fg_or_item  # type: ignore[assignment]
            comp = fg_or_item.component
            if not isinstance(comp, FunctionGenerator):
                raise TypeError("FunctionGeneratorDialog requires a FunctionGenerator component.")
            self.fg: FunctionGenerator = comp
        elif isinstance(fg_or_item, FunctionGenerator):
            self._item = None
            self.fg = fg_or_item
        else:
            raise TypeError("FunctionGeneratorDialog requires a FunctionGenerator component.")

        self._parameter_changed_callback = parameter_changed_callback
        self._closed_callback = closed_callback
        self._syncing: bool = False
        self._freq_unit_scale: float = 1.0  # 1.0=Hz, 1e3=kHz, 1e6=MHz

        self.setWindowTitle(f"Function Generator — {self.fg.name}")
        self.setMinimumWidth(460)
        self.setStyleSheet(
            """
            QDialog { background-color: #1A1F29; color: #E2E8F0; }
            QLabel { color: #CBD5E1; font-size: 11px; }
            QDoubleSpinBox {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #273244;
                color: #E2E8F0;
                border: 1px solid #3B4961;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #334155; }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ------------------------------------------------------------------
        # Dark Benchtop LCD Readout Header
        # ------------------------------------------------------------------
        lcd_frame = QFrame()
        lcd_frame.setStyleSheet(
            "QFrame { background-color: #0B1120; border: 2px solid #334155; border-radius: 6px; }"
        )
        lcd_layout = QVBoxLayout(lcd_frame)
        lcd_layout.setContentsMargins(12, 10, 12, 10)
        lcd_layout.setSpacing(4)

        top_lcd_row = QHBoxLayout()
        self.lcd_wave_label = QLabel("SINE")
        self.lcd_wave_label.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.lcd_wave_label.setStyleSheet("color: #22D3EE; border: none;")

        self.lcd_status_badge = QLabel("OUTPUT ON")
        self.lcd_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lcd_status_badge.setFixedWidth(104)

        top_lcd_row.addWidget(self.lcd_wave_label)
        top_lcd_row.addStretch(1)
        top_lcd_row.addWidget(self.lcd_status_badge)
        lcd_layout.addLayout(top_lcd_row)

        self.lcd_freq_label = QLabel("100 Hz")
        self.lcd_freq_label.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        self.lcd_freq_label.setStyleSheet("color: #4ADE80; border: none;")
        lcd_layout.addWidget(self.lcd_freq_label)

        self.lcd_sub_label = QLabel("5 Vpk (10 Vpp) | Offset: 0 V | Duty: 50.0% | Zout: Hi-Z")
        self.lcd_sub_label.setFont(QFont("Consolas", 10))
        self.lcd_sub_label.setStyleSheet("color: #94A3B8; border: none;")
        lcd_layout.addWidget(self.lcd_sub_label)
        root.addWidget(lcd_frame)

        # ------------------------------------------------------------------
        # Waveform Selector Bar
        # ------------------------------------------------------------------
        wf_row = QHBoxLayout()
        wf_row.setSpacing(6)
        self.waveform_buttons: dict[WaveformType, QPushButton] = {}
        for wf, caption in (
            (WaveformType.SINE, "~ SINE"),
            (WaveformType.SQUARE, "⊓ SQUARE"),
            (WaveformType.TRIANGLE, "∧ TRIANGLE"),
            (WaveformType.SAWTOOTH, "⊿ SAWTOOTH"),
        ):
            btn = QPushButton(caption)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _chk=False, w=wf: self.set_waveform(w))
            self.waveform_buttons[wf] = btn
            wf_row.addWidget(btn)
        root.addLayout(wf_row)

        # ------------------------------------------------------------------
        # Parameter Controls Grid
        # ------------------------------------------------------------------
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        # Frequency + Unit Range Buttons
        grid.addWidget(QLabel("Frequency:"), 0, 0)
        freq_box = QHBoxLayout()
        freq_box.setSpacing(4)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.000001, 1e7)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setSingleStep(10.0)
        self.freq_spin.valueChanged.connect(self._on_freq_spin_changed)
        freq_box.addWidget(self.freq_spin, 1)

        self.unit_buttons: dict[float, QPushButton] = {}
        for scale, u_lbl in ((1.0, "Hz"), (1e3, "kHz"), (1e6, "MHz")):
            ubtn = QPushButton(u_lbl)
            ubtn.setCheckable(True)
            ubtn.setFixedWidth(44)
            ubtn.clicked.connect(lambda _chk=False, sc=scale: self._set_freq_unit(sc))
            self.unit_buttons[scale] = ubtn
            freq_box.addWidget(ubtn)
        grid.addLayout(freq_box, 0, 1)

        # Amplitude (Vpk)
        grid.addWidget(QLabel("Amplitude (Vpk):"), 1, 0)
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1000.0)
        self.amp_spin.setDecimals(3)
        self.amp_spin.setSingleStep(0.5)
        self.amp_spin.setSuffix(" V")
        self.amp_spin.valueChanged.connect(lambda val: self._apply_updates({"amplitude": float(val)}))
        grid.addWidget(self.amp_spin, 1, 1)

        # DC Offset
        grid.addWidget(QLabel("DC Offset:"), 2, 0)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1000.0, 1000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.25)
        self.offset_spin.setSuffix(" V")
        self.offset_spin.valueChanged.connect(lambda val: self._apply_updates({"dc_offset": float(val)}))
        grid.addWidget(self.offset_spin, 2, 1)

        # Duty Cycle (%)
        grid.addWidget(QLabel("Duty / Symmetry:"), 3, 0)
        self.duty_spin = QDoubleSpinBox()
        self.duty_spin.setRange(5.0, 95.0)
        self.duty_spin.setDecimals(1)
        self.duty_spin.setSingleStep(5.0)
        self.duty_spin.setSuffix(" %")
        self.duty_spin.valueChanged.connect(lambda val: self._apply_updates({"duty_cycle": float(val) / 100.0}))
        grid.addWidget(self.duty_spin, 3, 1)

        root.addLayout(grid)

        # ------------------------------------------------------------------
        # Output Impedance & Master Output Switch
        # ------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        self.zout_btn = QPushButton("Load / Zout: Hi-Z")
        self.zout_btn.setCheckable(True)
        self.zout_btn.clicked.connect(self._on_zout_toggled)

        self.output_btn = QPushButton("OUTPUT: ON")
        self.output_btn.setCheckable(True)
        self.output_btn.clicked.connect(self._on_output_toggled)

        bottom_row.addWidget(self.zout_btn, 1)
        bottom_row.addWidget(self.output_btn, 1)
        root.addLayout(bottom_row)

        self._choose_initial_freq_unit()
        self.sync_from_component()

    def _choose_initial_freq_unit(self) -> None:
        freq = float(self.fg.frequency)
        if freq >= 1e6:
            self._freq_unit_scale = 1e6
        elif freq >= 1e3:
            self._freq_unit_scale = 1e3
        else:
            self._freq_unit_scale = 1.0

    def _set_freq_unit(self, scale: float) -> None:
        self._freq_unit_scale = float(scale)
        self.sync_from_component()

    def _on_freq_spin_changed(self, value: float) -> None:
        if self._syncing:
            return
        hz = max(0.1, min(1e7, float(value) * self._freq_unit_scale))
        self._apply_updates({"frequency": hz})

    def set_waveform(self, waveform: WaveformType | str) -> None:
        wf_str = waveform.value if isinstance(waveform, WaveformType) else str(waveform).strip().lower()
        self._apply_updates({"waveform": wf_str})

    def _on_zout_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        zout = 50.0 if checked else 0.0
        self._apply_updates({"output_impedance": zout})

    def _on_output_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        self._apply_updates({"output_enabled": bool(checked)})

    def _apply_updates(self, updates: dict[str, object]) -> None:
        if self._syncing:
            return
        applied: dict[str, object] = {}
        for k, v in updates.items():
            self.fg.set_parameter_value(k, v)  # type: ignore[arg-type]
            applied[k] = getattr(self.fg, k)
        if self._item is not None:
            self._item.live_display_value = None
            self._item.setToolTip(self._item.get_live_tooltip(is_running=False))
            self._item.update()
        self.sync_from_component()
        self.parameters_changed.emit(self.fg.name, applied)
        if self._parameter_changed_callback is not None:
            self._parameter_changed_callback(self.fg.name, applied)

    def sync_from_component(self) -> None:
        """Refresh controls and dark LCD readout from the underlying `FunctionGenerator`."""
        self._syncing = True
        try:
            wf = self.fg.waveform
            for w, btn in self.waveform_buttons.items():
                active = w == wf
                btn.setChecked(active)
                if active:
                    btn.setStyleSheet(
                        "background-color: #0284C7; color: #F8FAFC; border: 1px solid #38BDF8; border-radius: 4px; padding: 5px 10px; font-weight: bold;"
                    )
                else:
                    btn.setStyleSheet("")

            for sc, ubtn in self.unit_buttons.items():
                active_u = abs(sc - self._freq_unit_scale) < 0.1
                ubtn.setChecked(active_u)
                if active_u:
                    ubtn.setStyleSheet(
                        "background-color: #0F766E; color: #F8FAFC; border: 1px solid #2DD4BF; border-radius: 4px; font-weight: bold;"
                    )
                else:
                    ubtn.setStyleSheet("")

            min_hz, max_hz = 0.1, 1e7
            self.freq_spin.setRange(min_hz / self._freq_unit_scale, max_hz / self._freq_unit_scale)
            self.freq_spin.setValue(float(self.fg.frequency) / self._freq_unit_scale)
            self.amp_spin.setValue(float(self.fg.amplitude))
            self.offset_spin.setValue(float(self.fg.dc_offset))
            self.duty_spin.setValue(float(self.fg.duty_cycle) * 100.0)
            self.duty_spin.setEnabled(wf != WaveformType.SINE and wf != WaveformType.SAWTOOTH)

            is_50 = float(self.fg.output_impedance) >= 1.0
            self.zout_btn.setChecked(is_50)
            self.zout_btn.setText("Zout: 50 Ω" if is_50 else "Zout: Hi-Z (0 Ω)")

            out_on = bool(self.fg.output_enabled)
            self.output_btn.setChecked(out_on)
            self.output_btn.setText("OUTPUT: ON" if out_on else "OUTPUT: OFF")
            if out_on:
                self.output_btn.setStyleSheet(
                    "background-color: #16A34A; color: #FFFFFF; border: 1px solid #4ADE80; border-radius: 4px; padding: 6px 10px; font-weight: bold;"
                )
            else:
                self.output_btn.setStyleSheet(
                    "background-color: #475569; color: #CBD5E1; border: 1px solid #64748B; border-radius: 4px; padding: 6px 10px; font-weight: bold;"
                )

            wf_symbols = {
                WaveformType.SINE: "~ SINE",
                WaveformType.SQUARE: "⊓ SQUARE",
                WaveformType.TRIANGLE: "∧ TRIANGLE",
                WaveformType.SAWTOOTH: "⊿ SAWTOOTH",
                WaveformType.PULSE: "⎍ PULSE",
            }
            self.lcd_wave_label.setText(wf_symbols.get(wf, wf.value.upper()))
            self.lcd_freq_label.setText(format_engineering_value(self.fg.frequency, "Hz", sig_figs=4))
            amp_s = format_engineering_value(self.fg.amplitude, "V", sig_figs=3)
            vpp_s = format_engineering_value(2.0 * self.fg.amplitude, "V", sig_figs=3)
            off_s = format_engineering_value(self.fg.dc_offset, "V", sig_figs=3)
            z_s = "50 Ω" if abs(self.fg.output_impedance - 50.0) < 0.5 else (
                "Hi-Z" if self.fg.output_impedance < 1e-6 else format_engineering_value(self.fg.output_impedance, "Ω", sig_figs=3)
            )
            self.lcd_sub_label.setText(
                f"{amp_s}pk ({vpp_s}pp) | Offset: {off_s} | Duty: {self.fg.duty_cycle * 100.0:.1f}% | Zout: {z_s}"
            )
            if out_on:
                self.lcd_status_badge.setText("OUTPUT ON")
                self.lcd_status_badge.setStyleSheet(
                    "color: #052E16; background-color: #4ADE80; border-radius: 4px; padding: 3px 6px; font-weight: bold;"
                )
            else:
                self.lcd_status_badge.setText("OUTPUT OFF")
                self.lcd_status_badge.setStyleSheet(
                    "color: #F8FAFC; background-color: #DC2626; border-radius: 4px; padding: 3px 6px; font-weight: bold;"
                )
        finally:
            self._syncing = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._closed_callback is not None:
            try:
                self._closed_callback(self)
            except Exception:
                pass
        super().closeEvent(event)
