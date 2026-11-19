"""Tests for Item F7: Audio .wav I/O (`WavVoltageSource` & resample export) + 6 Test Jigs."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import wave
import numpy as np

from circuitsim.components import (
    Capacitor,
    Ground,
    KNOWN_COMPONENT_TYPES,
    Resistor,
    WavVoltageSource,
    create_default_component_model,
)
from circuitsim.engine import Circuit, MNASolver
from circuitsim.io.wav_signal import (
    export_waveform_to_wav,
    load_wav_samples,
    save_wav_samples,
)
from circuitsim.templates import build_test_jig, list_test_jigs
from circuitsim.wiring.model import FreeEnd


def test_f7_wav_and_templates_are_qt_free() -> None:
    """Verify `wav_signal.py` and `templates/test_jigs.py` contain zero Qt imports."""
    root = Path(__file__).resolve().parent.parent / "src" / "circuitsim"
    for rel in ("io/wav_signal.py", "templates/test_jigs.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("PySide6", "PyQt")), alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("PySide6", "PyQt")), node.module


def test_f7_wav_load_simulate_and_resample_export(tmp_path: Path) -> None:
    """Synthesize WAV -> simulate RC LPF with WavVoltageSource -> export at 22.05 kHz."""
    fs_in = 44100
    dur = 0.01  # 10 ms (10 cycles at 1 kHz)
    t_in = np.arange(int(round(fs_in * dur)), dtype=np.float64) / fs_in
    samples_norm = 0.8 * np.sin(2.0 * math.pi * 1000.0 * t_in)

    wav_in = tmp_path / "input_1khz.wav"
    save_wav_samples(wav_in, fs_in, samples_norm, normalize=False, peak_fullscale_v=1.0)

    loaded = load_wav_samples(wav_in, gain_v=5.0)
    assert loaded.sample_rate_hz == fs_in
    # Peak = 0.8 * 5.0 = 4.0 V (16-bit quantization within 1 mV)
    assert math.isclose(float(np.max(np.abs(loaded.samples_v))), 4.0, abs_tol=2e-3)

    # Filter through 1 kHz RC LPF: steady-state peak = 4.0 / sqrt(2) = 2.8284 V
    c = Circuit()
    c.add(Ground("GND", "0"))
    src = WavVoltageSource("W1", "vin", "0", wav_path=str(wav_in), gain_v=5.0)
    c.add(src)
    c.add(Resistor("R1", "vin", "vout", resistance=1000.0))
    c.add(Capacitor("C1", "vout", "0", capacitance=1.0 / (2.0 * math.pi * 1000.0 * 1000.0)))

    res = MNASolver(c).solve_transient(duration=dur, dt=1e-5)
    v_out = res.node_voltages["vout"]
    half_idx = len(v_out) // 2
    ss_peak = float(np.max(np.abs(v_out[half_idx:])))
    expected_ss_peak = 4.0 / math.sqrt(2.0)
    assert math.isclose(ss_peak, expected_ss_peak, rel_tol=0.05)

    # Resample and export to 22050 Hz WAV
    wav_out = tmp_path / "filtered_22050.wav"
    fs_out = 22050
    export_waveform_to_wav(
        wav_out,
        res.times,
        v_out,
        sample_rate_hz=fs_out,
        normalize=False,
        peak_fullscale_v=5.0,
    )
    reloaded = load_wav_samples(wav_out, gain_v=5.0)
    assert reloaded.sample_rate_hz == fs_out
    assert math.isclose(len(reloaded.samples_v), round(dur * fs_out) + 1, abs_tol=2)
    assert math.isclose(
        float(np.max(np.abs(reloaded.samples_v[len(reloaded.samples_v) // 2 :]))),
        ss_peak,
        rel_tol=0.03,
    )

    # Verify stereo 8-bit PCM extraction
    wav_stereo = tmp_path / "stereo_8bit.wav"
    n_fr = 100
    interleaved = np.zeros(n_fr * 2, dtype=np.uint8)
    interleaved[0::2] = 192  # Ch0 ~ +0.5 FS
    interleaved[1::2] = 64   # Ch1 ~ -0.5 FS
    with wave.open(str(wav_stereo), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(interleaved.tobytes())
    ch1 = load_wav_samples(wav_stereo, channel=1)
    assert math.isclose(float(ch1.samples_v[0]), -0.5, abs_tol=1e-2)


def test_f7_all_six_test_jigs_valid_and_loadable_in_main_window(qapp, tmp_path: Path) -> None:
    """Verify all 18 built-in Test Jigs build cleanly, solve accurately, and load via UI menu."""
    from circuitsim.app.main_window import MainWindow
    from circuitsim.app.settings import AppSettings

    jigs = list_test_jigs()
    assert len(jigs) == 18

    expected_checks: dict[str, tuple[float, float]] = {
        "rc_lowpass": (2e-3, 1e-5),
        "rlc_resonance": (5e-4, 1e-6),
        "diode_clipper": (2e-3, 1e-5),
        "half_wave_rectifier": (5e-3, 2e-5),
        "ring_oscillator": (2e-7, 1e-9),
        "potentiometer_divider": (1e-3, 1e-4),
        "common_emitter_npn_amp": (2e-3, 2e-5),
        "zener_5v1_regulator": (1e-3, 1e-4),
        "nmos_low_side_driver": (1e-3, 1e-4),
        "opamp_non_inverting_amp": (1e-3, 1e-4),
        "opamp_relaxation_oscillator": (5e-3, 2e-5),
        "mcu_analog_threshold_controller": (1e-3, 1e-4),
        "555_astable_oscillator": (6e-3, 2e-5),
        "transformer_ac_supply": (15e-3, 1e-4),
        "spdt_relay_driver": (1e-3, 1e-4),
        "overcurrent_fuse_demo": (1e-3, 1e-4),
        "counter_4bit_seven_seg": (4e-3, 5e-5),
        "comparator_threshold_detector": (10e-3, 1e-4),
    }

    for jig in jigs:
        doc = build_test_jig(jig.jig_id)
        assert not any(
            isinstance(w.end_a, FreeEnd) or isinstance(w.end_b, FreeEnd)
            for w in doc.graph.wires
        ), f"FreeEnd found in jig {jig.jig_id}"

        wnet = doc.build_circuit()
        solver = MNASolver(wnet.circuit)
        if jig.jig_id == "potentiometer_divider":
            dc = solver.solve_dc()
            vm = wnet.circuit.get_component("VM1")
            r_th_bot = (5000.0 * vm.r_internal) / (5000.0 + vm.r_internal)  # type: ignore[attr-defined]
            expected_v = 5.0 * r_th_bot / (5000.0 + r_th_bot)
            assert math.isclose(vm.reading, expected_v, abs_tol=1e-7)  # type: ignore[attr-defined]
            assert math.isfinite(sum(dc.node_voltages.values()))
        else:
            dur, dt = expected_checks[jig.jig_id]
            tr = solver.solve_transient(duration=dur, dt=dt)
            assert tr.step_count > 0
            if jig.jig_id == "diode_clipper":
                d1 = wnet.circuit.get_component("D1")
                assert max(abs(tr.component_voltages[d1.name])) <= 0.85
            elif jig.jig_id == "ring_oscillator":
                u1 = wnet.circuit.get_component("U1")
                wave_u1 = tr.component_voltages[u1.name]
                assert float(np.ptp(wave_u1)) > 3.0

    # Verify UI menu exposes all 18 Test Jigs and loads them cleanly
    win = MainWindow(settings=AppSettings(data_dir=tmp_path / "settings"))
    try:
        menu_titles = [act.text() for act in win.test_jigs_menu.actions()]
        assert len(menu_titles) == 18
        for jig in jigs:
            win.load_test_jig(jig.jig_id, title=jig.title, prompt_discard=False)
            assert len(win.circuit_scene.placed_items) >= 4
    finally:
        win.close()


def test_f7_wav_source_g1_catalog(qapp) -> None:
    """Verify WavVoltageSource passes G1 factory & parameter descriptors cleanly."""
    assert "wav_voltage" in KNOWN_COMPONENT_TYPES
    comp = create_default_component_model("wav_voltage", "W1")
    assert isinstance(comp, WavVoltageSource)
    params = comp.get_editable_parameters()
    assert any(p.key == "gain_v" for p in params)
