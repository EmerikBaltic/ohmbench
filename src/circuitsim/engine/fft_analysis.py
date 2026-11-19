"""Qt-free single-sided Hann-windowed FFT spectrum & THD% analyzer (H8)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class FFTSpectrumResult:
    """Single-sided frequency spectrum and Total Harmonic Distortion (THD) metrics."""

    freqs_hz: tuple[float, ...]
    magnitude_dbv: tuple[float, ...]
    magnitude_v_rms: tuple[float, ...]
    fundamental_freq_hz: float
    fundamental_rms_v: float
    peak_dbv: float
    thd_percent: float


def compute_fft_spectrum(
    times: Sequence[float] | np.ndarray,
    voltages: Sequence[float] | np.ndarray,
    *,
    max_harmonics: int = 7,
) -> FFTSpectrumResult:
    """Compute a Hann-windowed single-sided FFT spectrum (dBV) and THD%.

    Handles non-uniform time grids by linear interpolation, subtracts DC mean,
    compensates Hann window coherent gain/energy, and integrates mainlobe energy
    around the fundamental and each harmonic `2*f1 .. max_harmonics*f1`.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(voltages, dtype=np.float64)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v):
        raise ValueError("times and voltages must be 1D sequences of identical length.")
    if len(t) < 8:
        raise ValueError("Need at least 8 samples to compute an FFT spectrum.")

    t_span = float(t[-1] - t[0])
    if t_span <= 0.0:
        raise ValueError("Time span must be strictly positive.")

    dt_diffs = np.diff(t)
    dt_mean = float(np.mean(dt_diffs))
    if np.max(np.abs(dt_diffs - dt_mean)) > 1e-4 * max(abs(dt_mean), 1e-15):
        t_uniform = np.linspace(float(t[0]), float(t[-1]), len(t), dtype=np.float64)
        v_uniform = np.interp(t_uniform, t, v)
        dt = float(t_uniform[1] - t_uniform[0])
    else:
        v_uniform = v
        dt = dt_mean

    n = len(v_uniform)
    v_ac = v_uniform - float(np.mean(v_uniform))

    # Periodic Hann window
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n, dtype=np.float64) / float(n))
    cg = 0.5
    win_sq_sum = float(np.sum(window * window))

    spec = np.fft.rfft(v_ac * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Peak voltage per bin -> RMS
    mag_peak = (2.0 * np.abs(spec)) / (float(n) * cg)
    mag_peak[0] *= 0.5
    if n % 2 == 0 and len(mag_peak) > 1:
        mag_peak[-1] *= 0.5
    mag_rms = mag_peak / math.sqrt(2.0)
    mag_dbv = 20.0 * np.log10(np.maximum(mag_rms, 1e-12))

    # Find dominant AC bin (ignoring DC bin 0)
    if len(mag_rms) <= 1 or float(np.max(mag_rms[1:])) < 1e-12:
        return FFTSpectrumResult(
            freqs_hz=tuple(float(x) for x in freqs),
            magnitude_dbv=tuple(float(x) for x in mag_dbv),
            magnitude_v_rms=tuple(float(x) for x in mag_rms),
            fundamental_freq_hz=0.0,
            fundamental_rms_v=0.0,
            peak_dbv=-240.0,
            thd_percent=0.0,
        )

    k1 = int(np.argmax(mag_rms[1:])) + 1
    # Parabolic interpolation on linear amplitude/log magnitude for sub-bin fundamental accuracy
    delta = 0.0
    if 1 <= k1 < len(mag_dbv) - 1:
        y_m, y_0, y_p = float(mag_dbv[k1 - 1]), float(mag_dbv[k1]), float(mag_dbv[k1 + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if abs(denom) > 1e-12:
            delta = max(-0.5, min(0.5, 0.5 * (y_m - y_p) / denom))
    f1 = (float(k1) + delta) * df

    # Integrate mainlobe RMS energy safely via Parseval's theorem
    lobe_r = min(2, max(1, k1 // 3))
    n_bins = len(spec)

    def _lobe_rms_sq(center_idx: int, exclude_lo: int = -1, exclude_hi: int = -1) -> float:
        lo = max(1, center_idx - lobe_r)
        hi = min(n_bins - 1, center_idx + lobe_r)
        total_sq = 0.0
        for b in range(lo, hi + 1):
            if exclude_lo <= b <= exclude_hi:
                continue
            scale = 2.0 if (0 < b < (n // 2 if n % 2 == 0 else n)) else 1.0
            total_sq += (scale * (abs(complex(spec[b])) ** 2)) / (float(n) * max(win_sq_sum, 1e-30))
        return total_sq

    fund_lo = max(1, k1 - lobe_r)
    fund_hi = min(n_bins - 1, k1 + lobe_r)
    v1_rms_sq = _lobe_rms_sq(k1)
    v1_rms = math.sqrt(max(v1_rms_sq, 1e-30))
    peak_dbv = 20.0 * math.log10(max(v1_rms, 1e-12))

    harm_rms_sq = 0.0
    for h in range(2, max(2, int(max_harmonics)) + 1):
        f_h = h * f1
        k_h = int(round(f_h / df))
        if k_h >= n_bins:
            break
        harm_rms_sq += _lobe_rms_sq(k_h, exclude_lo=fund_lo, exclude_hi=fund_hi)

    thd_pct = 100.0 * math.sqrt(max(harm_rms_sq, 0.0)) / max(v1_rms, 1e-15)
    return FFTSpectrumResult(
        freqs_hz=tuple(float(x) for x in freqs),
        magnitude_dbv=tuple(float(x) for x in mag_dbv),
        magnitude_v_rms=tuple(float(x) for x in mag_rms),
        fundamental_freq_hz=float(f1),
        fundamental_rms_v=float(v1_rms),
        peak_dbv=float(peak_dbv),
        thd_percent=float(thd_pct),
    )
