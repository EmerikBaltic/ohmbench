"""Horizontals, timebase presets, solver step coupling, and peak-detect resampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

DEFAULT_HORIZONTAL_DIVISIONS = 10
DEFAULT_POINTS_PER_DIV = 100
ROLL_MODE_THRESHOLD_S = 0.1  # >= 100 ms/div enters Roll mode

TIMEBASE_STEPS_S: tuple[float, ...] = (
    10e-9,
    20e-9,
    50e-9,
    100e-9,
    200e-9,
    500e-9,
    1e-6,
    2e-6,
    5e-6,
    10e-6,
    20e-6,
    50e-6,
    100e-6,
    200e-6,
    500e-6,
    1e-3,
    2e-3,
    5e-3,
    10e-3,
    20e-3,
    50e-3,
    100e-3,
    200e-3,
    500e-3,
    1.0,
    2.0,
    5.0,
    10.0,
)


def snap_timebase(seconds_per_div: float) -> float:
    """Return the nearest standard 1-2-5 timebase step in seconds/div."""
    val = float(seconds_per_div)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"seconds_per_div must be positive and finite, got {seconds_per_div!r}")
    return min(TIMEBASE_STEPS_S, key=lambda step: abs(math.log10(step) - math.log10(val)))


def step_timebase(seconds_per_div: float, delta: int) -> float:
    """Move `delta` steps along `TIMEBASE_STEPS_S` from the nearest preset."""
    snapped = snap_timebase(seconds_per_div)
    idx = TIMEBASE_STEPS_S.index(snapped)
    new_idx = max(0, min(len(TIMEBASE_STEPS_S) - 1, idx + int(delta)))
    return TIMEBASE_STEPS_S[new_idx]


@dataclass(frozen=True)
class TimebaseConfig:
    """Horizontal timebase configuration (always Qt-free and immutable)."""

    seconds_per_div: float = 1e-3
    points_per_div: int = DEFAULT_POINTS_PER_DIV
    horizontal_divisions: int = DEFAULT_HORIZONTAL_DIVISIONS
    position_s: float = 0.0
    roll_threshold_s: float = ROLL_MODE_THRESHOLD_S

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds_per_div) or self.seconds_per_div <= 0.0:
            raise ValueError(f"seconds_per_div must be > 0, got {self.seconds_per_div!r}")
        if self.points_per_div < 1:
            raise ValueError(f"points_per_div must be >= 1, got {self.points_per_div!r}")
        if self.horizontal_divisions < 1:
            raise ValueError(
                f"horizontal_divisions must be >= 1, got {self.horizontal_divisions!r}"
            )
        if not math.isfinite(self.position_s):
            raise ValueError(f"position_s must be finite, got {self.position_s!r}")
        if not math.isfinite(self.roll_threshold_s) or self.roll_threshold_s <= 0.0:
            raise ValueError(f"roll_threshold_s must be > 0, got {self.roll_threshold_s!r}")

    @property
    def display_points(self) -> int:
        return self.points_per_div * self.horizontal_divisions

    @property
    def window_duration_s(self) -> float:
        return self.seconds_per_div * self.horizontal_divisions

    @property
    def dt_scope(self) -> float:
        return self.seconds_per_div / self.points_per_div

    @property
    def is_roll_mode(self) -> bool:
        return self.seconds_per_div >= self.roll_threshold_s


def compute_coupled_solver_dt(
    timebase: float | TimebaseConfig,
    dt_circuit: float,
    *,
    points_per_div: int = DEFAULT_POINTS_PER_DIV,
) -> float:
    """Couple scope sample period with circuit time constant: ``min(dt_scope, dt_circuit)``."""
    if isinstance(timebase, TimebaseConfig):
        dt_scope = timebase.dt_scope
    else:
        tb_val = float(timebase)
        if not math.isfinite(tb_val) or tb_val <= 0.0:
            raise ValueError(f"timebase must be positive and finite, got {timebase!r}")
        if points_per_div < 1:
            raise ValueError(f"points_per_div must be >= 1, got {points_per_div!r}")
        dt_scope = tb_val / points_per_div

    circ_val = float(dt_circuit)
    if not math.isfinite(circ_val) or circ_val <= 0.0:
        raise ValueError(f"dt_circuit must be positive and finite, got {dt_circuit!r}")
    return min(dt_scope, circ_val)


@dataclass(frozen=True)
class ResampledTrace:
    """Display-resolution trace carrying both interpolated samples and peak-detect extrema."""

    t: np.ndarray
    y: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    peak_detect: bool

    def __post_init__(self) -> None:
        self._lock_arrays()

    def _lock_arrays(self) -> None:
        for arr in (self.t, self.y, self.y_min, self.y_max):
            arr.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock_arrays()


def resample_for_display(
    t: np.ndarray,
    y: np.ndarray,
    t_start: float,
    t_end: float,
    num_points: int = DEFAULT_HORIZONTAL_DIVISIONS * DEFAULT_POINTS_PER_DIV,
) -> ResampledTrace:
    """Resample ``(t, y)`` onto ``[t_start, t_end)`` with ``num_points`` buckets.

    When multiple raw samples fall into one display bucket (``dt_circuit < dt_scope``),
    vectorized ``np.minimum.reduceat`` / ``np.maximum.reduceat`` computes exact per-bucket
    min/max extrema (peak-detect) so narrow spikes or glitches between display grid ticks
    are never smoothed away.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points!r}")
    if not math.isfinite(t_start) or not math.isfinite(t_end) or t_end <= t_start:
        raise ValueError(f"Invalid display window [{t_start!r}, {t_end!r}]")

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if t_arr.ndim != 1 or y_arr.ndim != 1 or len(t_arr) != len(y_arr):
        raise ValueError("t and y must be 1D arrays of identical length")

    dt_bin = (t_end - t_start) / num_points
    t_grid = t_start + np.arange(num_points, dtype=np.float64) * dt_bin

    if len(t_arr) == 0:
        zeros = np.zeros(num_points, dtype=np.float64)
        return ResampledTrace(
            t=t_grid,
            y=zeros,
            y_min=zeros.copy(),
            y_max=zeros.copy(),
            peak_detect=False,
        )

    y_interp = np.interp(t_grid, t_arr, y_arr)
    y_min = y_interp.copy()
    y_max = y_interp.copy()

    i0 = int(np.searchsorted(t_arr, t_start, side="left"))
    i1 = int(np.searchsorted(t_arr, t_end, side="right"))
    peak_detect = False

    if i1 > i0:
        t_in = t_arr[i0:i1]
        y_in = y_arr[i0:i1]
        bin_idx = np.clip(np.floor((t_in - t_start) / dt_bin).astype(np.intp), 0, num_points - 1)

        is_new = np.empty(len(bin_idx), dtype=bool)
        is_new[0] = True
        if len(bin_idx) > 1:
            is_new[1:] = bin_idx[1:] != bin_idx[:-1]
        first_idx = np.flatnonzero(is_new)
        unique_bins = bin_idx[first_idx]

        b_min = np.minimum.reduceat(y_in, first_idx)
        b_max = np.maximum.reduceat(y_in, first_idx)
        y_min[unique_bins] = np.minimum(y_min[unique_bins], b_min)
        y_max[unique_bins] = np.maximum(y_max[unique_bins], b_max)

        counts = np.diff(np.append(first_idx, len(bin_idx)))
        peak_detect = bool(len(counts) > 0 and counts.max() > 1)

    return ResampledTrace(
        t=t_grid,
        y=y_interp,
        y_min=y_min,
        y_max=y_max,
        peak_detect=peak_detect,
    )
