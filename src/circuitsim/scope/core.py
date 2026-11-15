"""Qt-free 4-channel oscilloscope state machine (RUN / STOP / SINGLE / Roll / Auto Set)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import TYPE_CHECKING, Any
import numpy as np

from circuitsim.scope.capture import RingBuffer, SampleBlock
from circuitsim.scope.channels import (
    CHANNEL_NAMES,
    ChannelConfig,
    ChannelCoupling,
    default_channels,
    process_channel_signal,
    snap_volts_per_div,
)
from circuitsim.scope.math_fft import (
    CursorReadout,
    FFTResult,
    MathConfig,
    MathOp,
    XYTrace,
    compute_cursors,
    compute_fft,
    compute_math_trace,
    compute_xy,
)
from circuitsim.scope.measurements import (
    ChannelMeasurements,
    measure_channel,
    measure_phase_lag_deg,
)
from circuitsim.scope.timebase import (
    DEFAULT_HORIZONTAL_DIVISIONS,
    DEFAULT_POINTS_PER_DIV,
    ResampledTrace,
    TimebaseConfig,
    resample_for_display,
    snap_timebase,
)
from circuitsim.scope.trigger import (
    TriggerConfig,
    TriggerCoupling,
    TriggerEdge,
    TriggerMode,
    find_trigger_crossings,
    select_window_trigger,
)

if TYPE_CHECKING:
    from circuitsim.engine.simulation_session import SimulationSession


class ScopeRunState(str, Enum):
    """Oscilloscope acquisition state."""

    RUN = "run"
    STOP = "stop"
    SINGLE = "single"


class TriggerStatus(str, Enum):
    """Oscilloscope trigger / display status badge."""

    TRIGGERED = "triggered"
    WAITING = "waiting"
    AUTO_FREE_RUN = "auto_free_run"
    ROLL = "roll"
    STOPPED = "stopped"
    NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class ChannelFrameTrace:
    """Processed, display-resampled trace and measurements for one active channel."""

    channel: str
    node_name: str
    config: ChannelConfig
    volts: np.ndarray
    volts_min: np.ndarray
    volts_max: np.ndarray
    display_div_y: np.ndarray
    peak_detect: bool
    measurements: ChannelMeasurements

    def __post_init__(self) -> None:
        self._lock_arrays()

    def _lock_arrays(self) -> None:
        for arr in (self.volts, self.volts_min, self.volts_max, self.display_div_y):
            arr.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock_arrays()


@dataclass(frozen=True)
class ScopeFrame:
    """Immutable, picklable oscilloscope snapshot containing only read-only arrays."""

    run_state: ScopeRunState
    status: TriggerStatus
    status_message: str
    sim_time: float
    t_rel: np.ndarray
    t_abs: np.ndarray
    timebase: TimebaseConfig
    trigger: TriggerConfig
    trigger_time_s: float | None = None
    trigger_edge: TriggerEdge | None = None
    traces: dict[str, ChannelFrameTrace] | None = None
    math_trace: np.ndarray | None = None
    math_config: MathConfig | None = None
    fft: FFTResult | None = None
    xy: XYTrace | None = None
    phase_lag_ch1_ch2_deg: float | None = None
    phase_reason: str | None = None
    cursors: CursorReadout | None = None
    overflow: bool = False
    missed_frames: int = 0

    def __post_init__(self) -> None:
        if self.traces is None:
            object.__setattr__(self, "traces", {})
        self._lock_arrays()

    def _lock_arrays(self) -> None:
        self.t_rel.setflags(write=False)
        self.t_abs.setflags(write=False)
        if self.math_trace is not None:
            self.math_trace.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock_arrays()


class ScopeCore:
    """Qt-free 4-channel oscilloscope signal processing core and acquisition state machine."""

    def __init__(
        self,
        *,
        buffer_capacity: int = 300_000,
        timebase: TimebaseConfig | None = None,
        trigger: TriggerConfig | None = None,
    ) -> None:
        self._channels: dict[str, ChannelConfig] = default_channels()
        self._timebase: TimebaseConfig = timebase if timebase is not None else TimebaseConfig()
        self._trigger: TriggerConfig = trigger if trigger is not None else TriggerConfig()
        self._math: MathConfig = MathConfig()
        self._fft_channel: str | None = None
        self._fft_window: str = "hann"
        self._xy_enabled: bool = False
        self._xy_x: str = "CH1"
        self._xy_y: str = "CH2"
        self._cursor_spec: tuple[str, float, float] | None = None

        self._run_state: ScopeRunState = ScopeRunState.RUN
        self._buffer = RingBuffer(capacity=buffer_capacity)
        self._frozen_frame: ScopeFrame | None = None
        self._waiting_since_sim_s: float | None = None
        self._last_sim_time: float = 0.0
        self._missed_frames: int = 0
        self._overflow: bool = False
        self._session: SimulationSession | None = None

    # --- Properties & configuration mutators ---

    @property
    def run_state(self) -> ScopeRunState:
        return self._run_state

    @property
    def timebase(self) -> TimebaseConfig:
        return self._timebase

    @property
    def trigger(self) -> TriggerConfig:
        return self._trigger

    @property
    def channels(self) -> dict[str, ChannelConfig]:
        return dict(self._channels)

    @property
    def math_config(self) -> MathConfig:
        return self._math

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    @property
    def overflow(self) -> bool:
        return self._overflow

    @overflow.setter
    def overflow(self, value: bool) -> None:
        self._overflow = bool(value)

    def set_channel(self, name: str, **kwargs: Any) -> ChannelConfig:
        if name not in self._channels:
            raise ValueError(f"Unknown channel {name!r}; expected CH1..CH4")
        updated = replace(self._channels[name], **kwargs)
        self._channels[name] = updated
        return updated

    def assign_probe(
        self,
        channel: str,
        node_name: str | None,
        *,
        enabled: bool | None = None,
    ) -> ChannelConfig:
        if enabled is None:
            enabled = node_name is not None
        return self.set_channel(channel, node_name=node_name, enabled=enabled)

    def set_timebase(
        self,
        seconds_per_div: float | None = None,
        *,
        points_per_div: int | None = None,
        position_s: float | None = None,
    ) -> TimebaseConfig:
        updates: dict[str, Any] = {}
        if seconds_per_div is not None:
            updates["seconds_per_div"] = float(seconds_per_div)
        if points_per_div is not None:
            updates["points_per_div"] = int(points_per_div)
        if position_s is not None:
            updates["position_s"] = float(position_s)
        self._timebase = replace(self._timebase, **updates)
        if self._session is not None:
            self._session.set_timebase(
                self._timebase.seconds_per_div,
                points_per_div=self._timebase.points_per_div,
            )
        return self._timebase

    def set_trigger(self, **kwargs: Any) -> TriggerConfig:
        self._trigger = replace(self._trigger, **kwargs)
        return self._trigger

    def set_math(
        self,
        enabled: bool | None = None,
        op: MathOp | str | None = None,
        *,
        source_a: str | None = None,
        source_b: str | None = None,
    ) -> MathConfig:
        updates: dict[str, Any] = {}
        if enabled is not None:
            updates["enabled"] = bool(enabled)
        if op is not None:
            updates["op"] = op
        if source_a is not None:
            updates["source_a"] = source_a
        if source_b is not None:
            updates["source_b"] = source_b
        self._math = replace(self._math, **updates)
        return self._math

    def set_fft(self, channel: str | None, *, window: str = "hann") -> None:
        if channel is not None and channel not in CHANNEL_NAMES and channel != "MATH":
            raise ValueError(f"FFT source must be CH1..CH4, 'MATH', or None, got {channel!r}")
        self._fft_channel = channel
        self._fft_window = window

    def set_xy(
        self, enabled: bool, *, x_channel: str = "CH1", y_channel: str = "CH2"
    ) -> None:
        if x_channel not in CHANNEL_NAMES or y_channel not in CHANNEL_NAMES:
            raise ValueError("XY channels must be in CH1..CH4")
        self._xy_enabled = bool(enabled)
        self._xy_x = x_channel
        self._xy_y = y_channel

    def set_cursors(self, channel: str | None, t1_rel: float = 0.0, t2_rel: float = 0.0) -> None:
        if channel is None:
            self._cursor_spec = None
        else:
            if channel not in CHANNEL_NAMES and channel != "MATH":
                raise ValueError(f"Cursor source must be CH1..CH4, 'MATH', or None, got {channel!r}")
            self._cursor_spec = (channel, float(t1_rel), float(t2_rel))

    # --- Acquisition control ---

    def run(self) -> None:
        self._run_state = ScopeRunState.RUN
        self._frozen_frame = None
        self._waiting_since_sim_s = None

    def stop(self) -> None:
        if self._frozen_frame is None and self._buffer.size > 0:
            self._frozen_frame = self._build_frame(mark_stopped=True)
        self._run_state = ScopeRunState.STOP

    def single(self) -> None:
        self._run_state = ScopeRunState.SINGLE
        self._frozen_frame = None
        self._waiting_since_sim_s = None
        self._trigger = replace(self._trigger, mode=TriggerMode.SINGLE)

    def clear(self) -> None:
        self._buffer.clear()
        self._frozen_frame = None
        self._waiting_since_sim_s = None
        self._last_sim_time = 0.0

    def bind_session(self, session: SimulationSession) -> None:
        """Attach this scope core as the session's sink and couple solver dt."""
        self._session = session
        session.set_timebase(
            self._timebase.seconds_per_div,
            points_per_div=self._timebase.points_per_div,
        )
        session.attach_scope_sink(self)

    def unbind_session(self) -> None:
        if self._session is not None:
            self._session.detach_scope_sink(self)
            self._session.set_timebase(None)
            self._session = None

    # --- Sample sink ingestion ---

    def ingest_block(self, block: SampleBlock) -> None:
        if block.n_steps == 0:
            return
        if self._run_state == ScopeRunState.STOP:
            return

        self._buffer.append_block(block)
        self._last_sim_time = block.t_end

        if self._run_state == ScopeRunState.SINGLE:
            # Capture immediately once a complete trigger window is present
            frame = self._try_capture_single()
            if frame is not None:
                self._frozen_frame = frame
                self._run_state = ScopeRunState.STOP

    def on_sample_block(self, block: SampleBlock) -> None:
        self.ingest_block(block)

    def auto_set(self) -> None:
        """Automatically choose vertical scale, offset, timebase, and trigger level."""
        t_all, d_all, names = self._buffer.get_arrays()
        if len(t_all) < 2 or not names:
            return

        active_chs = [
            ch for ch in (self._channels[n] for n in CHANNEL_NAMES)
            if ch.enabled and ch.node_name in names
        ]
        if not active_chs:
            # Auto-bind CH1 to first non-ground net if available
            candidate_nets = [n for n in names if n != "0"]
            if not candidate_nets:
                return
            self.assign_probe("CH1", candidate_nets[0], enabled=True)
            active_chs = [self._channels["CH1"]]

        best_ch: ChannelConfig | None = None
        best_vpp = -1.0
        best_mid = 0.0
        best_period: float | None = None

        for ch in active_chs:
            assert ch.node_name is not None
            col = d_all[:, names.index(ch.node_name)]
            meas = measure_channel(t_all, col)
            v_mid = 0.5 * (meas.vmax + meas.vmin)
            target_v_per_div = max(meas.vpp / 5.0, 1e-3) if meas.vpp > 1e-6 else max(abs(v_mid) / 2.0, 1.0)
            new_vdiv = snap_volts_per_div(target_v_per_div)
            new_offset = round((-v_mid / new_vdiv) * 2.0) / 2.0
            self.set_channel(ch.name, volts_per_div=new_vdiv, offset_div=new_offset)
            if meas.vpp > best_vpp:
                best_vpp = meas.vpp
                best_ch = ch
                best_mid = v_mid
                best_period = meas.period_s

        if best_period is not None and best_period > 0.0:
            # Target ~2.5 cycles across 10 horizontal divisions
            new_s_per_div = snap_timebase(0.25 * best_period)
            self.set_timebase(new_s_per_div)

        if best_ch is not None:
            self.set_trigger(
                source=best_ch.name,
                level_v=float(best_mid),
                edge=TriggerEdge.RISING,
            )

    # --- Internal frame building ---

    def _window_slice_and_extract(
        self, t_all: np.ndarray, t_start: float, t_end: float
    ) -> tuple[int, int]:
        # Pad 1 sample outside [t_start, t_end] so np.interp at boundaries stays exact
        i0 = max(0, int(np.searchsorted(t_all, t_start, side="left")) - 1)
        i1 = min(len(t_all), int(np.searchsorted(t_all, t_end, side="right")) + 1)
        return i0, i1

    def _try_capture_single(self) -> ScopeFrame | None:
        if self._timebase.is_roll_mode:
            if self._buffer.size >= 2:
                return self._build_frame(mark_stopped=False, force_single_captured=True)
            return None

        t_all, d_all, names = self._buffer.get_arrays()
        if len(t_all) < 2 or not names:
            return None

        trig_ch = self._channels[self._trigger.source]
        if not trig_ch.enabled or trig_ch.node_name not in names:
            return None

        col_idx = names.index(trig_ch.node_name)
        v_trig = d_all[:, col_idx]
        w = self._timebase.window_duration_s
        pre = self._trigger.pre_trigger_fraction
        t_max = float(t_all[-1])
        events = find_trigger_crossings(
            t_all, v_trig, self._trigger, volts_per_div=trig_ch.volts_per_div
        )
        post = (1.0 - pre) * w
        eps = 1e-12 * max(1.0, w)
        if not any(ev.time_s <= t_max - post + eps for ev in events):
            return None
        return self._build_frame(mark_stopped=False, force_single_captured=True)

    def _empty_frame(self, status: TriggerStatus, msg: str) -> ScopeFrame:
        n_pts = self._timebase.display_points
        w = self._timebase.window_duration_s
        if self._timebase.is_roll_mode:
            t_rel = -w + np.arange(n_pts, dtype=np.float64) * self._timebase.dt_scope
        else:
            t_rel = (
                -self._trigger.pre_trigger_fraction * w
                + np.arange(n_pts, dtype=np.float64) * self._timebase.dt_scope
            )
        t_abs = t_rel + self._last_sim_time
        return ScopeFrame(
            run_state=self._run_state,
            status=status,
            status_message=msg,
            sim_time=self._last_sim_time,
            t_rel=t_rel,
            t_abs=t_abs,
            timebase=self._timebase,
            trigger=self._trigger,
            overflow=self._overflow,
            missed_frames=self._missed_frames,
        )

    def produce_frame(self) -> ScopeFrame:
        """Process buffered samples and emit an immutable, read-only `ScopeFrame`."""
        if self._session is not None:
            self._overflow = bool(self._session.overflow)

        if self._run_state == ScopeRunState.STOP:
            if self._frozen_frame is not None:
                return self._frozen_frame
            return self._empty_frame(TriggerStatus.STOPPED, "Stopped")

        return self._build_frame(mark_stopped=False)

    def _build_frame(
        self,
        *,
        mark_stopped: bool = False,
        force_single_captured: bool = False,
    ) -> ScopeFrame:
        tb = self._timebase
        trig = self._trigger
        n_pts = tb.display_points
        w = tb.window_duration_s

        t_all, d_all, names = self._buffer.get_arrays()
        if len(t_all) < 2 or not names:
            return self._empty_frame(TriggerStatus.NO_SIGNAL, "No signal")

        enabled_chs = [
            self._channels[n] for n in CHANNEL_NAMES if self._channels[n].enabled
        ]
        if not any(ch.node_name in names or ch.node_name == "0" for ch in enabled_chs):
            return self._empty_frame(TriggerStatus.NO_SIGNAL, "No signal")

        t_min = float(t_all[0])
        t_max = float(t_all[-1])
        self._last_sim_time = t_max

        trig_time_s: float | None = None
        trig_edge: TriggerEdge | None = None

        if tb.is_roll_mode:
            t_end = t_max
            t_start = t_max - w
            t_rel_origin = t_max
            status = TriggerStatus.ROLL
            status_msg = "Roll"
        else:
            trig_ch = self._channels[trig.source]
            events = []
            if trig_ch.enabled and (trig_ch.node_name in names or trig_ch.node_name == "0"):
                if trig_ch.node_name == "0":
                    v_trig = np.zeros(len(t_all), dtype=np.float64)
                else:
                    v_trig = d_all[:, names.index(trig_ch.node_name)]
                events = find_trigger_crossings(
                    t_all, v_trig, trig, volts_per_div=trig_ch.volts_per_div
                )

            selected = select_window_trigger(
                events,
                t_min,
                t_max,
                w,
                pre_trigger_fraction=trig.pre_trigger_fraction,
            )

            if selected is not None:
                trig_time_s = selected.time_s
                trig_edge = selected.edge
                t_start = selected.time_s - trig.pre_trigger_fraction * w + tb.position_s
                t_end = t_start + w
                t_rel_origin = selected.time_s
                status = TriggerStatus.TRIGGERED
                status_msg = "Triggered" if not force_single_captured else "Single triggered"
                self._waiting_since_sim_s = None
            else:
                if self._waiting_since_sim_s is None:
                    self._waiting_since_sim_s = t_min
                wait_elapsed = t_max - self._waiting_since_sim_s
                if trig.mode == TriggerMode.AUTO and wait_elapsed >= trig.auto_timeout_s:
                    t_end = t_max
                    t_start = t_max - w
                    t_rel_origin = t_start + trig.pre_trigger_fraction * w
                    status = TriggerStatus.AUTO_FREE_RUN
                    status_msg = "Auto (free run)"
                else:
                    return self._empty_frame(TriggerStatus.WAITING, "Waiting for trigger")

        if mark_stopped:
            status = TriggerStatus.STOPPED
            status_msg = "Stopped"

        i0, i1 = self._window_slice_and_extract(t_all, t_start, t_end)
        t_win = t_all[i0:i1]
        in_strict = (t_win >= t_start) & (t_win <= t_end)
        t_strict = t_win[in_strict] if np.any(in_strict) else t_win

        traces: dict[str, ChannelFrameTrace] = {}
        full_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        t_abs_grid: np.ndarray | None = None

        for ch in enabled_chs:
            if ch.node_name is None:
                continue
            if ch.node_name == "0":
                v_raw = np.zeros(len(t_win), dtype=np.float64)
            elif ch.node_name in names:
                v_raw = d_all[i0:i1, names.index(ch.node_name)]
            else:
                continue

            v_full, _ = process_channel_signal(v_raw, t_win, ch)
            v_strict = v_full[in_strict] if np.any(in_strict) else v_full
            meas = measure_channel(t_strict, v_strict)
            resamp: ResampledTrace = resample_for_display(
                t_win, v_full, t_start, t_end, num_points=n_pts
            )
            if t_abs_grid is None:
                t_abs_grid = resamp.t

            disp_y = (resamp.y / ch.volts_per_div) + ch.offset_div
            traces[ch.name] = ChannelFrameTrace(
                channel=ch.name,
                node_name=ch.node_name,
                config=ch,
                volts=resamp.y,
                volts_min=resamp.y_min,
                volts_max=resamp.y_max,
                display_div_y=disp_y,
                peak_detect=resamp.peak_detect,
                measurements=meas,
            )
            full_signals[ch.name] = (t_strict, v_strict)

        if t_abs_grid is None:
            return self._empty_frame(TriggerStatus.NO_SIGNAL, "No signal")

        t_rel_grid = t_abs_grid - t_rel_origin

        # Derived Math trace
        math_trace: np.ndarray | None = None
        if (
            self._math.enabled
            and self._math.source_a in traces
            and self._math.source_b in traces
        ):
            math_trace = compute_math_trace(
                traces[self._math.source_a].volts,
                traces[self._math.source_b].volts,
                self._math.op,
            )

        # Phase lag CH1 vs CH2 using full-res window when both enabled
        phase_lag: float | None = None
        phase_reason: str | None = None
        if "CH1" in full_signals and "CH2" in full_signals:
            t_c1, v_c1 = full_signals["CH1"]
            _t_c2, v_c2 = full_signals["CH2"]
            if len(v_c1) == len(v_c2) and len(v_c1) >= 3:
                phase_lag, phase_reason = measure_phase_lag_deg(t_c1, v_c1, v_c2)

        # FFT (use full-resolution window when <= 8192 pts for wide Nyquist, else display trace)
        fft_res: FFTResult | None = None
        if self._fft_channel is not None:
            if self._fft_channel == "MATH" and math_trace is not None:
                fft_res = compute_fft(t_abs_grid, math_trace, window=self._fft_window)
            elif self._fft_channel in full_signals:
                t_f, v_f = full_signals[self._fft_channel]
                if len(t_f) <= 8192:
                    fft_res = compute_fft(t_f, v_f, window=self._fft_window)
                else:
                    fft_res = compute_fft(
                        t_abs_grid,
                        traces[self._fft_channel].volts,
                        window=self._fft_window,
                    )

        # XY trace
        xy_res: XYTrace | None = None
        if self._xy_enabled and self._xy_x in traces and self._xy_y in traces:
            xy_res = compute_xy(
                traces[self._xy_x].volts,
                traces[self._xy_y].volts,
                x_channel=self._xy_x,
                y_channel=self._xy_y,
            )

        # Cursors (evaluated against `t_rel_grid` in display units or seconds)
        cursors_res: CursorReadout | None = None
        if self._cursor_spec is not None:
            c_src, c_t1, c_t2 = self._cursor_spec
            if c_src == "MATH" and math_trace is not None:
                cursors_res = compute_cursors(t_rel_grid, math_trace, c_t1, c_t2)
            elif c_src in traces:
                cursors_res = compute_cursors(t_rel_grid, traces[c_src].volts, c_t1, c_t2)

        if self._overflow and "slow" not in status_msg.lower():
            status_msg = f"{status_msg} | Sim slower than real-time"

        out_state = ScopeRunState.STOP if (mark_stopped or force_single_captured) else self._run_state
        return ScopeFrame(
            run_state=out_state,
            status=status,
            status_message=status_msg,
            sim_time=t_max,
            t_rel=t_rel_grid,
            t_abs=t_abs_grid,
            timebase=tb,
            trigger=trig,
            trigger_time_s=trig_time_s,
            trigger_edge=trig_edge,
            traces=traces,
            math_trace=math_trace,
            math_config=self._math if self._math.enabled else None,
            fft=fft_res,
            xy=xy_res,
            phase_lag_ch1_ch2_deg=phase_lag,
            phase_reason=phase_reason,
            cursors=cursors_res,
            overflow=self._overflow,
            missed_frames=self._missed_frames,
        )
