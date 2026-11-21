"""Background QThread simulation worker wrapping the pure-Python `SimulationSession`."""

from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtCore import (
    QCoreApplication,
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)

from circuitsim.canvas.connectivity import (
    PlacedComponentSpec,
    PointPx,
    WireSpec,
)
from circuitsim.engine.simulation_session import (
    SimulationFrame,
    SimulationSession,
    SimulationSpeed,
    SimulationState,
)


class _SessionThreadWorker(QObject):
    """Worker object living inside the background `QThread`."""

    frame_emitted = Signal(object)  # Emits immutable SimulationFrame
    sample_block_emitted = Signal(object)  # Emits immutable SampleBlock chunks

    def __init__(self) -> None:
        super().__init__()
        self.session: SimulationSession = SimulationSession()
        self.session.attach_scope_sink(self._on_sample_block)
        self._tick_timer: QTimer | None = None
        self.tick_interval_ms: int = 33  # ~30 FPS UI update rate

    def _on_sample_block(self, block: object) -> None:
        self.sample_block_emitted.emit(block)

    def _ensure_timer(self) -> QTimer:
        if self._tick_timer is None:
            self._tick_timer = QTimer(self)
            self._tick_timer.setInterval(self.tick_interval_ms)
            self._tick_timer.timeout.connect(self._on_tick)
        return self._tick_timer

    @Slot(object, object, object, int)
    def handle_load_schematic(
        self,
        placed_specs: object,
        wire_specs: object,
        junctions: object,
        min_gen: int = 0,
    ) -> None:
        timer = self._ensure_timer()
        frame = self.session.load_schematic(
            placed_specs,  # type: ignore[arg-type]
            wire_specs,  # type: ignore[arg-type]
            junctions,  # type: ignore[arg-type]
        )
        self.session._generation = max(self.session._generation, int(min_gen))
        if frame.generation < self.session._generation:
            frame = self.session.build_frame(node_voltages=frame.node_voltages)
        if frame.state == SimulationState.RUNNING:
            timer.start()
        else:
            timer.stop()
        self.frame_emitted.emit(frame)

    @Slot(object, int)
    def handle_load_connectivity(self, connectivity: object, min_gen: int = 0) -> None:
        timer = self._ensure_timer()
        frame = self.session.load_connectivity(connectivity)  # type: ignore[arg-type]
        self.session._generation = max(self.session._generation, int(min_gen))
        if frame.generation < self.session._generation:
            frame = self.session.build_frame(node_voltages=frame.node_voltages)
        if frame.state == SimulationState.RUNNING:
            timer.start()
        else:
            timer.stop()
        self.frame_emitted.emit(frame)

    @Slot()
    def handle_start(self) -> None:
        timer = self._ensure_timer()
        frame = self.session.start()
        if frame.state == SimulationState.RUNNING:
            timer.start()
        else:
            timer.stop()
        self.frame_emitted.emit(frame)

    @Slot()
    def handle_pause(self) -> None:
        timer = self._ensure_timer()
        timer.stop()
        frame = self.session.pause()
        self.frame_emitted.emit(frame)

    @Slot()
    def handle_stop(self) -> None:
        timer = self._ensure_timer()
        timer.stop()
        frame = self.session.stop()
        self.frame_emitted.emit(frame)

    @Slot(str)
    def handle_set_speed(self, speed_value: str) -> None:
        self.session.speed = SimulationSpeed(speed_value)

    @Slot(object, int)
    def handle_set_timebase(self, seconds_per_div: object, points_per_div: int = 100) -> None:
        sec_div = float(seconds_per_div) if seconds_per_div is not None else None
        self.session.set_timebase(sec_div, points_per_div=int(points_per_div))

    @Slot(str, int)
    def handle_toggle_switch(self, switch_name: str, min_gen: int = 0) -> None:
        try:
            frame = self.session.toggle_switch(switch_name)
            self.session._generation = max(self.session._generation, int(min_gen))
            if frame.generation < self.session._generation:
                frame = self.session.build_frame(node_voltages=frame.node_voltages)
            self.frame_emitted.emit(frame)
        except Exception:
            pass

    @Slot(str, int)
    def handle_reset_fuse(self, fuse_name: str, min_gen: int = 0) -> None:
        try:
            frame = self.session.reset_fuse(fuse_name)
            self.session._generation = max(self.session._generation, int(min_gen))
            if frame.generation < self.session._generation:
                frame = self.session.build_frame(node_voltages=frame.node_voltages)
            self.frame_emitted.emit(frame)
        except Exception:
            pass

    @Slot(str, str, object, int)
    def handle_update_parameter(
        self, comp_name: str, param_key: str, raw_value: object, min_gen: int = 0
    ) -> None:
        try:
            frame = self.session.update_component_parameter(
                comp_name, param_key, raw_value  # type: ignore[arg-type]
            )
            self.session._generation = max(self.session._generation, int(min_gen))
            if frame.generation < self.session._generation:
                frame = self.session.build_frame(node_voltages=frame.node_voltages)
            self.frame_emitted.emit(frame)
        except Exception:
            pass

    @Slot()
    def _on_tick(self) -> None:
        if self.session.state != SimulationState.RUNNING:
            if self._tick_timer is not None:
                self._tick_timer.stop()
            return

        wall_dt = self.tick_interval_ms / 1000.0
        target_sim_advance = wall_dt * self.session.speed.multiplier
        dt = max(1e-9, self.session.dt)
        steps = max(1, min(2500, int(round(target_sim_advance / dt))))
        frame = self.session.step_chunk(steps=steps, dt=dt)
        if frame.state != SimulationState.RUNNING and self._tick_timer is not None:
            self._tick_timer.stop()
        self.frame_emitted.emit(frame)

    @Slot()
    def cleanup_for_shutdown(self) -> None:
        """Stop timer inside worker's thread and return worker affinity to main thread."""
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer.deleteLater()
            self._tick_timer = None
        self.session.stop()
        app = QCoreApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            self.moveToThread(app.thread())


class SimulationWorkerController(QObject):
    """Main-thread facade managing the background `QThread` simulation worker.

    All communication with `_SessionThreadWorker` occurs via queued Qt signals
    carrying immutable/picklable dataclasses so the GUI event loop never blocks.
    """

    frame_ready = Signal(object)  # Emits SimulationFrame on the main thread
    sample_block_ready = Signal(object)  # Emits SampleBlock chunks on the main thread

    # Internal queued signals to the worker thread
    _sig_load = Signal(object, object, object, int)
    _sig_load_connectivity = Signal(object, int)
    _sig_start = Signal()
    _sig_pause = Signal()
    _sig_stop = Signal()
    _sig_speed = Signal(str)
    _sig_set_timebase = Signal(object, int)
    _sig_toggle_switch = Signal(str, int)
    _sig_reset_fuse = Signal(str, int)
    _sig_update_param = Signal(str, str, object, int)
    _sig_cleanup = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _SessionThreadWorker()
        self._worker_moved_to_thread: bool = False
        self._is_shut_down: bool = False
        self._min_generation: int = 0

        self._sig_load.connect(
            self._worker.handle_load_schematic, Qt.ConnectionType.QueuedConnection
        )
        self._sig_load_connectivity.connect(
            self._worker.handle_load_connectivity, Qt.ConnectionType.QueuedConnection
        )
        self._sig_start.connect(
            self._worker.handle_start, Qt.ConnectionType.QueuedConnection
        )
        self._sig_pause.connect(
            self._worker.handle_pause, Qt.ConnectionType.QueuedConnection
        )
        self._sig_stop.connect(
            self._worker.handle_stop, Qt.ConnectionType.QueuedConnection
        )
        self._sig_speed.connect(
            self._worker.handle_set_speed, Qt.ConnectionType.QueuedConnection
        )
        self._sig_set_timebase.connect(
            self._worker.handle_set_timebase, Qt.ConnectionType.QueuedConnection
        )
        self._sig_toggle_switch.connect(
            self._worker.handle_toggle_switch, Qt.ConnectionType.QueuedConnection
        )
        self._sig_reset_fuse.connect(
            self._worker.handle_reset_fuse, Qt.ConnectionType.QueuedConnection
        )
        self._sig_update_param.connect(
            self._worker.handle_update_parameter, Qt.ConnectionType.QueuedConnection
        )
        self._sig_cleanup.connect(
            self._worker.cleanup_for_shutdown,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        self._worker.frame_emitted.connect(
            self._on_worker_frame, Qt.ConnectionType.QueuedConnection
        )
        self._worker.sample_block_emitted.connect(
            self._on_worker_sample_block, Qt.ConnectionType.QueuedConnection
        )
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def _ensure_thread_running(self) -> None:
        if self._is_shut_down:
            return
        if not self._worker_moved_to_thread:
            self._worker.moveToThread(self._thread)
            self._worker_moved_to_thread = True
        if not self._thread.isRunning():
            self._thread.start()

    @Slot(object)
    def _on_worker_frame(self, frame: object) -> None:
        """Drop stale pre-edit/pre-reset frames using the monotonic generation counter."""
        if getattr(frame, "generation", 0) < self._min_generation:
            return
        self.frame_ready.emit(frame)

    @Slot(object)
    def _on_worker_sample_block(self, block: object) -> None:
        """Forward high-rate oscilloscope SampleBlock chunks to main-thread listeners."""
        if not self._is_shut_down:
            self.sample_block_ready.emit(block)

    def load_schematic(
        self,
        placed_components: Sequence[PlacedComponentSpec],
        wires: Sequence[WireSpec],
        explicit_junctions: Iterable[PointPx] = (),
    ) -> None:
        """Send an updated schematic snapshot to the worker thread."""
        self._ensure_thread_running()
        self._min_generation += 1
        self._sig_load.emit(
            tuple(placed_components),
            tuple(wires),
            tuple(explicit_junctions),
            self._min_generation,
        )

    def load_connectivity(self, connectivity: object) -> None:
        """Send a pre-extracted `ConnectivityResult` (built from `WireGraph`) to the worker."""
        self._ensure_thread_running()
        self._min_generation += 1
        self._sig_load_connectivity.emit(connectivity, self._min_generation)

    def start(self) -> None:
        """Start or resume simulation on the worker thread."""
        self._ensure_thread_running()
        self._sig_start.emit()

    def pause(self) -> None:
        """Pause simulation on the worker thread."""
        if self._thread.isRunning():
            self._sig_pause.emit()

    def stop(self) -> None:
        """Stop simulation and reset states on the worker thread."""
        if self._thread.isRunning():
            self._sig_stop.emit()

    def set_speed(self, speed: SimulationSpeed) -> None:
        """Change target simulation speed on the worker thread."""
        self._ensure_thread_running()
        self._sig_speed.emit(speed.value)

    def set_timebase(
        self, seconds_per_div: float | None, points_per_div: int = 100
    ) -> None:
        """Couple worker solver dt to an active DSO timebase without resetting state."""
        self._ensure_thread_running()
        self._sig_set_timebase.emit(seconds_per_div, int(points_per_div))

    def toggle_switch(self, switch_name: str) -> None:
        """Toggle a switch live on the worker thread."""
        self._ensure_thread_running()
        self._min_generation += 1
        self._sig_toggle_switch.emit(switch_name, self._min_generation)

    def reset_fuse(self, fuse_name: str) -> None:
        """Reset a blown fuse live on the worker thread and ignore stale pre-reset frames."""
        self._ensure_thread_running()
        self._min_generation += 1
        self._sig_reset_fuse.emit(fuse_name, self._min_generation)

    def update_component_parameter(
        self, comp_name: str, param_key: str, value: object
    ) -> None:
        """Update a component parameter live on the worker thread."""
        self._ensure_thread_running()
        self._min_generation += 1
        self._sig_update_param.emit(comp_name, param_key, value, self._min_generation)

    def shutdown(self) -> None:
        """Cleanly stop the worker timer and join the background `QThread`."""
        if self._is_shut_down:
            return
        self._is_shut_down = True

        app = QCoreApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.disconnect(self.shutdown)
            except (RuntimeError, TypeError):
                pass

        if self._worker is not None:
            try:
                self._worker.frame_emitted.disconnect(self._on_worker_frame)
            except (RuntimeError, TypeError):
                pass
            try:
                self._worker.sample_block_emitted.disconnect(self._on_worker_sample_block)
            except (RuntimeError, TypeError):
                pass

        if self._thread.isRunning():
            if self._worker_moved_to_thread and QThread.currentThread() is not self._thread:
                try:
                    self._sig_cleanup.emit()
                except RuntimeError:
                    pass
            self._thread.quit()
            self._thread.wait(2000)
        elif self._worker is not None and not self._worker_moved_to_thread:
            self._worker.cleanup_for_shutdown()

        if self._worker is not None:
            try:
                self._worker.deleteLater()
            except RuntimeError:
                pass
