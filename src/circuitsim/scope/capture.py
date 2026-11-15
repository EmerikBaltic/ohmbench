"""Qt-free oscilloscope sample blocks, sink protocol, and vectorized ring buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ScopeSink(Protocol):
    """Protocol for receivers of immutable oscilloscope `SampleBlock` chunks."""

    def on_sample_block(self, block: "SampleBlock") -> None: ...


@dataclass(frozen=True)
class SampleBlock:
    """Immutable, picklable batch of node voltages emitted by a solver chunk."""

    t0: float
    dt: float
    node_names: tuple[str, ...]
    data: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.t0):
            raise ValueError(f"t0 must be finite, got {self.t0!r}")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be > 0 and finite, got {self.dt!r}")
        names = tuple(self.node_names)
        object.__setattr__(self, "node_names", names)

        arr = np.asarray(self.data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"SampleBlock.data must be 2D, got shape {arr.shape!r}")
        if arr.shape[1] != len(names):
            raise ValueError(
                f"SampleBlock column count {arr.shape[1]} != len(node_names) {len(names)}"
            )
        if arr is not self.data:
            object.__setattr__(self, "data", arr)
        arr.setflags(write=False)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.data.setflags(write=False)

    @property
    def n_steps(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.data.shape[1])

    @property
    def t_end(self) -> float:
        return self.t0 + max(0, self.n_steps - 1) * self.dt

    def timestamps(self) -> np.ndarray:
        """Return 1D float64 array of step timestamps in seconds."""
        t = self.t0 + np.arange(self.n_steps, dtype=np.float64) * self.dt
        t.setflags(write=False)
        return t

    def column(self, node_name: str) -> np.ndarray:
        """Return 1D read-only voltage series for ``node_name`` (ground '0' defaults to 0 V)."""
        try:
            idx = self.node_names.index(node_name)
        except ValueError as exc:
            if node_name == "0":
                z = np.zeros(self.n_steps, dtype=np.float64)
                z.setflags(write=False)
                return z
            raise KeyError(f"Node {node_name!r} not present in SampleBlock") from exc
        return self.data[:, idx]


class RingBuffer:
    """Thread-safe, fixed-capacity vectorized circular buffer of multi-node samples."""

    def __init__(self, capacity: int = 200_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity!r}")
        self._capacity: int = int(capacity)
        self._node_names: tuple[str, ...] = ()
        self._node_to_col: dict[str, int] = {}
        self._t: np.ndarray = np.empty(0, dtype=np.float64)
        self._data: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._head: int = 0
        self._size: int = 0
        self._total_written: int = 0
        self._last_dt: float = 0.0
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def node_names(self) -> tuple[str, ...]:
        with self._lock:
            return self._node_names

    @property
    def last_dt(self) -> float:
        with self._lock:
            return self._last_dt

    def clear(self) -> None:
        with self._lock:
            self._head = 0
            self._size = 0
            self._total_written = 0
            self._last_dt = 0.0

    def _allocate_for_nodes(self, node_names: tuple[str, ...]) -> None:
        self._node_names = node_names
        self._node_to_col = {name: i for i, name in enumerate(node_names)}
        self._t = np.empty(self._capacity, dtype=np.float64)
        self._data = np.empty((self._capacity, len(node_names)), dtype=np.float64)
        self._head = 0
        self._size = 0

    def append_block(self, block: SampleBlock) -> None:
        """Append all samples of `block` using O(N) vectorized NumPy slices."""
        n = block.n_steps
        if n == 0:
            return

        t_block = block.timestamps()
        with self._lock:
            if block.node_names != self._node_names:
                self._allocate_for_nodes(block.node_names)

            cap = self._capacity
            if n >= cap:
                tail = block.data[-cap:]
                self._t[:] = t_block[-cap:]
                if len(self._node_names) > 0:
                    self._data[:, :] = tail
                self._head = 0
                self._size = cap
            else:
                head = self._head
                first = min(n, cap - head)
                self._t[head : head + first] = t_block[:first]
                if len(self._node_names) > 0:
                    self._data[head : head + first, :] = block.data[:first, :]
                if n > first:
                    rem = n - first
                    self._t[:rem] = t_block[first:]
                    if len(self._node_names) > 0:
                        self._data[:rem, :] = block.data[first:, :]
                self._head = (head + n) % cap
                self._size = min(cap, self._size + n)

            self._total_written += n
            self._last_dt = block.dt

    def on_sample_block(self, block: SampleBlock) -> None:
        self.append_block(block)

    def get_arrays(
        self, max_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return contiguous chronological ``(t, data, node_names)`` arrays."""
        with self._lock:
            if self._size == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty((0, len(self._node_names)), dtype=np.float64),
                    self._node_names,
                )

            cap = self._capacity
            n_want = self._size if max_samples is None else min(self._size, max(0, int(max_samples)))
            if n_want == 0:
                return (
                    np.empty(0, dtype=np.float64),
                    np.empty((0, len(self._node_names)), dtype=np.float64),
                    self._node_names,
                )

            start = (self._head - n_want) % cap
            if start < self._head:
                t_out = self._t[start : self._head].copy()
                d_out = self._data[start : self._head, :].copy()
            else:
                # Wrapped across end of buffer
                t_out = np.concatenate((self._t[start:], self._t[: self._head]))
                d_out = np.concatenate((self._data[start:, :], self._data[: self._head, :]), axis=0)
            return t_out, d_out, self._node_names


@dataclass
class SampleRecorder:
    """Utility sink that records incoming `SampleBlock` batches in chronological order."""

    blocks: list[SampleBlock] = field(default_factory=list)

    def on_sample_block(self, block: SampleBlock) -> None:
        self.blocks.append(block)

    def clear(self) -> None:
        self.blocks.clear()

    def concatenated(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Concatenate all recorded blocks into ``(t, data, node_names)``."""
        if not self.blocks:
            return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64), ()
        names = self.blocks[-1].node_names
        t_all = np.concatenate([b.timestamps() for b in self.blocks])
        d_all = np.concatenate([b.data for b in self.blocks], axis=0)
        return t_all, d_all, names
