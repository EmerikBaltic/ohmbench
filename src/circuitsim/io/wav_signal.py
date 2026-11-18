"""Qt-free PCM WAV audio signal loader, generator, and uniform-rate resampler."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
import wave
import numpy as np


@dataclass(frozen=True)
class WavSignalData:
    """Loaded mono audio waveform mapped to Volts."""

    sample_rate_hz: int
    samples_v: np.ndarray
    duration_s: float
    channels_original: int
    sample_width_bytes: int

    def voltage_at(self, time_s: float, *, loop: bool = False) -> float:
        """Linearly interpolate sample voltage at `time_s` (seconds)."""
        n = len(self.samples_v)
        if n == 0 or self.sample_rate_hz <= 0:
            return 0.0
        if time_s < 0.0:
            return float(self.samples_v[0])
        if loop and self.duration_s > 0.0:
            t_eff = time_s % self.duration_s
        else:
            t_eff = time_s
        pos = t_eff * float(self.sample_rate_hz)
        if pos >= n - 1:
            return float(self.samples_v[-1]) if not loop else float(self.samples_v[0])
        idx = int(math.floor(pos))
        frac = pos - idx
        v0 = float(self.samples_v[idx])
        v1 = float(self.samples_v[idx + 1])
        return v0 + frac * (v1 - v0)


def _decode_pcm_frames(
    raw: bytes, *, sampwidth: int, n_channels: int, channel: int
) -> np.ndarray:
    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        mono = (arr[channel::n_channels] - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        mono = arr[channel::n_channels] / 32768.0
    elif sampwidth == 3:
        n_total_samples = len(raw) // 3
        vals = np.empty(n_total_samples, dtype=np.float64)
        for i in range(n_total_samples):
            b0, b1, b2 = raw[3 * i : 3 * i + 3]
            signed = struct.unpack("<i", bytes((b0, b1, b2, 0xFF if b2 & 0x80 else 0x00)))[0]
            vals[i] = float(signed) / 8388608.0
        mono = vals[channel::n_channels]
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        mono = arr[channel::n_channels] / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes.")
    return mono


def load_wav_samples(
    path: str | os.PathLike[str],
    *,
    gain_v: float = 1.0,
    dc_offset_v: float = 0.0,
    normalize_peak: bool = False,
    channel: int = 0,
) -> WavSignalData:
    """Read an uncompressed PCM `.wav` file into floating-point Volt samples."""
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        comptype = wf.getcomptype()
        if comptype != "NONE":
            raise ValueError(f"Compressed WAV format {comptype!r} is not supported.")
        if framerate <= 0:
            raise ValueError(f"Invalid WAV sample rate {framerate}.")
        if not (0 <= channel < n_channels):
            raise ValueError(f"Requested channel {channel} out of range [0, {n_channels - 1}].")
        raw = wf.readframes(n_frames)

    mono = _decode_pcm_frames(raw, sampwidth=sampwidth, n_channels=n_channels, channel=channel)
    if normalize_peak and len(mono) > 0:
        peak = float(np.max(np.abs(mono)))
        if peak > 1e-12:
            mono = mono / peak
    samples_v = mono * float(gain_v) + float(dc_offset_v)
    samples_v.setflags(write=False)
    duration_s = float(len(samples_v)) / float(framerate)
    return WavSignalData(
        sample_rate_hz=int(framerate),
        samples_v=samples_v,
        duration_s=duration_s,
        channels_original=int(n_channels),
        sample_width_bytes=int(sampwidth),
    )


def save_wav_samples(
    path: str | os.PathLike[str],
    sample_rate_hz: int,
    samples_v: np.ndarray,
    *,
    normalize: bool = False,
    peak_fullscale_v: float = 1.0,
    sample_width_bytes: int = 2,
) -> Path:
    """Write a 1D mono signal vector to an uncompressed PCM `.wav` file."""
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}.")
    if peak_fullscale_v <= 0.0:
        raise ValueError(f"peak_fullscale_v must be > 0, got {peak_fullscale_v}.")
    if sample_width_bytes not in (1, 2):
        raise ValueError("save_wav_samples supports 8-bit (1) or 16-bit (2) PCM WAV.")

    arr = np.asarray(samples_v, dtype=np.float64).reshape(-1)
    if len(arr) == 0:
        scaled = np.zeros(0, dtype=np.float64)
    elif normalize:
        peak = float(np.max(np.abs(arr)))
        scaled = arr / peak if peak > 1e-12 else np.zeros_like(arr)
    else:
        scaled = arr / float(peak_fullscale_v)
    scaled = np.clip(scaled, -1.0, 1.0)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(int(sample_rate_hz))
        if sample_width_bytes == 1:
            pcm = np.clip(np.round(scaled * 127.0 + 128.0), 0, 255).astype(np.uint8)
            wf.writeframes(pcm.tobytes())
        else:
            pcm = np.clip(np.round(scaled * 32767.0), -32767, 32767).astype("<i2")
            wf.writeframes(pcm.tobytes())
    return out_path


def export_waveform_to_wav(
    path: str | os.PathLike[str],
    times_s: np.ndarray,
    values_v: np.ndarray,
    *,
    sample_rate_hz: int = 44100,
    normalize: bool = True,
    peak_fullscale_v: float = 1.0,
) -> Path:
    """Resample solver time/voltage arrays to uniform `sample_rate_hz` PCM `.wav`."""
    t_arr = np.asarray(times_s, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(values_v, dtype=np.float64).reshape(-1)
    if len(t_arr) < 2 or len(t_arr) != len(v_arr):
        raise ValueError("times_s and values_v must have equal length >= 2.")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}.")

    t0 = float(t_arr[0])
    t1 = float(t_arr[-1])
    duration = max(0.0, t1 - t0)
    n_out = max(1, int(round(duration * float(sample_rate_hz))) + 1)
    t_uniform = t0 + np.arange(n_out, dtype=np.float64) / float(sample_rate_hz)
    resampled = np.interp(t_uniform, t_arr, v_arr)
    return save_wav_samples(
        path,
        sample_rate_hz,
        resampled,
        normalize=normalize,
        peak_fullscale_v=peak_fullscale_v,
    )
