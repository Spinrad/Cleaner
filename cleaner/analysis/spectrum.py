"""
Spectrum analysis — Time-averaged FFT, room mode detection by temporal
persistence (mean/variance stationarity score), harshness band energy.

Pure scipy + numpy + soundfile. No librosa, no numba, no resampy.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import scipy.signal
import soundfile as sf

logger = logging.getLogger(__name__)

ANALYSIS_SR: int = 16000
MAX_DURATION_S: float = 60.0
N_FFT: int = 8192
HOP_LENGTH: int = 1024
ROOM_MODE_LOW_HZ: float = 100.0
ROOM_MODE_HIGH_HZ: float = 800.0
ROOM_MODE_COUNT: int = 3

FALLBACK = {
    "room_modes_hz": [300.0, 450.0, 600.0],
    "room_mode_qs": [5.0, 5.0, 5.0],
    "room_mode_gains_db": [3.0, 3.0, 3.0],
    "harshness_band_energy_db": -20.0,
    "spectral_centroid_hz": 2000.0,
    "low_mid_energy_db": -18.0,
}


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Downsample with anti-aliasing via scipy."""
    if orig_sr == target_sr:
        return y.astype(np.float32)
    num_samples = int(len(y) * target_sr / orig_sr)
    return scipy.signal.resample(y, num_samples).astype(np.float32)


def load_analysis_audio(source_path: str) -> tuple[np.ndarray, float]:
    """Load 60s max, mono, 16 kHz via soundfile + scipy."""
    logger.info("Loading: %s", source_path)
    info = sf.info(source_path)
    orig_sr = info.samplerate
    max_frames = min(info.frames, int(MAX_DURATION_S * orig_sr))
    y, _ = sf.read(source_path, frames=max_frames, always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != ANALYSIS_SR:
        y = _resample(y, orig_sr, ANALYSIS_SR)
    dur = len(y) / ANALYSIS_SR
    if dur < 1.0:
        raise ValueError(f"Audio too short: {dur:.2f}s")
    logger.info("Loaded %.2fs (%d samples)", dur, len(y))
    return y.astype(np.float32), dur


def compute_stft(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """STFT via scipy, returns (freqs_hz, S_linear) shape (n_bins, n_frames)."""
    f, _, Zxx = scipy.signal.stft(
        y, fs=ANALYSIS_SR, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH,
        window="hann", boundary=None, padded=False,
    )
    return f, np.abs(Zxx)


def compute_stationarity_scores(S: np.ndarray, freqs: np.ndarray):
    """score = mean / (std + eps) per bin in [100, 800] Hz."""
    mask = (freqs >= ROOM_MODE_LOW_HZ) & (freqs <= ROOM_MODE_HIGH_HZ)
    band_f, band_S = freqs[mask], S[mask, :]
    if band_S.shape[0] < 5:
        return band_f, np.zeros_like(band_f)
    mu = np.mean(band_S, axis=1)
    std = np.std(band_S, axis=1, ddof=1)
    eps = np.finfo(np.float32).eps
    return band_f, mu / (std + eps)


def detect_room_modes_by_persistence(freqs: np.ndarray, S: np.ndarray):
    """Top 3 persistent peaks in [100,800] Hz."""
    bf, scores = compute_stationarity_scores(S, freqs)
    if len(bf) < 5 or float(np.max(scores)) < 0.1:
        return FALLBACK["room_modes_hz"][:3], FALLBACK["room_mode_qs"][:3], FALLBACK["room_mode_gains_db"][:3]
    res = float(bf[1] - bf[0]) if len(bf) > 1 else 2.0
    dist = max(3, int(15.0 / res))
    peaks, props = scipy.signal.find_peaks(scores, prominence=0.05 * (float(np.max(scores)) - float(np.min(scores)) + 1e-10), distance=dist)
    if len(peaks) == 0:
        return FALLBACK["room_modes_hz"][:3], FALLBACK["room_mode_qs"][:3], FALLBACK["room_mode_gains_db"][:3]
    prom = props.get("prominences", np.ones(len(peaks)))
    top = peaks[np.argsort(prom)[::-1][:ROOM_MODE_COUNT]]
    avg_mag = np.mean(S, axis=1)
    band_avg = avg_mag[(freqs >= ROOM_MODE_LOW_HZ) & (freqs <= ROOM_MODE_HIGH_HZ)]
    fh, qs, gs = [], [], []
    for idx in top:
        f0 = float(bf[idx])
        # Prominence: peak magnitude minus local median (±1/3 octave ≈ factor 1.26 each side)
        # This measures how much the peak sticks out, not its absolute level.
        oct_third_bins = max(1, int(f0 * 0.26 / (bf[1]-bf[0])))
        lo = max(0, idx - oct_third_bins)
        hi = min(len(band_avg) - 1, idx + oct_third_bins)
        local_median = float(np.median(band_avg[lo:hi+1]))
        peak_val = band_avg[idx]
        # Prominence relative to the band's median level (not local ratio of tiny values)
        band_median = float(np.median(band_avg[band_avg > 1e-10])) if np.any(band_avg > 1e-10) else 1e-10
        ref_level = max(band_median, 1e-10)
        peak_db = 20.0 * np.log10(max(peak_val, 1e-10))
        ref_db = 20.0 * np.log10(ref_level)
        prominence_db = max(0.0, peak_db - ref_db)
        # Q estimation: -3 dB bandwidth
        target = band_avg[idx] / np.sqrt(2)
        l, r = int(idx), int(idx)
        while l > 0 and band_avg[l] > target: l -= 1
        while r < len(band_avg) - 1 and band_avg[r] > target: r += 1
        df = float(bf[min(r, len(bf)-1)] - bf[max(l, 0)])
        q = f0 / df if df > 0 else 10.0
        fh.append(round(f0, 1))
        qs.append(round(q, 1))
        gs.append(round(prominence_db, 1))
    while len(fh) < ROOM_MODE_COUNT:
        i = len(fh)
        fh.append(FALLBACK["room_modes_hz"][i])
        qs.append(FALLBACK["room_mode_qs"][i])
        gs.append(FALLBACK["room_mode_gains_db"][i])
    return fh, qs, gs


def measure_band_energy(freqs: np.ndarray, S: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask): return -60.0
    return round(float(20.0 * np.log10(max(np.mean(np.mean(S[mask, :], axis=1)), 1e-10))), 1)


def compute_spectral_centroid(freqs: np.ndarray, S: np.ndarray) -> float:
    avg = np.mean(S, axis=1)
    total = np.sum(avg)
    return round(float(np.sum(freqs * avg) / total), 1) if total > 1e-10 else 2000.0


def analyse_spectrum(source_path: str) -> dict:
    y, dur = load_analysis_audio(source_path)
    freqs, S = compute_stft(y)
    modes, qs, gains = detect_room_modes_by_persistence(freqs, S)
    harsh_db = measure_band_energy(freqs, S, 2500.0, 4500.0)
    low_db = measure_band_energy(freqs, S, 120.0, 480.0)
    cent = compute_spectral_centroid(freqs, S)
    del y, S; gc.collect()
    return {
        "duration_s": round(dur, 2), "sample_rate": ANALYSIS_SR,
        "spectral_centroid_hz": cent, "harshness_band_energy_db": harsh_db,
        "low_mid_energy_db": low_db, "room_modes_hz": modes,
        "room_mode_qs": qs, "room_mode_gains_db": gains,
    }
