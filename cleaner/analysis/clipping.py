"""Digital clipping detector. Pure soundfile+numpy."""

from __future__ import annotations
import gc, logging
import numpy as np, soundfile as sf

logger = logging.getLogger(__name__)
ANALYSIS_SR = 48000; MAX_DURATION_S = 60.0
CLIP_DBFS = -0.1; CLIP_RATIO = 0.015

def detect_clipping(source_path: str) -> dict:
    clip_amp = 10.0 ** (CLIP_DBFS / 20.0)
    try:
        info = sf.info(source_path)
        max_f = min(info.frames, int(MAX_DURATION_S * info.samplerate))
        y, sr = sf.read(source_path, frames=max_f, always_2d=True, dtype="float32")
        if y.ndim == 1: y = y.reshape(-1, 1)
        if y.shape[1] > 2: y = y[:, :2]
        if info.samplerate != ANALYSIS_SR:
            from scipy.signal import resample
            y = resample(y, int(y.shape[0] * ANALYSIS_SR / info.samplerate), axis=0)
    except Exception as exc:
        logger.warning("Clipping scan failed: %s", exc)
        return {"is_heavily_clipped": False, "clip_ratio": 0.0, "max_consecutive_clips": 0, "total_clipped_samples": 0, "total_samples": 0}
    total = y.size
    clipped = np.sum(np.abs(y) >= clip_amp)
    max_cons = 0; cur = 0
    for v in np.abs(y).flatten():
        if v >= clip_amp: cur += 1; max_cons = max(max_cons, cur)
        else: cur = 0
    ratio = float(clipped) / total if total > 0 else 0.0
    is_heavy = ratio > CLIP_RATIO
    del y; gc.collect()
    return {"is_heavily_clipped": is_heavy, "clip_ratio": round(ratio, 5), "max_consecutive_clips": int(max_cons), "total_clipped_samples": int(clipped), "total_samples": int(total)}
