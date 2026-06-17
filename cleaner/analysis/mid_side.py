"""M/S analysis — Phase correlation + cymbal phase profiling. Pure scipy+numpy+soundfile."""

from __future__ import annotations
import gc, logging
import numpy as np, scipy.signal, soundfile as sf

logger = logging.getLogger(__name__)
from cleaner.constants import HF_CORR_LOW_HZ, HF_CORR_HIGH_HZ, HF_CORR_ORDER
from cleaner.constants import (
    MS_CORR_AVG_DEFAULT, HARSHNESS_INDEX_DEFAULT, HF_CORRELATION_DEFAULT,
)
ANALYSIS_SR = 48000; MAX_DURATION_S = 60.0

def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr: return y
    return scipy.signal.resample(y, int(y.shape[-1]*target_sr/orig_sr), axis=-1)

def load_stereo_audio(source_path: str) -> np.ndarray:
    info = sf.info(source_path)
    max_f = min(info.frames, int(MAX_DURATION_S * info.samplerate))
    y, _ = sf.read(source_path, frames=max_f, always_2d=True, dtype="float32")
    if y.ndim == 1: y = np.column_stack([y, y.copy()])
    if y.shape[1] != 2: y = y.T
    if info.samplerate != ANALYSIS_SR: y = _resample(y, info.samplerate, ANALYSIS_SR)
    return y.T.astype(np.float32)  # (2, n)

def compute_ms_correlation(left, right, window=4096, hop=2048):
    n = len(left); corrs = []
    for start in range(0, n-window+1, hop):
        l, r = left[start:start+window], right[start:start+window]
        if np.std(l) < 1e-8 or np.std(r) < 1e-8: continue
        c = np.corrcoef(l, r)[0, 1]
        if not np.isnan(c): corrs.append(float(c))
    return round(float(np.mean(corrs)), 3) if corrs else MS_CORR_AVG_DEFAULT

def compute_side_energy_ratio(left, right):
    side = (left - right) / 2
    rl, rr, rs = float(np.sqrt(np.mean(left**2))), float(np.sqrt(np.mean(right**2))), float(np.sqrt(np.mean(side**2)))
    total = rl**2 + rr**2
    return round(min(max(rs**2/total, 0.0), 1.0), 3) if total > 1e-12 else 0.0

def _design_bp(low, high, sr, order):
    nyq = sr/2; return scipy.signal.butter(order, [low/nyq, high/nyq], btype="band", output="sos")

def analyse_cymbal_phase(left, right, sr=ANALYSIS_SR, overall_rms_lin=None):
    try:
        sos = _design_bp(HF_CORR_LOW_HZ, HF_CORR_HIGH_HZ, sr, HF_CORR_ORDER)
        lf = scipy.signal.sosfiltfilt(sos, left)
        rf = scipy.signal.sosfiltfilt(sos, right)
    except Exception as exc:
        logger.warning("cymbal phase analysis failed: %s", exc)
        return {"hf_correlation": HF_CORRELATION_DEFAULT, "harshness_index": HARSHNESS_INDEX_DEFAULT}
    hf_corr = compute_ms_correlation(lf, rf)
    energy_lin = float(np.sqrt(np.mean(lf**2 + rf**2)))
    decorr = max(1.0 - hf_corr, 0.0)
    if overall_rms_lin is not None and overall_rms_lin > 1e-10:
        energy_ratio = min(energy_lin / overall_rms_lin, 1.0)
    else:
        energy_ratio = min(energy_lin * 100.0, 1.0)
    harshness = round(decorr * energy_ratio, 3)
    del lf, rf; gc.collect()
    return {"hf_correlation": round(hf_corr, 3), "harshness_index": harshness}

def analyse_mid_side(source_path: str) -> dict:
    y = load_stereo_audio(source_path)
    left, right = y[0, :], y[1, :]
    overall_rms = float(np.sqrt(np.mean(left**2 + right**2)))
    corr = compute_ms_correlation(left, right)
    sr_ = compute_side_energy_ratio(left, right)
    cym = analyse_cymbal_phase(left, right, overall_rms_lin=overall_rms)
    del y, left, right; gc.collect()
    return {"ms_correlation_avg": corr, "side_energy_ratio": sr_, **cym}
