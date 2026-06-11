"""M/S analysis — Phase correlation + cymbal phase profiling. Pure scipy+numpy+soundfile."""

from __future__ import annotations
import gc, logging
import numpy as np, scipy.signal, soundfile as sf

logger = logging.getLogger(__name__)
ANALYSIS_SR = 48000; MAX_DURATION_S = 60.0
FALLBACK = {"ms_correlation_avg": 0.5, "side_energy_ratio": 0.3, "hf_correlation": 0.4, "harshness_index": 0.0}
HF_LOW, HF_HIGH, HF_ORDER = 5000.0, 10000.0, 4

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
    return round(float(np.mean(corrs)), 3) if corrs else FALLBACK["ms_correlation_avg"]

def compute_side_energy_ratio(left, right):
    side = (left - right) / 2
    rl, rr, rs = float(np.sqrt(np.mean(left**2))), float(np.sqrt(np.mean(right**2))), float(np.sqrt(np.mean(side**2)))
    total = rl**2 + rr**2
    return round(min(max(rs**2/total, 0.0), 1.0), 3) if total > 1e-12 else 0.0

def _design_bp(low, high, sr, order):
    nyq = sr/2; return scipy.signal.butter(order, [low/nyq, high/nyq], btype="band", output="sos")

def analyse_cymbal_phase(left, right, sr=ANALYSIS_SR):
    try:
        sos = _design_bp(HF_LOW, HF_HIGH, sr, HF_ORDER)
        lf = scipy.signal.sosfiltfilt(sos, left)
        rf = scipy.signal.sosfiltfilt(sos, right)
    except Exception:
        return {"hf_correlation": FALLBACK["hf_correlation"], "harshness_index": FALLBACK["harshness_index"]}
    hf_corr = compute_ms_correlation(lf, rf)
    energy_lin = float(np.sqrt(np.mean(lf**2 + rf**2)))
    energy_db = 20 * np.log10(max(energy_lin, 1e-10))
    decorr = max(1.0 - hf_corr, 0.0)
    harshness = round(decorr * np.log10(max(1 + energy_db + 60, 1.0)), 3)
    del lf, rf; gc.collect()
    return {"hf_correlation": round(hf_corr, 3), "harshness_index": harshness}

def analyse_mid_side(source_path: str) -> dict:
    y = load_stereo_audio(source_path)
    left, right = y[0, :], y[1, :]
    corr = compute_ms_correlation(left, right)
    sr_ = compute_side_energy_ratio(left, right)
    cym = analyse_cymbal_phase(left, right)
    del y, left, right; gc.collect()
    return {"ms_correlation_avg": corr, "side_energy_ratio": sr_, **cym}
