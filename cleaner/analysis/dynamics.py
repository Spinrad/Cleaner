"""Dynamics analysis — Crest factor, transient detection, AGC recovery. Pure scipy+numpy+soundfile."""

from __future__ import annotations
import gc, logging
import numpy as np, scipy.signal, soundfile as sf
from cleaner.constants import ANALYSIS_SR_LO as ANALYSIS_SR, ANALYSIS_DURATION_S as MAX_DURATION_S
from cleaner.constants import (
    TRANSIENT_CREST_LOCAL_DB_DEFAULT,
    AGC_RECOVERY_MS_DEFAULT,
    TRANSIENT_ATTACK_MS_DEFAULT,
)

logger = logging.getLogger(__name__)

def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr: return y.astype(np.float32)
    return scipy.signal.resample(y, int(len(y)*target_sr/orig_sr)).astype(np.float32)

def load_mono_audio(source_path: str) -> np.ndarray:
    info = sf.info(source_path)
    max_f = min(info.frames, int(MAX_DURATION_S * info.samplerate))
    y, _ = sf.read(source_path, frames=max_f, always_2d=False, dtype="float32")
    if y.ndim > 1: y = np.mean(y, axis=1)
    if info.samplerate != ANALYSIS_SR: y = _resample(y, info.samplerate, ANALYSIS_SR)
    return y.astype(np.float32)

def compute_peak_dbfs(y): p = float(np.max(np.abs(y))); return -120.0 if p < 1e-10 else round(float(20*np.log10(p)), 1)
def compute_rms_db(y): r = float(np.sqrt(np.mean(y**2))); return -120.0 if r < 1e-10 else round(float(20*np.log10(r)), 1)
def compute_crest_factor_db(pk, rm): return round(pk - rm, 2)

def detect_onsets(y, sr=ANALYSIS_SR):
    """Simple energy-based onset detection."""
    frame = int(sr * 0.02)
    hop = frame // 2
    rms_env = np.array([np.sqrt(np.mean(y[i:i+frame]**2)) for i in range(0, len(y)-frame, hop)])
    if len(rms_env) < 3: return np.array([])
    rms_env = np.concatenate([rms_env, [rms_env[-1]]])
    flux = np.diff(rms_env)
    flux = np.maximum(flux, 0)
    med = np.median(flux) + 0.5 * np.std(flux)
    peaks, _ = scipy.signal.find_peaks(flux, height=med, distance=max(3, int(0.1 * sr / hop)))
    onset_samples = peaks * hop
    return onset_samples / sr

def measure_attack_times(y, sr, onset_times_s, max_attack_ms=100.0):
    max_s = int(max_attack_ms/1000*sr)
    env = np.abs(y)
    times = []
    for os in onset_times_s:
        oc = int(os*sr)
        seg = env[max(0,oc-max_s):oc+1]
        if len(seg) < 10: continue
        pk = seg[-1]
        if pk < 1e-8: continue
        th10, th90 = 0.1*pk, 0.9*pk
        i90, i10 = None, None
        for i in range(len(seg)-1,-1,-1):
            if i90 is None and seg[i] >= th90: i90 = i
            if i10 is None and seg[i] <= th10 and i90 is not None: i10 = i; break
        if i90 is not None and i10 is not None and i90 > i10:
            ms = (i90-i10)/sr*1000
            if 0.1 < ms < max_attack_ms: times.append(round(ms, 2))
    return times

def compute_local_crest_factor(y, sr, onset_times_s, window_ms=50.0):
    if len(onset_times_s) == 0: return TRANSIENT_CREST_LOCAL_DB_DEFAULT
    half = int(window_ms/1000*sr)//2
    crests = []
    for os in onset_times_s:
        c = int(os*sr)
        seg = y[max(0,c-half):min(len(y),c+half)]
        if len(seg) < 10: continue
        pk = float(np.max(np.abs(seg))); rm = float(np.sqrt(np.mean(seg**2)))
        if rm > 1e-10: crests.append(20*np.log10(pk/rm))
    return round(float(np.median(crests)), 2) if crests else TRANSIENT_CREST_LOCAL_DB_DEFAULT

def measure_agc_recovery(y, sr, onset_times_s, ambient_rms):
    if len(onset_times_s) == 0: return AGC_RECOVERY_MS_DEFAULT
    rec_win = int(0.5 * sr); rms_win = int(0.01 * sr) or 4
    thresh = 1.5 * ambient_rms
    times = []
    for os in onset_times_s:
        start = int(os*sr); end = min(len(y), start+rec_win)
        if end-start < rms_win: continue
        seg = y[start:end]
        rms_env = np.sqrt(np.convolve(seg**2, np.ones(rms_win)/rms_win, mode="same"))
        pk_idx = int(np.argmax(rms_env)); pk_val = rms_env[pk_idx]
        rec_idx = None
        for i in range(pk_idx, len(rms_env)):
            if rms_env[i] <= thresh and pk_val > thresh: rec_idx = i; break
        if rec_idx is not None:
            ms = (rec_idx-pk_idx)/sr*1000
            if 1.0 < ms < 500.0: times.append(ms)
    return round(float(np.median(times)), 1) if times else AGC_RECOVERY_MS_DEFAULT

def analyse_dynamics(source_path: str) -> dict:
    y = load_mono_audio(source_path)
    pk = compute_peak_dbfs(y); rm = compute_rms_db(y); cr = compute_crest_factor_db(pk, rm)
    amb = float(np.sqrt(np.mean(y**2)))
    onsets = detect_onsets(y); attacks = measure_attack_times(y, ANALYSIS_SR, onsets)
    local_cr = compute_local_crest_factor(y, ANALYSIS_SR, onsets)
    trans_atk = round(float(np.median(attacks)), 1) if attacks else TRANSIENT_ATTACK_MS_DEFAULT
    agc = measure_agc_recovery(y, ANALYSIS_SR, onsets, amb)
    if cr < 6.0: logger.warning("Low Crest Factor (%.1f dB)", cr)
    del y; gc.collect()
    return {"peak_db": pk, "rms_db": rm, "crest_factor_db": cr, "transient_attack_ms": trans_atk, "transient_crest_local_db": local_cr, "agc_recovery_ms": agc}
