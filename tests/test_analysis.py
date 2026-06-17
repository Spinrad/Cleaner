"""Analysis tests — pure scipy+numpy+soundfile. No librosa."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np, pytest, soundfile as sf

def _wav(path, audio, sr=48000):
    if audio.ndim == 1: audio = np.column_stack([audio, audio.copy()])
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_24")

def _sine(path, freq=440, dur=2.0, amp=0.5, sr=48000):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    _wav(path, amp * np.sin(2*np.pi*freq*t), sr)

def _persistent_modes(path, dur=5.0, sr=48000):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    persistent = 0.3*np.sin(2*np.pi*250*t) + 0.25*np.sin(2*np.pi*400*t) + 0.2*np.sin(2*np.pi*550*t)
    third = len(t)//3
    musical = np.zeros_like(t); musical[third:2*third] = 0.5*np.sin(2*np.pi*330*t[third:2*third])
    noise = np.random.default_rng(42).normal(0, 0.02, len(t))
    _wav(path, persistent + musical + noise, sr)

def _clipped(path, dur=3.0, clip_frac=0.03, sr=48000):
    rng = np.random.default_rng(99)
    s = rng.normal(0, 0.8, int(sr*dur))
    s = np.clip(s, -0.95, 0.95)
    n = int(len(s)*clip_frac)
    idx = rng.choice(len(s), n, replace=False)
    s[idx] = np.sign(rng.normal(0, 1, n))
    _wav(path, s, sr)

def _agc_wav(path, dur=4.0, sr=48000):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    signal = np.random.default_rng(7).normal(0, 0.02, len(t))
    for onset_s, decay_ms in [(0.8, 60), (1.6, 80), (2.4, 50), (3.2, 70)]:
        idx = int(onset_s*sr)
        if idx >= len(signal): continue
        signal[idx] = 1.0
        end = min(idx + int(decay_ms/1000*sr), len(signal))
        signal[idx:end] += np.exp(-np.linspace(0, 5, end-idx)) * 0.8
    _wav(path, np.clip(signal, -1, 1).astype(np.float32), sr)

class TestSpectrum:
    def test_load(self):
        from cleaner.analysis.spectrum import load_analysis_audio
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"t.wav"; _sine(w, 440, 2)
            y, dur = load_analysis_audio(str(w))
            assert dur >= 1.0 and y.ndim == 1

    def test_stationarity(self):
        from cleaner.analysis.spectrum import load_analysis_audio, compute_stft, compute_stationarity_scores
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"m.wav"; _persistent_modes(w, 5)
            y, _ = load_analysis_audio(str(w))
            f, S = compute_stft(y)
            bf, scores = compute_stationarity_scores(S, f)
            assert len(bf) > 10
            top = bf[np.argmax(scores)]
            assert any(abs(top - m) < 20 for m in [250, 400, 550])

    def test_room_modes(self):
        from cleaner.analysis.spectrum import load_analysis_audio, compute_stft, detect_room_modes_by_persistence
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"m.wav"; _persistent_modes(w, 5)
            y, _ = load_analysis_audio(str(w))
            f, S = compute_stft(y)
            modes, qs, gains = detect_room_modes_by_persistence(f, S)
            assert len(modes) == 3

    def test_full(self):
        from cleaner.analysis.spectrum import analyse_spectrum
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"m.wav"; _persistent_modes(w, 5)
            r = analyse_spectrum(str(w))
            assert "room_modes_hz" in r and len(r["room_modes_hz"]) == 3

class TestClipping:
    def test_clean(self):
        from cleaner.analysis.clipping import detect_clipping
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"c.wav"; _sine(w, 440, 2, 0.5)
            r = detect_clipping(str(w))
            assert r["is_heavily_clipped"] is False

    def test_heavy(self):
        from cleaner.analysis.clipping import detect_clipping
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"c.wav"; _clipped(w, 3, 0.04)
            r = detect_clipping(str(w))
            assert r["is_heavily_clipped"] is True

class TestDynamics:
    def test_crest(self):
        from cleaner.analysis.dynamics import compute_peak_dbfs, compute_rms_db, compute_crest_factor_db
        y = (0.7 * np.sin(2*np.pi*440*np.linspace(0,1,48000))).astype(np.float32)
        c = compute_crest_factor_db(compute_peak_dbfs(y), compute_rms_db(y))
        assert 2.0 < c < 4.5

    def test_agc(self):
        from cleaner.analysis.dynamics import analyse_dynamics
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"a.wav"; _agc_wav(w, 4)
            r = analyse_dynamics(str(w))
            assert "agc_recovery_ms" in r and r["agc_recovery_ms"] > 0

class TestMidSide:
    def test_mono_corr(self):
        from cleaner.analysis.mid_side import compute_ms_correlation
        t = np.linspace(0, 2, 96000); x = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
        assert compute_ms_correlation(x, x.copy()) > 0.95

    def test_cymbal(self):
        from cleaner.analysis.mid_side import analyse_cymbal_phase
        rng = np.random.default_rng(42)
        l = rng.normal(0, 0.3, 96000).astype(np.float32)
        r = rng.normal(0, 0.3, 96000).astype(np.float32)
        res = analyse_cymbal_phase(l, r)
        assert "hf_correlation" in res and res["hf_correlation"] < 0.5

class TestGlobalAnalysis:
    def test_full(self):
        from cleaner.analysis.global_analysis import get_global_analysis
        from cleaner.analysis.derived import compute_derived_params
        from cleaner.types import MasteringSettings
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"f.wav"; _persistent_modes(w, 5)
            r = get_global_analysis(str(w))
            settings = MasteringSettings()
            derived = compute_derived_params(r, settings)
            # AnalysisReport keys
            for k in ["crest_factor_db", "room_modes_hz"]:
                assert hasattr(r, k), f"Missing {k} in analysis"
            # DerivedParams keys
            for k in ["comp_threshold_linear", "notch_freq_1", "notch_q_1", "notch_gain_1", "limiter_ceiling_linear"]:
                assert hasattr(derived, k), f"Missing {k} in derived"

    def test_clip_penalty(self):
        from cleaner.analysis.global_analysis import get_global_analysis
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)/"c.wav"; _clipped(w, 3, 0.04)
            r = get_global_analysis(str(w))
            assert r["is_heavily_clipped"] is True
