"""Tests for the ffmpeg filterchain builder."""
from cleaner.ffmpeg_chain import build_filtergraph

def test_build_graph_structure():
    report = {
        "comp_threshold_linear": 0.18, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.0, "notch_q_1": 20.0,
        "notch_gain_1": -6.0, "notch_freq_2": 450.0, "notch_q_2": 20.0,
        "notch_gain_2": -5.0, "notch_freq_3": 600.0, "notch_q_3": 20.0,
        "notch_gain_3": -4.0, "limiter_ceiling_linear": 0.88,
        "deharsher_threshold_linear": 5.0, "deharsher_filter_ratio": 3.0,
        "deharsher_attack_ms": 3.0, "deharsher_release_ms": 30.0,
        "expander_threshold_linear": 0.05, "expander_ratio": 2.0,
        "expander_attack_ms": 5.0, "expander_release_ms": 40.0,
        "sat_threshold_linear": 0.85, "sat_softclip_type": 0,
        "_air_db": 0.0, "_width": 0.0,
        "bus_threshold_linear": 0.18, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.0,
    }
    g = build_filtergraph(report)
    # Must contain key filter names
    assert "highpass=f=35" in g
    assert "stereotools=mode=lr>ms" in g
    assert "stereotools=mode=ms>lr" in g
    assert "highpass=f=150" in g
    assert "sidechaincompress" in g
    assert "anequalizer" in g
    assert "alimiter" in g
    assert "agate=mode=upward" in g
    assert "adynamicequalizer" in g
    assert "asoftclip" in g
    assert "[out]" in g

def test_notch_frequencies_in_graph():
    report = {
        "comp_threshold_linear": 0.05, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.8, "notch_q_1": 20.0,
        "notch_gain_1": -6.0, "notch_freq_2": 525.4, "notch_q_2": 20.0,
        "notch_gain_2": -5.0, "notch_freq_3": 552.7, "notch_q_3": 20.0,
        "notch_gain_3": -4.0, "limiter_ceiling_linear": 0.88,
        "_air_db": 0.0, "_width": 0.0,
        "bus_threshold_linear": 0.18, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.0,
    }
    g = build_filtergraph(report)
    assert "f=300.8" in g
    assert "f=525.4" in g
    assert "f=552.7" in g

def test_sidechain_structure():
    report = {
        "comp_threshold_linear": 0.05, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.0, "notch_q_1": 20.0,
        "notch_gain_1": -6.0, "notch_freq_2": 450.0, "notch_q_2": 20.0,
        "notch_gain_2": -5.0, "notch_freq_3": 600.0, "notch_q_3": 20.0,
        "notch_gain_3": -4.0, "limiter_ceiling_linear": 0.88,
        "deharsher_threshold_linear": 5.0, "deharsher_filter_ratio": 3.0,
        "deharsher_attack_ms": 3.0, "deharsher_release_ms": 30.0,
        "expander_threshold_linear": 0.05, "expander_ratio": 2.0,
        "expander_attack_ms": 5.0, "expander_release_ms": 40.0,
        "sat_threshold_linear": 0.85, "sat_softclip_type": 0,
        "_air_db": 0.0, "_width": 0.0,
        "bus_threshold_linear": 0.18, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.0,
    }
    g = build_filtergraph(report)
    # new pad names after refactor (asplit for sidechain pad sharing)
    assert "[side]" in g
    assert "[mid]" in g
    assert "[mid]asplit=2[mid_sc][mid_main]" in g
    assert "[side_hp][mid_sc]sidechaincompress" in g
    assert "[side_ducked]" in g
    assert "[mid_main][side_ducked]amerge" in g


def test_hp_disabled():
    report = {
        "comp_threshold_linear": 0.05, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.0, "notch_q_1": 20.0,
        "notch_gain_1": -6.0, "notch_freq_2": 450.0, "notch_q_2": 20.0,
        "notch_gain_2": -5.0, "notch_freq_3": 600.0, "notch_q_3": 20.0,
        "notch_gain_3": -4.0, "limiter_ceiling_linear": 0.88,
        "deharsher_threshold_linear": 5.0, "deharsher_filter_ratio": 3.0,
        "deharsher_attack_ms": 3.0, "deharsher_release_ms": 30.0,
        "expander_threshold_linear": 0.05, "expander_ratio": 2.0,
        "expander_attack_ms": 5.0, "expander_release_ms": 40.0,
        "expander_range_linear": 0.25,
        "sat_threshold_linear": 0.85, "sat_softclip_type": 0,
        "_air_db": 0.0, "_width": 0.0,
        "bus_threshold_linear": 0.18, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.0,
    }
    # All stages off except structure
    stages = {"expander": False, "ducking": False, "deharsher": False,
              "notches": False, "saturation": False, "limiter": False,
              "hp35": False, "hp150": False}
    g = build_filtergraph(report, stages)
    assert "highpass" not in g  # no highpass at all
    assert "stereotools" in g  # MS processing still there
    assert "alimiter" not in g
    assert "agate" not in g
    assert "adynamicequalizer" not in g
    assert "asoftclip" not in g


def test_mastering_flags():
    report = {
        "comp_threshold_linear": 0.18, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.0, "notch_q_1": 20.0,
        "notch_gain_1": -3.0, "notch_freq_2": 450.0, "notch_q_2": 20.0,
        "notch_gain_2": -4.0, "notch_freq_3": 600.0, "notch_q_3": 20.0,
        "notch_gain_3": -5.0, "limiter_ceiling_linear": 0.88,
        "deharsher_threshold_linear": 30.0, "deharsher_filter_ratio": 2.5,
        "deharsher_attack_ms": 3.0, "deharsher_release_ms": 30.0,
        "expander_threshold_linear": 0.4, "expander_ratio": 1.4,
        "expander_attack_ms": 1.0, "expander_release_ms": 15.0,
        "expander_range_linear": 0.12,
        "sat_threshold_linear": 0.85, "sat_softclip_type": 0,
        "_air_db": 2.0, "_width": 0.3, "sat_glue": 0.5,
        "bus_threshold_linear": 0.18, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.0,
    }
    stages = {"glue": False, "air": True, "width": True,
              "expander": False, "ducking": False, "deharsher": False,
              "notches": False, "saturation": False, "limiter": False}
    g = build_filtergraph(report, stages)
    assert "equalizer" in g
    assert "gain=2.0" in g
    assert "stereotools=mode=lr>lr" in g
    assert "base=0.3" in g

def test_bus_comp_enabled():
    report = {
        "comp_threshold_linear": 0.18, "comp_ratio": 4, "comp_attack_ms": 2.0,
        "comp_release_ms": 60, "notch_freq_1": 300.0, "notch_q_1": 20.0,
        "notch_gain_1": -3.0, "notch_freq_2": 450.0, "notch_q_2": 20.0,
        "notch_gain_2": -4.0, "notch_freq_3": 600.0, "notch_q_3": 20.0,
        "notch_gain_3": -5.0, "limiter_ceiling_linear": 0.88,
        "deharsher_threshold_linear": 30.0, "deharsher_filter_ratio": 2.5,
        "deharsher_attack_ms": 3.0, "deharsher_release_ms": 30.0,
        "expander_threshold_linear": 0.4, "expander_ratio": 1.4,
        "expander_attack_ms": 1.0, "expander_release_ms": 15.0,
        "expander_range_linear": 0.12,
        "sat_threshold_linear": 0.95, "sat_softclip_type": 0,
        "_air_db": 0.0, "_width": 0.0, "sat_glue": 0.15,
        "bus_threshold_linear": 0.1, "bus_ratio": 2, "bus_attack_ms": 10,
        "bus_release_ms": 100, "bus_mix": 0.5,
    }
    stages = {"bus_comp": True}
    g = build_filtergraph(report, stages)
    assert "acompressor=mode=downward" in g
    assert "mix=0.5" in g
    assert "ratio=2" in g
    assert "attack=10" in g
    assert "knee=4" in g


def test_ms_roundtrip_neutral():
    """M/S encode → decode must be gain-neutral for mono input with ducking off."""
    import subprocess, tempfile, struct, math
    sample_rate = 48000
    duration_s = 1.0
    freq = 1000.0
    n_samples = int(sample_rate * duration_s)
    samples = [math.sin(2 * math.pi * freq * i / sample_rate) * 0.5 for i in range(n_samples)]
    raw = b"".join(struct.pack("<f", s) for s in samples)
    raw_stereo = b"".join(struct.pack("<f", s) + struct.pack("<f", s) for s in samples)
    
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        f.write(raw_stereo)
        raw_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        out_path = f.name
    
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "f32le", "-ar", str(sample_rate), "-ac", "2",
            "-i", raw_path,
            "-filter_complex",
            "[0:a]stereotools=mode=lr>ms,stereotools=mode=ms>lr[out]",
            "-map", "[out]", "-f", "f32le", "-c:a", "pcm_f32le", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        assert proc.returncode == 0, f"ffmpeg failed: {proc.stderr[-500:].decode()}"
        
        with open(out_path, "rb") as f:
            out_data = f.read()
        out_samples = struct.unpack(f"<{n_samples * 2}f", out_data)
        
        max_diff = 0.0
        for i in range(n_samples):
            diff_l = abs(samples[i] - out_samples[i * 2])
            diff_r = abs(samples[i] - out_samples[i * 2 + 1])
            max_diff = max(max_diff, diff_l, diff_r)
        
        assert max_diff < 0.02, f"M/S round-trip gain error: max diff = {max_diff:.6f}"
    finally:
        import os
        os.unlink(raw_path)
        os.unlink(out_path)
