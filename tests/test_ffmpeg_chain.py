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
    assert "treble" in g
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
