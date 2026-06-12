"""Global analysis orchestrator. Calls 4 sub-modules, computes ffmpeg params."""

from __future__ import annotations
import gc, logging
from typing import Any
from cleaner.analysis.spectrum import analyse_spectrum, FALLBACK as SF
from cleaner.analysis.clipping import detect_clipping
from cleaner.analysis.dynamics import analyse_dynamics, FALLBACK as DF
from cleaner.analysis.mid_side import analyse_mid_side, FALLBACK as MF

from cleaner.lv2_params import db_to_linear_gain

logger = logging.getLogger(__name__)
AnalysisReport = dict[str, Any]


def compute_ffmpeg_params(report: AnalysisReport) -> AnalysisReport:
    crest = report.get("crest_factor_db", 12.0)
    rms = report.get("rms_db", -18.0)
    attack_ms = report.get("transient_attack_ms", 10.0)
    agc_rec = report.get("agc_recovery_ms", 80.0)

    # --- Expander (agate upward) ---
    # Must be extremely gentle — only nudge transients, never saturate.
    # Albini philosophy: respect dynamics, don't add artificial punch.
    peak_db = report.get("peak_db", -3.0)
    # Threshold very close to peak — only the loudest 3-6 dB get expanded
    exp_thresh_db = peak_db - 3.0
    report["expander_threshold_linear"] = round(
        10.0 ** (exp_thresh_db / 20.0), 4
    )
    # Very gentle ratio: 1.1-1.5
    report["expander_ratio"] = round(max(1.1, min(1.5, 1.6 - crest * 0.03)), 1)
    # Fast attack to catch transients
    report["expander_attack_ms"] = round(max(min(attack_ms * 0.5, 10.0), 1.0), 1)
    # Quick release to avoid pumping
    report["expander_release_ms"] = round(max(min(agc_rec * 0.8, 50.0), 15.0), 1)
    # Range: more expansion when signal is compressed (low crest)
    # 0.4 when crest<10 (AGC probable), 0.15 when crest>14 (dynamic), scaled by intensity
    intensity = report.get("_intensity", 0.5)
    if crest < 8:
        expander_range = 0.45 * intensity
    elif crest < 10:
        expander_range = 0.35 * intensity
    elif crest < 14:
        expander_range = 0.20 * intensity
    else:
        expander_range = 0.10 * intensity
    report["expander_range_linear"] = round(max(expander_range, 0.05), 2)

    # --- Sidechain compressor ---
    # Sidechain ducking threshold: only trigger on LOUD transients (kick/snare),
    # NOT on normal program material. Side channel carries spatial/air frequencies.
    comp_thresh_db = rms + 6.0  # trigger when Mid exceeds RMS + 6dB
    report["comp_threshold_linear"] = round(10.0 ** (comp_thresh_db / 20.0), 4)
    report["comp_release_ms"] = round(max(agc_rec * 1.5, 40.0), 1)
    report["comp_ratio"] = 4
    report["comp_attack_ms"] = 2.0

    # --- De-harsher (adynamicequalizer) ---
    # threshold is 0-100 scale. Default=0 (always active).
    # Higher threshold = activates only on louder harshness = less processing.
    # We want it gentle: only cut when harshness band is genuinely harsh.
    deharsh_linear = max(8.0, min(50.0, crest * 2.0))
    report["deharsher_threshold_linear"] = round(deharsh_linear, 1)
    # Gentler ratio: 1.5-3.0 (was 2.0-5.0)
    report["deharsher_filter_ratio"] = round(min(1.5 + crest * 0.06, 3.0), 1)
    report["deharsher_attack_ms"] = round(max(min(attack_ms * 0.3, 8.0), 2.0), 1)
    report["deharsher_release_ms"] = round(max(attack_ms * 2.5, 40.0), 1)
    # Apply tame_cymbals delta
    tame_delta = report.get("_tame_cymbals", 0.0)
    report["deharsher_threshold_linear"] = round(
        max(0.5, report["deharsher_threshold_linear"] + tame_delta * 0.5), 1
    )
    report["deharsher_display_threshold"] = round(deharsh_linear, 1)

    # --- Limiter ---
    ceiling = report.get("_ceiling_db", -1.1)
    report["limiter_ceiling_linear"] = round(10.0 ** (ceiling / 20.0), 4)

    # --- Saturation (drive + makeup) ---
    glue = report.get("_glue", 0.15)
    intensity = report.get("_intensity", 0.5)
    eff_glue = glue * (0.3 + intensity * 0.7)  # intensity scales glue effect
    report["sat_drive_db"] = round(eff_glue * 12.0, 1)  # 0→0 dB, 0.5→+6, 1→+12
    report["sat_threshold_linear"] = round(0.92 - eff_glue * 0.35, 3)  # 0.92→0.57
    report["sat_makeup_db"] = round(-eff_glue * 12.0 * 0.5, 1)  # compensate half
    report["sat_glue"] = glue
    report["sat_softclip_type"] = 0

    # --- Mastering air & width ---
    report["_air_db"] = report.get("_air", 1.5)
    report["_width"] = report.get("_width", 0.0)

    # --- Bus compressor (SSL-style glue) ---
    bus = report.get("_bus_comp", 0.0)
    # Threshold: compress the body, not the transients
    # Higher bus_comp = lower threshold = more compression
    bus_thresh_db = rms - crest * 0.3 + (1.0 - bus) * 12.0  # 0→-8dB, 1→-20dB
    report["bus_threshold_linear"] = round(10.0 ** (bus_thresh_db / 20.0), 4)
    report["bus_mix"] = round(bus, 2)  # parallel compression
    report["bus_ratio"] = 2  # SSL classic
    report["bus_attack_ms"] = 10  # slow, lets transients through
    report["bus_release_ms"] = 100  # smooth

    # --- Notches ---
    modes_hz = list(report.get("room_modes_hz", [300, 450, 600]))
    while len(modes_hz) < 3: modes_hz.append(450)
    modes_q = list(report.get("room_mode_qs", [5, 5, 5]))
    while len(modes_q) < 3: modes_q.append(5)
    prominences = list(report.get("room_mode_gains_db", [3, 3, 3]))
    while len(prominences) < 3: prominences.append(3)
    mult = report.get("_notch_multiplier", 1.0)
    intensity = report.get("_intensity", 0.5)
    for i in range(3):
        # Wider Q: clamp to [3, 10] for musically useful bandwidth
        q = min(max(modes_q[i], 3.0), 10.0)
        prom = abs(prominences[i])
        # Skip if prominence < 3 dB (not a real mode, just spectral noise)
        if prom < 3.0:
            g = 0.0
        else:
            # Depth proportional to prominence: 0.5× prominence, bounded [-9, -2] dB
            depth = min(prom * 0.5, 9.0)
            depth = max(depth, 2.0)
            g = -(depth * mult * intensity)
        g = max(g, -12.0)  # hard floor
        report[f"notch_freq_{i+1}"] = round(modes_hz[i], 1)
        report[f"notch_q_{i+1}"] = round(q, 1)
        report[f"notch_gain_{i+1}"] = round(g, 1)

    # --- Clipping penalty ---
    if report.get("is_heavily_clipped", False):
        report["comp_threshold_linear"] = round(report["comp_threshold_linear"] * 0.8, 4)
        report["expander_ratio"] = round(max(report["expander_ratio"] * 0.6, 1.1), 1)
        report["expander_range_linear"] = 0.1  # almost no expansion on clipped audio
        report["sat_threshold_linear"] = min(report["sat_threshold_linear"] + 0.05, 0.99)
        logger.warning("Clipping penalty applied: expander/saturation reduced")

    return report


def get_global_analysis(source_path: str) -> AnalysisReport:
    logger.info("=== Phase 1 ===")
    report: AnalysisReport = {}
    failures = []

    for name, func, fb in [
        ("spectral", analyse_spectrum, {
            "room_modes_hz": [300, 450, 600], "room_mode_qs": [5, 5, 5],
            "room_mode_gains_db": [3, 3, 3], "harshness_band_energy_db": -20.0,
            "spectral_centroid_hz": 2000.0, "low_mid_energy_db": -18.0,
        }),
        ("clipping", detect_clipping, {"is_heavily_clipped": False, "clip_ratio": 0.0}),
        ("dynamics", analyse_dynamics, {
            "peak_db": -3.0, "rms_db": -15.0, "crest_factor_db": 12.0,
            "transient_attack_ms": 10.0, "transient_crest_local_db": 12.0,
            "agc_recovery_ms": 80.0,
        }),
        ("mid_side", analyse_mid_side, {
            "ms_correlation_avg": 0.5, "side_energy_ratio": 0.3,
            "hf_correlation": 0.4, "harshness_index": 0.0,
        }),
    ]:
        try:
            report.update(func(source_path))
            logger.info("[OK] %s", name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            logger.warning("[FAIL] %s: %s", name, exc)
            for k, v in fb.items():
                report.setdefault(k, v)

    if len(failures) >= 4:
        raise ValueError("All 4 modules failed.\n" + "\n".join(failures))
    report.setdefault("duration_s", 0.0)
    report.setdefault("sample_rate", 48000)
    if failures: report["_analysis_warnings"] = failures
    gc.collect()
    logger.info("=== Phase 1 Complete: %d keys ===", len(report))
    return report


def compute_native_saturation_params(report: AnalysisReport) -> dict[str, float]:
    """Compute native ffmpeg asoftclip saturation params (drive + makeup).
    
    Uses asoftclip=type=tanh with proper drive so the signal ENTERS
    the non-linear zone. Returns params for ffmpeg_chain format:
    sat_drive_db, sat_makeup_db, sat_threshold_linear.
    """
    glue = report.get("_glue", 0.15)
    intensity = report.get("_intensity", 0.5)
    eff_glue = glue * (0.3 + intensity * 0.7)
    
    drive_db = eff_glue * 16.0
    
    threshold_linear = round(0.92 - eff_glue * 0.35, 3)
    
    makeup_db = round(-drive_db * 0.4, 1)
    
    return {
        "sat_drive_db": round(drive_db, 1),
        "sat_makeup_db": makeup_db,
        "sat_threshold_linear": threshold_linear,
    }


def compute_expander_lsp_params(report: AnalysisReport, tracker=None) -> dict[str, float]:
    """Compute parameters for LSP expander_stereo (anti-AGC, Mode=Up).
    
    Replaces agate=mode=upward. Position: after HP35, before M/S encode.
    Uses GainTracker for initial levels only (expander is first in chain).
    """
    crest = report.get("crest_factor_db", 12.0)
    peak_db = report.get("peak_db", -3.0)
    attack_ms = report.get("transient_attack_ms", 10.0)
    agc_rec = report.get("agc_recovery_ms", 80.0)
    intensity = report.get("_intensity", 0.5)
    
    # Mode: Up (anti-AGC expansion)
    em = 1.0
    
    # Attack level (threshold): peak - 3 dB, converted to linear G
    exp_thresh_db = peak_db - 3.0
    al = db_to_linear_gain(exp_thresh_db)
    
    # Ratio: DECREASES with crest (more expansion when compressed)
    # crest=4 (AGC) -> 1.48, crest=12 -> 1.24, crest=18 -> 1.1
    base_ratio = max(1.1, 1.6 - crest * 0.03)
    # Intensity scales the amount ABOVE 1.0: er = 1.0 + (base_ratio - 1.0) * intensity
    er = 1.0 + (base_ratio - 1.0) * intensity
    er = max(1.05, min(1.5, er))  # clamp to real port range
    
    # Attack: fast to catch transients
    at_val = max(1.0, min(attack_ms * 0.5, 10.0))
    
    # Release: AGC recovery based, clamped
    rt_val = max(15.0, min(agc_rec * 0.8, 50.0))
    
    # Knee: moderate
    kn = 0.5
    
    # Makeup: unity (don't add gain here)
    mk = 1.0
    
    return {
        "em": 1.0,  # Upward mode
        "al": round(al, 4),
        "er": round(er, 1),
        "at": round(at_val, 1),
        "rt": round(rt_val, 1),
        "kn": round(kn, 3),
        "mk": 1.0,
        "g_in": 1.0,
        "g_out": 1.0,
        "scm": 1.0,   # RMS sidechain
        "sla": 0.0,   # no lookahead
    }


def compute_eq_lsp_params(report: AnalysisReport, tracker=None) -> dict[str, float]:
    """Compute parameters for LSP para_equalizer_x16_stereo (notches + air).
    
    Uses up to 3 notch bands for room modes + 1 hi-shelf for air.
    Band is disabled (gain=1.0 = 0 dB) if prominence < 3 dB.
    Position: after de-harsher, before saturator.
    """
    modes_hz = list(report.get("room_modes_hz", [300, 450, 600]))
    while len(modes_hz) < 3:
        modes_hz.append(450)
    modes_q = list(report.get("room_mode_qs", [5, 5, 5]))
    while len(modes_q) < 3:
        modes_q.append(5)
    prominences = list(report.get("room_mode_gains_db", [3, 3, 3]))
    while len(prominences) < 3:
        prominences.append(3)
    
    mult = report.get("_notch_multiplier", 1.0)
    intensity = report.get("_intensity", 0.5)
    air_db = report.get("_air", 1.5)
    
    params: dict[str, float] = {
        "mode": 0.0,   # stereo mode
        "g_in": 1.0,
        "g_out": 1.0,
    }
    
    # Notch bands (0, 1, 2)
    for i in range(3):
        f0 = modes_hz[i]
        q_val = min(max(modes_q[i], 3.0), 10.0)
        prom = abs(prominences[i])
        
        if prom < 3.0:
            # Disable band
            params[f"s_{i}"] = 0.0
            params[f"g_{i}"] = 1.0  # 0 dB = linear gain 1.0
            params[f"f_{i}"] = round(f0, 1)
            params[f"w_{i}"] = 4.0
            params[f"q_{i}"] = 0.0
            params[f"ft_{i}"] = 0.0  # off
            params[f"fm_{i}"] = 0.0
        else:
            depth_db = min(prom * 0.5, 9.0)
            depth_db = max(depth_db, 2.0)
            gain_db = -(depth_db * mult * intensity)
            gain_db = max(gain_db, -12.0)
            
            params[f"s_{i}"] = 0.0  # not soloed
            params[f"ft_{i}"] = 4.0  # Bell filter type
            params[f"fm_{i}"] = 0.0  # default filter mode
            params[f"f_{i}"] = round(f0, 1)
            params[f"w_{i}"] = round(q_val / 2.5, 1)  # Q to width mapping
            params[f"q_{i}"] = round(q_val, 1)
            params[f"g_{i}"] = round(db_to_linear_gain(gain_db), 4)
    
    # Air band (band 3): hi-shelf at 8 kHz
    air_enabled = air_db > 0.01
    params["s_3"] = 0.0
    if air_enabled:
        params["ft_3"] = 7.0   # High-shelf type (6=LowShelf, 7=HighShelf)
        params["fm_3"] = 0.0
        params["f_3"] = 8000.0
        params["w_3"] = 2.8    # Q≈0.7 mapped to width
        params["q_3"] = 0.7
        params["g_3"] = round(db_to_linear_gain(air_db), 4)
    else:
        params["ft_3"] = 0.0
        params["fm_3"] = 0.0
        params["f_3"] = 8000.0
        params["w_3"] = 4.0
        params["q_3"] = 0.0
        params["g_3"] = 1.0
    
    # Remaining bands (4-15): disabled
    for i in range(4, 16):
        params[f"s_{i}"] = 0.0
        params[f"ft_{i}"] = 0.0
        params[f"fm_{i}"] = 0.0
        params[f"f_{i}"] = 100.0 + i * 200.0
        params[f"w_{i}"] = 4.0
        params[f"q_{i}"] = 0.0
        params[f"g_{i}"] = 1.0
    
    return params


def compute_compressor_lsp_params(report: AnalysisReport, tracker=None) -> dict[str, float]:
    """Compute parameters for LSP compressor_stereo (bus/glue).
    
    SSL-style glue compressor with parallel dry/wet mix.
    Position: after saturator, before limiter.
    Uses --bus-comp for threshold and mix (NOT --glue).
    """
    crest = report.get("crest_factor_db", 12.0)
    if tracker is not None:
        rms_db = tracker.current_rms_dbfs
    else:
        rms_db = report.get("rms_db", -15.0)
    bus_comp = report.get("_bus_comp", 0.0)
    
    # Threshold: compress the body, not transients
    bus_thresh_db = rms_db - crest * 0.3 + (1.0 - bus_comp) * 12.0
    al = db_to_linear_gain(bus_thresh_db)
    
    # Dry/wet for parallel compression
    cdr = max(0.0, 1.0 - bus_comp)  # dry
    cwt = bus_comp                     # wet
    
    return {
        "cm": 0.0,      # Downward compression
        "al": round(al, 4),
        "cr": 2.0,      # SSL classic 2:1
        "at": 10.0,     # slow attack, lets transients through
        "rt": 100.0,    # smooth release
        "kn": 0.5,      # medium knee
        "mk": 1.0,      # unity makeup
        "cdr": round(cdr, 2),
        "cwt": round(cwt, 2),
        "g_in": 1.0,
        "g_out": 1.0,
        "scm": 1.0,     # RMS sidechain
        "sla": 0.0,
    }


def compute_limiter_lsp_params(report: AnalysisReport, tracker=None) -> dict[str, float]:
    """Compute parameters for LSP limiter_stereo (true-peak musical limiter).
    
    Replaces alimiter. Position: after compressor, before LUFS measurement.
    """
    ceiling = report.get("_ceiling_db", -1.1)
    
    # Threshold = ceiling (brickwall), NOT predicted peak
    th_val = db_to_linear_gain(ceiling)
    
    return {
        "mode": 0.0,    # default mode
        "th": round(th_val, 4),
        "knee": 1.0,    # soft knee
        "boost": 1.0,   # boost enabled
        "lk": 5.0,      # 5 ms lookahead
        "at": 5.0,      # 5 ms attack
        "rt": 5.0,      # 5 ms release
        "ovs": 4.0,     # 4x oversampling
        "alr": 1.0,     # adaptive release enabled
        "g_in": 1.0,
        "g_out": 1.0,
        "scp": 1.0,     # sidechain preamp
    }


def compute_deharsher_lsp_params(report: AnalysisReport, tracker=None) -> dict[str, float]:
    """Compute parameters for LSP sc_compressor_stereo as de-harsher.
    
    Uses internal sidechain bandpass filter (shpf/slpf) to target
    the harshness band (2.5-4.5 kHz). Opt-in via --deharsher.
    Position: after M/S decode, before EQ.
    """
    harshness_index = report.get("harshness_index", 0.0)
    tame_delta = report.get("_tame_cymbals", 0.0)
    
    # Convert harshness_index (decorr * log energy) to threshold
    # Higher harshness -> lower threshold -> more reduction
    base_threshold = max(0.01, 1.0 - harshness_index * 2.0)
    threshold = max(0.005, base_threshold + tame_delta * 0.05)
    
    # Ratio: moderate, 1.5-3.0
    ratio = max(1.5, min(3.0, 1.5 + harshness_index * 2.0 + abs(tame_delta) * 0.3))
    
    return {
        "cm": 0.0,      # Downward (cut)
        "al": round(threshold, 4),
        "cr": round(ratio, 1),
        "at": 5.0,      # moderate attack
        "rt": 30.0,     # moderate release
        "kn": 0.5,
        "mk": 1.0,
        "g_in": 1.0,
        "g_out": 1.0,
        "scm": 1.0,     # RMS sidechain
        "sct": 1.0,     # Internal sidechain (use shpf/slpf)
        "shpf": 2500.0, # HPF: bottom of harshness band
        "slpf": 4500.0, # LPF: top of harshness band
        "sla": 0.0,
        "cdr": 0.0,
        "cwt": 1.0,
    }
