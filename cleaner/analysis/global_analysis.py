"""Global analysis orchestrator. Calls 4 sub-modules, computes ffmpeg params."""

from __future__ import annotations
import gc, logging
from typing import Any
from cleaner.analysis.spectrum import analyse_spectrum
from cleaner.analysis.clipping import detect_clipping
from cleaner.analysis.dynamics import analyse_dynamics
from cleaner.analysis.mid_side import analyse_mid_side

from cleaner.lv2_params import db_to_linear_gain
from cleaner.constants import (
    SAT_DRIVE_MULTIPLIER, SAT_MAKEUP_RATIO, SAT_THRESHOLD_BASE, SAT_THRESHOLD_SLOPE,
    SAT_CLIP_PENALTY,
    EXP_THRESH_DELTA_DB, EXP_RATIO_BASE, EXP_RATIO_SLOPE, EXP_RATIO_MIN, EXP_RATIO_MAX,
    EXP_ATTACK_FRAC, EXP_RELEASE_FRAC, EXP_CLIP_RATIO_FACTOR,
    NOTCH_PROM_DISABLE_DB, NOTCH_DEPTH_RATIO, NOTCH_DEPTH_MIN_DB, NOTCH_DEPTH_MAX_DB,
    NOTCH_GAIN_FLOOR_DB, NOTCH_Q_MIN, NOTCH_Q_MAX, NOTCH_DEFAULT_HZ, NOTCH_DEFAULT_Q,
    NOTCH_DEFAULT_GAIN_DB,
    AIR_FREQ_HZ, AIR_Q, CLEAN_FREQ_HZ, CLEAN_Q,
    COMP_DUCK_THRESH_OFFSET_DB, COMP_DUCK_RATIO, COMP_DUCK_ATTACK_MS,
    COMP_DUCK_RELEASE_FACTOR, COMP_DUCK_RELEASE_MIN_MS,
    BUS_RATIO, BUS_ATTACK_MS, BUS_RELEASE_MS, BUS_THRESH_CREST_FACTOR, BUS_THRESH_OFFSET,
    LIMITER_LOOKAHEAD_S, LIMITER_ATTACK_S, LIMITER_RELEASE_S, LIMITER_OVERSAMPLING,
    INTENSITY_GLUE_OFFSET, INTENSITY_GLUE_SLOPE,
    DEHARSH_THRESH_MIN, DEHARSH_THRESH_MAX, DEHARSH_CREST_FACTOR,
    DEHARSH_BAND_LOW_HZ, DEHARSH_BAND_HIGH_HZ,
    CLIP_PENALTY_COMP, CLIP_PENALTY_EXP_RATIO, CLIP_PENALTY_EXP_RANGE, CLIP_PENALTY_SAT,
    MAX_ROOM_MODES,
)

logger = logging.getLogger(__name__)

# Backward compat alias — being phased out in favor of cleaner.types.AnalysisReport.
from cleaner.types import AnalysisReport as _AnalysisReportDataclass
AnalysisReport = _AnalysisReportDataclass  # type alias, not the dict anymore
from cleaner.types import DerivedParams


def compute_ffmpeg_params(report: AnalysisReport) -> AnalysisReport:
    # Convert to mutable dict for legacy mutation (being phased out).
    report = report.to_dict() if hasattr(report, 'to_dict') else dict(report)
    crest = report.get("crest_factor_db", 12.0)
    rms = report.get("rms_db", -15.0)
    attack_ms = report.get("transient_attack_ms", 10.0)
    agc_rec = report.get("agc_recovery_ms", 80.0)

    # --- Expander (agate upward) ---
    # Must be extremely gentle — only nudge transients, never saturate.
    # Albini philosophy: respect dynamics, don't add artificial punch.
    peak_db = report.get("peak_db", -3.0)
    # Threshold very close to peak — only the loudest few dB get expanded
    exp_thresh_db = peak_db - EXP_THRESH_DELTA_DB
    report["expander_threshold_linear"] = round(
        10.0 ** (exp_thresh_db / 20.0), 4
    )
    # Very gentle ratio: 1.1-1.5
    report["expander_ratio"] = round(max(EXP_RATIO_MIN, min(EXP_RATIO_MAX, EXP_RATIO_BASE - crest * EXP_RATIO_SLOPE)), 1)
    # Fast attack to catch transients
    report["expander_attack_ms"] = round(max(min(attack_ms * EXP_ATTACK_FRAC, 10.0), 1.0), 1)
    # Quick release to avoid pumping
    report["expander_release_ms"] = round(max(min(agc_rec * EXP_RELEASE_FRAC, 50.0), 15.0), 1)
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
    comp_thresh_db = rms + COMP_DUCK_THRESH_OFFSET_DB  # trigger when Mid exceeds RMS + 6dB
    report["comp_threshold_linear"] = round(10.0 ** (comp_thresh_db / 20.0), 4)
    report["comp_release_ms"] = round(max(agc_rec * COMP_DUCK_RELEASE_FACTOR, COMP_DUCK_RELEASE_MIN_MS), 1)
    report["comp_ratio"] = COMP_DUCK_RATIO
    report["comp_attack_ms"] = COMP_DUCK_ATTACK_MS

    # --- De-harsher (adynamicequalizer) ---
    deharsh_linear = max(DEHARSH_THRESH_MIN, min(DEHARSH_THRESH_MAX, crest * DEHARSH_CREST_FACTOR))
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
    eff_glue = glue * (INTENSITY_GLUE_OFFSET + intensity * INTENSITY_GLUE_SLOPE)
    report["sat_drive_db"] = round(eff_glue * SAT_DRIVE_MULTIPLIER, 1)
    report["sat_threshold_linear"] = round(SAT_THRESHOLD_BASE - eff_glue * SAT_THRESHOLD_SLOPE, 3)
    report["sat_makeup_db"] = round(-eff_glue * SAT_DRIVE_MULTIPLIER * SAT_MAKEUP_RATIO, 1)
    report["sat_glue"] = glue
    report["sat_softclip_type"] = 0

    # --- Mastering air & width ---
    report["_air_db"] = report.get("_air", 0.0)
    report["_width"] = report.get("_width", 0.0)

    # --- Bus compressor (SSL-style glue) ---
    bus = report.get("_bus_comp", 0.0)
    # Threshold: compress the body, not the transients
    bus_thresh_db = rms - crest * BUS_THRESH_CREST_FACTOR + (1.0 - bus) * BUS_THRESH_OFFSET
    report["bus_threshold_linear"] = round(10.0 ** (bus_thresh_db / 20.0), 4)
    report["bus_mix"] = round(bus, 2)  # parallel compression
    report["bus_ratio"] = BUS_RATIO  # SSL classic
    report["bus_attack_ms"] = BUS_ATTACK_MS  # slow, lets transients through
    report["bus_release_ms"] = BUS_RELEASE_MS  # smooth

    # --- Notches ---
    modes_hz = list(report.get("room_modes_hz", [300, 450, 600]))
    while len(modes_hz) < MAX_ROOM_MODES: modes_hz.append(450)
    modes_q = list(report.get("room_mode_qs", [5, 5, 5]))
    while len(modes_q) < MAX_ROOM_MODES: modes_q.append(5)
    prominences = list(report.get("room_mode_gains_db", [3, 3, 3]))
    while len(prominences) < MAX_ROOM_MODES: prominences.append(3)
    mult = report.get("_notch_multiplier", 1.0)
    intensity = report.get("_intensity", 0.5)
    for i in range(MAX_ROOM_MODES):
        # Wider Q: clamp to [3, 10] for musically useful bandwidth
        q = min(max(modes_q[i], NOTCH_Q_MIN), NOTCH_Q_MAX)
        prom = abs(prominences[i])
        # Skip if prominence < 3 dB (not a real mode, just spectral noise)
        if prom < NOTCH_PROM_DISABLE_DB:
            g = 0.0
        else:
            # Depth proportional to prominence: 0.5× prominence, bounded [2, 9] dB
            depth = min(prom * NOTCH_DEPTH_RATIO, NOTCH_DEPTH_MAX_DB)
            depth = max(depth, NOTCH_DEPTH_MIN_DB)
            g = -(depth * mult * intensity)
        g = max(g, NOTCH_GAIN_FLOOR_DB)  # hard floor
        report[f"notch_freq_{i+1}"] = round(modes_hz[i], 1)
        report[f"notch_q_{i+1}"] = round(q, 1)
        report[f"notch_gain_{i+1}"] = round(g, 1)

    # --- Clipping penalty ---
    if report.get("is_heavily_clipped", False):
        report["comp_threshold_linear"] = round(report["comp_threshold_linear"] * CLIP_PENALTY_COMP, 4)
        report["expander_ratio"] = round(max(report["expander_ratio"] * CLIP_PENALTY_EXP_RATIO, EXP_RATIO_MIN), 1)
        report["expander_range_linear"] = CLIP_PENALTY_EXP_RANGE
        report["sat_threshold_linear"] = min(report["sat_threshold_linear"] + CLIP_PENALTY_SAT, 0.99)
        logger.warning("Clipping penalty applied: expander/saturation reduced")

    return report


def get_global_analysis(source_path: str) -> AnalysisReport:
    logger.info("=== Phase 1 ===")
    data: dict[str, Any] = {}
    failures = []

    modules = [
        ("spectral", analyse_spectrum),
        ("clipping", detect_clipping),
        ("dynamics", analyse_dynamics),
        ("mid_side", analyse_mid_side),
    ]
    for name, func in modules:
        try:
            data.update(func(source_path))
            logger.info("[OK] %s", name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            logger.warning("[FAIL] %s: %s", name, exc)
            # Fields not produced by failed module get AnalysisReport defaults

    if len(failures) >= 4:
        raise ValueError("All 4 modules failed.\n" + "\n".join(failures))
    data.setdefault("duration_s", 0.0)
    data.setdefault("sample_rate", 48000)
    if failures:
        data["_analysis_warnings"] = failures

    gc.collect()
    # Build dataclass from explicit keys
    known_fields = {f.name for f in AnalysisReport.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    result = AnalysisReport(**filtered)
    logger.info("=== Phase 1 Complete: %d keys ===", len(filtered))
    return result


def compute_native_saturation_params(report: AnalysisReport,
                                       derived: DerivedParams | None = None) -> dict[str, float]:
    """Native ffmpeg asoftclip saturation params. Reads DerivedParams if available."""
    if derived:
        return {
            "sat_drive_db": derived.sat_drive_db,
            "sat_makeup_db": derived.sat_makeup_db,
            "sat_threshold_linear": derived.sat_threshold_linear,
        }
    glue = report.get("_glue", 0.15)
    intensity = report.get("_intensity", 0.5)
    eff_glue = glue * (INTENSITY_GLUE_OFFSET + intensity * INTENSITY_GLUE_SLOPE)
    drive_db = eff_glue * SAT_DRIVE_MULTIPLIER
    threshold_linear = round(SAT_THRESHOLD_BASE - eff_glue * SAT_THRESHOLD_SLOPE, 3)
    if report.get("is_heavily_clipped", False):
        threshold_linear = min(threshold_linear + SAT_CLIP_PENALTY, 0.99)
    makeup_db = round(-drive_db * SAT_MAKEUP_RATIO, 1)
    return {
        "sat_drive_db": round(drive_db, 1),
        "sat_makeup_db": makeup_db,
        "sat_threshold_linear": threshold_linear,
    }


def compute_expander_lsp_params(report: AnalysisReport, tracker=None,
                                 derived: DerivedParams | None = None) -> dict[str, float]:
    """LSP expander_stereo port mapping. Reads DerivedParams if available."""
    if derived:
        al = round(derived.expander_threshold_linear, 4)
        er = round(derived.expander_ratio, 1)
        at_val = round(derived.expander_attack_ms, 1)
        rt_val = round(derived.expander_release_ms, 1)
    else:
        peak_db = report.get("peak_db", -3.0)
        attack_ms = report.get("transient_attack_ms", 10.0)
        agc_rec = report.get("agc_recovery_ms", 80.0)
        intensity = report.get("_intensity", 0.5)
        crest = report.get("crest_factor_db", 12.0)
        exp_thresh_db = peak_db - EXP_THRESH_DELTA_DB
        al = db_to_linear_gain(exp_thresh_db)
        base_ratio = max(EXP_RATIO_MIN, EXP_RATIO_BASE - crest * EXP_RATIO_SLOPE)
        er = 1.0 + (base_ratio - 1.0) * intensity
        er = max(1.05, min(EXP_RATIO_MAX, er))
        if report.get("is_heavily_clipped", False):
            er = max(1.05, er * EXP_CLIP_RATIO_FACTOR)
        at_val = max(1.0, min(attack_ms * EXP_ATTACK_FRAC, 10.0))
        rt_val = max(15.0, min(agc_rec * EXP_RELEASE_FRAC, 50.0))
        al = round(al, 4)
        er = round(er, 1)
        at_val = round(at_val, 1)
        rt_val = round(rt_val, 1)

    return {
        "em": 1.0,  # Upward mode
        "al": al,
        "er": er,
        "at": at_val,
        "rt": rt_val,
        "kn": 0.5,   # moderate knee
        "mk": 1.0,   # unity makeup
        "g_in": 1.0,
        "g_out": 1.0,
        "scm": 1.0,  # RMS sidechain
        "sla": 0.0,  # no lookahead
    }


def compute_eq_lsp_params(report: AnalysisReport, tracker=None,
                          derived: DerivedParams | None = None) -> dict[str, float]:
    """LSP para_equalizer_x16_stereo port mapping. Reads DerivedParams if available."""
    params: dict[str, float] = {
        "mode": 0.0, "g_in": 1.0, "g_out": 1.0,
    }

    if derived:
        # Use pre-computed DerivedParams — single source of truth
        for i in range(MAX_ROOM_MODES):
            g_db = getattr(derived, f"notch_gain_{i+1}")
            f0 = getattr(derived, f"notch_freq_{i+1}")
            q_val = getattr(derived, f"notch_q_{i+1}")
            params[f"s_{i}"] = 0.0
            if g_db == 0.0:
                params[f"ft_{i}"] = 0.0
                params[f"fm_{i}"] = 0.0
                params[f"f_{i}"] = round(f0, 1)
                params[f"w_{i}"] = 4.0
                params[f"q_{i}"] = 0.0
                params[f"g_{i}"] = 1.0
            else:
                params[f"ft_{i}"] = 1.0
                params[f"fm_{i}"] = 0.0
                params[f"f_{i}"] = round(f0, 1)
                params[f"w_{i}"] = round(q_val / 2.5, 1)
                params[f"q_{i}"] = round(q_val, 1)
                params[f"g_{i}"] = round(db_to_linear_gain(g_db), 4)

        # Air
        air_db_val = derived.air_db
        params["s_3"] = 0.0
        if abs(air_db_val) > 0.01:
            params["ft_3"] = 1.0
            params["fm_3"] = 0.0
            params["f_3"] = AIR_FREQ_HZ
            params["w_3"] = round(AIR_Q / 2.5, 1)
            params["q_3"] = AIR_Q
            params["g_3"] = round(db_to_linear_gain(air_db_val), 4)
        else:
            params["ft_3"] = 0.0; params["fm_3"] = 0.0
            params["f_3"] = AIR_FREQ_HZ; params["w_3"] = 4.0
            params["q_3"] = 0.0; params["g_3"] = 1.0

        # Clean-mediums
        clean_db = report.get("_clean_mediums", 0.0)
        params["s_4"] = 0.0
        if clean_db < -0.01:
            params["ft_4"] = 1.0; params["fm_4"] = 0.0
            params["f_4"] = CLEAN_FREQ_HZ; params["w_4"] = 4.0
            params["q_4"] = CLEAN_Q
            params["g_4"] = round(db_to_linear_gain(clean_db), 4)
        else:
            params["ft_4"] = 0.0; params["fm_4"] = 0.0
            params["f_4"] = CLEAN_FREQ_HZ; params["w_4"] = 4.0
            params["q_4"] = 0.0; params["g_4"] = 1.0
    else:
        # Legacy path — compute from report dict
        modes_hz = list(report.get("room_modes_hz", [300, 450, 600]))
        while len(modes_hz) < MAX_ROOM_MODES:
            modes_hz.append(450)
        modes_q = list(report.get("room_mode_qs", [5, 5, 5]))
        while len(modes_q) < MAX_ROOM_MODES:
            modes_q.append(5)
        prominences = list(report.get("room_mode_gains_db", [3, 3, 3]))
        while len(prominences) < MAX_ROOM_MODES:
            prominences.append(3)
        mult = report.get("_notch_multiplier", 1.0)
        intensity = report.get("_intensity", 0.5)
        air_db = report.get("_air", 0.0)

        # Notch bands (0, 1, 2)
        for i in range(MAX_ROOM_MODES):
            f0 = modes_hz[i]
            q_val = min(max(modes_q[i], NOTCH_Q_MIN), NOTCH_Q_MAX)
            prom = abs(prominences[i])

            if prom < NOTCH_PROM_DISABLE_DB:
                params[f"s_{i}"] = 0.0
                params[f"g_{i}"] = 1.0
                params[f"f_{i}"] = round(f0, 1)
                params[f"w_{i}"] = 4.0
                params[f"q_{i}"] = 0.0
                params[f"ft_{i}"] = 0.0
                params[f"fm_{i}"] = 0.0
            else:
                depth_db = min(prom * NOTCH_DEPTH_RATIO, NOTCH_DEPTH_MAX_DB)
                depth_db = max(depth_db, NOTCH_DEPTH_MIN_DB)
                gain_db = -(depth_db * mult * intensity)
                gain_db = max(gain_db, NOTCH_GAIN_FLOOR_DB)
                params[f"s_{i}"] = 0.0
                params[f"ft_{i}"] = 1.0
                params[f"fm_{i}"] = 0.0
                params[f"f_{i}"] = round(f0, 1)
                params[f"w_{i}"] = round(q_val / 2.5, 1)
                params[f"q_{i}"] = round(q_val, 1)
                params[f"g_{i}"] = round(db_to_linear_gain(gain_db), 4)

        # Air band
        params["s_3"] = 0.0
        if abs(air_db) > 0.01:
            q_air = AIR_Q
            params["ft_3"] = 1.0; params["fm_3"] = 0.0
            params["f_3"] = AIR_FREQ_HZ
            params["w_3"] = round(q_air / 2.5, 1)
            params["q_3"] = q_air
            params["g_3"] = round(db_to_linear_gain(air_db), 4)
        else:
            params["ft_3"] = 0.0; params["fm_3"] = 0.0
            params["f_3"] = AIR_FREQ_HZ; params["w_3"] = 4.0
            params["q_3"] = 0.0; params["g_3"] = 1.0

        # Clean-mediums
        clean_db = report.get("_clean_mediums", 0.0)
        params["s_4"] = 0.0
        if clean_db < -0.01:
            params["ft_4"] = 1.0; params["fm_4"] = 0.0
            params["f_4"] = CLEAN_FREQ_HZ; params["w_4"] = 4.0
            params["q_4"] = CLEAN_Q
            params["g_4"] = round(db_to_linear_gain(clean_db), 4)
        else:
            params["ft_4"] = 0.0; params["fm_4"] = 0.0
            params["f_4"] = CLEAN_FREQ_HZ; params["w_4"] = 4.0
            params["q_4"] = 0.0; params["g_4"] = 1.0

    # Bands 5-15: disabled
    for i in range(5, 16):
        params[f"s_{i}"] = 0.0
        params[f"ft_{i}"] = 0.0
        params[f"fm_{i}"] = 0.0
        params[f"f_{i}"] = 100.0 + i * 200.0
        params[f"w_{i}"] = 4.0
        params[f"q_{i}"] = 0.0
        params[f"g_{i}"] = 1.0
    
    return params


def compute_compressor_lsp_params(report: AnalysisReport, tracker=None,
                                   derived: DerivedParams | None = None) -> dict[str, float]:
    """LSP compressor_stereo port mapping. Uses tracker when available for RMS."""
    if derived and tracker is None:
        al = round(derived.bus_threshold_linear, 4)
        cdr = max(0.0, 1.0 - derived.bus_mix)
        cwt = derived.bus_mix
    else:
        crest = report.get("crest_factor_db", 12.0)
        if tracker is not None:
            rms_db = tracker.current_rms_dbfs
        else:
            rms_db = report.get("rms_db", -15.0)
        bus_comp = report.get("_bus_comp", 0.0)
        bus_thresh_db = rms_db - crest * BUS_THRESH_CREST_FACTOR + (1.0 - bus_comp) * BUS_THRESH_OFFSET
        al = round(db_to_linear_gain(bus_thresh_db), 4)
        cdr = max(0.0, 1.0 - bus_comp)
        cwt = bus_comp
    
    return {
        "cm": 0.0, "al": al,
        "cr": BUS_RATIO, "at": BUS_ATTACK_MS, "rt": BUS_RELEASE_MS,
        "kn": 0.5, "mk": 1.0,
        "cdr": round(cdr, 2), "cwt": round(cwt, 2),
        "g_in": 1.0, "g_out": 1.0, "scm": 1.0, "sla": 0.0,
    }


def compute_limiter_lsp_params(report: AnalysisReport, tracker=None,
                                derived: DerivedParams | None = None) -> dict[str, float]:
    """LSP limiter_stereo port mapping. Reads DerivedParams if available."""
    if derived:
        th_val = round(derived.limiter_ceiling_linear, 4)
    else:
        ceiling = report.get("_ceiling_db", -1.1)
        th_val = db_to_linear_gain(ceiling)
    return {
        "mode": 0.0, "th": th_val, "knee": 1.0, "boost": 1.0,
        "lk": LIMITER_LOOKAHEAD_S, "at": LIMITER_ATTACK_S, "rt": LIMITER_RELEASE_S,
        "ovs": LIMITER_OVERSAMPLING, "alr": 1.0,
        "g_in": 1.0, "g_out": 1.0, "scp": 1.0,
    }


def compute_deharsher_lsp_params(report: AnalysisReport, tracker=None,
                                  derived: DerivedParams | None = None) -> dict[str, float]:
    """LSP sc_compressor_stereo as de-harsher port mapping. Reads DerivedParams if available."""
    if derived:
        threshold = max(0.005, derived.deharsher_threshold_linear)
        ratio = derived.deharsher_filter_ratio
    else:
        harshness_index = report.get("harshness_index", 0.0)
        tame_delta = report.get("_tame_cymbals", 0.0)
        base_threshold = max(0.01, 1.0 - harshness_index * 2.0)
        threshold = max(0.005, base_threshold + tame_delta * 0.05)
        ratio = max(1.5, min(3.0, 1.5 + harshness_index * 2.0 + abs(tame_delta) * 0.3))
    return {
        "cm": 0.0, "al": round(threshold, 4), "cr": round(ratio, 1),
        "at": 5.0, "rt": 30.0, "kn": 0.5, "mk": 1.0,
        "g_in": 1.0, "g_out": 1.0, "scm": 1.0,
        "sct": 1.0, "shpf": DEHARSH_BAND_LOW_HZ, "slpf": DEHARSH_BAND_HIGH_HZ,
        "sla": 0.0, "cdr": 0.0, "cwt": 1.0,
    }
