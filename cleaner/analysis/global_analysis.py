"""Global analysis orchestrator. Calls 4 sub-modules, computes ffmpeg params."""

from __future__ import annotations
import gc, logging
from typing import Any
from cleaner.analysis.spectrum import analyse_spectrum, FALLBACK as SF
from cleaner.analysis.clipping import detect_clipping
from cleaner.analysis.dynamics import analyse_dynamics, FALLBACK as DF
from cleaner.analysis.mid_side import analyse_mid_side, FALLBACK as MF

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
    report["expander_ratio"] = round(min(1.1 + crest * 0.02, 1.5), 1)
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
    report = compute_ffmpeg_params(report)
    gc.collect()
    logger.info("=== Phase 1 Complete: %d keys ===", len(report))
    return report
