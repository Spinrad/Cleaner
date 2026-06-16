"""Compute DerivedParams — single source of truth for both builders.

Replaces compute_ffmpeg_params and the physics inside compute_*_lsp_params.
Reads AnalysisReport + MasteringSettings attributes directly — no dict.get().
"""

from __future__ import annotations

import logging

from cleaner.types import AnalysisReport, MasteringSettings, DerivedParams
from cleaner.lv2_params import db_to_linear_gain
from cleaner.constants import (
    SAT_DRIVE_MULTIPLIER, SAT_MAKEUP_RATIO, SAT_THRESHOLD_BASE, SAT_THRESHOLD_SLOPE,
    SAT_CLIP_PENALTY,
    EXP_THRESH_DELTA_DB, EXP_RATIO_BASE, EXP_RATIO_SLOPE, EXP_RATIO_MIN, EXP_RATIO_MAX,
    EXP_ATTACK_FRAC, EXP_RELEASE_FRAC, EXP_CLIP_RATIO_FACTOR,
    NOTCH_PROM_DISABLE_DB, NOTCH_DEPTH_RATIO, NOTCH_DEPTH_MIN_DB, NOTCH_DEPTH_MAX_DB,
    NOTCH_GAIN_FLOOR_DB, NOTCH_Q_MIN, NOTCH_Q_MAX, MAX_ROOM_MODES,
    COMP_DUCK_THRESH_OFFSET_DB, COMP_DUCK_RATIO, COMP_DUCK_ATTACK_MS,
    COMP_DUCK_RELEASE_FACTOR, COMP_DUCK_RELEASE_MIN_MS,
    BUS_RATIO, BUS_ATTACK_MS, BUS_RELEASE_MS, BUS_THRESH_CREST_FACTOR, BUS_THRESH_OFFSET,
    DEHARSH_THRESH_MIN, DEHARSH_THRESH_MAX, DEHARSH_CREST_FACTOR,
    INTENSITY_GLUE_OFFSET, INTENSITY_GLUE_SLOPE,
    CLIP_PENALTY_COMP, CLIP_PENALTY_EXP_RATIO, CLIP_PENALTY_EXP_RANGE, CLIP_PENALTY_SAT,
    LIMITER_OVERSAMPLING, LIMITER_LOOKAHEAD_S, LIMITER_ATTACK_S, LIMITER_RELEASE_S,
    CEILING_DEFAULT_DBFS,
)

logger = logging.getLogger(__name__)


def compute_derived_params(analysis: AnalysisReport | dict,
                           settings: MasteringSettings) -> DerivedParams:
    """Compute all DSP parameters from analysis and settings.

    Args:
        analysis: AnalysisReport dataclass, or backward-compat dict.
        settings: User-facing configuration.

    Returns:
        DerivedParams with all computed values for both builders.
    """
    d = DerivedParams()

    # ── Unpack analysis ─────────────────────────────────────────────
    crest = analysis.crest_factor_db if hasattr(analysis, 'crest_factor_db') else analysis.get("crest_factor_db", 12.0)
    rms = analysis.rms_db if hasattr(analysis, 'rms_db') else analysis.get("rms_db", -15.0)
    peak_db = analysis.peak_db if hasattr(analysis, 'peak_db') else analysis.get("peak_db", -3.0)
    attack_ms = analysis.transient_attack_ms if hasattr(analysis, 'transient_attack_ms') else analysis.get("transient_attack_ms", 10.0)
    agc_rec = analysis.agc_recovery_ms if hasattr(analysis, 'agc_recovery_ms') else analysis.get("agc_recovery_ms", 80.0)
    is_clip = analysis.is_heavily_clipped if hasattr(analysis, 'is_heavily_clipped') else analysis.get("is_heavily_clipped", False)

    # ── Expander ────────────────────────────────────────────────────
    exp_thresh_db = peak_db - EXP_THRESH_DELTA_DB
    d.expander_threshold_linear = round(10.0 ** (exp_thresh_db / 20.0), 4)
    er = round(max(EXP_RATIO_MIN, min(EXP_RATIO_MAX, EXP_RATIO_BASE - crest * EXP_RATIO_SLOPE)), 1)
    d.expander_attack_ms = round(max(min(attack_ms * EXP_ATTACK_FRAC, 10.0), 1.0), 1)
    d.expander_release_ms = round(max(min(agc_rec * EXP_RELEASE_FRAC, 50.0), 15.0), 1)

    # Intensity macro: expander range
    intensity = settings.intensity
    if crest < 8:
        expander_range = 0.45 * intensity
    elif crest < 10:
        expander_range = 0.35 * intensity
    elif crest < 14:
        expander_range = 0.20 * intensity
    else:
        expander_range = 0.10 * intensity
    d.expander_range_linear = round(max(expander_range, 0.05), 2)

    # ── Sidechain ducking ───────────────────────────────────────────
    comp_thresh_db = rms + COMP_DUCK_THRESH_OFFSET_DB
    d.comp_threshold_linear = round(10.0 ** (comp_thresh_db / 20.0), 4)
    d.comp_release_ms = round(max(agc_rec * COMP_DUCK_RELEASE_FACTOR, COMP_DUCK_RELEASE_MIN_MS), 1)
    d.comp_ratio = COMP_DUCK_RATIO
    d.comp_attack_ms = COMP_DUCK_ATTACK_MS

    # ── De-harsher ──────────────────────────────────────────────────
    deharsh_linear = max(DEHARSH_THRESH_MIN, min(DEHARSH_THRESH_MAX, crest * DEHARSH_CREST_FACTOR))
    d.deharsher_threshold_linear = round(deharsh_linear, 1)
    d.deharsher_filter_ratio = round(min(1.5 + crest * 0.06, 3.0), 1)
    d.deharsher_attack_ms = round(max(min(attack_ms * 0.3, 8.0), 2.0), 1)
    d.deharsher_release_ms = round(max(attack_ms * 2.5, 40.0), 1)
    tame_delta = settings.tame_cymbals
    d.deharsher_threshold_linear = round(max(0.5, d.deharsher_threshold_linear + tame_delta * 0.5), 1)
    d.deharsher_display_threshold = round(deharsh_linear, 1)

    # ── Limiter ─────────────────────────────────────────────────────
    ceiling = settings.ceiling_db
    d.limiter_ceiling_linear = round(10.0 ** (ceiling / 20.0), 4)

    # ── Saturation ──────────────────────────────────────────────────
    glue = settings.glue
    eff_glue = glue * (INTENSITY_GLUE_OFFSET + intensity * INTENSITY_GLUE_SLOPE)
    d.sat_drive_db = round(eff_glue * SAT_DRIVE_MULTIPLIER, 1)
    d.sat_threshold_linear = round(SAT_THRESHOLD_BASE - eff_glue * SAT_THRESHOLD_SLOPE, 3)
    d.sat_makeup_db = round(-eff_glue * SAT_DRIVE_MULTIPLIER * SAT_MAKEUP_RATIO, 1)
    d.sat_glue = glue

    # ── Air ─────────────────────────────────────────────────────────
    d.air_db = settings.air
    d.width = settings.width

    # ── Bus compressor ──────────────────────────────────────────────
    bus = settings.bus_comp
    bus_thresh_db = rms - crest * BUS_THRESH_CREST_FACTOR + (1.0 - bus) * BUS_THRESH_OFFSET
    d.bus_threshold_linear = round(10.0 ** (bus_thresh_db / 20.0), 4)
    d.bus_mix = round(bus, 2)
    d.bus_ratio = BUS_RATIO
    d.bus_attack_ms = BUS_ATTACK_MS
    d.bus_release_ms = BUS_RELEASE_MS

    # ── Notches ─────────────────────────────────────────────────────
    # Read room modes: prefer dataclass, fallback to dict
    if hasattr(analysis, 'room_modes_hz'):
        modes_hz = list(analysis.room_modes_hz)
        modes_q = list(analysis.room_mode_qs)
        prom_db = list(analysis.room_mode_gains_db)
    else:
        modes_hz = list(analysis.get("room_modes_hz", [300, 450, 600]))
        modes_q = list(analysis.get("room_mode_qs", [5, 5, 5]))
        prom_db = list(analysis.get("room_mode_gains_db", [3, 3, 3]))

    while len(modes_hz) < MAX_ROOM_MODES:
        modes_hz.append(450)
        modes_q.append(5)
        prom_db.append(3)

    mult = settings.notch_multiplier
    for i in range(MAX_ROOM_MODES):
        q_val = min(max(modes_q[i], NOTCH_Q_MIN), NOTCH_Q_MAX)
        prom = abs(prom_db[i])
        if prom < NOTCH_PROM_DISABLE_DB:
            g = 0.0
        else:
            depth = min(prom * NOTCH_DEPTH_RATIO, NOTCH_DEPTH_MAX_DB)
            depth = max(depth, NOTCH_DEPTH_MIN_DB)
            g = -(depth * mult * intensity)
        g = max(g, NOTCH_GAIN_FLOOR_DB)
        # Set notch fields dynamically
        setattr(d, f"notch_freq_{i+1}", round(modes_hz[i], 1))
        setattr(d, f"notch_q_{i+1}", round(q_val, 1))
        setattr(d, f"notch_gain_{i+1}", round(g, 1))

    # ── Clipping penalty ────────────────────────────────────────────
    if is_clip:
        d.comp_threshold_linear = round(d.comp_threshold_linear * CLIP_PENALTY_COMP, 4)
        er = max(er * CLIP_PENALTY_EXP_RATIO, EXP_RATIO_MIN)
        d.expander_range_linear = CLIP_PENALTY_EXP_RANGE
        d.sat_threshold_linear = min(d.sat_threshold_linear + CLIP_PENALTY_SAT, 0.99)
        logger.warning("Clipping penalty applied: expander/saturation reduced")

    d.expander_ratio = round(er, 1)

    # ── Limiter LSP defaults ────────────────────────────────────────
    d.limiter_lk = LIMITER_LOOKAHEAD_S
    d.limiter_at = LIMITER_ATTACK_S
    d.limiter_rt = LIMITER_RELEASE_S
    d.limiter_ovs = LIMITER_OVERSAMPLING

    return d
