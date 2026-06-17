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
    AIR_FREQ_HZ, AIR_Q, CLEAN_FREQ_HZ, CLEAN_Q, MAX_ROOM_MODES,
    BUS_RATIO, BUS_ATTACK_MS, BUS_RELEASE_MS, BUS_THRESH_CREST_FACTOR, BUS_THRESH_OFFSET,
    LIMITER_LOOKAHEAD_S, LIMITER_ATTACK_S, LIMITER_RELEASE_S, LIMITER_OVERSAMPLING,
    DEHARSH_BAND_LOW_HZ, DEHARSH_BAND_HIGH_HZ,
)

logger = logging.getLogger(__name__)

from cleaner.types import AnalysisReport as _AnalysisReportDataclass
AnalysisReport = _AnalysisReportDataclass
from cleaner.types import DerivedParams, MasteringSettings


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


def compute_expander_lsp_params(derived: DerivedParams) -> dict[str, float]:
    """LSP expander_stereo port mapping from DerivedParams."""
    return {
        "em": 1.0,
        "al": round(derived.expander_threshold_linear, 4),
        "er": round(derived.expander_ratio, 1),
        "at": round(derived.expander_attack_ms, 1),
        "rt": round(derived.expander_release_ms, 1),
        "kn": 0.5, "mk": 1.0,
        "g_in": 1.0, "g_out": 1.0,
        "scm": 1.0, "sla": 0.0,
    }


def compute_eq_lsp_params(derived: DerivedParams,
                          settings: MasteringSettings) -> dict[str, float]:
    """LSP para_equalizer_x16_stereo port mapping from DerivedParams + MasteringSettings."""
    params: dict[str, float] = {
        "mode": 0.0, "g_in": 1.0, "g_out": 1.0,
    }

    # Notch bands (0, 1, 2)
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
            params[f"w_{i}"] = 4.0
            params[f"q_{i}"] = round(q_val, 1)
            params[f"g_{i}"] = round(db_to_linear_gain(g_db), 4)

    # Air band (band 3): Bell at 10 kHz
    air_db_val = derived.air_db
    params["s_3"] = 0.0
    if abs(air_db_val) > 0.01:
        params["ft_3"] = 1.0; params["fm_3"] = 0.0
        params["f_3"] = AIR_FREQ_HZ
        params["w_3"] = 4.0
        params["q_3"] = AIR_Q
        params["g_3"] = round(db_to_linear_gain(air_db_val), 4)
    else:
        params["ft_3"] = 0.0; params["fm_3"] = 0.0
        params["f_3"] = AIR_FREQ_HZ; params["w_3"] = 4.0
        params["q_3"] = 0.0; params["g_3"] = 1.0

    # Clean-mediums band (band 4): Bell at 600 Hz
    clean_db = settings.clean_mediums
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
        params[f"ft_{i}"] = 0.0; params[f"fm_{i}"] = 0.0
        params[f"f_{i}"] = 100.0 + i * 200.0
        params[f"w_{i}"] = 4.0; params[f"q_{i}"] = 0.0
        params[f"g_{i}"] = 1.0

    return params


def compute_native_saturation_params(derived: DerivedParams) -> dict[str, float]:
    """Native asoftclip saturation params from DerivedParams."""
    return {
        "sat_drive_db": derived.sat_drive_db,
        "sat_makeup_db": derived.sat_makeup_db,
        "sat_threshold_linear": derived.sat_threshold_linear,
    }


def compute_compressor_lsp_params(derived: DerivedParams,
                                   tracker=None) -> dict[str, float]:
    """LSP compressor_stereo port mapping. Uses tracker RMS when available."""
    # Note: bus threshold depends on pre-compressor level; use tracker if present.
    if tracker and hasattr(tracker, 'current_rms_dbfs'):
        # Recompute with tracked level
        al = round(db_to_linear_gain(
            tracker.current_rms_dbfs - 12.0 * BUS_THRESH_CREST_FACTOR + (1.0 - derived.bus_mix) * BUS_THRESH_OFFSET
        ), 4)
    else:
        al = round(derived.bus_threshold_linear, 4)
    return {
        "cm": 0.0, "al": al,
        "cr": BUS_RATIO, "at": BUS_ATTACK_MS, "rt": BUS_RELEASE_MS,
        "kn": 0.5, "mk": 1.0,
        "cdr": round(max(0.0, 1.0 - derived.bus_mix), 2),
        "cwt": round(derived.bus_mix, 2),
        "g_in": 1.0, "g_out": 1.0, "scm": 1.0, "sla": 0.0,
    }


def compute_limiter_lsp_params(derived: DerivedParams) -> dict[str, float]:
    """LSP limiter_stereo port mapping from DerivedParams."""
    return {
        "mode": 0.0, "th": round(derived.limiter_ceiling_linear, 4),
        "knee": 1.0, "boost": 1.0,
        "lk": LIMITER_LOOKAHEAD_S, "at": LIMITER_ATTACK_S, "rt": LIMITER_RELEASE_S,
        "ovs": LIMITER_OVERSAMPLING, "alr": 1.0,
        "g_in": 1.0, "g_out": 1.0, "scp": 1.0,
    }


def compute_deharsher_lsp_params(derived: DerivedParams) -> dict[str, float]:
    """LSP sc_compressor_stereo de-harsher port mapping from DerivedParams."""
    threshold = max(0.005, derived.deharsher_threshold_linear)
    ratio = derived.deharsher_filter_ratio
    return {
        "cm": 0.0, "al": round(threshold, 4), "cr": round(ratio, 1),
        "at": 5.0, "rt": 30.0, "kn": 0.5, "mk": 1.0,
        "g_in": 1.0, "g_out": 1.0, "scm": 1.0,
        "sct": 1.0, "shpf": DEHARSH_BAND_LOW_HZ, "slpf": DEHARSH_BAND_HIGH_HZ,
        "sla": 0.0, "cdr": 0.0, "cwt": 1.0,
    }
