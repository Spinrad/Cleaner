"""Typed structures replacing the dict[str, Any] report.

Three layers, each with a single source of truth for defaults:
  1. AnalysisReport (frozen) — raw measurements from audio analysis.
  2. MasteringSettings — user-facing configuration knobs.
  3. DerivedParams — computed DSP parameters for both builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cleaner.constants import (
    NOTCH_DEFAULT_HZ, NOTCH_DEFAULT_Q, NOTCH_DEFAULT_GAIN_DB, MAX_ROOM_MODES,
)


@dataclass(frozen=True)
class AnalysisReport:
    """Immutable audio measurements.  Defaults match the old FALLBACK dicts."""

    # Dynamique
    peak_db: float = -3.0
    rms_db: float = -15.0
    crest_factor_db: float = 12.0
    transient_attack_ms: float = 10.0
    transient_crest_local_db: float = 12.0
    agc_recovery_ms: float = 80.0

    # Modes de salle
    room_modes_hz: tuple[float, ...] = NOTCH_DEFAULT_HZ
    room_mode_qs: tuple[float, ...] = NOTCH_DEFAULT_Q
    room_mode_gains_db: tuple[float, ...] = NOTCH_DEFAULT_GAIN_DB

    # Spectral
    harshness_band_energy_db: float = -20.0
    spectral_centroid_hz: float = 2000.0
    low_mid_energy_db: float = -18.0
    harshness_index: float = 0.0

    # Stéréo
    ms_correlation_avg: float = 0.5
    side_energy_ratio: float = 0.3
    hf_correlation: float = 0.4

    # Écrêtage
    is_heavily_clipped: bool = False
    clip_ratio: float = 0.0
    max_consecutive_clips: int = 0
    total_clipped_samples: int = 0
    total_samples: int = 1

    # Méta
    duration_s: float = 0.0
    sample_rate: int = 48000

    # Warnings (non-frozen field for mutation after construction)
    _analysis_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """For backward compat with code that still expects a dict."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class MasteringSettings:
    """User-facing configuration.  Defaults match CLI defaults."""

    glue: float = 0.15
    air: float = 0.0
    width: float = 0.0
    bus_comp: float = 0.0
    intensity: float = 0.5
    ceiling_db: float = -1.1
    target_lufs: float = -14.0
    notch_multiplier: float = 1.0
    tame_cymbals: float = 0.0
    clean_mediums: float = 0.0


@dataclass
class DerivedParams:
    """DSP parameters computed from AnalysisReport + MasteringSettings."""

    # Expander (native + display)
    expander_threshold_linear: float = 0.1
    expander_ratio: float = 1.1
    expander_range_linear: float = 0.1
    expander_attack_ms: float = 5.0
    expander_release_ms: float = 40.0

    # Sidechain ducking
    comp_threshold_linear: float = 0.1
    comp_ratio: float = 4.0
    comp_attack_ms: float = 2.0
    comp_release_ms: float = 60.0

    # Notches
    notch_freq_1: float = NOTCH_DEFAULT_HZ[0]
    notch_q_1: float = NOTCH_DEFAULT_Q[0]
    notch_gain_1: float = 0.0
    notch_freq_2: float = NOTCH_DEFAULT_HZ[1]
    notch_q_2: float = NOTCH_DEFAULT_Q[1]
    notch_gain_2: float = 0.0
    notch_freq_3: float = NOTCH_DEFAULT_HZ[2]
    notch_q_3: float = NOTCH_DEFAULT_Q[2]
    notch_gain_3: float = 0.0

    # Saturation
    sat_drive_db: float = 0.0
    sat_makeup_db: float = 0.0
    sat_threshold_linear: float = 0.92
    sat_glue: float = 0.15

    # Air
    air_db: float = 0.0

    # Width
    width: float = 0.0

    # Bus compressor
    bus_threshold_linear: float = 0.18
    bus_mix: float = 0.0
    bus_ratio: float = 2.0
    bus_attack_ms: float = 10.0
    bus_release_ms: float = 100.0

    # Limiter
    limiter_ceiling_linear: float = 0.88

    # De-harsher (native only)
    deharsher_threshold_linear: float = 10.0
    deharsher_filter_ratio: float = 2.0
    deharsher_attack_ms: float = 5.0
    deharsher_release_ms: float = 40.0
    deharsher_display_threshold: float = 10.0
