"""
FFmpeg native filterchain builder — used by --force-native.

Builds a pure ffmpeg filter_complex string from AnalysisReport.
This is the legacy v0.1 native path; the default LSP path uses
lsp_chain_builder.py instead.

Full native chain:
  HP 35Hz -> Expander (agate) -> MS encode -> split M/S
    -> Side: HP 150Hz -> sidechaincompress (Mid triggers Side)
    -> Merge -> MS decode -> De-harsher (adynamicequalizer)
    -> Notches x3 (anequalizer) -> Saturation (asoftclip)
    -> Limiter (alimiter) -> Output
"""

from __future__ import annotations
from typing import Any


def build_filtergraph(report: dict[str, Any] | None = None,
                      stages: dict[str, bool] | None = None,
                      derived=None) -> str:
    """Build the complete ffmpeg filter_complex string.

    Args:
        report: Legacy AnalysisReport dict (optional, prefer derived).
        stages: Dict of stage_name → enabled. If None, all enabled.
        derived: DerivedParams with pre-computed values (preferred).
    """
    if stages is None:
        stages = {}

    # Stages that default to OFF (mastering flags)
    _defaults_off = {"glue": False, "air": False, "width": False, "bus_comp": False}

    def on(name: str) -> bool:
        return stages.get(name, _defaults_off.get(name, True))

    # ── Parameters (derived preferred, report fallback) ─────────────
    d = derived
    r = report or {}

    # Expander
    exp_ratio = d.expander_ratio if d else r.get("expander_ratio", 2.0)
    exp_thresh = d.expander_threshold_linear if d else r.get("expander_threshold_linear", 0.05)
    exp_attack = d.expander_attack_ms if d else r.get("expander_attack_ms", 5.0)
    exp_release = d.expander_release_ms if d else r.get("expander_release_ms", 40.0)
    exp_range = r.get("expander_range_linear", 0.25) if not d else 0.25

    # Sidechain ducking
    comp_thresh = d.comp_threshold_linear if d else r.get("comp_threshold_linear", 0.05)
    comp_ratio = d.comp_ratio if d else r.get("comp_ratio", 6)
    comp_attack = d.comp_attack_ms if d else r.get("comp_attack_ms", 0.5)
    comp_release = d.comp_release_ms if d else r.get("comp_release_ms", 60.0)

    # De-harsher
    deharsh_thresh = d.deharsher_threshold_linear if d else r.get("deharsher_threshold_linear", 5.0)
    deharsh_ratio = d.deharsher_filter_ratio if d else r.get("deharsher_filter_ratio", 3.0)
    deharsh_attack = d.deharsher_attack_ms if d else r.get("deharsher_attack_ms", 3.0)
    deharsh_release = d.deharsher_release_ms if d else r.get("deharsher_release_ms", 30.0)

    # Notches — use derived for type safety
    if d:
        nf1, nq1, ng1 = d.notch_freq_1, d.notch_q_1, d.notch_gain_1
        nf2, nq2, ng2 = d.notch_freq_2, d.notch_q_2, d.notch_gain_2
        nf3, nq3, ng3 = d.notch_freq_3, d.notch_q_3, d.notch_gain_3
    else:
        nf1, nq1, ng1 = r.get("notch_freq_1", 300), r.get("notch_q_1", 20), r.get("notch_gain_1", -6)
        nf2, nq2, ng2 = r.get("notch_freq_2", 450), r.get("notch_q_2", 20), r.get("notch_gain_2", -5)
        nf3, nq3, ng3 = r.get("notch_freq_3", 600), r.get("notch_q_3", 20), r.get("notch_gain_3", -4)

    # Saturation
    sat_thresh = d.sat_threshold_linear if d else r.get("sat_threshold_linear", 0.85)
    sat_drive_db = d.sat_drive_db if d else r.get("sat_drive_db", 1.2)
    sat_makeup_db = d.sat_makeup_db if d else r.get("sat_makeup_db", -0.7)

    # Bus compressor
    bus_thresh = d.bus_threshold_linear if d else r.get("bus_threshold_linear", 0.18)
    bus_ratio = d.bus_ratio if d else r.get("bus_ratio", 2)
    bus_attack = d.bus_attack_ms if d else r.get("bus_attack_ms", 10)
    bus_release = d.bus_release_ms if d else r.get("bus_release_ms", 100)
    bus_mix = d.bus_mix if d else r.get("bus_mix", 0.0)

    # Limiter
    limit_lin = d.limiter_ceiling_linear if d else r.get("limiter_ceiling_linear", 0.88)

    # ── Build chains ────────────────────────────────────────────
    parts: list[str] = []

    # Chain start: resample to 48k
    chain = "[0:a]aresample=48000"
    if on("hp35"):
        chain += ",highpass=f=35:t=o:p=2"

    # Stage 1: Expander (anti-AGC)
    if on("expander"):
        chain += (
            f",agate=mode=upward:threshold={exp_thresh}:ratio={exp_ratio}:"
            f"attack={exp_attack}:release={exp_release}:range={exp_range}:makeup=1"
        )

    # Stage 2: MS encode
    chain += ",stereotools=mode=lr>ms[ms]"

    if on("ducking"):
        # Split MS, process side with mid sidechain
        # Need asplit because sidechaincompress won't share its sidechain pad
        parts.append(f"[ms]channelsplit=channel_layout=stereo[mid][side]")
        parts.append(f"[mid]asplit=2[mid_sc][mid_main]")
        if on("hp150"):
            parts.append(f"[side]highpass=f=150:t=o:p=2[side_hp]")
        else:
            parts.append(f"[side]anull[side_hp]")
        parts.append(
            f"[side_hp][mid_sc]sidechaincompress="
            f"threshold={comp_thresh}:ratio={comp_ratio}:"
            f"attack={comp_attack}:release={comp_release}:level_sc=1"
            f"[side_ducked]"
        )
        parts.append(
            f"[mid_main][side_ducked]amerge=inputs=2,"
            f"channelmap=0|1:channel_layout=stereo"
            f"[ms_dec]"
        )
        tail = "[ms_dec]stereotools=mode=ms>lr"
    else:
        # No ducking: just pass MS through, HP 150 on Side if enabled
        parts.append(f"[ms]channelsplit=channel_layout=stereo[mid_raw][side_raw]")
        if on("hp150"):
            parts.append(f"[side_raw]highpass=f=150:t=o:p=2[side_hp2]")
        else:
            parts.append(f"[side_raw]anull[side_hp2]")
        parts.append(
            f"[mid_raw][side_hp2]amerge=inputs=2,"
            f"channelmap=0|1:channel_layout=stereo"
            f"[ms_nodec]"
        )
        tail = "[ms_nodec]stereotools=mode=ms>lr"

    # Stage 4: De-harsher
    if on("deharsher"):
        tail += (
            f",adynamicequalizer="
            f"dftype=highpass:dfrequency=2500:dqfactor=0.7:"
            f"tfrequency=3500:tqfactor=1.0:tftype=bell:"
            f"mode=cut:ratio={deharsh_ratio}:threshold={deharsh_thresh}:"
            f"attack={deharsh_attack}:release={deharsh_release}:"
            f"range=6:makeup=0"
        )

    # Stage 5: Notch filters
    if on("notches"):
        tail += (
            f",anequalizer="
            f"c0 f={nf1} w={nq1} g={ng1} t=7|"
            f"c1 f={nf2} w={nq2} g={ng2} t=7|"
            f"c2 f={nf3} w={nq3} g={ng3} t=7"
        )

    # Stage 6: Saturation (volume → asoftclip → volume)
    if on("saturation"):
        tail += (
            f",volume={sat_drive_db}dB,"
            f"asoftclip=type=tanh:threshold={sat_thresh}:output=1.0:oversample=4,"
            f"volume={sat_makeup_db}dB"
        )

    # Stage: Air (Bell at 10 kHz, Q=2.0)
    air_db = d.air_db if d else r.get("_air_db", 0.0)
    if on("air"):
        tail += f",equalizer=frequency=10000:width_type=q:width=2.0:gain={air_db}"

    # Stage: Width (stereo widening via stereotools)
    w = d.width if d else r.get("_width", 0.0)
    if on("width"):
        tail += f",stereotools=mode=lr>lr:base={w}:slev=1:mlev=1"

    # Stage: Bus compressor (SSL glue, parallel)
    if on("bus_comp"):
        tail += (
            f",acompressor=mode=downward:threshold={bus_thresh}:"
            f"ratio={bus_ratio}:attack={bus_attack}:release={bus_release}:"
            f"knee=4:makeup=1:detection=rms:link=average:mix={bus_mix}"
        )

    # Stage 7: Limiter
    if on("limiter"):
        tail += f",alimiter=limit={limit_lin}:attack=0.1:release=30:level=true"

    # Output pad
    tail += "[out]"

    # Assemble: sub-chain parts + main head + main tail
    graph = ";".join(parts + [chain, tail])

    return graph
