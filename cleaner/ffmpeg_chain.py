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


def build_filtergraph(stages: dict[str, bool] | None = None,
                      derived=None) -> str:
    """Build the complete ffmpeg filter_complex string.

    Args:
        stages: Dict of stage_name → enabled. If None, all enabled.
        derived: DerivedParams with pre-computed values.
    """
    if stages is None:
        stages = {}

    _defaults_off = {"glue": False, "air": False, "width": False, "bus_comp": False}

    def on(name: str) -> bool:
        return stages.get(name, _defaults_off.get(name, True))

    d = derived

    # ── Parameters (from DerivedParams) ─────────────────────────────
    d = derived

    # Expander
    exp_ratio = d.expander_ratio
    exp_thresh = d.expander_threshold_linear
    exp_attack = d.expander_attack_ms
    exp_release = d.expander_release_ms
    exp_range = d.expander_range_linear

    # Sidechain ducking
    comp_thresh = d.comp_threshold_linear
    comp_ratio = d.comp_ratio
    comp_attack = d.comp_attack_ms
    comp_release = d.comp_release_ms

    # De-harsher
    deharsh_thresh = d.deharsher_threshold_linear
    deharsh_ratio = d.deharsher_filter_ratio
    deharsh_attack = d.deharsher_attack_ms
    deharsh_release = d.deharsher_release_ms

    # Notches
    nf1, nq1, ng1 = d.notch_freq_1, d.notch_q_1, d.notch_gain_1
    nf2, nq2, ng2 = d.notch_freq_2, d.notch_q_2, d.notch_gain_2
    nf3, nq3, ng3 = d.notch_freq_3, d.notch_q_3, d.notch_gain_3

    # Saturation
    sat_thresh = d.sat_threshold_linear
    sat_drive_db = d.sat_drive_db
    sat_makeup_db = d.sat_makeup_db

    # Bus compressor
    bus_thresh = d.bus_threshold_linear
    bus_ratio = d.bus_ratio
    bus_attack = d.bus_attack_ms
    bus_release = d.bus_release_ms
    bus_mix = d.bus_mix

    # Limiter
    limit_lin = d.limiter_ceiling_linear

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
    air_db = d.air_db
    if on("air"):
        tail += f",equalizer=frequency=10000:width_type=q:width=2.0:gain={air_db}"

    # Stage: Width (stereo widening via stereotools)
    w = d.width
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
