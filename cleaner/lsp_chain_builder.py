"""
LSP/LV2 filter_complex chain builder.

Builds ffmpeg filter_complex segments using lv2=... nodes.
Handles URI escaping and control parameter formatting.
"""

from __future__ import annotations


def build_lv2_node(uri: str, params: dict[str, float]) -> str:
    """Build a single lv2 filter node for ffmpeg filter_complex.
    
    Args:
        uri: LV2 plugin URI (e.g. http://lsp-plug.in/plugins/lv2/compressor_stereo)
        params: dict of symbol -> value (in the port's native unit,
                already converted and clamped).
    
    Returns:
        A string like: lv2=p='http\\\\://lsp-plug.in/...':c=sym1=val1|sym2=val2
    
    The URI is escaped (every ':' becomes '\\\\:') for ffmpeg's parser.
    Control values are joined with '|' and separated from the URI by ':'.
    """
    escaped_uri = uri.replace(":", "\\\\:")
    controls = "|".join(f"{sym}={val}" for sym, val in sorted(params.items()))
    return f"lv2=p={escaped_uri}:c={controls}"


def build_lv2_node_help(uri: str) -> str:
    """Build an lv2 node in help mode (c=help), used for introspection."""
    escaped_uri = uri.replace(":", "\\\\:")
    return f"lv2=p={escaped_uri}:c=help"


def build_filtergraph_preamble() -> str:
    """Build the native ffmpeg preamble shared by LSP and native chains.
    
    Returns the head chain string (resample, HP35, M/S encode).
    Does NOT include the expander (handled per-builder).
    """
    return "[0:a]aresample=48000,highpass=f=35:t=o:p=2"


def build_ms_sidechain_block(ducking_enabled: bool,
                              comp_threshold: float,
                              comp_ratio: float,
                              comp_attack: float,
                              comp_release: float,
                              hp150_enabled: bool = True) -> tuple[list[str], str]:
    """Build the M/S encode + sidechain ducking + M/S decode block.
    
    Returns (sub_chain_parts, tail_prefix) where:
    - sub_chain_parts are filtergraph segments joined with ';'
    - tail_prefix is the pad name to chain subsequent filters onto
    
    This block is ALWAYS native ffmpeg — no LSP involved.
    """
    parts: list[str] = []
    
    if ducking_enabled:
        parts.append("[ms]channelsplit=channel_layout=stereo[mid][side]")
        parts.append("[mid]asplit=2[mid_sc][mid_main]")
        if hp150_enabled:
            parts.append("[side]highpass=f=150:t=o:p=2[side_hp]")
        else:
            parts.append("[side]anull[side_hp]")
        parts.append(
            f"[side_hp][mid_sc]sidechaincompress="
            f"threshold={comp_threshold}:ratio={comp_ratio}:"
            f"attack={comp_attack}:release={comp_release}:level_sc=1"
            f"[side_ducked]"
        )
        parts.append(
            f"[mid_main][side_ducked]amerge=inputs=2,"
            f"channelmap=0|1:channel_layout=stereo"
            f"[ms_dec]"
        )
        tail_prefix = "[ms_dec]stereotools=mode=ms>lr"
    else:
        parts.append("[ms]channelsplit=channel_layout=stereo[mid_raw][side_raw]")
        if hp150_enabled:
            parts.append("[side_raw]highpass=f=150:t=o:p=2[side_hp2]")
        else:
            parts.append("[side_raw]anull[side_hp2]")
        parts.append(
            f"[mid_raw][side_hp2]amerge=inputs=2,"
            f"channelmap=0|1:channel_layout=stereo"
            f"[ms_nodec]"
        )
        tail_prefix = "[ms_nodec]stereotools=mode=ms>lr"
    
    return parts, tail_prefix


def build_postamble(target_lufs: float, ceiling: float) -> str:
    """Build the LUFS measurement + post-limiter tail (always native)."""
    return (
        f"ebur128=peak=true:framelog=quiet,"
        f"volume=0dB,"  # gain applied later in io_adapter
        f"alimiter=limit={10.0**(ceiling/20.0):.4f}:attack=0.1:release=30:level=true"
        f"[out]"
    )


def build_lsp_filtergraph(report: dict, stages: dict[str, bool]) -> str:
    """Build the complete LSP filter_complex graph.
    
    Architecture:
      Preamble (native) → Expander (LSP) → M/S+ducking (native) → De-harsher (LSP)
      → EQ (LSP) → Saturator (LSP) → Compressor (LSP) → Limiter (LSP)
      → Postamble (native LUFS + alimiter)
    
    Args:
        report: AnalysisReport with LSP params already computed.
        stages: Stage enable/disable dict.
    
    Returns:
        A complete filter_complex string.
    """
    from cleaner.analysis.global_analysis import (
        compute_expander_lsp_params,
        compute_eq_lsp_params,
        compute_compressor_lsp_params,
        compute_limiter_lsp_params,
        compute_deharsher_lsp_params,
        compute_loud_comp_lsp_params,
    )
    from cleaner.gain_tracking import GainTracker
    from cleaner.lv2_introspect import get_plugin_info
    from cleaner.lv2_params import clamp_to_port
    from cleaner.lsp_uris import (EXPANDER_URI, EQ_URI, SATURATOR_URI,
                                  COMPRESSOR_URI, LIMITER_URI, DEHARSHER_URI)
    
    peak_db = report.get("peak_db", -3.0)
    rms_db = report.get("rms_db", -15.0)
    tracker = GainTracker(peak_db, rms_db)
    
    def on(name: str) -> bool:
        defaults_off = {"glue": False, "air": False, "width": False, "bus_comp": False,
                         "deharsher": False, "intensity": True}
        return stages.get(name, not defaults_off.get(name, True))
    
    def _clamped_lv2_node(uri, compute_fn, *args):
        """Build a clamped LV2 node from a compute function."""
        params = compute_fn(*args)
        plugin = get_plugin_info(uri)
        clamped = {}
        for sym, val in params.items():
            port = plugin.ports.get(sym) if plugin else None
            if port:
                clamped[sym] = clamp_to_port(val, port, convert_unit=False)
            else:
                clamped[sym] = val
        return build_lv2_node(uri, clamped)
    
    parts: list[str] = []
    
    # ── Preamble ──
    chain = build_filtergraph_preamble()
    
    # ── Stage 1: Expander (LSP, replaces agate) ──
    if on("expander"):
        chain += "," + _clamped_lv2_node(EXPANDER_URI, compute_expander_lsp_params, report, tracker)
        tracker.commit("expander", 0.0, "anti-AGC")
    
    # ── M/S encode ──
    chain += ",stereotools=mode=lr>ms[ms]"
    
    # ── Sidechain ducking (native) ──
    if on("ducking"):
        comp_thresh = report.get("comp_threshold_linear", 0.05)
        comp_ratio = report.get("comp_ratio", 4)
        comp_attack = report.get("comp_attack_ms", 2.0)
        comp_release = report.get("comp_release_ms", 60.0)
        hp150_on = on("hp150")
    else:
        comp_thresh = 0.05
        comp_ratio = 4
        comp_attack = 2.0
        comp_release = 60.0
        hp150_on = False
    
    ms_parts, tail_prefix = build_ms_sidechain_block(
        on("ducking"), comp_thresh, comp_ratio, comp_attack, comp_release, hp150_on
    )
    parts.extend(ms_parts)
    
    tail = tail_prefix
    tracker.commit("ms_ducking", 0.0, "M/S sidechain")
    
    # ── Stage: De-harsher (LSP, opt-in, before saturator) ──
    if on("deharsher"):
        tail += "," + _clamped_lv2_node(DEHARSHER_URI, compute_deharsher_lsp_params, report, tracker)
        tracker.commit("deharsher", -0.5, "band cut 2.5-4.5kHz")
    
    # ── Stage: EQ notches + air (LSP) ──
    if on("notches") or on("air"):
        tail += "," + _clamped_lv2_node(EQ_URI, compute_eq_lsp_params, report, tracker)
        notch_gain = -1.0 if on("notches") else 0.0
        air_gain = report.get("_air", 1.5) if on("air") else 0.0
        tracker.commit("eq", notch_gain + air_gain, "notches + air")
    
    # ── Stage: Saturator (LSP loud_comp) ──
    if on("saturation") and on("glue"):
        tail += "," + _clamped_lv2_node(SATURATOR_URI, compute_loud_comp_lsp_params, report, tracker)
        glue = report.get("_glue", 0.15)
        intensity = report.get("_intensity", 0.5)
        eff_glue = glue * (0.3 + intensity * 0.7)
        sat_net_gain = eff_glue * 16.0 * 0.6
        tracker.commit("saturator", sat_net_gain, "drive + clip + makeup")
    
    # ── Stage: Bus Compressor (LSP) ──
    if on("bus_comp"):
        tail += "," + _clamped_lv2_node(COMPRESSOR_URI, compute_compressor_lsp_params, report, tracker)
        bus = report.get("_bus_comp", 0.0)
        comp_gain = -bus * 4.0
        tracker.commit("compressor", comp_gain, "bus glue")
    
    # ── Stage: Limiter (LSP) ──
    if on("limiter"):
        tail += "," + _clamped_lv2_node(LIMITER_URI, compute_limiter_lsp_params, report, tracker)
        tracker.commit("limiter", 0.0, "peak ceiling")
    
    # ── Stage: Width (native stereotools) ──
    w = report.get("_width", 0.0)
    if on("width"):
        tail += f",stereotools=mode=lr>lr:base={w}:slev=1:mlev=1"
    
    # ── Postamble: LUFS + safety limiter (native) ──
    tail += ",ebur128=peak=true:framelog=quiet"
    ceiling = report.get('_ceiling_db', -1.1)
    post_ceiling = ceiling + 0.3 if on("limiter") else ceiling
    post_ceiling = min(post_ceiling, -0.3)  # never above -0.3 dBFS
    tail += f",alimiter=limit={10.0**(post_ceiling/20.0):.4f}:attack=0.1:release=30:level=true"
    tail += "[out]"
    
    # ── Assemble ──
    graph = ";".join(parts + [chain, tail])
    return graph
