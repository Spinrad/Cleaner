"""
LSP/LV2 filter_complex chain builder.

Builds ffmpeg filter_complex segments using lv2=... nodes.
Handles URI escaping and control parameter formatting.
"""

from __future__ import annotations

from cleaner.analysis.global_analysis import (
    compute_expander_lsp_params,
    compute_eq_lsp_params,
    compute_deharsher_lsp_params,
    compute_compressor_lsp_params,
    compute_limiter_lsp_params,
    compute_native_saturation_params,
)
from cleaner.gain_tracking import GainTracker
from cleaner.lv2_introspect import get_plugin_info
from cleaner.lv2_params import clamp_to_port
from cleaner.lsp_uris import (EXPANDER_URI, EQ_URI,
                              DEHARSHER_URI, COMPRESSOR_URI, LIMITER_URI)
from cleaner.constants import POST_LIMITER_ATTACK_MS, POST_LIMITER_RELEASE_MS


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


def build_lsp_filtergraph(report: dict, stages: dict[str, bool],
                          derived=None, settings=None) -> str:
    """Build the complete LSP filter_complex graph.
    
    Args:
        report: AnalysisReport dict (for tracker init only, being phased out).
        stages: Stage enable/disable dict.
        derived: DerivedParams for pre-computed values.
        settings: MasteringSettings for user config.
    """

    peak_db = report.get("peak_db", -3.0)
    rms_db = report.get("rms_db", -15.0)
    tracker = GainTracker(peak_db, rms_db)
    
    def on(name: str) -> bool:
        defaults_off = {"glue": False, "air": False, "width": False, "bus_comp": False,
                         "deharsher": False, "intensity": True}
        return stages.get(name, not defaults_off.get(name, True))
    
    def _clamped_lv2_node(uri, compute_fn, *args):
        """Build a clamped LV2 node from a compute function.
        
        Raises RuntimeError if any symbol is not found in the plugin's ports.
        """
        params = compute_fn(*args)
        plugin = get_plugin_info(uri)
        if plugin is None or not plugin.ports:
            raise RuntimeError(
                f"LV2 plugin introspection failed for {uri}. "
                f"Cannot validate port symbols."
            )
        clamped = {}
        for sym, val in params.items():
            port = plugin.ports.get(sym)
            if port is None:
                raise RuntimeError(
                    f"Unknown port symbol '{sym}' for plugin {uri}. "
                    f"Available ports: {sorted(plugin.ports.keys())}"
                )
            clamped[sym] = clamp_to_port(val, port, convert_unit=True)
        return build_lv2_node(uri, clamped)
    
    parts: list[str] = []
    
    # ── Preamble ──
    chain = build_filtergraph_preamble()
    
    # ── Stage 1: Expander (LSP, replaces agate) ──
    if on("expander"):
        chain += "," + _clamped_lv2_node(EXPANDER_URI, compute_expander_lsp_params, derived)
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
        tail += "," + _clamped_lv2_node(DEHARSHER_URI, compute_deharsher_lsp_params, derived)
        deharsh_reduction = -max(0.1, report.get("harshness_index", 0.0) * 1.5)
        tracker.commit("deharsher", deharsh_reduction, "band cut 2.5-4.5kHz")
    
    # ── Stage: EQ notches + air (LSP) ──
    if on("notches") or on("air"):
        tail += "," + _clamped_lv2_node(EQ_URI, compute_eq_lsp_params, derived, settings)
        # Derive actual RMS impact from computed notch gains
        notch_gain = 0.0
        if on("notches"):
            for j in (1, 2, 3):
                g = report.get(f"notch_gain_{j}", 0.0)
                if g < -0.5:
                    notch_gain += g
        air_gain = 0.0  # Bell at 10 kHz has negligible RMS impact
        tracker.commit("eq", notch_gain + air_gain, "notches + air")
    
    # ── Stage: Saturation (native asoftclip tanh) ──
    if on("saturation") and on("glue"):
        sat_params = compute_native_saturation_params(derived)
        tail += (
            f",volume={sat_params['sat_drive_db']}dB,"
            f"asoftclip=type=tanh:threshold={sat_params['sat_threshold_linear']}:output=1.0:oversample=4,"
            f"volume={sat_params['sat_makeup_db']}dB"
        )
        sat_net_gain = sat_params['sat_drive_db'] + sat_params['sat_makeup_db']
        # Tanh clipper reduces RMS by ~15% of drive amount
        sat_rms_gain = sat_net_gain - abs(sat_params['sat_drive_db']) * 0.15
        tracker.commit("saturation", sat_rms_gain, "drive + tanh + makeup")
    
    # ── Stage: Bus Compressor (LSP) ──
    if on("bus_comp"):
        tail += "," + _clamped_lv2_node(COMPRESSOR_URI, compute_compressor_lsp_params, derived, tracker)
        bus = report.get("_bus_comp", 0.0)
        comp_gain = -bus * 2.0  # moderate compression gain reduction
        tracker.commit("compressor", comp_gain, "bus glue")
    
    # ── Stage: Width (native stereotools) ──
    w = report.get("_width", 0.0)
    if on("width"):
        tail += f",stereotools=mode=lr>lr:base={w}:slev=1:mlev=1"
    
    # ── Stage: Limiter (LSP) ──
    if on("limiter"):
        tail += "," + _clamped_lv2_node(LIMITER_URI, compute_limiter_lsp_params, derived)
        tracker.commit("limiter", 0.0, "peak ceiling")
    
    # ── Postamble: safety limiter (native) ──
    ceiling = report.get('_ceiling_db', -1.1)
    post_ceiling = ceiling + 0.3 if on("limiter") else ceiling
    post_ceiling = min(post_ceiling, -0.3)  # never above -0.3 dBFS
    tail += f",alimiter=limit={10.0**(post_ceiling/20.0):.4f}:attack={POST_LIMITER_ATTACK_MS}:release={POST_LIMITER_RELEASE_MS}:level=true"
    tail += "[out]"
    
    # ── Assemble ──
    graph = ";".join(parts + [chain, tail])
    return graph
