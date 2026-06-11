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
