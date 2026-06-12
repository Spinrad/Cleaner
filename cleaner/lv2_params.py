"""Parameter conversion and clamping for LV2 ports."""

from __future__ import annotations
import math
from cleaner.lv2_introspect import PortInfo


def db_to_linear_gain(db: float) -> float:
    """Convert dB to linear gain multiplier G (1.0 = 0 dB)."""
    return math.pow(10.0, db / 20.0)


def linear_gain_to_db(g: float) -> float:
    """Convert linear gain multiplier G to dB."""
    if g <= 1e-10:
        return -200.0
    return 20.0 * math.log10(g)


def ms_to_s(ms: float) -> float:
    """Convert milliseconds to seconds."""
    return ms / 1000.0


def s_to_ms(s: float) -> float:
    """Convert seconds to milliseconds."""
    return s * 1000.0


def clamp_to_port(value: float, port: PortInfo, convert_unit: bool = True) -> float:
    """Clamp a value to a port's [min, max] range, optionally converting units.
    
    Args:
        value: The value in dB, ms, or linear as appropriate.
        port: The PortInfo with unit and range.
        convert_unit: If True, convert value to the port's native unit.
    
    Returns:
        Clamped value in the port's native unit.
    """
    if convert_unit:
        if port.unit == "linear_gain":
            value = db_to_linear_gain(value)
        elif port.unit == "s":
            value = ms_to_s(value)
        # dB, ms, Hz, ratio, bool, enum pass through
    
    return max(port.min_val, min(port.max_val, value))


def clamp_to_db_port(value_db: float, port: PortInfo) -> float:
    """Convenience: clamp a dB value for a linear_gain port, converting to G."""
    if port.unit == "linear_gain":
        g = db_to_linear_gain(value_db)
        return max(port.min_val, min(port.max_val, g))
    else:
        return max(port.min_val, min(port.max_val, value_db))


# ── Explicit unit table (confirmed via lv2info/ffmpeg introspection) ──
# Overrides heuristic _infer_unit for ports we pilot.
# Format: {symbol: unit_string} where unit is one of:
#   "s", "ms", "dB", "linear_gain", "Hz", "ratio", "bool", "enum"

EXPLICIT_UNITS: dict[str, str] = {
    # --- expander_stereo ---
    "em": "enum",       # 0=Down, 1=Up
    "al": "linear_gain", # threshold as G multiplier
    "er": "ratio",
    "at": "ms",         # 0-2000 ms
    "rt": "ms",         # 0-5000 ms
    "kn": "linear_gain",
    "mk": "linear_gain",
    "g_in": "linear_gain",
    "g_out": "linear_gain",
    "scm": "enum",
    "sla": "s",         # 0-20 seconds

    # --- para_equalizer_x16_stereo ---
    "mode": "enum",
    "ft_0": "enum", "ft_1": "enum", "ft_2": "enum", "ft_3": "enum",
    "fm_0": "enum", "fm_1": "enum", "fm_2": "enum", "fm_3": "enum",
    "s_0": "enum", "s_1": "enum", "s_2": "enum", "s_3": "enum",
    "f_0": "Hz", "f_1": "Hz", "f_2": "Hz", "f_3": "Hz",
    "w_0": "ratio", "w_1": "ratio", "w_2": "ratio", "w_3": "ratio",
    "g_0": "linear_gain", "g_1": "linear_gain", "g_2": "linear_gain", "g_3": "linear_gain",
    "q_0": "ratio", "q_1": "ratio", "q_2": "ratio", "q_3": "ratio",

    # --- compressor_stereo ---
    "cm": "enum",       # 0=Down, 1=Up, 2=Boost
    "cr": "ratio",
    "cdr": "linear_gain",
    "cwt": "linear_gain",
    # (al, at, rt, kn, mk, g_in, g_out, scm, sla already covered above)

    # --- limiter_stereo ---
    "th": "linear_gain", # threshold as G
    "knee": "linear_gain",
    "boost": "linear_gain",
    "lk": "s",          # 0.1-20 seconds
    "ovs": "enum",      # 0-20, oversampling factor
    "alr": "bool",      # 0-1, adaptive release
    "scp": "linear_gain",
    # (g_in, g_out, mode already covered; limiter at/rt are in seconds, override)

    # --- sc_compressor_stereo ---
    "sct": "enum",      # sidechain type
    "shpf": "Hz",
    "slpf": "Hz",
    # (cm, al, cr, at, rt, kn, mk, g_in, g_out, scm, sla, cdr, cwt already covered)
}

# Override: limiter at/rt are in SECONDS (range 0-20), not ms
EXPLICIT_UNITS["limiter_at"] = "s"  # not a real symbol, just documentation
# Actual override: these symbols appear in limiter context
_LIMITER_TIME_PORTS = {"at", "rt", "lk", "alr_at", "alr_rt"}
for _sym in _LIMITER_TIME_PORTS:
    EXPLICIT_UNITS[_sym] = "s"  # limiter uses seconds for time ports


def get_port_unit(symbol: str, fallback_infer_fn=None) -> str:
    """Get the confirmed unit for a port symbol.

    Checks explicit unit table first, falls back to inference.
    """
    if symbol in EXPLICIT_UNITS:
        return EXPLICIT_UNITS[symbol]
    if fallback_infer_fn:
        return fallback_infer_fn(symbol)
    return ""
