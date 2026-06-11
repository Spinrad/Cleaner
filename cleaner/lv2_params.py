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
