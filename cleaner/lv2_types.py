"""Shared LV2 types — no imports from lv2_introspect or lv2_params.

Extracted to break the import cycle:
  lv2_params ⇢ lv2_introspect (PortInfo)
  lv2_introspect ⇢ lv2_params (EXPLICIT_UNITS)

Both modules now depend on lv2_types instead of each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortInfo:
    symbol: str
    name: str = ""
    min_val: float = 0.0
    max_val: float = 1.0
    default_val: float = 0.0
    unit: str = ""


@dataclass
class PluginInfo:
    uri: str
    ports: dict[str, PortInfo] = field(default_factory=dict)
