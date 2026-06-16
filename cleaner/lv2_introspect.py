"""
LV2 plugin discovery and introspection.

Uses lv2file for discovery, ffmpeg for port introspection.
Caches results to avoid repeated subprocess calls.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path("/tmp/cleaner_lv2_cache.json")
LSP_URI_PREFIX = "http://lsp-plug.in/plugins/lv2/"


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


def _run_lv2ls_list() -> list[str]:
    """Run lv2ls and return list of URIs."""
    try:
        result = subprocess.run(
            ["lv2ls"], capture_output=True, text=True, timeout=15
        )
        uris = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("http") or line.startswith("urn:"):
                uris.append(line)
        if not uris and result.stderr:
            for line in result.stderr.splitlines():
                line = line.strip()
                if line.startswith("http") or line.startswith("urn:"):
                    uris.append(line)
        return uris
    except FileNotFoundError:
        return _run_lv2file_list()
    except Exception as exc:
        logger.warning("lv2ls failed: %s", exc)
        return _run_lv2file_list()


def _run_lv2file_list() -> list[str]:
    """Fallback discovery via lv2file -l."""
    try:
        result = subprocess.run(
            ["lv2file", "-l"], capture_output=True, text=True, timeout=15
        )
        uris = []
        output = result.stdout if result.stdout else result.stderr
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                line = line.split("\t", 1)[-1]
            if line.startswith("http") or line.startswith("urn:"):
                uris.append(line)
        return uris
    except FileNotFoundError:
        logger.warning("lv2file not found — LV2 discovery disabled")
        return []
    except Exception as exc:
        logger.warning("lv2file -l failed: %s", exc)
        return []


def discover_plugins(prefix: str = LSP_URI_PREFIX) -> dict[str, str]:
    all_uris = _run_lv2ls_list()
    found: dict[str, str] = {}
    for uri in all_uris:
        if uri.startswith(prefix):
            slug = uri.rsplit("/", 1)[-1]
            found[slug] = uri
    logger.info("Discovered %d plugins matching %s", len(found), prefix)
    return found


def _ffmpeg_escape_uri(uri: str) -> str:
    return uri.replace(":", "\\\\:")


def _infer_unit(symbol: str, desc: str, min_v: float, max_v: float, default_v: float) -> str:
    combined = f"{symbol} {desc}".lower()
    if any(w in combined for w in ("time", "attack", "release", "lookahead", "delay")):
        return "ms" if max_v > 20 else "s"
    if any(w in combined for w in ("freq", "frequency", "hz", "pitch")):
        return "Hz"
    if any(w in combined for w in ("gain", "makeup", "boost", "level", "volume",
                                     "input", "output", "drive", "dry", "wet")):
        return "linear_gain" if max_v >= 10 or (0 <= min_v and max_v <= 2) else "dB"
    if any(w in combined for w in ("ratio",)):
        return "ratio"
    if any(w in combined for w in ("threshold", "knee", "ceiling", "limit", "thresh")):
        if min_v < -1:
            return "dB"
        return "linear_gain"
    if max_v <= 1.01 and min_v >= -0.01:
        return "bool"
    if max_v <= 50 and min_v >= 0 and max_v == int(max_v):
        return "enum"
    return ""


def _parse_ffmpeg_lv2_help(stderr: str) -> dict[str, PortInfo]:
    ports: dict[str, PortInfo] = {}
    pattern = re.compile(
        r"\]\s+(\S+)\s+<float>\s+\(from\s+([\d.\-e]+)\s+to\s+([\d.\-e]+)\)\s+"
        r"\(default\s+([\d.\-e]+)\)\s*(.*)"
    )
    for line in stderr.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        symbol, min_s, max_s, def_s, desc = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5).strip()
        try:
            min_v, max_v, def_v = float(min_s), float(max_s), float(def_s)
        except ValueError:
            continue
        from cleaner.lv2_params import EXPLICIT_UNITS
        if symbol in EXPLICIT_UNITS:
            unit = EXPLICIT_UNITS[symbol]
        else:
            unit = _infer_unit(symbol, desc, min_v, max_v, def_v)
        ports[symbol] = PortInfo(
            symbol=symbol, name=desc,
            min_val=min_v, max_val=max_v, default_val=def_v,
            unit=unit,
        )
    return ports


def introspect_plugin(uri: str) -> PluginInfo:
    escaped = _ffmpeg_escape_uri(uri)
    filter_str = f"lv2=p={escaped}:c=help"
    cmd = [
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc",
        "-filter_complex", filter_str,
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        ports = _parse_ffmpeg_lv2_help(result.stderr)
        logger.info("Introspected %s: %d ports", uri, len(ports))
        return PluginInfo(uri=uri, ports=ports)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found")
    except subprocess.TimeoutExpired:
        logger.warning("Introspection timed out for %s", uri)
        return PluginInfo(uri=uri)
    except Exception as exc:
        logger.warning("Introspection failed for %s: %s", uri, exc)
        return PluginInfo(uri=uri)


def _get_version_info() -> dict[str, str]:
    info = {}
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        info["ffmpeg"] = r.stdout.splitlines()[0] if r.stdout else "unknown"
    except Exception as exc:
        logger.warning("ffmpeg -version failed: %s", exc)
        info["ffmpeg"] = "unknown"
    try:
        r = subprocess.run(["dpkg", "-s", "lsp-plugins-lv2"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if line.startswith("Version:"):
                info["lsp"] = line.split(":", 1)[-1].strip()
    except Exception as exc:
        logger.warning("dpkg lsp-plugins-lv2 failed: %s", exc)
        info["lsp"] = "unknown"
    return info


# Module-level memoization — avoids repeated subprocess calls.
_cached_version_info: Optional[dict[str, str]] = None
_cached_plugins: Optional[dict[str, PluginInfo]] = None


def _cached_version():
    global _cached_version_info
    if _cached_version_info is None:
        _cached_version_info = _get_version_info()
    return _cached_version_info


def load_cache() -> dict[str, PluginInfo]:
    global _cached_plugins
    if _cached_plugins is not None:
        return _cached_plugins
    if not CACHE_PATH.exists():
        _cached_plugins = {}
        return _cached_plugins
    try:
        data = json.loads(CACHE_PATH.read_text())
        cached_version = data.get("_version", {})
        if cached_version != _cached_version():
            logger.info("Cache invalidated: version mismatch")
            CACHE_PATH.unlink(missing_ok=True)
            _cached_plugins = {}
            return _cached_plugins
        plugins: dict[str, PluginInfo] = {}
        for uri, port_dict in data.get("plugins", {}).items():
            ports = {sym: PortInfo(**info) for sym, info in port_dict.items()}
            plugins[uri] = PluginInfo(uri=uri, ports=ports)
        logger.info("Loaded %d plugins from cache", len(plugins))
        _cached_plugins = plugins
        return _cached_plugins
    except Exception as exc:
        logger.warning("Cache read failed: %s", exc)
        _cached_plugins = {}
        return _cached_plugins


def save_cache(plugins: dict[str, PluginInfo]) -> None:
    try:
        data = {
            "_version": _cached_version(),
            "plugins": {
                uri: {sym: {
                    "symbol": p.symbol, "name": p.name,
                    "min_val": p.min_val, "max_val": p.max_val,
                    "default_val": p.default_val, "unit": p.unit,
                } for sym, p in info.ports.items()}
                for uri, info in plugins.items()
            }
        }
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d plugins to cache", len(plugins))
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


def get_plugin_info(uri: str) -> Optional[PluginInfo]:
    cache = load_cache()
    if uri in cache:
        return cache[uri]
    info = introspect_plugin(uri)
    if info and info.ports:
        cache[uri] = info
        save_cache(cache)
    return info if info.ports else None


def _plugin_exists(uri: str) -> bool:
    all_uris = _run_lv2ls_list()
    return uri in all_uris


def ensure_plugins(uris: list[str]) -> dict[str, PluginInfo]:
    result: dict[str, PluginInfo] = {}
    missing: list[str] = []
    failed: list[str] = []
    for uri in uris:
        info = get_plugin_info(uri)
        if info is None or not info.ports:
            if not _plugin_exists(uri):
                missing.append(uri)
            else:
                failed.append(uri)
        else:
            result[uri] = info
    if missing or failed:
        msgs = []
        if missing:
            msgs.append(f"LV2 plugins not found: {', '.join(missing)}")
        if failed:
            msgs.append(f"LV2 introspection failed: {', '.join(failed)}")
        msgs.append("Install LSP plugins: sudo apt install lsp-plugins-lv2")
        msgs.append("Or use --force-native for ffmpeg-native processing.")
        raise RuntimeError("\n".join(msgs))
    return result
