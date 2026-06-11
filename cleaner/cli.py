"""CLI for cleaner v0.1.0 — Complete DSP chain with per-stage control."""

from __future__ import annotations
import logging, sys
from pathlib import Path

import click
from cleaner.pipeline import run_pipeline

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

# ── Mastering presets ────────────────────────────────────────────────

PRESETS: dict = {
    "transparent": {
        "glue": 0.05, "air": 0.5, "width": 0.0,
        "bus_comp": 0.0, "target_lufs": -14.0,
        "desc": "Corrective only. No coloration. Archival/forensic.",
    },
    "warm": {
        "glue": 0.5, "air": 0.0, "width": -0.15,
        "bus_comp": 0.3, "target_lufs": -13.0,
        "desc": "Tape/vintage warmth. Saturated mids, tight stereo.",
    },
    "open": {
        "glue": 0.1, "air": 2.5, "width": 0.35,
        "bus_comp": 0.15, "target_lufs": -14.0,
        "desc": "Wide and airy. Preserves room ambience. Live recording.",
    },
    "punchy": {
        "glue": 0.4, "air": 2.0, "width": 0.1,
        "bus_comp": 0.5, "target_lufs": -11.0,
        "desc": "Aggressive & forward. Heavy bus comp. Post-punk/hardcore.",
    },
    "loud": {
        "glue": 0.6, "air": 3.0, "width": 0.2,
        "bus_comp": 0.7, "target_lufs": -9.0,
        "desc": "Maximum density. War volume. Demos & loudness war.",
    },
}

@click.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output WAV (default: <source>_clean.wav)")
@click.option("--keep-temp", is_flag=True, default=False,
              help="Keep intermediate files.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Analyse + show filtergraph (no render).")
@click.option("--target-lufs", type=float, default=-14.0,
              help="Target LUFS [-20.0, -8.0].")
@click.option("--ceiling", type=float, default=-1.1,
              help="Limiter ceiling dBFS [-3.0, -0.3].")
@click.option("--notch-intensity", type=float, default=1.0,
              help="Room mode attenuation multiplier [0.0, 2.0].")
@click.option("--tame-cymbals", type=float, default=0.0,
              help="De-harsher threshold delta [-12.0, 0.0] dB.")
@click.option("--timeout", type=int, default=3600,
              help="Render timeout seconds.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Verbose logging.")
@click.option("--preset", type=click.Choice(list(PRESETS.keys()), case_sensitive=False),
              default=None, help="Mastering preset. Overrides color defaults.")
# Mastering color (floats -- preset-aware)
@click.option("--glue", type=float, default=0.15,
              help="Saturation drive (0.0=off, 0.5=medium, 1.0=max).")
@click.option("--air", type=float, default=1.5,
              help="High-shelf brilliance at 8kHz in dB [0.0, 5.0].")
@click.option("--width", type=float, default=0.0,
              help="Stereo width [-1.0, 1.0]. +widens, -narrows.")
@click.option("--bus-comp", type=float, default=0.0,
              help="SSL bus compressor drive [0.0, 1.0].")
@click.option("--intensity", type=float, default=0.5,
              help="Global intensity [0.0, 1.0]. Scales glue, notches, expander. 0=transparent, 1=maximum.")
@click.option("--force-native", is_flag=True, default=False,
              help="Use full-native ffmpeg chain (no LSP/LV2 plugins).")

# Stage toggles (booleans)
@click.option("--expander/--no-expander", default=True, help="Anti-AGC upward expansion.")
@click.option("--ducking/--no-ducking", default=True, help="Mid->Side room ducking.")
@click.option("--deharsher/--no-deharsher", default=False, help="Dynamic de-harsher (experimental).")
@click.option("--notches/--no-notches", default=True, help="Room mode notch filters.")
@click.option("--saturation/--no-saturation", default=True, help="Tape saturation.")
@click.option("--limiter/--no-limiter", default=True, help="True Peak limiter.")
@click.option("--lufs/--no-lufs", default=True, help="LUFS normalisation.")
@click.option("--hp35/--no-hp35", default=True, help="35Hz high-pass.")
@click.option("--hp150/--no-hp150", default=True, help="150Hz Side high-pass.")
@click.version_option(version="0.1.0", prog_name="cleaner")
def main(source, output, keep_temp, dry_run, target_lufs, preset, ceiling,
         notch_intensity, tame_cymbals, timeout, verbose,
         glue, air, width, bus_comp, intensity, force_native,
         expander, ducking, deharsher, notches, saturation, limiter, lufs,
         hp35, hp150):
    """CLEANER -- Restore live recordings.

    SOURCE is the audio file (any ffmpeg format: mp3, m4a, wav, flac...).

    Architecture: Python analysis + hybrid ffmpeg + LSP/LV2 filter_complex chain.
    LSP/LV2 plugins required for coloration stages (optional: --force-native for pure ffmpeg).

    \b
    Full chain (all enabled by default):
      HP 35Hz -> Expander (LSP, anti-AGC) -> M/S encode -> Sidechain ducking
      -> M/S decode -> De-harsher (LSP) -> 3x Notch (LSP EQ, room modes)
      -> Tape saturation (LSP) -> Bus compressor (LSP) -> Limiter (LSP) -> LUFS normalize

    \b
    Examples:
      cleaner recording.m4a
      cleaner recording.m4a --no-ducking --no-notches
      cleaner recording.m4a --preset punchy
      cleaner recording.m4a --preset warm --air 3.0
      cleaner recording.m4a --dry-run
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if output is None:
        output = source.parent / f"{source.stem}_clean.wav"

    errors = []
    if not (-12.0 <= tame_cymbals <= 0.0):
        errors.append(f"--tame-cymbals range [-12,0], got {tame_cymbals}")
    if not (0.0 <= notch_intensity <= 2.0):
        errors.append(f"--notch-intensity range [0,2], got {notch_intensity}")
    if not (-20.0 <= target_lufs <= -8.0):
        errors.append(f"--target-lufs range [-20,-8], got {target_lufs}")
    if not (-3.0 <= ceiling <= -0.3):
        errors.append(f"--ceiling range [-3,-0.3], got {ceiling}")
    if not (0.0 <= glue <= 1.0):
        errors.append(f"--glue range [0,1], got {glue}")
    if not (0.0 <= air <= 5.0):
        errors.append(f"--air range [0,5], got {air}")
    if not (-1.0 <= width <= 1.0):
        errors.append(f"--width range [-1,1], got {width}")
    if not (0.0 <= bus_comp <= 1.0):
        errors.append(f"--bus-comp range [0,1], got {bus_comp}")
    if not (0.0 <= intensity <= 1.0):
        errors.append(f"--intensity range [0,1], got {intensity}")
    if errors:
        for e in errors: click.echo(f"  Error: {e}", err=True)
        sys.exit(1)

    # -- Apply preset ----
    if preset:
        p = PRESETS[preset]
        click.echo(f"\n  Preset: {preset} -- {p['desc']}")
        ctx = click.get_current_context()
        vals = {"glue": glue, "air": air, "width": width,
                "bus_comp": bus_comp, "target_lufs": target_lufs}
        for key in vals:
            try:
                src = ctx.get_parameter_source(key)
                if src is None or src.name == "DEFAULT":
                    vals[key] = p[key]
            except Exception:
                vals[key] = p[key]
        glue, air, width, bus_comp, target_lufs = (
            vals["glue"], vals["air"], vals["width"],
            vals["bus_comp"], vals["target_lufs"])

    stages = {
        "expander": expander, "ducking": ducking,
        "deharsher": deharsher, "notches": notches,
        "saturation": saturation, "limiter": limiter,
        "lufs": lufs, "hp35": hp35, "hp150": hp150,
        "glue": glue > 0.01, "air": air > 0.01,
        "width": abs(width) > 0.001, "bus_comp": bus_comp > 0.01,
        "intensity": True,
    }

    result = run_pipeline(
        source=source, output=output, keep_temp=keep_temp,
        dry_run=dry_run, timeout=timeout,
        target_lufs=target_lufs, ceiling=ceiling,
        tame_cymbals=tame_cymbals, notch_intensity=notch_intensity,
        glue=glue, air=air, width=width,
        bus_comp=bus_comp, intensity=intensity,
        stages=stages, force_native=force_native,
    )
    if not result.success: sys.exit(1)
