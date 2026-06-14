"""Pipeline orchestrator — hybrid ffmpeg + LSP/LV2, before/after metrics, rich output."""

from __future__ import annotations
import logging, shutil, subprocess, uuid
from pathlib import Path
from typing import Optional
import numpy as np
import click
from cleaner.analysis.global_analysis import get_global_analysis, compute_ffmpeg_params, AnalysisReport
from cleaner.ffmpeg_chain import build_filtergraph
from cleaner.lsp_chain_builder import build_lsp_filtergraph
from cleaner.io_adapter import (convert_to_wav, measure_lufs, apply_lufs_gain,
                                FFmpegNotFoundError, SourceDecodeError,
                                LUFSMeasurementError, require_ffmpeg)

logger = logging.getLogger(__name__)

class PipelineResult:
    def __init__(self):
        self.success = False
        self.output_path: Optional[Path] = None
        self.report: AnalysisReport = {}
        self.measured_lufs: Optional[float] = None
        self.applied_gain_db: float = 0.0
        self.input_metrics: dict = {}
        self.output_metrics: dict = {}
        self.temp_dir: Optional[Path] = None


def _measure_file(wav_path: Path) -> dict:
    """Measure peak, RMS, crest, clipping on a WAV file."""
    try:
        import soundfile as sf
        y, sr = sf.read(str(wav_path), always_2d=True, dtype='float32')
        peak = 20 * np.log10(max(np.max(np.abs(y)), 1e-10))
        rms = 20 * np.log10(np.sqrt(max(np.mean(y ** 2), 1e-10)))
        clipped = int(np.sum(np.abs(y) >= 0.9885))
        return {
            "peak_db": round(peak, 1), "rms_db": round(rms, 1),
            "crest_db": round(peak - rms, 1), "clipped": clipped,
            "total": int(y.size), "sr": int(sr), "channels": y.shape[1],
            "duration_s": round(y.shape[0] / sr, 1),
        }
    except Exception:
        return {}


def _lsp_available() -> bool:
    """Check if required LSP plugins are available."""
    try:
        from cleaner.lv2_introspect import get_plugin_info
        from cleaner.lsp_uris import REQUIRED_URIS
        required = REQUIRED_URIS
        for uri in required:
            info = get_plugin_info(uri)
            if info is None or not info.ports:
                return False
        return True
    except Exception:
        return False


def _box_header(text: str) -> None:
    """Print a boxed header."""
    w = 58
    click.secho(f"  +{'-'*w}+", fg="yellow")
    click.secho(f"  | {text:<{w}} |", fg="yellow", bold=True)
    click.secho(f"  +{'-'*w}+", fg="yellow")


def _box_line(text: str, color: str = "yellow") -> None:
    w = 58
    click.secho(f"  | {text:<{w}} |", fg=color)


def _box_close() -> None:
    click.secho(f"  +{'-'*58}+", fg="yellow")


def _print_analysis_report(report):
    """Detailed Phase 1 analysis."""
    _box_header("RAPPORT D'ANALYSE (Phase 1)")
    crest = report.get('crest_factor_db', 0)
    tag = ("tres dynamique" if crest > 12 else
           "AGC smartphone probable" if crest < 8 else "dynamique moderee")
    _box_line(f"Crest Factor  : {crest:.1f} dB  ({tag})")
    _box_line(f"Peak / RMS    : {report.get('peak_db',0):.1f} / {report.get('rms_db',0):.1f} dBFS")
    _box_line(f"AGC recovery  : {report.get('agc_recovery_ms',0):.0f} ms")
    _box_line(f"Attack median : {report.get('transient_attack_ms',0):.1f} ms")
    _box_line("")
    _box_line("Resonances (Room Modes) :", "yellow")
    modes = report.get('room_modes_hz', [])
    qs = report.get('room_mode_qs', [])
    gains = report.get('room_mode_gains_db', [])
    for i in range(min(3, len(modes))):
        _box_line(f"  Mode {i+1}: {modes[i]:.1f} Hz  Q={qs[i]:.0f}  {gains[i]:+.1f} dB")
    _box_line("")
    _box_line("Stereo :", "yellow")
    _box_line(f"  Correlation M/S  : {report.get('ms_correlation_avg',0):.3f}  (1=mono, 0=large)")
    _box_line(f"  Side energy      : {report.get('side_energy_ratio',0)*100:.1f}%")
    _box_line(f"  HF correlation   : {report.get('hf_correlation',0):.3f}  (cymbales >5kHz)")
    hf = report.get('hf_correlation', 0.5)
    if hf < 0.3:
        _box_line("  -> Chaos de phase aigus detecte !")
    _box_line("")
    _box_line("Ecretage :", "yellow")
    is_clip = report.get('is_heavily_clipped', False)
    ratio = report.get('clip_ratio', 0)
    if is_clip:
        _box_line(f"  FORT EC RETAGE: {ratio*100:.1f}%  -> penalite appliquee")
    else:
        _box_line(f"  {ratio*100:.2f}%  -- acceptable")
    _box_close()
    click.echo()


def _print_chain_summary(report, stages, use_lsp=False):
    """Human-readable filterchain summary."""
    _box_header("CHAINE DSP APPLIQUEE")
    i = 0
    if stages.get("expander", True):
        i += 1
        if use_lsp:
            _box_line(f"{i}. HP 35Hz + Expander LSP (anti-AGC)", "cyan")
            _box_line(f"   mode=Up  ratio={report.get('expander_ratio',2):.1f}  "
                      f"attack={report.get('expander_attack_ms',5):.0f}ms  "
                      f"release={report.get('expander_release_ms',40):.0f}ms", "cyan")
        else:
            _box_line(f"{i}. HP 35Hz + Expander (anti-AGC)", "cyan")
            exp_th = 20*np.log10(max(report.get('expander_threshold_linear', 0.1), 1e-10))
            _box_line(f"   seuil={exp_th:.1f}dB  ratio={report.get('expander_ratio',2):.1f}  "
                      f"attack={report.get('expander_attack_ms',5):.0f}ms  "
                      f"release={report.get('expander_release_ms',40):.0f}ms  "
                      f"range=+{20*np.log10(1+report.get('expander_range_linear',0.25)):.1f}dB", "cyan")
    else:
        _box_line(f"   Expander DESACTIVE", "cyan")
    hp35_on = stages.get("hp35", True)
    hp150_on = stages.get("hp150", True)
    if hp35_on:
        _box_line(f"   HP 35Hz active", "cyan")
    else:
        _box_line(f"   HP 35Hz DESACTIVE", "cyan")
    if not hp150_on:
        _box_line(f"   HP 150Hz Side DESACTIVE", "cyan")
    if stages.get("ducking", True):
        i += 1
        th = 20*np.log10(max(report.get('comp_threshold_linear', 0.1), 1e-10))
        _box_line(f"{i}. Sidechain Ducking (Mid ecrase Side)", "cyan")
        _box_line(f"   seuil={th:.1f}dB  ratio={report.get('comp_ratio',4)}:1  "
                  f"attack={report.get('comp_attack_ms',2.0):.0f}ms  "
                  f"release={report.get('comp_release_ms',60):.0f}ms", "cyan")
    else:
        _box_line(f"   Ducking DESACTIVE", "cyan")
    if stages.get("deharsher", False):
        i += 1
        if use_lsp:
            _box_line(f"{i}. De-harsher LSP [2.5-4.5 kHz] (experimental)", "cyan")
            _box_line(f"   seuil adaptatif  "
                      f"ratio={report.get('harshness_index',0.3)*2+1.5:.1f}", "cyan")
        else:
            _box_line(f"{i}. De-harsher dynamique [2.5-4.5 kHz] (experimental)", "cyan")
            _box_line(f"   seuil={report.get('deharsher_display_threshold',5):.1f}  "
                      f"ratio={report.get('deharsher_filter_ratio',3):.1f}", "cyan")
    else:
        _box_line(f"   De-harsher DESACTIVE (opt-in avec --deharsher)", "cyan")
    if stages.get("notches", True):
        i += 1
        _box_line(f"{i}. Notch filters x3 (Room Modes)", "cyan")
        for j in range(1, 4):
            f = report.get(f'notch_freq_{j}', 0)
            g = report.get(f'notch_gain_{j}', 0)
            _box_line(f"   Notch {j}: {f:.1f} Hz  gain={g:.1f} dB", "cyan")
    else:
        _box_line(f"   Notches DESACTIVES", "cyan")
    if stages.get("saturation", True):
        i += 1
        if use_lsp:
            import math
            glue = report.get('_glue', 0.15)
            intensity = report.get('_intensity', 0.5)
            eff = glue * (0.3 + intensity * 0.7)
            drive_db = eff * 16.0
            makeup_db = -drive_db * 0.4
            _box_line(f"{i}. Saturation LSP (drive + hard-clip)", "cyan")
            _box_line(f"   drive=+{drive_db:.1f}dB  makeup={makeup_db:.1f}dB  "
                      f"hclip={min(1.0, eff*1.2):.2f}", "cyan")
        else:
            _box_line(f"{i}. Tape Saturation (drive+makeup, 4x oversample)", "cyan")
            _box_line(f"   drive=+{report.get('sat_drive_db',1.2):.1f}dB  seuil={report.get('sat_threshold_linear',0.85):.2f}  makeup={report.get('sat_makeup_db',-0.7):.1f}dB", "cyan")
    else:
        _box_line(f"   Saturation DESACTIVEE", "cyan")
    if stages.get("air", False):
        _box_line(f"   Air boost: +{report.get('_air_db', 1.5):.1f} dB @ 8kHz", "cyan")
    if stages.get("width", False):
        w = report.get("_width", 0.0)
        _box_line(f"   Width: {w:+.1f} ({'elargi' if w > 0 else 'resserre' if w < 0 else 'neutre'})", "cyan")
    if stages.get("bus_comp", False):
        if use_lsp:
            bus = report.get('_bus_comp', 0.0)
            _box_line(f"   Bus Comp LSP: seuil auto  ratio=2:1  "
                      f"attack=10ms  mix={bus:.0%}  dry/wet", "cyan")
        else:
            bth = 20*np.log10(max(report.get('bus_threshold_linear', 0.18), 1e-10))
            _box_line(f"   Bus Comp SSL: seuil={bth:.1f}dB  ratio=2:1  "
                      f"attack=10ms  mix={report.get('bus_mix',0):.0%}", "cyan")
    if stages.get("limiter", True):
        i += 1
        if use_lsp:
            _box_line(f"{i}. Limiter LSP (true-peak, 4x oversampling, ALR)", "cyan")
        else:
            _box_line(f"{i}. True Peak Limiter  ceiling=-1.1 dBFS", "cyan")
    else:
        _box_line(f"   Limiter DESACTIVE", "cyan")
    if stages.get("lufs", True):
        _box_line(f"   + Normalisation LUFS (EBU R128)", "cyan")
    _box_close()
    click.echo()


def _print_metrics(title, metrics, color="green"):
    """Print a before/after metrics box."""
    _box_header(title)
    if metrics:
        _box_line(f"Peak      : {metrics.get('peak_db',0):.1f} dBFS", color)
        _box_line(f"RMS       : {metrics.get('rms_db',0):.1f} dBFS", color)
        _box_line(f"Crest     : {metrics.get('crest_db',0):.1f} dB", color)
        c = metrics.get('clipped', 0); t = max(metrics.get('total', 1), 1)
        _box_line(f"Ecrete    : {c}/{t} ({100*c/t:.3f}%)", color)
        _box_line(f"Duree     : {metrics.get('duration_s',0):.1f}s  "
                  f"{metrics.get('sr',0)}Hz  {metrics.get('channels',0)}ch", color)
    _box_close()
    click.echo()


def _print_comparison(before, after):
    """Side-by-side before/after comparison."""
    click.secho(f"  +{'='*58}+", fg="magenta")
    click.secho(f"  | {'':15} {'AVANT':>15} {'APRES':>15} {'DELTA':>11} |", fg="magenta", bold=True)
    click.secho(f"  +{'='*58}+", fg="magenta")
    for label, key, fmt in [
        ("Peak (dBFS)", "peak_db", ".1f"),
        ("RMS (dBFS)", "rms_db", ".1f"),
        ("Crest (dB)", "crest_db", ".1f"),
    ]:
        b = before.get(key, 0); a = after.get(key, 0); d = a - b
        sign = "+" if d > 0 else ""
        click.secho(f"  | {label:<15} {b:>15{fmt}} {a:>15{fmt}} {sign}{d:>10{fmt}} |", fg="magenta")
    bc = before.get('clipped', 0); ac = after.get('clipped', 0)
    click.secho(f"  | Clipped       {bc:>15} {ac:>15} {'':>11} |", fg="magenta")
    click.secho(f"  +{'='*58}+", fg="magenta")
    click.echo()


def run_pipeline(source, output, *, keep_temp=False, dry_run=False, timeout=3600,
                 target_lufs=-14.0, ceiling=-1.1, tame_cymbals=0.0,
                 notch_intensity=1.0, glue=0.15, air=1.5, width=0.0,
                 bus_comp=0.0, intensity=0.5,
                 stages=None, force_native=False,
                 punchin_s=None, punchout_s=None):
    if stages is None:
        stages = {}
    result = PipelineResult()
    click.echo()
    use_lsp = not force_native and _lsp_available()
    if use_lsp:
        click.secho("  cleaner v0.1.0 -- hybrid ffmpeg + LSP/LV2 chain", fg="cyan", bold=True)
    else:
        click.secho("  cleaner v0.1.0 -- ffmpeg-native DSP chain", fg="cyan", bold=True)

    try:
        ffmpeg = require_ffmpeg()
    except FFmpegNotFoundError as exc:
        click.secho(f"  Erreur: {exc}", fg="red", err=True)
        return result

    temp_dir = Path("/tmp") / f"cleaner_{uuid.uuid4().hex[:12]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    result.temp_dir = temp_dir

    try:
        # 1. Convert
        click.echo("  [1/5] Conversion...")
        input_wav = temp_dir / "input.wav"
        convert_to_wav(source, input_wav, start_s=punchin_s, end_s=punchout_s)
        result.input_metrics = _measure_file(input_wav)

        # 2. Analyse
        click.echo("  [2/5] Analyse (Phase 1)...")
        report = get_global_analysis(str(input_wav))
        result.report = report
        _print_analysis_report(report)

        # 3. Deltas + params
        import copy
        report = copy.deepcopy(report)
        report["_ceiling_db"] = ceiling
        report["_target_lufs"] = target_lufs
        report["_notch_multiplier"] = notch_intensity
        report["_tame_cymbals"] = tame_cymbals
        report["_glue"] = glue
        report["_air"] = air
        report["_width"] = width
        report["_bus_comp"] = bus_comp
        report["_intensity"] = intensity

        # Re‑compute ffmpeg params now that mastering flags are set
        report = compute_ffmpeg_params(report)

        # 4. Filtergraph
        click.echo("  [3/5] Construction de la chaine DSP...")
        if use_lsp:
            click.secho("  Mode: LSP/LV2 (plugins detectes)", fg="cyan")
            graph = build_lsp_filtergraph(report, stages)
        else:
            if not force_native and not _lsp_available():
                click.secho("  [!] LSP plugins non trouves — fallback natif.", fg="yellow")
                click.secho("  Utilisez --force-native pour supprimer cet avertissement.", fg="yellow")
            else:
                click.secho("  Mode: ffmpeg natif", fg="cyan")
            graph = build_filtergraph(report, stages)
        _print_chain_summary(report, stages, use_lsp)

        if dry_run:
            click.echo(f"  Filtergraph ({len(graph)} chars):")
            click.echo(f"  {graph[:300]}...")
            if use_lsp:
                click.secho("  Mode: LSP/LV2", fg="cyan")
            else:
                click.secho("  Mode: ffmpeg natif", fg="cyan")
            click.secho("\n  Dry run termine.", fg="green")
            result.success = True
            return result

        # 5. Render
        click.echo("  [4/5] Rendu ffmpeg...")
        rendered_wav = temp_dir / "rendered.wav"
        cmd = [str(ffmpeg), "-y", "-i", str(input_wav),
               "-filter_complex", graph, "-map", "[out]",
               "-c:a", "pcm_s24le", "-ar", "48000", str(rendered_wav)]
        logger.debug("ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = proc.stderr[-1500:] if len(proc.stderr) > 1500 else proc.stderr
            raise RuntimeError(f"ffmpeg render failed (exit {proc.returncode}):\n{tail}")
        click.echo("  [OK] Rendu termine")

        # 6. LUFS
        if stages.get("lufs", True):
            click.echo(f"  [5/5] Normalisation LUFS (cible: {target_lufs})...")
            measured = measure_lufs(rendered_wav)
            result.measured_lufs = measured
            final_output, gain_db = apply_lufs_gain(rendered_wav, output, target_lufs=target_lufs)
            result.applied_gain_db = gain_db
            final_lufs = measured + gain_db
            if abs(final_lufs - target_lufs) < 0.5:
                click.echo(f"  [OK] LUFS: {measured:.1f} + {gain_db:+.1f} dB = {final_lufs:.1f} LUFS (cible {target_lufs})")
            else:
                click.secho(f"  [!] LUFS: {measured:.1f} + {gain_db:+.1f} dB = {final_lufs:.1f} LUFS", fg="yellow")
                click.secho(f"      Cible {target_lufs} non atteinte (gain plafonne a {gain_db:+.1f} dB)", fg="yellow")

            # If gain pushed peaks above ceiling, re-apply limiter
            if gain_db > 0.5 and stages.get("limiter", True):
                pre_limit = temp_dir / "pre_limit.wav"
                shutil.move(str(output), str(pre_limit))
                limit_cmd = [
                    str(ffmpeg), "-y", "-i", str(pre_limit),
                    "-af", f"alimiter=limit={10.0**(ceiling/20.0):.4f}:attack=0.1:release=30:level=true",
                    "-c:a", "pcm_s24le", str(output),
                ]
                subprocess.run(limit_cmd, capture_output=True, timeout=timeout)
                click.echo(f"  [OK] Limiteur reapplique post-LUFS (ceiling={ceiling} dBFS)")
        else:
            click.echo("  [5/5] LUFS DESACTIVE -- copie directe")
            shutil.copy2(rendered_wav, output)
            final_output = output

        result.output_path = final_output
        result.output_metrics = _measure_file(final_output)

        # Before/after comparison
        click.echo()
        _print_comparison(result.input_metrics, result.output_metrics)

        result.success = True
        click.secho(f"\n  +{'='*58}+", fg="green")
        click.secho(f"  | PIPELINE TERMINE  : {final_output.name}", fg="green", bold=True)
        click.secho(f"  +{'='*58}+", fg="green")
        click.echo()

    except (FFmpegNotFoundError, SourceDecodeError, LUFSMeasurementError,
            ValueError, RuntimeError) as exc:
        click.secho(f"\n  Erreur: {exc}", fg="red", err=True)
    except Exception as exc:
        click.secho(f"\n  Erreur inattendue: {exc}", fg="red", err=True)
        logger.exception("Unhandled exception")
    finally:
        if not keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    return result
