"""Phase 0 integration tests — LSP loud_comp as saturator, end-to-end."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cleaner.lv2_introspect import get_plugin_info
from cleaner.lv2_params import db_to_linear_gain, clamp_to_port
from cleaner.lsp_chain_builder import build_lv2_node
from cleaner.analysis.global_analysis import compute_loud_comp_lsp_params

LOUD_COMP_URI = "http://lsp-plug.in/plugins/lv2/loud_comp_stereo"


def _has_lsp() -> bool:
    """Check if LSP plugins are available."""
    try:
        info = get_plugin_info(LOUD_COMP_URI)
        return info is not None and len(info.ports) > 0
    except Exception:
        return False


def _generate_sweep(path: Path, dur: float = 3.0, sr: int = 48000,
                    amp: float = 0.3, f0: float = 100.0, f1: float = 10000.0):
    """Generate a logarithmic sine sweep."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Log sweep: instantaneous frequency = f0 * (f1/f0)^(t/dur)
    phase = 2 * math.pi * f0 * dur / math.log(f1 / f0) * (
        (f1 / f0) ** (t / dur) - 1
    )
    y = amp * np.sin(phase)
    stereo = np.column_stack([y, y.copy()])
    sf.write(str(path), stereo.astype(np.float32), sr, subtype="PCM_24")


def _measure_harmonics(wav_path: Path, fund_freq: float = 440.0) -> dict[int, float]:
    """Measure energy at fundamental and first 5 harmonics.
    
    Returns dict[harmonic_number, energy_dB].
    """
    y, sr = sf.read(str(wav_path), always_2d=True, dtype='float32')
    y_mono = np.mean(y, axis=1)
    n = len(y_mono)
    fft = np.abs(np.fft.rfft(y_mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    
    energies = {}
    for h in range(1, 7):
        target = fund_freq * h
        # Find bin closest to target
        idx = np.argmin(np.abs(freqs - target))
        # Average energy in a small window around the bin
        win = max(1, int(sr / n * 5))
        lo, hi = max(0, idx - win), min(len(fft) - 1, idx + win)
        energy = np.mean(fft[lo:hi + 1] ** 2)
        energies[h] = 20 * math.log10(max(energy, 1e-15))
    
    return energies


@pytest.mark.skipif(not _has_lsp(), reason="LSP plugins not installed")
class TestSaturatorAudibility:
    """End-to-end: generate signal → LSP saturator → measure harmonics."""

    def test_harmonics_added_by_saturation(self):
        """A 440 Hz tone at -10 dBFS should gain harmonics when saturated."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "input.wav"
            output_wav = Path(tmp) / "output.wav"
            
            # Generate a 440 Hz tone at -10 dBFS
            dur = 2.0
            sr = 48000
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            amp = 10 ** (-10 / 20)  # -10 dBFS
            y = amp * np.sin(2 * math.pi * 440 * t)
            stereo = np.column_stack([y, y.copy()])
            sf.write(str(input_wav), stereo.astype(np.float32), sr, subtype="PCM_24")
            
            # Compute saturator params (glue=0.8 for clearly audible saturation)
            report = {"_glue": 0.8, "_intensity": 1.0}
            params = compute_loud_comp_lsp_params(report)
            
            # Get plugin info for clamping
            plugin_info = get_plugin_info(LOUD_COMP_URI)
            assert plugin_info is not None, "loud_comp_stereo not found"
            
            # Clamp params to port ranges
            clamped = {}
            for sym, val in params.items():
                port = plugin_info.ports.get(sym)
                if port:
                    clamped[sym] = clamp_to_port(val, port, convert_unit=False)
            
            # Build the LV2 node
            lv2_node = build_lv2_node(LOUD_COMP_URI, clamped)
            
            # Build filter_complex: simple chain with just the saturator
            graph = (
                f"[0:a]aresample={sr},"
                f"{lv2_node},"
                f"volume=0dB[out]"
            )
            
            # Render
            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", str(input_wav),
                "-filter_complex", graph,
                "-map", "[out]",
                "-c:a", "pcm_s24le",
                str(output_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert proc.returncode == 0, (
                f"ffmpeg failed:\n{proc.stderr[-800:]}"
            )
            assert output_wav.exists(), "Output file not created"
            
            # Measure harmonics in input and output
            in_harm = _measure_harmonics(input_wav)
            out_harm = _measure_harmonics(output_wav)
            
            print(f"\nInput harmonics:  {in_harm}")
            print(f"Output harmonics: {out_harm}")
            
            # H2 should increase (saturation adds even harmonics)
            h2_diff = out_harm[2] - in_harm[2]
            h3_diff = out_harm[3] - in_harm[3]
            print(f"H2 delta: {h2_diff:+.1f} dB")
            print(f"H3 delta: {h3_diff:+.1f} dB")
            
            # At least one harmonic should increase by >3 dB
            total_harmonic_increase = sum(
                max(0, out_harm[h] - in_harm[h])
                for h in range(2, 7)
            )
            print(f"Total harmonic increase: {total_harmonic_increase:.1f} dB")
            
            assert total_harmonic_increase > 3.0, (
                f"Saturation is inaudible: only {total_harmonic_increase:.1f} dB "
                f"of harmonic increase across H2-H6"
            )

    def test_sweep_saturation_visible(self):
        """A sine sweep through the saturator should show broadband THD increase."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "sweep_in.wav"
            output_wav = Path(tmp) / "sweep_out.wav"
            
            _generate_sweep(input_wav, dur=3.0, amp=0.3)
            
            # Heavy saturation to make it obvious
            report = {"_glue": 0.9, "_intensity": 1.0}
            params = compute_loud_comp_lsp_params(report)
            plugin_info = get_plugin_info(LOUD_COMP_URI)
            clamped = {}
            for sym, val in params.items():
                port = plugin_info.ports.get(sym)
                if port:
                    clamped[sym] = clamp_to_port(val, port, convert_unit=False)
            
            lv2_node = build_lv2_node(LOUD_COMP_URI, clamped)
            graph = f"[0:a]aresample=48000,{lv2_node}[out]"
            
            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", str(input_wav),
                "-filter_complex", graph,
                "-map", "[out]",
                "-c:a", "pcm_s24le",
                str(output_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert proc.returncode == 0, f"ffmpeg failed:\n{proc.stderr[-500:]}"
            
            # Compare broadband energy above 5 kHz (where harmonics of sweeps land)
            y_in, sr = sf.read(str(input_wav), always_2d=True, dtype='float32')
            y_out, _ = sf.read(str(output_wav), always_2d=True, dtype='float32')
            
            # Simple approach: RMS of the difference contains the added harmonics
            y_in_mono = np.mean(y_in, axis=1)
            y_out_mono = np.mean(y_out, axis=1)
            
            # Normalize output to match input level (compensate makeup)
            rms_in = np.sqrt(np.mean(y_in_mono ** 2))
            rms_out = np.sqrt(np.mean(y_out_mono ** 2))
            y_out_norm = y_out_mono * (rms_in / rms_out) if rms_out > 1e-10 else y_out_mono
            
            # Difference energy
            diff = y_out_norm - y_in_mono
            diff_rms = np.sqrt(np.mean(diff ** 2))
            diff_db = 20 * math.log10(max(diff_rms, 1e-15))
            signal_db = 20 * math.log10(max(rms_in, 1e-15))
            
            print(f"\nInput RMS: {signal_db:.1f} dBFS")
            print(f"Diff RMS: {diff_db:.1f} dB (should be > -40 dB for audible saturation)")
            
            # The difference should be significant (> -40 dB relative to signal)
            assert diff_db > -40.0, (
                f"Saturation produces no measurable change: diff={diff_db:.1f} dB"
            )


class TestLSPIntrospection:
    """Tests that don't require actual plugin rendering."""

    def test_loud_comp_discoverable(self):
        """loud_comp_stereo must be discovered if LSP is installed."""
        if not _has_lsp():
            pytest.skip("LSP plugins not installed")
        info = get_plugin_info(LOUD_COMP_URI)
        assert info is not None
        assert len(info.ports) > 5
        # Check key ports exist
        assert "input" in info.ports
        assert "volume" in info.ports
        assert "hclip" in info.ports

    def test_param_clamping_works(self):
        """Params from compute_loud_comp_lsp_params clamp to port ranges."""
        if not _has_lsp():
            pytest.skip("LSP plugins not installed")
        info = get_plugin_info(LOUD_COMP_URI)
        report = {"_glue": 1.5, "_intensity": 2.0}  # out of range
        params = compute_loud_comp_lsp_params(report)
        for sym, val in params.items():
            port = info.ports.get(sym)
            if port:
                clamped = clamp_to_port(val, port, convert_unit=False)
                assert port.min_val <= clamped <= port.max_val, (
                    f"{sym}: {clamped} not in [{port.min_val}, {port.max_val}]"
                )

    def test_lv2_node_builds(self):
        """build_lv2_node produces a syntactically valid string."""
        params = {"input": 2.0, "volume": -3.0, "hclip": 0.5}
        node = build_lv2_node(LOUD_COMP_URI, params)
        assert "lv2=p=" in node
        assert "input=2.0" in node
        assert "volume=-3.0" in node
        assert "hclip=0.5" in node
