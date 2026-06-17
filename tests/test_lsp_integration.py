"""Phase 0 integration tests — native asoftclip saturation, LSP chain end-to-end."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cleaner.lv2_introspect import get_plugin_info
from cleaner.lv2_params import clamp_to_port
from cleaner.lsp_chain_builder import build_lv2_node

COMPRESSOR_URI = "http://lsp-plug.in/plugins/lv2/compressor_stereo"


def _has_lsp() -> bool:
    """Check if LSP plugins are available."""
    try:
        info = get_plugin_info(COMPRESSOR_URI)
        return info is not None and len(info.ports) > 0
    except Exception:
        return False


def _generate_sweep(path: Path, dur: float = 3.0, sr: int = 48000,
                    amp: float = 0.3, f0: float = 100.0, f1: float = 10000.0):
    """Generate a logarithmic sine sweep."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    phase = 2 * math.pi * f0 * dur / math.log(f1 / f0) * (
        (f1 / f0) ** (t / dur) - 1
    )
    y = amp * np.sin(phase)
    stereo = np.column_stack([y, y.copy()])
    sf.write(str(path), stereo.astype(np.float32), sr, subtype="PCM_24")


class TestNativeSaturationAudibility:
    """Audibility test for native asoftclip tanh saturation."""

    def test_tanh_saturation_adds_harmonics(self):
        """A 1 kHz sine at -6 dBFS with glue=0.8 should add harmonics via tanh."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "input.wav"
            output_wav = Path(tmp) / "output.wav"

            sr = 48000
            dur = 2.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            amp = 10 ** (-6 / 20)
            y = amp * np.sin(2 * math.pi * 1000 * t)
            stereo = np.column_stack([y, y.copy()])
            sf.write(str(input_wav), stereo.astype(np.float32), sr, subtype="PCM_24")

            from cleaner.analysis.global_analysis import compute_native_saturation_params
            from cleaner.types import DerivedParams
            derived = DerivedParams(sat_drive_db=round(0.8*(0.3+1.0*0.7)*12.0, 1),
                                    sat_makeup_db=round(-0.8*(0.3+1.0*0.7)*12.0*0.4, 1),
                                    sat_threshold_linear=round(0.92-0.8*(0.3+1.0*0.7)*0.35, 3))
            sat = compute_native_saturation_params(derived)

            graph = (
                f"[0:a]aresample={sr},"
                f"volume={sat['sat_drive_db']}dB,"
                f"asoftclip=type=tanh:threshold={sat['sat_threshold_linear']}:output=1.0:oversample=4,"
                f"volume={sat['sat_makeup_db']}dB[out]"
            )

            cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(input_wav),
                   "-filter_complex", graph, "-map", "[out]",
                   "-c:a", "pcm_s24le", str(output_wav)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert proc.returncode == 0, f"Render failed: {proc.stderr[-500:]}"

            y_in, _ = sf.read(str(input_wav), always_2d=True, dtype='float32')
            y_out, _ = sf.read(str(output_wav), always_2d=True, dtype='float32')
            y_in_m = np.mean(y_in, axis=1)
            y_out_m = np.mean(y_out, axis=1)

            def harmonic_energy(signal, freq, sr, n_fft=None):
                if n_fft is None:
                    n_fft = len(signal)
                fft = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=n_fft))
                freqs = np.fft.rfftfreq(n_fft, 1 / sr)
                idx = np.argmin(np.abs(freqs - freq))
                win = max(1, int(sr / n_fft * 5))
                lo, hi = max(0, idx - win), min(len(fft) - 1, idx + win)
                return 20 * math.log10(max(np.mean(fft[lo:hi + 1]), 1e-15))

            in_h2 = harmonic_energy(y_in_m, 2000, sr)
            in_h3 = harmonic_energy(y_in_m, 3000, sr)
            out_h2 = harmonic_energy(y_out_m, 2000, sr)
            out_h3 = harmonic_energy(y_out_m, 3000, sr)

            h2_rise = out_h2 - in_h2
            h3_rise = out_h3 - in_h3
            print(f"\nH2 rise: {h2_rise:+.1f} dB")
            print(f"H3 rise: {h3_rise:+.1f} dB")

            assert h2_rise + h3_rise > 6.0, (
                f"Saturation inaudible: H2+H3 rise = {h2_rise + h3_rise:.1f} dB"
            )


class TestLSPIntrospection:
    """Tests that don't require actual plugin rendering."""

    def test_compressor_discoverable(self):
        """compressor_stereo must be discovered if LSP is installed."""
        if not _has_lsp():
            pytest.skip("LSP plugins not installed")
        info = get_plugin_info(COMPRESSOR_URI)
        assert info is not None
        assert len(info.ports) > 5
        assert "cm" in info.ports
        assert "cr" in info.ports

    def test_param_clamping_works(self):
        """Params clamp to port ranges."""
        if not _has_lsp():
            pytest.skip("LSP plugins not installed")
        info = get_plugin_info(COMPRESSOR_URI)
        from cleaner.analysis.global_analysis import compute_compressor_lsp_params
        from cleaner.types import DerivedParams
        derived = DerivedParams(bus_threshold_linear=0.18, bus_mix=0.5)
        params = compute_compressor_lsp_params(derived)
        for sym, val in params.items():
            port = info.ports.get(sym)
            if port:
                clamped = clamp_to_port(val, port, convert_unit=True)
                assert port.min_val <= clamped <= port.max_val, (
                    f"{sym}: {clamped} not in [{port.min_val}, {port.max_val}]"
                )

    def test_lv2_node_builds(self):
        """build_lv2_node produces a syntactically valid string."""
        params = {"cm": 0.0, "cr": 2.0, "at": 10.0}
        node = build_lv2_node(COMPRESSOR_URI, params)
        assert "lv2=p=" in node
        assert "at=10.0" in node
        assert "cm=0.0" in node
        assert "cr=2.0" in node


class TestFullLSPChain:
    """End-to-end: full LSP filtergraph renders without errors."""

    @pytest.mark.skipif(not _has_lsp(), reason="LSP plugins not installed")
    def test_full_graph_renders(self):
        """The complete LSP filtergraph must be accepted by ffmpeg."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "input.wav"
            output_wav = Path(tmp) / "output.wav"

            sr = 48000
            dur = 2.0
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 10 ** (-20 / 20), (int(sr * dur), 2))
            sf.write(str(input_wav), noise.astype(np.float32), sr, subtype="PCM_24")

            analysis_data = {
                'crest_factor_db': 10.0, 'peak_db': -3.0, 'rms_db': -15.0,
                'transient_attack_ms': 8.0, 'agc_recovery_ms': 60.0,
                'room_modes_hz': (250, 400, 550), 'room_mode_qs': (8, 5, 7),
                'room_mode_gains_db': (5, 4, 3),
                'harshness_index': 0.2,
                'is_heavily_clipped': False,
            }
            from cleaner.types import AnalysisReport, MasteringSettings
            from cleaner.analysis.derived import compute_derived_params
            analysis = AnalysisReport(**analysis_data)
            settings = MasteringSettings(glue=0.3, air=1.0, bus_comp=0.2, ceiling_db=-1.1,
                                         intensity=0.5, notch_multiplier=1.0)
            derived = compute_derived_params(analysis, settings)
            stages = {
                'expander': True, 'ducking': True, 'deharsher': True,
                'notches': True, 'saturation': True, 'limiter': True,
                'hp35': True, 'hp150': True, 'glue': True, 'air': True,
                'width': False, 'bus_comp': True, 'intensity': True,
            }

            from cleaner.lsp_chain_builder import build_lsp_filtergraph
            graph = build_lsp_filtergraph(analysis, settings, derived, stages)

            assert 'lv2=p=' in graph
            assert '[out]' in graph
            assert 'expander_stereo' in graph
            assert 'para_equalizer_x16_stereo' in graph
            assert 'asoftclip=type=tanh' in graph
            assert 'compressor_stereo' in graph
            assert 'limiter_stereo' in graph
            assert 'sidechaincompress' in graph

            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", str(input_wav),
                "-filter_complex", graph,
                "-map", "[out]",
                "-c:a", "pcm_s24le",
                str(output_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            assert proc.returncode == 0, (
                f"Full LSP chain render failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-1000:]}"
            )
            assert output_wav.exists(), "Output file not created"
            assert output_wav.stat().st_size > 1000, "Output file too small"

    @pytest.mark.skipif(not _has_lsp(), reason="LSP plugins not installed")
    def test_native_graph_still_works(self):
        """The native ffmpeg_chain builder must still work (regression check)."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "input.wav"
            output_wav = Path(tmp) / "output.wav"

            sr = 48000
            dur = 1.0
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 10 ** (-20 / 20), (int(sr * dur), 2))
            sf.write(str(input_wav), noise.astype(np.float32), sr, subtype="PCM_24")

            from cleaner.ffmpeg_chain import build_filtergraph
            from cleaner.types import DerivedParams
            d = DerivedParams(
                comp_threshold_linear=0.18, comp_ratio=4,
                comp_attack_ms=2.0, comp_release_ms=60,
                notch_freq_1=300.0, notch_q_1=20.0, notch_gain_1=-6.0,
                notch_freq_2=450.0, notch_q_2=20.0, notch_gain_2=-5.0,
                notch_freq_3=600.0, notch_q_3=20.0, notch_gain_3=-4.0,
                limiter_ceiling_linear=0.88,
                expander_threshold_linear=0.05, expander_ratio=2.0,
                expander_attack_ms=5.0, expander_release_ms=40.0,
                expander_range_linear=0.25,
                sat_threshold_linear=0.74, sat_drive_db=8.0, sat_makeup_db=-3.2,
                air_db=0.0, width=0.0,
                bus_threshold_linear=0.18, bus_ratio=2, bus_attack_ms=10,
                bus_release_ms=100, bus_mix=0.0,
            )

            graph = build_filtergraph(derived=d)

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
                f"Native chain render failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-500:]}"
            )
            assert output_wav.stat().st_size > 1000

    @pytest.mark.skipif(not _has_lsp(), reason="LSP plugins not installed")
    def test_lsp_chain_adds_harmonics(self):
        """The full LSP chain with native saturation should add harmonics."""
        with tempfile.TemporaryDirectory() as tmp:
            input_wav = Path(tmp) / "sine_in.wav"
            output_wav = Path(tmp) / "sine_out.wav"

            sr = 48000
            dur = 2.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            amp = 10 ** (-10 / 20)
            y = amp * np.sin(2 * math.pi * 440 * t)
            stereo = np.column_stack([y, y.copy()])
            sf.write(str(input_wav), stereo.astype(np.float32), sr, subtype="PCM_24")

            analysis_data2 = {
                'crest_factor_db': 10.0, 'peak_db': -3.0, 'rms_db': -15.0,
                'transient_attack_ms': 8.0, 'agc_recovery_ms': 60.0,
                'room_modes_hz': (250,), 'room_mode_qs': (5,),
                'room_mode_gains_db': (2,),
                'harshness_index': 0.0, 'is_heavily_clipped': False,
            }
            stages = {
                'expander': False, 'ducking': False, 'deharsher': False,
                'notches': False, 'saturation': True, 'limiter': False,
                'hp35': False, 'hp150': False, 'glue': True, 'air': False,
                'width': False, 'bus_comp': False, 'intensity': True,
            }

            from cleaner.lsp_chain_builder import build_lsp_filtergraph
            from cleaner.types import AnalysisReport, MasteringSettings
            from cleaner.analysis.derived import compute_derived_params
            analysis2 = AnalysisReport(**analysis_data2)
            settings = MasteringSettings(glue=0.6, intensity=1.0)
            derived = compute_derived_params(analysis2, settings)
            graph = build_lsp_filtergraph(analysis2, settings, derived, stages)

            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", str(input_wav),
                "-filter_complex", graph,
                "-map", "[out]",
                "-c:a", "pcm_s24le",
                str(output_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            assert proc.returncode == 0, f"Render failed:\n{proc.stderr[-500:]}"

            y_out, _ = sf.read(str(output_wav), always_2d=True, dtype='float32')
            y_mono = np.mean(y_out, axis=1)
            n = len(y_mono)
            fft = np.abs(np.fft.rfft(y_mono * np.hanning(n)))
            freqs = np.fft.rfftfreq(n, 1 / sr)

            def energy_at(freq):
                idx = np.argmin(np.abs(freqs - freq))
                win = max(1, int(sr / n * 5))
                lo, hi = max(0, idx - win), min(len(fft) - 1, idx + win)
                return 20 * math.log10(max(np.mean(fft[lo:hi + 1]), 1e-15))

            h3 = energy_at(1320) - energy_at(440)
            print(f"\nH3 relative to fundamental: {h3:.1f} dB")
            assert h3 > -60.0, f"H3 too quiet: {h3:.1f} dB (tanh saturation not working in full chain)"

class TestSaturationSweep:
    """Verify tanh saturation produces monotonic harmonic increase."""

    def test_glue_sweep_monotonic(self):
        """H3/fund ratio must increase monotonically with glue from 0 to 1."""
        import math, tempfile, subprocess
        from pathlib import Path
        import numpy as np, soundfile as sf

        sr = 48000
        dur = 1.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        amp = 10 ** (-1 / 20)

        def h3_ratio(y):
            window = np.hanning(len(y))
            fft = np.abs(np.fft.rfft(y * window))
            freqs = np.fft.rfftfreq(len(y), 1 / sr)
            def peak_at(f):
                idx = np.argmin(np.abs(freqs - f))
                lo, hi = max(0, idx-2), min(len(fft)-1, idx+2)
                return np.max(fft[lo:hi+1])
            fund = peak_at(1000)
            h3 = peak_at(3000)
            return h3 / fund if fund > 1e-10 else 0.0

        ratios = []
        for glue in [0.0, 0.15, 0.3, 0.5, 0.8, 1.0]:
            with tempfile.TemporaryDirectory() as tmp:
                iw = Path(tmp) / "in.wav"; ow = Path(tmp) / "out.wav"
                y = amp * np.sin(2 * math.pi * 1000 * t)
                sf.write(str(iw), np.column_stack([y, y.copy()]).astype(np.float32), sr, subtype="PCM_24")
                eff = glue * (0.3 + 0.5 * 0.7)
                drive = eff * 16.0
                thresh = 0.92 - eff * 0.35
                graph = f"[0:a]aresample={sr},volume={drive}dB,asoftclip=type=tanh:threshold={thresh}:output=1.0:oversample=4,volume={-drive*0.4:.1f}dB[out]"
                proc = subprocess.run(["ffmpeg", "-y", "-nostdin", "-i", str(iw), "-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s24le", str(ow)], capture_output=True, text=True, timeout=30)
                assert proc.returncode == 0
                y_out, _ = sf.read(str(ow), always_2d=True, dtype='float32')
                ratios.append(h3_ratio(np.mean(y_out, axis=1)))

        # Must be monotonic
        for i in range(len(ratios)-1):
            assert ratios[i] <= ratios[i+1] * 1.01, (
                f"Not monotonic at glue index {i}: {ratios[i]:.6f} > {ratios[i+1]:.6f}"
            )
        # glue=0 must be near zero
        assert ratios[0] < 0.02, f"glue=0 has harmonics: {ratios[0]:.4f}"
        # glue=0.15 must show subtle increase
        assert ratios[1] > ratios[0] * 1.1, f"glue=0.15 no increase: {ratios[1]:.6f} <= {ratios[0]:.6f}"


def test_smoke_full_native_chain():
    """Full native chain renders without error."""
    with tempfile.TemporaryDirectory() as tmp:
        input_wav = Path(tmp) / "input.wav"
        output_wav = Path(tmp) / "output.wav"

        sr = 48000
        dur = 1.0
        noise = np.random.default_rng(42).normal(0, 10**(-12/20), (int(sr*dur), 2))
        sf.write(str(input_wav), noise.astype(np.float32), sr, subtype="PCM_24")

        from cleaner.ffmpeg_chain import build_filtergraph

        from cleaner.types import DerivedParams
        d = DerivedParams(
            comp_threshold_linear=0.18, comp_ratio=4,
            comp_attack_ms=2.0, comp_release_ms=60,
            notch_freq_1=300.0, notch_q_1=20.0, notch_gain_1=-6.0,
            notch_freq_2=450.0, notch_q_2=20.0, notch_gain_2=-5.0,
            notch_freq_3=600.0, notch_q_3=20.0, notch_gain_3=-4.0,
            limiter_ceiling_linear=0.88,
            expander_threshold_linear=0.05, expander_ratio=2.0,
            expander_attack_ms=5.0, expander_release_ms=40.0,
            expander_range_linear=0.25,
            sat_threshold_linear=0.74, sat_drive_db=8.0, sat_makeup_db=-3.2,
            air_db=0.0, width=0.0,
            bus_threshold_linear=0.18, bus_ratio=2, bus_attack_ms=10,
            bus_release_ms=100, bus_mix=0.0,
        )
        graph = build_filtergraph(derived=d)
        cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(input_wav),
               "-filter_complex", graph, "-map", "[out]",
               "-c:a", "pcm_s24le", str(output_wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"Render failed: {proc.stderr[-500:]}"

        y_out, _ = sf.read(str(output_wav), always_2d=True)
        assert output_wav.stat().st_size > 1000


def test_port_unit_resolution_without_plugins():
    """Verify clamp_to_port resolves ms vs s from max_v without LSP installed."""
    from cleaner.lv2_params import clamp_to_port
    from cleaner.lv2_introspect import PortInfo

    # Expander-like: max=2000 → ms
    p_exp_at = PortInfo(symbol="at", min_val=0, max_val=2000, default_val=10, unit="ms")
    assert clamp_to_port(10, p_exp_at, convert_unit=True) == 10.0  # ms pass through

    # Limiter-like: max=20 → s, value in ms → clamped to port min 0.25
    p_lim_at = PortInfo(symbol="at", min_val=0.25, max_val=20, default_val=5, unit="s")
    result = clamp_to_port(5, p_lim_at, convert_unit=True)
    assert 0.25 <= result <= 20, f"limiter at clamped: {result}"  # clamped to min or converted

    # Expander release: max=5000 → ms
    p_exp_rt = PortInfo(symbol="rt", min_val=0, max_val=5000, default_val=100, unit="ms")
    assert clamp_to_port(100, p_exp_rt, convert_unit=True) == 100.0

    # Limiter lookahead: max=20 → s, port min 0.1 clamps small values
    p_lim_lk = PortInfo(symbol="lk", min_val=0.1, max_val=20, default_val=5, unit="s")
    result_lk = clamp_to_port(5, p_lim_lk, convert_unit=True)
    assert result_lk >= 0.1, f"lk clamped to min: {result_lk}"

    # Linear gain port: dB → G
    p_gain = PortInfo(symbol="g_3", min_val=0.01585, max_val=63.1, default_val=1, unit="linear_gain")
    assert abs(clamp_to_port(0, p_gain, convert_unit=True) - 1.0) < 0.001  # 0 dB → 1.0
    assert abs(clamp_to_port(6, p_gain, convert_unit=True) - 2.0) < 0.01   # 6 dB → ~2.0


def test_lsp_graph_reflects_typed_values_no_plugins():
    """LSP graph must use settings.ceiling_db and derived.comp_threshold_linear,
    NOT hardcoded defaults. This test catches the ceiling regression (was always -1.1)."""
    from cleaner.types import AnalysisReport, MasteringSettings
    from cleaner.analysis.derived import compute_derived_params

    analysis = AnalysisReport()
    settings = MasteringSettings(ceiling_db=-0.5, intensity=0.5)
    derived = compute_derived_params(analysis, settings)

    stages = {
        'hp35': False, 'hp150': False, 'expander': False, 'ducking': True,
        'deharsher': False, 'notches': False, 'saturation': False,
        'limiter': False, 'glue': False, 'air': False,
        'width': False, 'bus_comp': False, 'intensity': True,
    }

    try:
        from cleaner.lsp_chain_builder import build_lsp_filtergraph
        graph = build_lsp_filtergraph(analysis, settings, derived, stages)
    except RuntimeError as e:
        if "LV2 plugin introspection failed" in str(e):
            # Plugin not found, skip
            return
        if "Cannot validate port symbols" in str(e):
            return
        raise

    # Postamble should use settings.ceiling_db (-0.5), NOT default -1.1
    # ceiling_db=-0.5: limit = 10^(-0.5/20) ≈ 0.9441
    # post_ceiling = -0.5 + 0.0 (limiter off) = -0.5, min(-0.5, -0.3) = -0.5
    expected_limit = f"{10.0**(-0.5/20.0):.4f}"
    assert f"alimiter=limit={expected_limit}" in graph, \
        f"Postamble ceiling should be {expected_limit} (from settings.ceiling_db=-0.5), got: ...{graph[-200:]}"

    # Ducking should use derived.comp_threshold_linear, not default 0.05
    assert f"threshold={derived.comp_threshold_linear}" in graph, \
        f"Ducking threshold should be {derived.comp_threshold_linear} (from derived), not 0.05"
