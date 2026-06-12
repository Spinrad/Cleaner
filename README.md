# cleaner v0.1.0

Live recording restoration. Hybrid ffmpeg + LSP/LV2 DSP chain.

## What it does

Reads any audio file, analyses it (room modes, dynamics, clipping, stereo
phase), then runs a configurable DSP chain in a single ffmpeg `filter_complex`
pass:

- **Structural stages** (resample, HP, M/S encode/decode, sidechain ducking,
  LUFS measurement, post-limiter) use **native ffmpeg filters**.
- **Coloration stages** (expander anti-AGC, notches EQ, air shelf, de-harsher,
  bus compressor, musical limiter) use **LSP plugins via LV2**.
  Saturation is **native ffmpeg** (`asoftclip=type=tanh`).,
  loaded through ffmpeg's native `lv2` filter.

When LSP plugins are unavailable, `--force-native` falls back to a pure ffmpeg
chain (see `ffmpeg_chain.py`).

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11, ffmpeg ≥ 5.0, and LSP plugins:

```bash
sudo apt install ffmpeg lsp-plugins-lv2
```

Cleaner auto-detects LSP at startup. If plugins are missing, it falls back to
native ffmpeg with a warning. Use `--force-native` to suppress the warning.

## Usage

```bash
cleaner recording.m4a                          # hybrid LSP chain (default)
cleaner recording.m4a --dry-run                # analyse only, print filtergraph
cleaner recording.m4a --force-native           # pure ffmpeg chain, no LSP
cleaner recording.m4a --preset punchy          # aggressive post-punk preset
cleaner recording.m4a --preset warm --air 2.0  # preset + override air
```

## Pipeline (detail)

Every stage can be toggled with `--stage` / `--no-stage` boolean flags.
Enabled by default except where noted.

| # | Stage | Engine | Toggle flag |
|---|-------|--------|-------------|
| 1 | HP 35 Hz | `highpass` (Butterworth, order 2) [NATIF] | `--hp35` / `--no-hp35` |
| 2 | Expander (anti-AGC) | `expander_stereo` (LSP, Mode=Up) [LSP] | `--expander` / `--no-expander` |
| 3 | M/S encode | `stereotools=mode=lr>ms` [NATIF] | (structural — always on) |
| 4 | Sidechain ducking | `channelsplit` → HP 150 Hz Side → `sidechaincompress` (Mid→Side) → `amerge` [NATIF] | `--ducking` / `--no-ducking` |
| 5 | M/S decode | `stereotools=mode=ms>lr` [NATIF] | (structural — always on) |
| 6 | De-harsher | `sc_compressor_stereo` (LSP, bandpass 2.5-4.5 kHz) [LSP] | `--deharsher` / `--no-deharsher` (default OFF) |
| 7 | Room-mode notches (x3) | `para_equalizer_x16_stereo` (LSP, Bell type) [LSP] | `--notches` / `--no-notches` |
| 8 | Saturation | `asoftclip=type=tanh` (native, drive + makeup, 4x oversample) [NATIF] | `--saturation` / `--no-saturation` |
| 9 | Air shelf | `para_equalizer_x16_stereo` (LSP, hi-shelf 8 kHz, same EQ node) [LSP] | `--air` / (value ≤ 0.01 → off) |
| 10 | Stereo width | `stereotools=mode=lr>lr` (base spread) [NATIF] | `--width` / (abs ≤ 0.001 → off) |
| 11 | Bus compressor | `compressor_stereo` (LSP, SSL-style, parallel) [LSP] | `--bus-comp` / (value ≤ 0.01 → off) |
| 12 | Limiter | `limiter_stereo` (LSP, true-peak, oversampling, ALR) [LSP] | `--limiter` / `--no-limiter` |
| 13 | LUFS normalisation | `ebur128` → `volume` (gain clamped to [-3, +6] dB) [NATIF] | `--lufs` / `--no-lufs` |
| 14 | Post-LUFS re-limiter | `alimiter` (ceiling from `--ceiling`, only if LUFS gain > 0.5 dB) [NATIF] | (automatic) |

When `--force-native` is used, the [LSP] stages are replaced by native ffmpeg
equivalents: `agate=mode=upward` for expander, `anequalizer` for notches,
`acompressor` for bus compressor, `adynamicequalizer` for de-harsher,
and `alimiter` for the limiter.

## Presets

Presets override colour parameters (`--glue`, `--air`, `--width`,
`--bus-comp`, `--target-lufs`) unless the user explicitly passes a value.
Toggle flags (`--expander`, `--notches`…) are never touched by presets.

| Preset | Glue | Air | Width | Bus | LUFS | Use case |
|--------|------|-----|-------|-----|------|----------|
| `transparent` | 5% | +0.5 | 0.00 | 0% | -14 | Archival, zero colour |
| `warm` | 50% | 0.0 | -0.15 | 30% | -13 | Vintage, tape feel |
| `open` | 10% | +2.5 | +0.35 | 15% | -14 | Live ambience, wide |
| `punchy` | 40% | +2.0 | +0.10 | 50% | -11 | Aggressive post-punk |
| `loud` | 60% | +3.0 | +0.20 | 70% | -9 | Maximum density |

## Options

See `cleaner --help` for the full list. Key flags:

| Flag | Range | Default | Description |
|------|-------|---------|-------------|
| `--glue FLOAT` | 0-1 | 0.15 | Saturation drive (0 = off, 1 = max) |
| `--air FLOAT` | 0-5 dB | 1.5 | High-shelf boost at 8 kHz |
| `--width FLOAT` | -1 to 1 | 0.0 | Stereo width (+widens, -narrows) |
| `--bus-comp FLOAT` | 0-1 | 0.0 | SSL bus compressor drive (parallel) |
| `--intensity FLOAT` | 0-1 | 0.5 | Global intensity (scales expander ratio, notch depth, saturator drive) |
| `--target-lufs FLOAT` | -20 to -8 | -14 | Output loudness (EBU R128) |
| `--ceiling FLOAT` | -3.0 to -0.3 | -1.1 | Limiter ceiling in dBFS |
| `--notch-intensity FLOAT` | 0-2 | 1.0 | Room-mode attenuation multiplier |
| `--tame-cymbals FLOAT` | -12 to 0 | 0.0 | De-harsher threshold offset (negative = more reduction) |
| `--force-native` | flag | off | Use pure ffmpeg chain (no LSP/LV2) |

## Architecture

```
cleaner/
  cli.py                  Click CLI, presets, validation
  pipeline.py             Orchestrator, rich output, before/after metrics
  ffmpeg_chain.py         Native ffmpeg filter_complex builder (--force-native)
  lsp_chain_builder.py    Hybrid LSP/LV2 filter_complex builder (default)
  lsp_uris.py             Canonical LSP plugin URI list
  lv2_introspect.py       LV2 plugin discovery, introspection, caching
  lv2_params.py           Unit conversions (dB/linear/ms) and port clamping
  gain_tracking.py        Analytic gain tracking along the DSP chain
  io_adapter.py           ffmpeg subprocess: convert, measure LUFS, apply gain
  analysis/
    spectrum.py           STFT, room mode detection (temporal persistence)
    dynamics.py           Crest factor, transient analysis, AGC recovery
    mid_side.py           M/S correlation, cymbal phase profiling
    clipping.py           Digital clipping detector
    global_analysis.py    Orchestrator, LSP and ffmpeg parameter derivation
```

## How analysis drives parameters

Parameters come from two sources: measurements (dynamic) and musical
constants (fixed ratios that work across the genre).

| Measurement | Drives |
|-------------|--------|
| Crest factor + AGC recovery time | Expander ratio, attack, release |
| Peak level | Expander threshold (peak - 3 dB) |
| Room mode frequencies + Q + prominence | Notch EQ frequencies, Q, and gain (depth = prominence x 0.5, clamped) |
| RMS level + crest | Sidechain compressor threshold, bus compressor threshold |
| Harshness index (decorr x HF energy) | De-harsher threshold and ratio |
| Clipping ratio | Penalty: reduces expander and saturation |

Fixed parameters (musical constants, not measured):
- Expander ratio: 1.1-1.5 (from crest, conservatively mapped)
- Sidechain compressor ratio: 4:1, attack: 2 ms
- Bus compressor ratio: 2:1, attack: 10 ms, release: 100 ms
- Limiter attack: 5 ms, release: 5 ms, lookahead: 5 ms, oversampling: 4x

The `--intensity` macro scales expander ratio, notch depth, and saturator
drive simultaneously. `intensity=0` gives minimal processing; `intensity=1`
gives maximum effect.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Limitations (honest)

- **LSP plugins required.** The default chain needs `lsp-plugins-lv2` installed.
  Without them, the chain falls back to pure ffmpeg processing (with reduced
  fidelity: `agate` instead of LSP expander, `anequalizer` instead of
  parametric EQ, `alimiter` instead of true-peak LSP limiter).
  Saturation is always native tanh in both modes.
- **Conservative processing.** The analysis is richer than the treatment.
  Parameters are deliberately restrained — the goal is restoration, not
  re-creation.
- **Upward expansion cannot restore crushed transients.** A smartphone AGC
  has already destroyed information. The expander can only amplify what remains.
- **Saturation is native ffmpeg** (`asoftclip=type=tanh`, 4× oversampling).
  LSP v1.2.12 does not provide a saturator plugin. The tanh soft-clip is
  driven into its non-linear zone by `--glue` (0→0 dB drive, 1→+16 dB drive)
  and compensated with automatic makeup gain. At default settings
  (`glue=0.15`), the effect is subtle but measurable (+52% H3 at -1 dBFS).
- **LUFS gain is clamped to [-3, +6] dB.** If the processed file is very
  quiet or very loud relative to the target, the target may not be reached.
  The output always reports the actual value achieved.
- **De-harsher is experimental and disabled by default.** Enable with
  `--deharsher`. It uses `sc_compressor_stereo` with internal sidechain
  bandpass (2.5-4.5 kHz).
- **No adaptive stereo processing.** The width stage is a static spread,
  not content-aware. M/S correlation is measured but only displayed, not
  acted upon beyond the ducking stage.
- **Analysis runs on the first 60 seconds at 16 kHz mono.** Longer files
  may miss late-occurring modes; the assumption is that room acoustics are
  time-invariant.
