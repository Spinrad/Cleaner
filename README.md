# cleaner v3.1.0

Post‑Punk live recording restoration. Python analysis drives a native ffmpeg
`filter_complex` DSP chain. Zero external plugin dependencies.

## What it does

Reads any audio file, analyses it (room modes, dynamics, clipping, stereo
phase), then runs a configurable ffmpeg filtergraph in a single pass:

```
Input → HP 35 Hz → Expander (agate, anti‑AGC) → M/S encode → Sidechain ducking
      → M/S decode → De‑harsher (opt‑in) → 3× Notch EQ (room modes)
      → Tape saturation (tanh) → Air shelf → Stereo width → Bus compressor
      → True Peak limiter → LUFS normalisation → Output
```

Every DSP parameter is either measured or uses conservative fixed ratios
that have proven musical across Post‑Punk material. The philosophy is
restrained (Albini: don't over‑process). What you hear is the analysis
speaking — no presets that fight the source.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11 and ffmpeg ≥ 5.0 (`sudo apt install ffmpeg`).

## Usage

```bash
cleaner recording.m4a                       # full chain, default mastering
cleaner recording.m4a --dry-run             # analyse only, print filtergraph
cleaner recording.m4a --preset punchy       # aggressive post‑punk preset
cleaner recording.m4a --preset warm --air 2.0  # preset + override air
```

## Pipeline (detail)

Every stage can be toggled with `--stage` / `--no-stage` boolean flags.
Enabled by default except where noted.

| # | Stage | Engine | Toggle flag |
|---|-------|--------|-------------|
| 1 | HP 35 Hz | `highpass` (Butterworth, order 2) | `--hp35` / `--no-hp35` |
| 2 | Expander (anti‑AGC) | `agate=mode=upward` | `--expander` / `--no-expander` |
| 3 | M/S encode | `stereotools=mode=lr>ms` | (structural — always on) |
| 4 | Sidechain ducking | `channelsplit` → HP 150 Hz Side → `sidechaincompress` (Mid→Side) → `amerge` | `--ducking` / `--no-ducking` |
| 5 | M/S decode | `stereotools=mode=ms>lr` | (structural — always on) |
| 6 | De‑harsher | `adynamicequalizer` (dynamic bell cut, 2.5–4.5 kHz) | `--deharsher` / `--no-deharsher` (default OFF) |
| 7 | Room‑mode notches (×3) | `anequalizer` (Bell type, 100–800 Hz) | `--notches` / `--no-notches` |
| 8 | Tape saturation | `volume` → `asoftclip=type=tanh` (4× oversample) → `volume` | `--saturation` / `--no-saturation` |
| 9 | Air shelf | `treble` (high‑shelf, 8 kHz) | `--air` / (value ≤ 0.01 → off) |
| 10 | Stereo width | `stereotools=mode=lr>lr` (base spread) | `--width` / (abs ≤ 0.001 → off) |
| 11 | Bus compressor | `acompressor=mode=downward` (SSL‑style, parallel) | `--bus-comp` / (value ≤ 0.01 → off) |
| 12 | True Peak limiter | `alimiter` (attack 0.1 ms, release 30 ms) | `--limiter` / `--no-limiter` |
| 13 | LUFS normalisation | `ebur128` → `volume` (gain clamped to [‑3, +6] dB) | `--lufs` / `--no-lufs` |
| 14 | Post‑LUFS re‑limiter | `alimiter` (ceiling from `--ceiling`, applied only if LUFS gain > 0.5 dB) | (automatic, same ceiling) |

## Presets

Presets override colour parameters (`--glue`, `--air`, `--width`,
`--bus-comp`, `--target-lufs`) unless the user explicitly passes a value.
Toggle flags (`--expander`, `--notches`…) are never touched by presets.

| Preset | Glue | Air | Width | Bus | LUFS | Use case |
|--------|------|-----|-------|-----|------|----------|
| `transparent` | 5% | +0.5 | 0.00 | 0% | ‑14 | Archival, zero colour |
| `warm` | 50% | 0.0 | ‑0.15 | 30% | ‑13 | Vintage, tape feel |
| `open` | 10% | +2.5 | +0.35 | 15% | ‑14 | Live ambience, wide |
| `punchy` | 40% | +2.0 | +0.10 | 50% | ‑11 | Aggressive post‑punk |
| `loud` | 60% | +3.0 | +0.20 | 70% | ‑9 | Maximum density |

## Options

See `cleaner --help` for the full list. Key flags:

| Flag | Range | Default | Description |
|------|-------|---------|-------------|
| `--glue FLOAT` | 0–1 | 0.15 | Saturation drive (0 = off, 1 = max) |
| `--air FLOAT` | 0–5 dB | 1.5 | High‑shelf boost at 8 kHz |
| `--width FLOAT` | ‑1 to 1 | 0.0 | Stereo width (+widens, −narrows) |
| `--bus-comp FLOAT` | 0–1 | 0.0 | SSL bus compressor drive (parallel) |
| `--target-lufs FLOAT` | ‑20 to ‑8 | ‑14 | Output loudness (EBU R128) |
| `--ceiling FLOAT` | ‑3.0 to ‑0.3 | ‑1.1 | Limiter ceiling in dBFS |
| `--notch-intensity FLOAT` | 0–2 | 1.0 | Room‑mode attenuation multiplier |
| `--tame-cymbals FLOAT` | ‑12 to 0 | 0.0 | De‑harsher threshold offset (negative = more reduction) |
| `--intensity FLOAT` | 0–1 | 0.5 | Global intensity (scales expander range and notch depth) |

## Architecture

```
cleaner/
  cli.py             Click CLI, presets, validation
  pipeline.py        Orchestrator, rich output, before/after metrics
  ffmpeg_chain.py    Builds filter_complex string from analysis report
  io_adapter.py      ffmpeg subprocess: convert, measure LUFS, apply gain
  analysis/
    spectrum.py      STFT, room mode detection (temporal persistence)
    dynamics.py      Crest factor, transient analysis, AGC recovery
    mid_side.py      M/S correlation, cymbal phase profiling
    clipping.py      Digital clipping detector
    global_analysis.py  Orchestrator, parameter derivation
```

## How analysis drives parameters

Parameters come from two sources: measurements (dynamic) and musical
constants (fixed ratios that work across the genre).

| Measurement | Drives |
|-------------|--------|
| Crest factor + AGC recovery time | Expander threshold, attack, release |
| Peak level | Expander threshold (peak − 3 dB) and range |
| Room mode frequencies + Q | Notch EQ frequencies and Q (gain is a scaled, clamped depth derived from mode prominence) |
| RMS level + crest | Sidechain compressor threshold, bus compressor threshold |
| HF correlation (5–10 kHz) | De‑harsher threshold and ratio |
| Clipping ratio | Penalty: reduces expander and saturation |

Fixed parameters (musical constants, not measured):
- Expander ratio: 1.1–1.5 (from crest, conservatively mapped)
- Sidechain compressor ratio: 4:1, attack: 2 ms
- Bus compressor ratio: 2:1, attack: 10 ms, release: 100 ms
- Limiter attack: 0.1 ms, release: 30 ms

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Limitations (honest)

- **Conservative processing.** The analysis is richer than the treatment.
  Parameters are deliberately restrained — the goal is restoration, not
  re‑creation.
- **Upward expansion cannot restore crushed transients.** A smartphone AGC
  has already destroyed information. `agate` can only amplify what remains.
- **Tape saturation is subtle.** The tanh soft‑clip only engages on peaks
  above the threshold (default: −1.1 dBFS for `glue=0.15`). On typical
  recordings peaking at −6 to −3 dBFS, the effect is barely audible unless
  `--glue` is pushed above 0.5 (threshold drops to −2.6 dBFS at `glue=0.5,
  intensity=0.5`).
- **LUFS gain is clamped to [‑3, +6] dB.** If the processed file is very
  quiet or very loud relative to the target, the target may not be reached.
  The output always reports the actual value achieved.
- **De‑harsher is experimental and disabled by default.** Enable with
  `--deharsher`. It uses `adynamicequalizer` which offers limited control
  over the detection band shape.
- **No adaptive stereo processing.** The width stage is a static spread,
  not content‑aware. M/S correlation is measured but only displayed, not
  acted upon beyond the ducking stage.
- **Analysis runs on the first 60 seconds at 16 kHz mono.** Longer files
  may miss late‑occurring modes; the assumption is that room acoustics are
  time‑invariant.

## Roadmap (v4 — non implémenté)

The next major version will introduce **LSP (Linux Studio Plugins) / LV2**
for coloration stages, loaded via ffmpeg's native `lv2` filter, while
keeping structural processing (M/S, sidechain ducking, LUFS, post‑limiter)
in ffmpeg native. This allows:

- **Saturation with real drive** — replacing the tanh soft‑clip with a
  multi‑stage saturator that the signal actually hits.
- **Parametric EQ with per‑band control** — replacing `anequalizer` with
  a 16‑band LSP EQ for room modes and air shelf.
- **Full‑featured bus compressor** — SSL‑style glue with proper knee,
  dual release, and dry/wet mix.
- **Musical true‑peak limiter** — with oversampling and adaptive release.
- **De‑esser with configurable detection** — replacing the fixed
  `adynamicequalizer`.

LSP plugins are an external dependency (user‑installed). The current
v3.1.0 architecture remains fully functional via `--force-native`.

Status: **design phase** (see `CDC.md`). No LSP code has been written.
