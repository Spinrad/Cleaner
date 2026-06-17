"""
I/O adapter -- FFmpeg interface for audio conversion and LUFS normalisation.

- Source -> WAV 48 kHz / 24-bit temporary file.
- LUFS measurement via ffmpeg ebur128 filter.
- Gain application via ffmpeg volume filter.
- Streaming disk I/O -- RAM near zero regardless of file duration.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from cleaner.constants import LUFS_GAIN_MIN_DB, LUFS_GAIN_MAX_DB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not available in $PATH."""

class SourceDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode the source file."""

class LUFSMeasurementError(RuntimeError):
    """Raised when the ffmpeg ebur128 filter fails to measure LUFS."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 48000
TARGET_BIT_DEPTH = "pcm_s24le"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def require_ffmpeg() -> Path:
    """
    Verify that ffmpeg is available in $PATH.

    Returns:
        Absolute path to the ffmpeg executable.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not found.
    """

    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "ffmpeg is not installed or not in $PATH.\n"
            "Install with: sudo apt install ffmpeg\n"
            "ffmpeg >= 5.0 is required for full ebur128 support."
        )
    return Path(path)


def convert_to_wav(
    source_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    bit_depth: str = TARGET_BIT_DEPTH,
    overwrite: bool = True,
    start_s: float | None = None,
    end_s: float | None = None,
) -> Path:
    """
    Convert any audio source to a temporary WAV file via ffmpeg.

    The output format is:
        - Container: WAV (RIFF)
        - Codec: PCM 24-bit (pcm_s24le)
        - Sample rate: 48 000 Hz
        - Channels: 2 (stereo)

    Args:
        source_path: Path to the source audio file (any ffmpeg-supported format).
        output_path: Desired path for the output WAV file.
        sample_rate: Target sample rate in Hz (default: 48000).
        bit_depth: FFmpeg codec name (default: 'pcm_s24le').
        overwrite: If True, overwrite existing output. If False,
                   raise FileExistsError.

    Returns:
        Path to the converted WAV file.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not available.
        SourceDecodeError: If ffmpeg cannot decode the source.
    """
    ffmpeg_path = require_ffmpeg()

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    # Build ffmpeg command with optional trim.
    # -ss before -i for fast seeking, then -t for exact duration.
    cmd = [str(ffmpeg_path), "-y" if overwrite else "-n"]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(source_path)]
    if end_s is not None:
        if start_s is not None:
            cmd += ["-t", str(end_s - start_s)]
        else:
            cmd += ["-to", str(end_s)]
    cmd += ["-acodec", bit_depth, "-ar", str(sample_rate), "-ac", "2", str(output_path)]

    logger.info("Converting source to WAV: %s → %s", source_path.name, output_path.name)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr_tail = result.stderr[-1500:] if len(result.stderr) > 1500 else result.stderr
        raise SourceDecodeError(
            f"ffmpeg failed to decode source file: {source_path}\n"
            f"Stderr:\n{stderr_tail}"
        )

    if not output_path.exists():
        raise SourceDecodeError(
            f"ffmpeg completed but no output file was produced: {output_path}"
        )

    return output_path


def measure_lufs(wav_path: Path) -> float:
    """
    Measure the Integrated Loudness (LUFS) of a WAV file using ffmpeg ebur128.

    This is a streaming measurement — RAM footprint is near zero regardless
    of file duration. The ebur128 filter processes the file in one pass
    and outputs the integrated loudness value to stderr as text.

    Args:
        wav_path: Path to the WAV file to measure.

    Returns:
        Integrated loudness value in LUFS (e.g., -23.5).

    Raises:
        FFmpegNotFoundError: If ffmpeg is not available.
        LUFSMeasurementError: If measurement fails.
    """
    ffmpeg_path = require_ffmpeg()

    cmd = [
        str(ffmpeg_path),
        "-i", str(wav_path),
        "-filter_complex", "ebur128=peak=true:framelog=quiet",
        "-f", "null", "-",
    ]

    logger.info("Measuring LUFS: %s", wav_path.name)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    # ffmpeg outputs ebur128 results to stderr
    stderr = result.stderr

    # Parse the Integrated Loudness line.
    # Expected format: "I:         -14.2 LUFS"
    match = re.search(r"I:\s+([-\d.]+)\s+LUFS", stderr)
    if match:
        lufs = float(match.group(1))
        logger.info("Measured Integrated LUFS: %.1f", lufs)
        return lufs

    # If we couldn't parse the LUFS value, check if ffmpeg itself failed
    if result.returncode != 0:
        raise LUFSMeasurementError(
            f"ffmpeg ebur128 measurement failed (exit code {result.returncode}).\n"
            f"File: {wav_path}\n"
            f"Stderr:\n{stderr[-1500:]}"
        )

    raise LUFSMeasurementError(
        f"Could not parse Integrated LUFS from ffmpeg ebur128 output.\n"
        f"File: {wav_path}\n"
        f"Full stderr:\n{stderr}"
    )


def apply_lufs_gain(
    input_wav: Path,
    output_wav: Path,
    target_lufs: float = -14.0,
    overwrite: bool = True,
) -> tuple[Path, float]:
    """
    Measure the LUFS of a WAV file and apply gain to reach the target level.

    This is a two-pass operation:
        1. Measure integrated LUFS via ffmpeg ebur128.
        2. Apply gain via ffmpeg volume filter.

    The gain is clamped to [-6.0, +14.0] dB to avoid extreme changes.

    Args:
        input_wav: Path to the rendered WAV file.
        output_wav: Desired path for the loudness-normalised output.
        target_lufs: Target integrated LUFS level (default: -14.0).
        overwrite: If True, overwrite existing output.

    Returns:
        Tuple of (output_path, applied_gain_db).

    Raises:
        FFmpegNotFoundError: If ffmpeg is not available.
        LUFSMeasurementError: If LUFS measurement fails.
        SourceDecodeError: If gain application fails.
    """
    ffmpeg_path = require_ffmpeg()

    # Passe 1: measure
    measured_lufs = measure_lufs(input_wav)

    # Passe 2: calculate and clamp gain
    gain_db = target_lufs - measured_lufs
    gain_db = max(min(gain_db, LUFS_GAIN_MAX_DB), LUFS_GAIN_MIN_DB)

    logger.info(
        "LUFS normalisation: measured=%.1f LUFS, target=%.1f LUFS, gain=%.2f dB",
        measured_lufs, target_lufs, gain_db,
    )

    if abs(gain_db) < 0.05:
        shutil.copy2(input_wav, output_wav)
        logger.info("Gain < 0.05 dB — copying file without re-encoding.")
        return output_wav, 0.0

    # Apply gain
    cmd = [
        str(ffmpeg_path),
        "-y" if overwrite else "-n",
        "-i", str(input_wav),
        "-filter:a", f"volume={gain_db:.4f}dB",
        "-c:a", TARGET_BIT_DEPTH,
        str(output_wav),
    ]

    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr_tail = result.stderr[-1500:] if len(result.stderr) > 1500 else result.stderr
        raise SourceDecodeError(
            f"ffmpeg failed to apply LUFS gain.\n"
            f"Input: {input_wav}\n"
            f"Stderr:\n{stderr_tail}"
        )

    if not output_wav.exists():
        raise SourceDecodeError(
            f"ffmpeg completed but no output file was produced: {output_wav}"
        )

    return output_wav, gain_db
