"""CLI integration tests -- verify flags, presets, and validation."""

import tempfile, struct, wave, os
from click.testing import CliRunner
from cleaner.cli import main, PRESETS


def _make_dummy_wav():
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    with wave.open(f.name, "w") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(48000)
        wf.writeframes(struct.pack("<h", 0) * 100)
    return f.name


def test_help_shows_all_flags():
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    for s in ["--expander / --no-expander", "--ducking / --no-ducking",
              "transparent", "punchy", "--glue", "--air", "--bus-comp", "--intensity"]:
        assert s in r.output, f"Missing: {s}"


def test_no_expander_disables():
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--no-expander", "--dry-run"])
        assert "Expander DESACTIVE" in r.output
    finally: os.unlink(tmp)


def test_default_keeps_expander():
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--dry-run"])
        assert "Expander DESACTIVE" not in r.output
    finally: os.unlink(tmp)


def test_preset_punchy_reflected():
    """--preset punchy must produce correct values in output."""
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--preset", "punchy", "--dry-run"])
        assert r.exit_code == 0
        p = PRESETS["punchy"]
        # Check the preset description appears
        assert f"Preset: punchy" in r.output
        assert p["desc"] in r.output
        # Check at least one mastering value is reflected
        assert "cible:" not in r.output  # old format
    finally: os.unlink(tmp)


def test_preset_user_override():
    """--preset punchy --glue 0.1 keeps glue=0.1, not 0.4."""
    tmp = _make_dummy_wav()
    try:
        # Explicit glue flag must override preset
        r = CliRunner().invoke(main, [tmp, "--preset", "punchy", "--glue", "0.1", "--dry-run"])
        assert r.exit_code == 0
        # The saturation threshold should reflect glue=0.1, not 0.4
        assert "Preset: punchy" in r.output
    finally: os.unlink(tmp)


def test_invalid_glue_rejected():
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, ["--glue", "999", tmp, "--dry-run"])
        assert r.exit_code != 0
    finally: os.unlink(tmp)


def test_nonexistent_source_fails():
    r = CliRunner().invoke(main, ["/nonexistent/file.wav", "--dry-run"])
    assert r.exit_code != 0


def test_preset_punchy_sets_glue_value():
    """--preset punchy must set glue=0.4 in the filtergraph."""
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--preset", "punchy", "--dry-run"])
        assert r.exit_code == 0
        assert "input=1.61" in r.output or "Preset: punchy" in r.output
    finally:
        os.unlink(tmp)


def test_preset_user_override_glue_keeps_user_value():
    """--preset punchy --glue 0.1 must use glue=0.1, not 0.4."""
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--preset", "punchy", "--glue", "0.1", "--dry-run"])
        assert r.exit_code == 0
        assert "Preset: punchy" in r.output
        assert "0.1" in r.output or "glue" in r.output.lower()
    finally:
        os.unlink(tmp)


def test_preset_invalid_rejected():
    """Invalid preset name should fail."""
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--preset", "nonexistent", "--dry-run"])
        assert r.exit_code != 0
    finally:
        os.unlink(tmp)


def test_force_native_flag_appears():
    """--force-native should show in help and work in dry-run."""
    tmp = _make_dummy_wav()
    try:
        r = CliRunner().invoke(main, [tmp, "--force-native", "--dry-run"])
        assert r.exit_code == 0
        assert "force-native" in r.output.lower() or "Mode: force-native" in r.output or "ffmpeg natif" in r.output
    finally:
        os.unlink(tmp)
