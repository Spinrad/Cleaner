"""DSP constants — single source of truth, grouped by stage.

Comments describe intention and provenance, not arithmetic.
Where a constant was tuned by ear on reference material, this is stated honestly.
"""

# ── Analysis ──────────────────────────────────────────────────────────

ANALYSIS_SR = 48000       # Full-band rendering sample rate
ANALYSIS_SR_LO = 16000    # Downsampled analysis rate (spectral/dynamics)
ANALYSIS_DURATION_S = 60.0  # Analysis window
N_FFT = 8192              # FFT size for spectral analysis
ROOM_MODE_LOW_HZ = 100.0  # Minimum room mode frequency (spectrum.py)
ROOM_MODE_HIGH_HZ = 800.0  # Maximum room mode frequency (spectrum.py)
MAX_ROOM_MODES = 3        # Number of room mode bands

# ── Saturation ────────────────────────────────────────────────────────

SAT_DRIVE_MULTIPLIER = 12.0   # drive = eff_glue × 12 dB (0→0, 0.5→6, 1→12)
SAT_MAKEUP_RATIO = 0.4        # makeup = −drive × 0.4 dB
SAT_THRESHOLD_BASE = 0.92     # tanh threshold at glue=0
SAT_THRESHOLD_SLOPE = 0.35    # threshold reduction per eff_glue unit
SAT_CLIP_PENALTY = 0.05       # threshold offset when heavily clipped

# ── Expander ──────────────────────────────────────────────────────────

# Gentle crest relief — tuned by ear on post-punk rehearsal recordings.
# Only the top ~3 dB are expanded.
EXP_THRESH_DELTA_DB = 3.0     # threshold = peak − 3 dB (near peak)
EXP_RATIO_BASE = 1.6          # ratio at crest=0 (max expansion)
EXP_RATIO_SLOPE = 0.03        # ratio reduction per crest dB
EXP_RATIO_MIN = 1.1           # minimum ratio (high crest, almost no expansion)
EXP_RATIO_MAX = 1.5           # maximum ratio (low crest, AGC-like)
EXP_ATTACK_FRAC = 0.5         # attack = measured_attack × 0.5
EXP_RELEASE_FRAC = 0.8        # release = agc_recovery × 0.8
EXP_CLIP_RATIO_FACTOR = 0.6   # ratio multiplier when heavily clipped

# ── Notches ───────────────────────────────────────────────────────────

NOTCH_PROM_DISABLE_DB = 3.0   # Disable band if prominence < 3 dB
NOTCH_DEPTH_RATIO = 0.5        # depth = prominence × 0.5
NOTCH_DEPTH_MIN_DB = 2.0      # Minimum cut depth
NOTCH_DEPTH_MAX_DB = 9.0      # Maximum cut depth
NOTCH_GAIN_FLOOR_DB = -12.0   # Hard floor on notch gain
NOTCH_Q_MIN = 3.0             # Minimum Q (wide)
NOTCH_Q_MAX = 10.0            # Maximum Q (narrow)
NOTCH_DEFAULT_HZ = (300.0, 450.0, 600.0)  # Fallback room mode frequencies
NOTCH_DEFAULT_Q = (5.0, 5.0, 5.0)
NOTCH_DEFAULT_GAIN_DB = (3.0, 3.0, 3.0)

# ── Air ───────────────────────────────────────────────────────────────

AIR_FREQ_HZ = 10000.0         # Bell centre frequency
AIR_Q = 2.0                   # Bell Q (moderate width)

# ── Clean mediums ─────────────────────────────────────────────────────

CLEAN_FREQ_HZ = 600.0         # Bell centre for low-mid cleanup
CLEAN_Q = 1.5                 # Wide bell

# ── Compressor (ducking + bus) ───────────────────────────────────────

# Sidechain ducking — classic conservative settings.
COMP_DUCK_THRESH_OFFSET_DB = 6.0  # threshold = RMS + 6 dB
COMP_DUCK_RATIO = 4.0
COMP_DUCK_ATTACK_MS = 2.0
COMP_DUCK_RELEASE_FACTOR = 1.5    # release = agc_recovery × 1.5
COMP_DUCK_RELEASE_MIN_MS = 40.0

# Bus compressor — SSL bus compressor convention.
BUS_RATIO = 2.0               # SSL classic 2:1
BUS_ATTACK_MS = 10.0          # Slow, lets transients through
BUS_RELEASE_MS = 100.0        # Smooth release
BUS_THRESH_CREST_FACTOR = 0.3 # Threshold formula: RMS − crest×0.3 + ...
BUS_THRESH_OFFSET = 12.0      # (1−bus_comp)×12 dB range

# ── Limiter ───────────────────────────────────────────────────────────

LIMITER_LOOKAHEAD_S = 0.1     # 100 ms (port minimum, native unit=s)
LIMITER_ATTACK_S = 0.25       # 250 ms (port minimum)
LIMITER_RELEASE_S = 0.25      # 250 ms (port minimum)
LIMITER_OVERSAMPLING = 4      # 4x oversampling

# ── De-harsher ────────────────────────────────────────────────────────

DEHARSH_BAND_LOW_HZ = 2500.0    # Actual de-harsher target band
DEHARSH_BAND_HIGH_HZ = 4500.0
DEHARSH_THRESH_MIN = 8.0
DEHARSH_THRESH_MAX = 50.0
DEHARSH_CREST_FACTOR = 2.0

# ── HF correlation (mid_side analysis) ────────────────────────────────

HF_CORR_LOW_HZ = 5000.0
HF_CORR_HIGH_HZ = 10000.0
HF_CORR_ORDER = 4

# ── Intensity macro ───────────────────────────────────────────────────

# Intensity scales glue differently from notches/expander:
# glue uses 0.3 + intensity×0.7 (subtle at low intensity, full at 1.0)
# notches and expander use intensity linearly.
INTENSITY_GLUE_OFFSET = 0.3
INTENSITY_GLUE_SLOPE = 0.7

# ── LUFS ──────────────────────────────────────────────────────────────

LUFS_GAIN_MIN_DB = -6.0
LUFS_GAIN_MAX_DB = 14.0

# ── Post-LUFS re-limiter ──────────────────────────────────────────────

POST_LIMITER_ATTACK_MS = 0.1
POST_LIMITER_RELEASE_MS = 30.0

# ── Clipping detection ────────────────────────────────────────────────

CLIP_DBFS = -0.1              # Samples ≥ −0.1 dBFS are considered clipped
CLIP_RATIO_THRESHOLD = 0.015  # "Heavily clipped" if > 1.5% of samples
CLIP_PENALTY_COMP = 0.8       # Multiply comp_threshold by 0.8 when clipped
CLIP_PENALTY_EXP_RATIO = 0.6  # Multiply expander_ratio by 0.6
CLIP_PENALTY_EXP_RANGE = 0.1  # Force expander_range to 0.1
CLIP_PENALTY_SAT = 0.05       # Raise sat threshold by 0.05
