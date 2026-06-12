"""Canonical LSP plugin URIs — single source of truth."""

# ── Required plugins (checked at startup) ──
# Saturation is native ffmpeg (asoftclip type=tanh) — LSP has no saturator in v1.2.12
REQUIRED_URIS: list[str] = [
    "http://lsp-plug.in/plugins/lv2/expander_stereo",
    "http://lsp-plug.in/plugins/lv2/para_equalizer_x16_stereo",
    "http://lsp-plug.in/plugins/lv2/compressor_stereo",
    "http://lsp-plug.in/plugins/lv2/limiter_stereo",
    "http://lsp-plug.in/plugins/lv2/sc_compressor_stereo",
]

# ── Individual URIs ──
EXPANDER_URI = "http://lsp-plug.in/plugins/lv2/expander_stereo"
EQ_URI = "http://lsp-plug.in/plugins/lv2/para_equalizer_x16_stereo"
COMPRESSOR_URI = "http://lsp-plug.in/plugins/lv2/compressor_stereo"
LIMITER_URI = "http://lsp-plug.in/plugins/lv2/limiter_stereo"
DEHARSHER_URI = "http://lsp-plug.in/plugins/lv2/sc_compressor_stereo"
