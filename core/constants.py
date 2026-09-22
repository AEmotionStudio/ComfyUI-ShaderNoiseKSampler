"""
Constants and magic numbers for shader noise generation.

This module centralizes all constants used across the codebase to
ensure consistency and make values easy to tune.
"""

# Hash constant used in simplex noise and pseudo-random functions
HASH_CONSTANT = 43758.5453

# Default number of channels for latent space
DEFAULT_CHANNELS = 4

# Independent renders a generator draws to fill a latent's channel axis. Channels
# past this are mixtures of those renders, so a very wide latent does not cost one
# render per channel. 64 is the basis travel_mode's walk was measured at on
# MiniMax H3 (7661d7c), and it covers every latent ComfyUI ships short of LTXV's 128.
CHANNEL_BASIS = 64

# Maximum octaves for FBM noise to prevent DoS
MAX_OCTAVES = 20

# Minimum and maximum scale values
MIN_SCALE = 0.001
MAX_SCALE = 100.0

# Default parameter values
DEFAULT_SCALE = 1.0
DEFAULT_OCTAVES = 3.0
DEFAULT_WARP_STRENGTH = 0.5
DEFAULT_PHASE_SHIFT = 0.5
DEFAULT_SHAPE_STRENGTH = 1.0
DEFAULT_COLOR_INTENSITY = 0.8
DEFAULT_SHADER_STRENGTH = 0.3
DEFAULT_TIME = 0.0

# Supported blend modes for combining shader noise with base noise
SUPPORTED_BLEND_MODES = [
    "normal",
    "add", 
    "multiply",
    "screen",
    "overlay",
    "soft_light",
    "hard_light",
    "difference",
]

# Supported noise transforms
SUPPORTED_TRANSFORMS = [
    "none",
    "reverse",
    "inverse",
    "absolute",
    "square",
    "sqrt",
    "log",
    "sin",
    "cos",
]

# Supported stage distributions for multi-stage sampling
SUPPORTED_DISTRIBUTIONS = [
    "uniform",
    "linear_decrease",
    "linear_increase",
    "gaussian",
    "first_stronger",
    "last_stronger",
]

# High channel threshold for fast mode
HIGH_CHANNEL_THRESHOLD = 16

# Visualization types
VISUALIZATION_TYPES = {
    0: "arrows",
    1: "lines",
    2: "dots",
    3: "ellipses",
    4: "streamlines",
}

# Default visualization type
DEFAULT_VISUALIZATION_TYPE = 3  # ellipses
