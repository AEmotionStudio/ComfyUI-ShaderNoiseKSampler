"""
Constants and magic numbers for shader noise generation.

This module centralizes all constants used across the codebase to
ensure consistency and make values easy to tune.
"""

# Hash constant used in simplex noise and pseudo-random functions
HASH_CONSTANT = 43758.5453

# Default number of channels for latent space
DEFAULT_CHANNELS = 4

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

# Supported shader types
SUPPORTED_SHADER_TYPES = [
    "domain_warp",
    "tensor_field",
    "curl_noise",
    "temporal_coherent",
    "fbm_noise",
    "perlin",
    "waves",
    "gaussian",
    "heterogeneous_fbm",
    "interference",
    "spectral",
    "projection_3d",
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

# Model-specific channel counts
# Maps model class names to their latent channel counts
MODEL_CHANNEL_COUNTS = {
    # Standard models (4 channels)
    "SD1.5": 4,
    "SD2.1": 4,
    "SDXL": 4,
    "SD3": 16,
    
    # Flux models
    "Flux": 16,
    "FluxGuidance": 16,
    
    # Video models
    "WAN2.1": 16,
    "HunyuanVideo": 16,
    "LTXV": 128,
    "CosmosVideo": 16,
    "CogVideoX": 16,
    
    # Audio models
    "ACEStep": 8,
    
    # AnimateDiff
    "AnimateDiff": 4,
}

# Model name patterns for detection
# Used when exact class name isn't available
MODEL_NAME_PATTERNS = {
    "sd3": 16,
    "sdxl": 4,
    "sd1.5": 4,
    "sd2.1": 4,
    "flux": 16,
    "wan": 16,
    "hunyuan": 16,
    "ltxv": 128,
    "cosmos": 16,
    "cogvideo": 16,
    "animatediff": 4,
    "acestep": 8,
}

# High channel threshold for fast mode
HIGH_CHANNEL_THRESHOLD = 16

# Video model channel counts (subset of MODEL_CHANNEL_COUNTS for quick lookup)
VIDEO_MODEL_CHANNELS = {
    "WAN2.1": 16,
    "HunyuanVideo": 16,
    "LTXV": 128,
    "CosmosVideo": 16,
    "CogVideoX": 16,
    "Mochi": 12,
    "AnimateDiff": 4,
}

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
