"""
Core functionality for ComfyUI-ShaderNoiseKSampler.

This package provides the central logic for shader noise sampling:
- params: Parameter normalization and validation
- constants: Magic numbers and default values
- blending: Blend mode implementations
- transforms: Noise transforms
- model_compat: Model detection and channel handling
- sampler: Core sampling logic
"""

from .params import (
    PARAM_ALIASES,
    ShaderParams,
    normalize_param_name,
    get_param_value,
)

from .constants import (
    HASH_CONSTANT,
    DEFAULT_CHANNELS,
    MAX_OCTAVES,
    SUPPORTED_BLEND_MODES,
    SUPPORTED_TRANSFORMS,
    SUPPORTED_SHADER_TYPES,
    SUPPORTED_DISTRIBUTIONS,
    MODEL_CHANNEL_COUNTS,
    VIDEO_MODEL_CHANNELS,
)

from .blending import (
    blend_noises,
)

from .transforms import (
    apply_noise_transform,
    normalize_noise,
    resize_noise_spatial,
    resize_noise_channels,
    match_noise_shape,
)

from .model_compat import (
    get_model_channel_count,
    detect_latent_format,
    build_noise_shape,
)

from .sampler import (
    calculate_stage_strengths,
    calculate_step_ranges,
    calculate_step_points,
    generate_shader_noise,
)

from .debug import (
    StubVisualizer,
    StubDebugger,
    get_visualizer,
    get_debugger,
    set_debug_level,
)

from .logging_config import (
    setup_logging,
    get_logger,
    set_debug_logging,
    set_info_logging,
    set_warning_logging,
    set_quiet_logging,
)

__all__ = [
    # Parameters
    "PARAM_ALIASES",
    "ShaderParams",
    "normalize_param_name",
    "get_param_value",
    # Constants
    "HASH_CONSTANT",
    "DEFAULT_CHANNELS",
    "MAX_OCTAVES",
    "SUPPORTED_BLEND_MODES",
    "SUPPORTED_TRANSFORMS",
    "SUPPORTED_SHADER_TYPES",
    "SUPPORTED_DISTRIBUTIONS",
    "MODEL_CHANNEL_COUNTS",
    "VIDEO_MODEL_CHANNELS",
    # Blending
    "blend_noises",
    # Transforms
    "apply_noise_transform",
    "normalize_noise",
    "resize_noise_spatial",
    "resize_noise_channels",
    "match_noise_shape",
    # Model compatibility
    "get_model_channel_count",
    "detect_latent_format",
    "build_noise_shape",
    # Sampler
    "calculate_stage_strengths",
    "calculate_step_ranges",
    "calculate_step_points",
    "generate_shader_noise",
    # Debug
    "StubVisualizer",
    "StubDebugger",
    "get_visualizer",
    "get_debugger",
    "set_debug_level",
    # Logging
    "setup_logging",
    "get_logger",
    "set_debug_logging",
    "set_info_logging",
    "set_warning_logging",
    "set_quiet_logging",
]
