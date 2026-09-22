"""
Shared utilities for ComfyUI-ShaderNoiseKSampler.

This package provides common functionality used across shader generators:
- color_utils: Color scheme interpolation and application
- shape_masks: Shape mask generation and application
- noise_utils: the coordinate grid every generator starts from
"""

from .color_utils import (
    COLOR_SCHEMES,
    lerp,
    hsv_to_rgb,
    interpolate_colors,
    apply_color_scheme,
)

from .shape_masks import (
    SHAPE_TYPES,
    smoothstep,
    apply_shape_mask,
)

from .noise_utils import create_coordinate_grid

__all__ = [
    # Color utilities
    "COLOR_SCHEMES",
    "lerp",
    "hsv_to_rgb",
    "interpolate_colors",
    "apply_color_scheme",
    # Shape masks
    "SHAPE_TYPES",
    "smoothstep",
    "apply_shape_mask",
    # Noise utilities
    "create_coordinate_grid",
]
