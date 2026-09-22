"""
Shader noise generators for ComfyUI-ShaderNoiseKSampler.

One module per shader type, each registered under its name by the
@shader_generator decorator when the module is imported: domain_warp,
tensor_field, curl_noise, temporal_coherent, spectral, gaussian, fractal,
perlin, heterogeneous_fbm, interference, projection_3d, cellular and waves.
simplex.py holds the hashes and simplex primitives they share, fbm.py the
draw-and-fill skeleton of the scalar-field types.
"""

from .base import BaseNoiseGenerator
from .registry import (
    ShaderRegistry,
    register_shader,
    get_shader,
    list_shaders,
    shader_generator,
)

__all__ = [
    # Base class
    "BaseNoiseGenerator",
    # Registry
    "ShaderRegistry",
    "register_shader",
    "get_shader",
    "list_shaders",
    "shader_generator",
]
