"""
Shader noise generators for ComfyUI-ShaderNoiseKSampler.

This package provides various noise generation algorithms:
- domain_warp: Domain warping noise
- tensor_field: Tensor field noise
- curl_noise: Curl/fluid noise
- temporal_coherent: Temporally coherent noise for animations
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
