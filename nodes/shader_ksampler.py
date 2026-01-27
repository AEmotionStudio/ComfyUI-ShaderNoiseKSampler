"""
Shader Noise KSampler node for ComfyUI.

This module re-exports the ShaderNoiseKSampler class from the main module
for backward compatibility while the codebase is being refactored.
"""

# Import from the original location for now
# This will be refactored to use the new core modules in a future update
from ..shader_noise_ksampler import ShaderNoiseKSampler

__all__ = ["ShaderNoiseKSampler"]
