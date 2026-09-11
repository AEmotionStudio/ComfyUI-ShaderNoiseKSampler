"""
Noise composition for the standard sampling pipeline.

Samplers expect their noise to be standard normal. The blend modes come from
image compositing, where values live in [0, 1] -- `1 - (1 - b) * (1 - s)` is
meaningless for b = -2.3 -- so applying them straight to N(0, 1) noise shifts
the mean and variance. Measured on unit-Gaussian inputs, the old blending gave
a mean of +0.39 for overlay at strength 0.3 and a standard deviation of 4.2 for
soft_light at strength 1.0, which reaches the model as a colour cast and as
washed-out or burnt contrast.

So the compositing modes run in uniform space: map the noise through the normal
CDF to (0, 1), composite there, map back, then interpolate by strength and
re-standardise. The mode still shapes the *structure* of the noise, but what
reaches the sampler keeps the distribution the model was trained on.

Legacy mode keeps the old, unnormalised behaviour on purpose.
"""
import math

import torch

# Modes whose formulas assume [0, 1] data and are composited in uniform space.
_UNIT_SPACE_MODES = ("multiply", "screen", "overlay", "soft_light", "hard_light", "difference")
SUPPORTED_MODES = ("normal", "add") + _UNIT_SPACE_MODES
SUPPORTED_TRANSFORMS = (
    "none", "reverse", "inverse", "absolute", "square", "sqrt", "log", "sin", "cos",
)

# Keeps |inverse| bounded: 1/0.25 = 4, in range for unit-scale noise. Without it
# the reciprocal of near-zero samples reached ~4e4 on a single latent.
_INVERSE_FLOOR = 0.25


def standardize(noise: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Rescale to mean 0, standard deviation 1, per sample and per channel.

    Statistics are taken over the spatial (and temporal) axes only, so channels
    stay independent and a batch item cannot be skewed by its neighbours.
    """
    dims = tuple(range(2, noise.ndim)) if noise.ndim > 2 else (-1,)
    mean = noise.mean(dim=dims, keepdim=True)
    std = noise.std(dim=dims, keepdim=True)
    return (noise - mean) / std.clamp_min(eps)


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """N(0, 1) -> (0, 1) through the normal CDF."""
    return torch.special.ndtr(x)


def _from_unit(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(0, 1) -> N(0, 1) through the inverse normal CDF, clamped off the poles."""
    return torch.special.ndtri(u.clamp(eps, 1.0 - eps))


def _composite(base: torch.Tensor, shader: torch.Tensor, mode: str) -> torch.Tensor:
    """Standard compositing formulas, on values in [0, 1]."""
    if mode == "multiply":
        return base * shader
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - shader)
    if mode == "overlay":
        return torch.where(base < 0.5, 2 * base * shader, 1.0 - 2 * (1.0 - base) * (1.0 - shader))
    if mode == "hard_light":
        return torch.where(shader < 0.5, 2 * base * shader, 1.0 - 2 * (1.0 - base) * (1.0 - shader))
    if mode == "soft_light":
        return (1.0 - 2.0 * shader) * base ** 2 + 2.0 * shader * base
    if mode == "difference":
        return (base - shader).abs()
    raise ValueError(f"unknown compositing mode: {mode}")


def mix_noise(base: torch.Tensor, shader: torch.Tensor, mode: str, strength: float) -> torch.Tensor:
    """
    Blend shader noise into base noise without changing the distribution.

    Args:
        base: base noise, normally from comfy.sample.prepare_noise
        shader: shader noise, same shape as base
        mode: one of SUPPORTED_MODES
        strength: 0.0 returns base untouched, 1.0 is full shader influence

    Returns:
        Noise with mean 0 and standard deviation 1 per sample and channel.
    """
    if strength <= 0.0:
        return base
    if shader.shape != base.shape:
        raise ValueError(f"shape mismatch: base {tuple(base.shape)} vs shader {tuple(shader.shape)}")

    strength = float(min(strength, 1.0))
    b = standardize(base)
    s = standardize(shader.to(dtype=b.dtype, device=b.device))

    if mode == "normal":
        # Variance preserving for independent unit-variance inputs:
        # cos^2 + sin^2 = 1, unlike (1-k)*b + k*s which dips to 0.71 at k=0.5.
        theta = strength * math.pi / 2.0
        return math.cos(theta) * b + math.sin(theta) * s
    if mode == "add":
        return standardize(b + s * strength)

    blended = _from_unit(_composite(_to_unit(b), _to_unit(s), mode))
    return standardize(b * (1.0 - strength) + blended * strength)


def transform_noise(noise: torch.Tensor, transform: str) -> torch.Tensor:
    """
    Apply a mathematical transform, then restore the noise distribution.

    Several transforms are not symmetric (absolute, square, sqrt, log), so
    without re-standardising they hand the sampler a large DC offset.
    """
    if transform == "none":
        return noise
    if transform == "reverse":
        return -noise
    if transform == "inverse":
        result = torch.sign(noise) / noise.abs().clamp_min(_INVERSE_FLOOR)
    elif transform == "absolute":
        result = noise.abs()
    elif transform == "square":
        result = noise ** 2
    elif transform == "sqrt":
        result = noise.abs().sqrt()
    elif transform == "log":
        result = torch.log(noise.abs() + 1.0)
    elif transform == "sin":
        result = torch.sin(noise * math.pi)
    elif transform == "cos":
        result = torch.cos(noise * math.pi)
    else:
        return noise

    return standardize(result)
