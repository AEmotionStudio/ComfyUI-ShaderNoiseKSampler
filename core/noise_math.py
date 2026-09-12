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
CDF to (0, 1), composite there, map back, interpolate by strength, and give the
result the base noise's own mean and deviation. The mode still shapes the
*structure* of the noise, but what reaches the sampler keeps the distribution the
model was trained on.

Legacy mode keeps the old, unnormalised behaviour on purpose.
"""
import math

import torch

# Modes whose formulas assume [0, 1] data and are composited in uniform space.
_UNIT_SPACE_MODES = ("multiply", "screen", "overlay", "soft_light", "hard_light", "difference")
SUPPORTED_MODES = ("normal", "add") + _UNIT_SPACE_MODES

# How much shader the sampler actually receives, per mode, at strength 0.0 to 1.0
# in steps of 0.05. Measured as the cosine between the mixed noise and the shader
# noise: mix_noise standardises both operands, so the result lies on the unit
# sphere they span and that cosine is exactly the shader's share of it.
#
# The modes disagree wildly. At a strength of 0.5, `normal` hands over 0.71 of
# the shader and `difference` 0.03 -- a factor of twenty-three for the same
# number on the same slider. That is why one strength value cannot be carried
# from one mode to another, and it lines up with what the modes do to a picture:
# on MiniMax H3, `add` breaks down around 0.15 and `soft_light` is still clean at
# 0.3.
#
# Regenerate with tests/test_noise_math.py::test_blend_calibration_is_current,
# which re-measures this and fails on drift. The curves vary by at most 0.07
# across shader types, shape masks and latent ranks, so one table serves all.
BLEND_SHADER_FRACTION = {
    "normal": (0.0000, 0.0780, 0.1560, 0.2330, 0.3086, 0.3823, 0.4537, 0.5222, 0.5875, 0.6492, 0.7069, 0.7603, 0.8089, 0.8526, 0.8910, 0.9239, 0.9510, 0.9724, 0.9877, 0.9969, 1.0000),
    "add": (0.0000, 0.0495, 0.0990, 0.1479, 0.1957, 0.2421, 0.2869, 0.3299, 0.3710, 0.4100, 0.4469, 0.4816, 0.5142, 0.5447, 0.5732, 0.5997, 0.6245, 0.6474, 0.6688, 0.6886, 0.7069),
    "multiply": (0.0000, 0.0284, 0.0584, 0.0895, 0.1216, 0.1546, 0.1885, 0.2232, 0.2585, 0.2941, 0.3300, 0.3660, 0.4018, 0.4372, 0.4719, 0.5059, 0.5388, 0.5705, 0.6009, 0.6298, 0.6571),
    "screen": (0.0000, 0.0321, 0.0660, 0.1010, 0.1370, 0.1740, 0.2118, 0.2502, 0.2889, 0.3279, 0.3668, 0.4054, 0.4434, 0.4806, 0.5167, 0.5516, 0.5850, 0.6168, 0.6469, 0.6752, 0.7016),
    "overlay": (0.0000, 0.0295, 0.0591, 0.0883, 0.1169, 0.1450, 0.1725, 0.1992, 0.2252, 0.2505, 0.2750, 0.2987, 0.3215, 0.3436, 0.3648, 0.3852, 0.4048, 0.4237, 0.4417, 0.4591, 0.4756),
    "soft_light": (0.0000, 0.0182, 0.0367, 0.0551, 0.0733, 0.0914, 0.1092, 0.1269, 0.1443, 0.1615, 0.1784, 0.1951, 0.2115, 0.2276, 0.2435, 0.2590, 0.2743, 0.2893, 0.3039, 0.3183, 0.3324),
    "hard_light": (0.0000, 0.0555, 0.1134, 0.1726, 0.2325, 0.2925, 0.3518, 0.4095, 0.4652, 0.5181, 0.5678, 0.6139, 0.6563, 0.6950, 0.7299, 0.7612, 0.7892, 0.8139, 0.8358, 0.8551, 0.8720),
    "difference": (0.0000, 0.0016, 0.0040, 0.0065, 0.0093, 0.0122, 0.0154, 0.0188, 0.0224, 0.0260, 0.0295, 0.0330, 0.0362, 0.0391, 0.0416, 0.0437, 0.0453, 0.0466, 0.0475, 0.0480, 0.0484),
}

# The mode the normalised scale is expressed in, so a workflow that never leaves
# the default sees the same numbers it always did.
CALIBRATION_REFERENCE = "multiply"


def _interpolate(curve, position: float) -> float:
    """Read a value off a 21-point 0..1 curve."""
    if position <= 0.0:
        return curve[0]
    if position >= 1.0:
        return curve[-1]
    index = position * (len(curve) - 1)
    low = int(index)
    return curve[low] + (curve[low + 1] - curve[low]) * (index - low)


def _invert(curve, value: float) -> float:
    """Smallest strength on a monotone curve reaching `value`, 1.0 if unreachable."""
    if value <= curve[0]:
        return 0.0
    for i in range(1, len(curve)):
        if curve[i] >= value:
            span = curve[i] - curve[i - 1]
            frac = 1.0 if span <= 0 else (value - curve[i - 1]) / span
            return (i - 1 + frac) / (len(curve) - 1)
    return 1.0


def normalized_strength(mode: str, strength: float) -> float:
    """
    Re-express `strength` so it hands the sampler the same amount of shader
    whichever blend mode is chosen.

    The scale is the reference mode's, so at the default `multiply` this returns
    `strength` unchanged. A mode that cannot reach the requested share of shader
    saturates at 1.0 rather than reporting an error -- `difference` tops out at
    0.048, well under the reference's 0.657, which is why it stays subtle at any
    setting.
    """
    if mode == CALIBRATION_REFERENCE or mode not in BLEND_SHADER_FRACTION:
        return strength
    target = _interpolate(BLEND_SHADER_FRACTION[CALIBRATION_REFERENCE], strength)
    return _invert(BLEND_SHADER_FRACTION[mode], target)
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


def _match_statistics(result: torch.Tensor, reference: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Give `result` the per-sample, per-channel mean and deviation of `reference`.

    A reference channel with no spread keeps unit statistics instead, so a flat
    input cannot erase the blend.
    """
    dims = tuple(range(2, reference.ndim)) if reference.ndim > 2 else (-1,)
    mean = reference.mean(dim=dims, keepdim=True)
    std = reference.std(dim=dims, keepdim=True)
    flat = std <= eps
    mean = torch.where(flat, torch.zeros_like(mean), mean)
    std = torch.where(flat, torch.ones_like(std), std)
    return standardize(result, eps) * std + mean


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


def mix_noise(base: torch.Tensor, shader: torch.Tensor, mode: str, strength: float,
              normalize: bool = False) -> torch.Tensor:
    """
    Blend shader noise into base noise without changing the distribution.

    Args:
        base: base noise, normally from comfy.sample.prepare_noise
        shader: shader noise, same shape as base
        mode: one of SUPPORTED_MODES
        strength: 0.0 returns base untouched, 1.0 is full shader influence
        normalize: re-express `strength` on the reference mode's scale, so the
            same value hands over the same amount of shader in every mode

    Returns:
        Noise with the per-sample, per-channel mean and standard deviation of
        `base`.

    The result takes the base noise's own statistics rather than exactly 0 and 1,
    so it is continuous with strength: strength 0 returns base untouched, and as
    strength approaches 0 the result approaches base. Forcing 0 and 1 moved the
    noise by the base's own sampling offsets at any strength above 0, which on
    SD 1.5 moved the image as far as a whole 0.05 step of shader.
    """
    if strength <= 0.0:
        return base
    if shader.shape != base.shape:
        raise ValueError(f"shape mismatch: base {tuple(base.shape)} vs shader {tuple(shader.shape)}")

    strength = float(min(strength, 1.0))
    if normalize:
        strength = normalized_strength(mode, strength)
        if strength <= 0.0:
            return base
    b = standardize(base)
    s = standardize(shader.to(dtype=b.dtype, device=b.device))

    if mode == "normal":
        # Variance preserving for independent unit-variance inputs:
        # cos^2 + sin^2 = 1, unlike (1-k)*b + k*s which dips to 0.71 at k=0.5.
        theta = strength * math.pi / 2.0
        return _match_statistics(math.cos(theta) * b + math.sin(theta) * s, base)
    if mode == "add":
        return _match_statistics(b + s * strength, base)

    blended = _from_unit(_composite(_to_unit(b), _to_unit(s), mode))
    return _match_statistics(b * (1.0 - strength) + blended * strength, base)


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
