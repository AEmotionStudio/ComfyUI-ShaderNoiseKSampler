"""
The standard pipeline must hand the sampler noise it was trained on.

Measured on the pre-2.0 blending, with unit-Gaussian inputs: overlay had a mean
of +0.39 at strength 0.3, soft_light a standard deviation of 4.2 at strength
1.0, and the inverse transform reached ~4e4. Those are colour casts and blown
contrast at the model's input.
"""
import pytest
import torch

from snk.core.noise_math import (
    SUPPORTED_MODES,
    SUPPORTED_TRANSFORMS,
    mix_noise,
    standardize,
    transform_noise,
)

# Transforms that are exact pass-throughs: already zero-mean and symmetric.
PASSTHROUGH_TRANSFORMS = ("none", "reverse")


@pytest.fixture
def noise_pair():
    generator = torch.Generator().manual_seed(0)
    shape = (2, 4, 64, 64)
    return (torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
@pytest.mark.parametrize("strength", [0.1, 0.3, 0.7, 1.0])
def test_mix_keeps_the_noise_distribution(noise_pair, mode, strength):
    base, shader = noise_pair
    mixed = mix_noise(base, shader, mode, strength)

    assert mixed.shape == base.shape
    assert torch.isfinite(mixed).all()
    assert mixed.mean().abs() < 0.02, f"{mode}@{strength} shifted the mean"
    assert abs(mixed.std().item() - 1.0) < 0.05, f"{mode}@{strength} changed the variance"


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_zero_strength_returns_base_untouched(noise_pair, mode):
    base, shader = noise_pair
    assert torch.equal(mix_noise(base, shader, mode, 0.0), base)


def test_mix_actually_changes_the_noise(noise_pair):
    """A blend must do something: same distribution, different structure."""
    base, shader = noise_pair
    for mode in SUPPORTED_MODES:
        mixed = mix_noise(base, shader, mode, 0.5)
        assert not torch.allclose(mixed, base, atol=1e-3), f"{mode} had no effect"


def test_normal_mode_is_variance_preserving(noise_pair):
    """cos/sin mixing, not (1-k)*base + k*shader, which dips to std 0.71 at k=0.5."""
    base, shader = noise_pair
    mixed = mix_noise(base, shader, "normal", 0.5)
    assert abs(mixed.std().item() - 1.0) < 0.05

    naive = base * 0.5 + shader * 0.5
    assert naive.std().item() < 0.8  # what the old code handed the sampler


def test_mix_rejects_mismatched_shapes(noise_pair):
    base, _ = noise_pair
    with pytest.raises(ValueError):
        mix_noise(base, torch.randn(2, 4, 32, 32), "normal", 0.5)


@pytest.mark.parametrize("transform", SUPPORTED_TRANSFORMS)
def test_transforms_stay_in_distribution(transform):
    noise = torch.randn(2, 4, 64, 64, generator=torch.Generator().manual_seed(1))
    result = transform_noise(noise, transform)

    assert result.shape == noise.shape
    assert torch.isfinite(result).all()
    if transform in PASSTHROUGH_TRANSFORMS:
        assert torch.equal(result.abs(), noise.abs())
        return
    assert result.mean().abs() < 0.02, f"{transform} left a DC offset"
    assert abs(result.std().item() - 1.0) < 0.05, f"{transform} changed the variance"
    assert result.abs().max() < 50, f"{transform} produced extreme outliers"


def test_inverse_transform_is_bounded():
    """1/x on near-zero samples reached ~4e4 before; it is clamped now."""
    noise = torch.tensor([[[[1e-9, -1e-9, 0.5, -2.0]]]])
    assert transform_noise(noise, "inverse").abs().max() < 50


def test_standardize_is_per_sample_and_channel():
    noise = torch.randn(3, 4, 16, 16, generator=torch.Generator().manual_seed(2))
    noise[0, 0] = noise[0, 0] * 10 + 5  # one badly scaled channel

    result = standardize(noise)
    per_channel_mean = result.mean(dim=(2, 3))
    per_channel_std = result.std(dim=(2, 3))
    assert per_channel_mean.abs().max() < 1e-4
    assert (per_channel_std - 1.0).abs().max() < 1e-3


def test_video_latents_are_supported():
    video = torch.randn(1, 16, 5, 8, 8, generator=torch.Generator().manual_seed(3))
    mixed = mix_noise(video, torch.randn_like(video), "soft_light", 0.5)
    assert mixed.shape == video.shape
    assert abs(mixed.std().item() - 1.0) < 0.05
