"""
The standard pipeline must hand the sampler noise it was trained on.

Measured on the pre-2.0 blending, with unit-Gaussian inputs: overlay had a mean
of +0.39 at strength 0.3, soft_light a standard deviation of 4.2 at strength
1.0, and the inverse transform reached ~4e4. Those are colour casts and blown
contrast at the model's input.
"""
import pytest
import torch

from snk.core import noise_math
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


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_the_blend_is_continuous_from_zero_strength(noise_pair, mode):
    """
    Strength 0 returns base untouched, so a strength just above 0 must return almost
    exactly base too. Rescaling to mean 0 and deviation 1 instead moved the noise by
    the base's own sampling offsets at any strength above 0 -- on SD 1.5, as far as
    a whole 0.05 step of shader.
    """
    base, shader = noise_pair
    assert (mix_noise(base, shader, mode, 1e-6) - base).abs().max() < 1e-3


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_the_blend_keeps_the_base_noise_statistics(mode):
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(2, 4, 32, 32, generator=generator) * 1.2 + 0.1
    shader = torch.randn(2, 4, 32, 32, generator=generator)
    mixed = mix_noise(base, shader, mode, 0.5)
    assert torch.allclose(mixed.mean(dim=(2, 3)), base.mean(dim=(2, 3)), atol=1e-4)
    assert torch.allclose(mixed.std(dim=(2, 3)), base.std(dim=(2, 3)), atol=1e-4)


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


# --- blend-mode strength calibration -------------------------------------------------

def _shader_fraction(mixed, shader):
    """Cosine between the mixed noise and the shader: the shader's share of the result."""
    a = mixed.flatten().double(); b = shader.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _fixture(shape=(1, 8, 48, 48), shader_type="domain_warp", shape_type="none"):
    from snk.core import shader_noise
    params = {"scale": 1.0, "octaves": 2.0, "warp_strength": 0.7, "phase_shift": 0.5,
              "time": 0.0, "shape_type": shape_type, "color_scheme": "none", "intensity": 0.8}
    torch.manual_seed(0)
    base = torch.randn(shape)
    shader = noise_math.standardize(
        shader_noise.generate(shape, params, shader_type, 8888, torch.device("cpu")))
    return base, shader


def test_blend_modes_disagree_wildly_without_normalisation():
    """The reason the calibration exists: one number, twenty-plus different meanings."""
    base, shader = _fixture()
    fractions = {m: _shader_fraction(noise_math.mix_noise(base, shader, m, 0.5), shader)
                 for m in noise_math.SUPPORTED_MODES}

    assert fractions["normal"] > 0.6, fractions
    assert fractions["difference"] < 0.1, fractions
    assert fractions["normal"] / fractions["difference"] > 10, fractions


def test_normalisation_makes_one_strength_mean_one_thing():
    base, shader = _fixture()
    target = _shader_fraction(
        noise_math.mix_noise(base, shader, noise_math.CALIBRATION_REFERENCE, 0.3), shader)

    for mode in noise_math.SUPPORTED_MODES:
        got = _shader_fraction(
            noise_math.mix_noise(base, shader, mode, 0.3, normalize=True), shader)
        if mode == "difference":
            continue  # cannot reach the reference scale at all; saturates instead
        assert abs(got - target) < 0.05, f"{mode}: {got:.3f} vs target {target:.3f}"


def test_the_reference_mode_is_left_alone():
    """Workflows that never change blend_mode must reproduce exactly."""
    base, shader = _fixture()
    for k in (0.1, 0.3, 0.75):
        plain = noise_math.mix_noise(base, shader, noise_math.CALIBRATION_REFERENCE, k)
        normed = noise_math.mix_noise(base, shader, noise_math.CALIBRATION_REFERENCE, k, normalize=True)
        assert torch.equal(plain, normed)


def test_normalisation_is_off_by_default():
    base, shader = _fixture()
    for mode in ("add", "soft_light", "normal"):
        assert torch.equal(noise_math.mix_noise(base, shader, mode, 0.4),
                           noise_math.mix_noise(base, shader, mode, 0.4, normalize=False))


def test_a_saturating_mode_clamps_instead_of_failing():
    """difference tops out far below the reference; it must saturate, not raise."""
    base, shader = _fixture()
    assert noise_math.normalized_strength("difference", 0.9) == 1.0
    out = noise_math.mix_noise(base, shader, "difference", 0.9, normalize=True)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("shader_type,shape_type,shape", [
    ("domain_warp", "none", (1, 8, 48, 48)),
    ("tensor_field", "none", (1, 16, 32, 32)),
    ("curl_noise", "none", (1, 4, 40, 40)),
    ("domain_warp", "spiral", (1, 8, 48, 48)),
    ("domain_warp", "none", (1, 12, 5, 24, 24)),
])
def test_blend_calibration_is_current(shader_type, shape_type, shape):
    """
    Re-measure the table and fail on drift.

    One static table serves every shader type, shape mask and latent rank because
    mix_noise standardises its operands first, so the geometry barely moves. If a
    blend formula changes, this catches it rather than letting the calibration
    quietly lie.
    """
    base, shader = _fixture(shape, shader_type, shape_type)
    for mode, curve in noise_math.BLEND_SHADER_FRACTION.items():
        for index in (4, 10, 16):           # strengths 0.2, 0.5, 0.8
            k = index / (len(curve) - 1)
            got = _shader_fraction(noise_math.mix_noise(base, shader, mode, k), shader)
            if mode == "difference":
                # Its whole curve tops out below 0.05, under the tolerance itself,
                # so what it hands over is dominated by the shader's own geometry
                # and an absolute comparison cannot test it. What matters is that
                # it stays a small share of the shader at any strength.
                assert got < 0.2, f"difference at {k:.2f}: measured {got:.3f}"
                continue
            assert abs(got - curve[index]) < 0.08, (
                f"{mode} at {k:.2f}: measured {got:.3f}, table says {curve[index]:.3f}")
