"""
The shared simplex primitive, and exactly how far batching it stays exact.

shaders/simplex.py exists so the channel axis can be rendered in one call instead
of one call per channel. That is only worth doing if a batched draw equals the
draws it replaces, so this pins where it does and where it does not.
"""
import math

import pytest
import torch

from snk.shaders.simplex import simplex_2d

# Per-slice element counts that are and are not kind to the vectoriser. The
# aligned ones are the shapes real latents actually have: SD 1.5 at 512x512 is
# 64x64, H3 at 608x352 is 22x38, H3 at 1344x768 is 48x84.
ALIGNED = [(64, 64), (22, 38), (48, 84), (16, 16), (22, 40)]
RAGGED = [(22, 39), (37, 53), (23, 38), (17, 22)]


def _seeds(n):
    return [8888 + 6151 * c for c in range(n)]


def _scalar(p, seeds, rotate):
    return torch.stack([simplex_2d(p, s, rotate=rotate) for s in seeds])


def _batched(p, seeds, rotate):
    seed = torch.tensor(seeds, dtype=torch.int64).reshape(len(seeds), 1, 1, 1, 1)
    return simplex_2d(p, seed, rotate=rotate)


@pytest.mark.parametrize("hw", ALIGNED + RAGGED)
def test_batching_is_exact_without_the_rotation(hw):
    """
    curl_noise and tensor_field do not rotate their coordinates, so the shared
    coordinate tensor is never materialised per slice and a batched draw is the
    same arithmetic on the same memory. Exact at every shape.
    """
    torch.manual_seed(0)
    p = torch.randn(1, *hw, 2) * 3.0
    seeds = _seeds(24)
    assert torch.equal(_scalar(p, seeds, False), _batched(p, seeds, False))


@pytest.mark.parametrize("hw", ALIGNED)
def test_batching_is_exact_with_the_rotation_at_the_shapes_latents_have(hw):
    torch.manual_seed(0)
    p = torch.randn(1, *hw, 2) * 3.0
    seeds = _seeds(24)
    assert torch.equal(_scalar(p, seeds, True), _batched(p, seeds, True))


@pytest.mark.parametrize("hw", RAGGED)
def test_the_rotation_costs_an_ulp_at_ragged_shapes(hw):
    """
    domain_warp turns its coordinates by an angle drawn from the seed, so the
    coordinates genuinely differ per slice and the batched draw has to materialise
    them. That makes every downstream elementwise op run over N times as many
    elements, and where the per-slice count does not divide the vector width the
    tail is handled differently -- a one-ulp difference, 3e-08 to 1.2e-07.

    This is a real limit on batching domain_warp, not a bug to fix: it is float32
    rounding, invisible in an image, but it is not `torch.equal`, so batching that
    generator moves the golden fixtures. The non-rotating generators do not have
    this problem. Asserted rather than merely documented so that a future change
    which makes it worse is caught.
    """
    torch.manual_seed(0)
    p = torch.randn(1, *hw, 2) * 3.0
    seeds = _seeds(24)
    scalar, batched = _scalar(p, seeds, True), _batched(p, seeds, True)
    assert not torch.equal(scalar, batched), "exact now -- tighten this test"
    assert (scalar - batched).abs().max() < 2e-07


def test_a_tensor_seed_renders_the_same_channels_a_loop_would():
    """The seed axis is the channel axis: slice c must equal the draw at seed c."""
    torch.manual_seed(0)
    p = torch.randn(1, 22, 38, 2) * 3.0
    seeds = _seeds(8)
    batched = _batched(p, seeds, False)
    for index, seed in enumerate(seeds):
        assert torch.equal(batched[index], simplex_2d(p, seed, rotate=False))


def test_the_seed_is_coerced_to_int64():
    """
    `h*h*h` overflows int64 for realistic coordinates and the noise depends on how
    it wraps, so the hash is only the intended function when the seed arrives as
    int64. A caller handing in a float tensor must get the same draw as the int,
    not a quietly different one.
    """
    torch.manual_seed(0)
    p = torch.randn(1, 16, 16, 2) * 3.0
    seeds = [8888, 12345]
    as_int = simplex_2d(p, torch.tensor(seeds, dtype=torch.int64).reshape(2, 1, 1, 1, 1), False)
    as_float = simplex_2d(p, torch.tensor(seeds, dtype=torch.float32).reshape(2, 1, 1, 1, 1), False)
    assert torch.equal(as_int, as_float)
    assert torch.equal(as_int[0], simplex_2d(p, 8888, rotate=False))


@pytest.mark.parametrize("shader_type", ["domain_warp", "curl_noise",
                                         "temporal_coherent", "tensor_field", "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"])
@pytest.mark.parametrize("hw", RAGGED + ALIGNED)
def test_channel_zero_survives_batching_at_every_shape(shader_type, hw):
    """
    Channel 0 of a wide draw must be the one-channel draw at the same seed, and
    the existing test for that only checks vector-aligned shapes.

    That matters because batching materialises the channel axis, and a draw over
    N times as many elements vectorises its tail differently. tensor_field batched
    channel 0 along with the rest and broke this at 22x38 by 6e-07 -- small, but it
    is the identity every travel-mode basis rests on. The fix was to keep channel 0
    on the scalar path, which is what the other three already did by virtue of
    handing it to fill_channels as `base`.
    """
    from snk.core.shader_noise import generate

    params = {
        "scale": 1.0, "octaves": 3.0, "warp_strength": 0.5, "phase_shift": 0.5,
        "shape_type": "none", "color_scheme": "none", "color_intensity": 0.8,
        "shape_mask_strength": 1.0, "time": 0.0, "base_seed": 8888,
        "use_temporal_coherence": False,
    }
    wide = generate((1, 24) + hw, params, shader_type, 8888, torch.device("cpu"))
    single = generate((1, 1) + hw, params, shader_type, 8888, torch.device("cpu"))
    assert torch.equal(wide[:, :1], single)
