"""
Every registered generator must return [B, C, H, W] for any parameter combination.

Regression test for a tensor_field crash: the eigenvalue-difference branch
(viz_type == 1, reached whenever `(int(octaves) + channel_index) % 4 == 1`)
called `.unsqueeze(-1)` on an already-4D tensor, so `permute(0, 3, 1, 2)` raised
`RuntimeError`. tensor_field failed for 46 of 48 parameter combinations,
including the node's own defaults (octaves=1).
"""
import itertools

import pytest
import torch

SHADER_TYPES = ["domain_warp", "tensor_field", "curl_noise", "temporal_coherent", "spectral",
                "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"]


def make_params(octaves=1, channels=4, color_scheme="none", shape_type="none", temporal=False):
    from snk.core.params import ShaderParams

    return ShaderParams({
        "scale": 1.0, "octaves": octaves, "warp_strength": 0.5, "phase_shift": 0.5,
        "shape_type": shape_type, "shape_strength": 1.0,
        "color_scheme": color_scheme, "color_intensity": 0.8,
        "time": 0.0, "base_seed": 8888, "useTemporalCoherence": temporal,
        "target_channels": channels,
    }).validate()


def generate(shader_type, params, channels, size=16):
    from snk.shader_noise_ksampler import get_shader_generator

    return get_shader_generator(shader_type)(
        params=params, height=size, width=size, batch_size=1,
        device="cpu", seed=8888, target_channels=channels,
    )


@pytest.mark.parametrize("shader_type", SHADER_TYPES)
@pytest.mark.parametrize("octaves", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("channels", [4, 16])
def test_generator_returns_bchw(shader_type, octaves, channels):
    noise = generate(shader_type, make_params(octaves, channels), channels)
    assert tuple(noise.shape) == (1, channels, 16, 16)
    assert torch.isfinite(noise).all(), "generator produced NaN or Inf"


@pytest.mark.parametrize("shader_type", SHADER_TYPES)
@pytest.mark.parametrize(
    "color_scheme,shape_type",
    list(itertools.product(["none", "viridis"], ["none", "radial"])),
)
def test_generator_handles_masks_and_colors(shader_type, color_scheme, shape_type):
    params = make_params(channels=16, color_scheme=color_scheme, shape_type=shape_type)
    noise = generate(shader_type, params, 16)
    assert tuple(noise.shape) == (1, 16, 16, 16)
    assert torch.isfinite(noise).all(), "generator produced NaN or Inf"
