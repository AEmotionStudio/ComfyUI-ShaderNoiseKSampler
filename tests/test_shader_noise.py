"""
Shader noise must match the latent's own layout.

ComfyUI latents are [B, C, H, W] and [B, C, T, H, W]. Legacy guessed between
[B,C,F,H,W] and [B,F,C,H,W] and defaulted to the latter when both dimensions
matched the channel count -- which is exactly what a 61-frame Wan or Hunyuan
clip looks like: (61-1)//4+1 = 16 latent frames and 16 channels. Time evolution
then ran across channels, with no error raised.
"""
import pytest
import torch

from snk.core.shader_noise import generate, latent_layout

CPU = torch.device("cpu")
PARAMS = {
    "shader_type": "domain_warp", "scale": 1.0, "octaves": 1.0, "warp_strength": 0.5,
    "phase_shift": 0.5, "shape_type": "none", "color_scheme": "none", "time": 0.0,
    "base_seed": 8888,
}


def test_layout_reads_comfy_latent_shapes():
    assert latent_layout((2, 4, 32, 32)) == {"batch": 2, "channels": 4, "frames": 1, "height": 32, "width": 32}
    assert latent_layout((1, 16, 5, 8, 8)) == {"batch": 1, "channels": 16, "frames": 5, "height": 8, "width": 8}


@pytest.mark.parametrize("shape", [(1, 3), (1, 4, 8, 8, 8, 8)])
def test_layout_rejects_unsupported_ranks(shape):
    with pytest.raises(ValueError):
        latent_layout(shape)


@pytest.mark.parametrize("shape", [(1, 4, 32, 32), (2, 4, 16, 16), (1, 16, 5, 8, 8), (1, 16, 3, 16, 16)])
def test_generated_noise_matches_the_latent_shape(shape):
    noise = generate(shape, PARAMS, "domain_warp", 8888, CPU)
    assert tuple(noise.shape) == shape
    assert torch.isfinite(noise).all()


def test_video_frames_vary_along_the_time_axis():
    """The 61-frame Wan case: 16 latent frames and 16 channels."""
    noise = generate((1, 16, 16, 8, 8), PARAMS, "domain_warp", 8888, CPU)

    assert tuple(noise.shape) == (1, 16, 16, 8, 8)
    frame_delta = (noise[:, :, 1:] - noise[:, :, :-1]).abs().mean()
    assert frame_delta > 1e-6, "frames are identical: time is not being applied on dim 2"


def test_generation_leaves_the_callers_rng_alone():
    """Generators call torch.manual_seed internally; fork_rng contains that."""
    torch.manual_seed(1234)
    expected = torch.rand(3)

    torch.manual_seed(1234)
    generate((1, 4, 16, 16), PARAMS, "domain_warp", 999, CPU)
    assert torch.equal(torch.rand(3), expected)


def test_fractional_octaves_interpolate():
    """ShaderParams.validate() truncates octaves to int, so 1.5 used to equal 1.0."""
    low = generate((1, 4, 16, 16), {**PARAMS, "octaves": 1.0}, "domain_warp", 8888, CPU)
    mid = generate((1, 4, 16, 16), {**PARAMS, "octaves": 1.5}, "domain_warp", 8888, CPU)
    high = generate((1, 4, 16, 16), {**PARAMS, "octaves": 2.0}, "domain_warp", 8888, CPU)

    assert not torch.allclose(mid, low)
    assert not torch.allclose(mid, high)
    # Halfway between the neighbouring integer renders.
    assert torch.allclose(mid, torch.lerp(low, high, 0.5), atol=1e-5)


def test_temporal_coherence_holds_the_seed():
    without = generate((1, 16, 4, 8, 8), PARAMS, "domain_warp", 8888, CPU, temporal_coherence=False)
    with_tc = generate((1, 16, 4, 8, 8), PARAMS, "domain_warp", 8888, CPU, temporal_coherence=True)

    assert tuple(with_tc.shape) == (1, 16, 4, 8, 8)
    assert not torch.allclose(without, with_tc)


@pytest.mark.parametrize("shader_type", ["domain_warp", "tensor_field", "curl_noise", "temporal_coherent"])
def test_every_registered_generator_is_usable(shader_type):
    noise = generate((1, 4, 16, 16), {**PARAMS, "shader_type": shader_type}, shader_type, 8888, CPU)
    assert tuple(noise.shape) == (1, 4, 16, 16)
    assert torch.isfinite(noise).all()


def test_unknown_shader_type_raises_a_clear_error():
    """
    Better than silently generating unrelated noise. The legacy fallback routed
    to shader_params_reader.generate_noise_tensor, which raises for everything
    it is given -- including perlin, cellular and waves, archetypes named in the
    parameter vocabulary but not shipped in this repository.
    """
    with pytest.raises(ValueError, match="unknown shader type"):
        generate((1, 4, 16, 16), PARAMS, "not_a_real_shader", 8888, CPU)
