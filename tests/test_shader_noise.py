"""
Shader noise must match the latent's own layout.

ComfyUI latents are [B, C, H, W] and [B, C, T, H, W]. Legacy guessed between
[B,C,F,H,W] and [B,F,C,H,W] and defaulted to the latter when both dimensions
matched the channel count -- which is exactly what a 61-frame Wan or Hunyuan
clip looks like: (61-1)//4+1 = 16 latent frames and 16 channels. Time evolution
then ran across channels, with no error raised.
"""
import re

import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import (
    UnsupportedLatentError, effective_channel_rank, generate, latent_layout,
)

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
    with pytest.raises(UnsupportedLatentError):
        latent_layout(shape)


@pytest.mark.parametrize("shape", [(1, 64, 1024), (1, 128, 860), (1, 8192, 16)])
def test_layout_names_the_sequence_latents_it_cannot_paint(shape):
    """Stable Audio, MiniMax Music 3 and TripoSplat: rank 3, no spatial grid."""
    with pytest.raises(UnsupportedLatentError, match=rf"3D {re.escape(str(shape))}"):
        latent_layout(shape)


@pytest.mark.parametrize("shape", [
    (1, 4, 32, 32), (2, 4, 16, 16), (1, 16, 5, 8, 8), (1, 16, 3, 16, 16),
    # every channel count in ComfyUI's roster, image and video
    (1, 3, 16, 16),      # pixel space: Z-Image, PixelDiT, HiDream-O1, SenseNova, Chroma Radiance
    (1, 8, 16, 16),      # ACE-Step 1.0
    (1, 12, 3, 8, 8),    # Mochi
    (1, 24, 5, 8, 8),    # MiniMax H3, video stream
    (1, 32, 16, 16),     # Trellis2
    (1, 32, 5, 8, 8),    # HunyuanVideo 1.5
    (1, 48, 3, 8, 8),    # Wan 2.2
    (1, 64, 16, 16),     # HunyuanImage 2.1
    (1, 128, 16, 16),    # Flux 2, Ideogram 4, Lens, Ernie, MageFlow
    (1, 128, 3, 8, 8),   # LTXV
    (1, 256, 16, 16),    # Stable Audio 3's channel count
])
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


@pytest.mark.parametrize("shader_type", ["domain_warp", "tensor_field", "curl_noise", "temporal_coherent",
                                         "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"])
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


# --- channel width --------------------------------------------------------------------

ALL_SHADERS = ["domain_warp", "temporal_coherent", "curl_noise", "tensor_field", "spectral",
               "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"]


@pytest.mark.parametrize("shape", [(1, 4, 48, 48), (1, 16, 32, 32), (1, 24, 5, 16, 16)])
@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_generators_span_their_channel_axis(shape, shader_type):
    """
    Every channel carries a field of its own. domain_warp used to copy one field
    four times and temporal_coherent broadcast one field to every channel -- rank
    1.00 whatever the latent -- which is what made the shader overwrite the
    picture instead of steering it.
    """
    rank = effective_channel_rank(generate(shape, PARAMS, shader_type, 8888, CPU))
    assert rank > shape[1] * 0.6, f"{shader_type}: rank {rank:.2f} of {shape[1]}"


@pytest.mark.parametrize("shader_type", ["domain_warp", "temporal_coherent", "curl_noise", "gaussian"])
def test_a_latent_wider_than_the_basis_still_spans_the_basis(shader_type):
    """Past CHANNEL_BASIS the channels are mixtures, which must not re-collapse the draw."""
    rank = effective_channel_rank(generate((1, 128, 3, 8, 8), PARAMS, shader_type, 8888, CPU))
    assert rank > CHANNEL_BASIS * 0.8, f"{shader_type}: rank {rank:.2f}"


@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_channel_zero_is_the_one_channel_draw(shader_type):
    """
    The travel-mode basis is built from one-channel draws. Filling the other
    channels must not move channel 0, or jump and drift would change underneath
    the presets calibrated against them.
    """
    wide = generate((1, 24, 5, 16, 16), PARAMS, shader_type, 8888, CPU)
    single = generate((1, 1, 5, 16, 16), PARAMS, shader_type, 8888, CPU)
    assert torch.equal(wide[:, :1], single)


def test_filling_channels_leaves_the_global_rng_alone():
    """The generators reseed torch inside every render; that must not leak to the caller."""
    from snk.shaders.base import BaseNoiseGenerator

    def render(seed):
        torch.manual_seed(seed)
        return torch.rand(1, 1, 8, 8)

    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    BaseNoiseGenerator.fill_channels(render, torch.zeros(1, 1, 8, 8), 16, 7)
    assert torch.equal(torch.rand(4), expected)


# --- travel-mode remixing -------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 4, 48, 48), (1, 16, 32, 32), (1, 24, 5, 16, 16), (1, 128, 16, 16)])
@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_decorrelation_never_narrows_the_noise(shape, shader_type):
    """
    Asking for the widest basis must never leave the noise spanning fewer
    channels than leaving it off. A remix reaches only about 0.6 of its basis, so
    generate() keeps whichever is wider rather than trusting it.
    """
    stock = generate(shape, PARAMS, shader_type, 8888, CPU)
    fixed = generate(shape, PARAMS, shader_type, 8888, CPU, decorrelate=True)
    assert fixed.shape == stock.shape
    assert torch.isfinite(fixed).all()
    assert effective_channel_rank(fixed) >= effective_channel_rank(stock) - 1e-6


@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_widening_leaves_an_already_wide_draw_alone(shader_type):
    """
    The generators draw wider than a remix could make them, so walk hands their
    noise through untouched instead of rendering a basis and throwing it away.
    """
    shape = (1, 24, 5, 16, 16)
    stock = generate(shape, PARAMS, shader_type, 8888, CPU)
    assert torch.equal(stock, generate(shape, PARAMS, shader_type, 8888, CPU, decorrelate=True))


def test_widening_still_rescues_a_narrow_draw():
    """No shipped generator arrives narrow any more, but one that does must be widened."""
    def one_field_everywhere(params, height, width, batch_size, device, seed, target_channels):
        field = torch.randn(batch_size, 1, height, width, generator=torch.Generator().manual_seed(seed))
        return field.expand(-1, target_channels, -1, -1).clone()

    shape = (1, 24, 5, 16, 16)
    stock = generate(shape, PARAMS, "domain_warp", 8888, CPU, generator=one_field_everywhere)
    widened = generate(shape, PARAMS, "domain_warp", 8888, CPU, generator=one_field_everywhere,
                       decorrelate=True)
    assert effective_channel_rank(stock) < 1.5
    assert effective_channel_rank(widened) > 10


@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_a_narrow_basis_narrows_a_wide_draw(shader_type):
    """
    drift asks for four directions out of a draw spanning twenty-odd. The widening
    guard must not mistake that for a remix that failed to help and skip it.
    """
    shape = (1, 24, 5, 16, 16)
    stock = effective_channel_rank(generate(shape, PARAMS, shader_type, 8888, CPU))
    drifted = effective_channel_rank(
        generate(shape, PARAMS, shader_type, 8888, CPU, decorrelate=True, basis=4))
    assert drifted < 4.5 < stock, f"{shader_type}: stock {stock:.2f}, drift {drifted:.2f}"


def test_decorrelation_is_off_by_default_and_reproducible():
    shape = (1, 16, 32, 32)
    assert torch.equal(generate(shape, PARAMS, "domain_warp", 1, CPU),
                       generate(shape, PARAMS, "domain_warp", 1, CPU, decorrelate=False))
    assert torch.equal(generate(shape, PARAMS, "domain_warp", 1, CPU, decorrelate=True),
                       generate(shape, PARAMS, "domain_warp", 1, CPU, decorrelate=True))


@pytest.mark.parametrize("shader_type", ALL_SHADERS)
@pytest.mark.parametrize("shape", [(1, 24, 5, 16, 16), (1, 4, 32, 32), (1, 128, 3, 8, 8)])
def test_a_collapse_does_not_render_the_draw_it_discards(shape, shader_type, monkeypatch):
    """
    jump rebuilds every channel from one-channel draws and never reads the wide
    draw, so rendering one first was pure waste -- 5.99s of jump's 6.21s at H3's
    default latent, for a tensor that is dropped on the next line.

    Both halves matter. The values must not move, because jump and stamp are
    byte-identical to the pre-collapse-fix tag and the presets are calibrated on
    that; and the wide draw must genuinely not be rendered, or the fix is only a
    comment.
    """
    from snk.core import shader_noise

    collapsed = generate(shape, PARAMS, shader_type, 8888, CPU, decorrelate=True, basis=1)
    assert torch.equal(collapsed, shader_noise._decorrelate(
        shape, PARAMS, shader_type, 8888, CPU, torch.float32, False, None, basis_size=1))

    calls = []
    real = shader_noise._render
    monkeypatch.setattr(shader_noise, "_render",
                        lambda *a, **k: (calls.append(a[3]["channels"]), real(*a, **k))[1])
    generate(shape, PARAMS, shader_type, 8888, CPU, decorrelate=True, basis=1)
    assert calls, "nothing was rendered at all"
    assert set(calls) == {1}, f"a collapse rendered a {max(calls)}-channel draw it cannot use"


@pytest.mark.parametrize("shader_type", ["domain_warp", "curl_noise", "temporal_coherent", "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"])
def test_the_hash_generators_do_not_reseed_the_global_rng(shader_type, monkeypatch):
    """
    These three are pure coordinate hashes of their seed argument, so the
    torch.manual_seed calls they used to make changed nothing -- domain_warp's ran
    once per channel render, 888 times per draw at H3's default latent, and
    manual_seed reseeds every CUDA device as well as the CPU.

    tensor_field is deliberately not in this list: shaders/tensor_field.py draws
    torch.randn_like inside its channel loop and that value reaches the output, so
    its reseed is load-bearing. If that ever changes, add it here.
    """
    monkeypatch.setattr(torch, "manual_seed",
                        lambda *a, **k: pytest.fail(f"{shader_type} reseeded the global RNG"))
    generate((1, 24, 5, 16, 16), PARAMS, shader_type, 8888, CPU)


@pytest.mark.parametrize("shader_type", ALL_SHADERS)
def test_the_draw_hands_back_an_ordinary_tensor(shader_type):
    """
    The draw runs under inference mode, which is worth about a tenth of it. An
    inference tensor raises "Inference tensors cannot be saved for backward" the
    moment it reaches a grad-recording region, and this noise is handed on to
    whatever the workflow does next, so generate() must clone it back to a normal
    tensor on the way out.
    """
    noise = generate((1, 4, 32, 32), PARAMS, shader_type, 8888, CPU, decorrelate=True)
    assert not torch.is_inference(noise)
    noise.requires_grad_(True)  # raises if it is still an inference tensor
