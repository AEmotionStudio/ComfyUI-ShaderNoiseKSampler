"""
The fractal generator: what each knob does to a plain FBM.

It is the reference the other layered types are read against, so what is
pinned is that every knob reaches exactly the FBM parameter it is documented
to, and nothing else.
"""
import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 3.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _neighbour_correlation(noise):
    field = noise[0]
    pairs = torch.stack([field[:, :-1, :].flatten(), field[:, 1:, :].flatten()])
    return torch.corrcoef(pairs)[0, 1].item()


def _draw(**overrides):
    return generate((1, 8, 32, 32), dict(PARAMS, **overrides), "fractal", 8888, CPU)


def test_the_field_is_large_scale_not_white():
    noise = generate((1, 24, 22, 38), PARAMS, "fractal", 8888, CPU)
    assert _neighbour_correlation(noise) > 0.5
    assert noise.abs().max() <= 1.0


def test_octaves_add_detail():
    assert _neighbour_correlation(_draw(octaves=1.0)) > _neighbour_correlation(_draw(octaves=4.0))


def test_noise_scale_zooms():
    assert _neighbour_correlation(_draw(scale=0.3)) > _neighbour_correlation(_draw(scale=3.0))


def test_warp_strength_is_the_layer_spacing():
    """It spaces the layers, so with one layer there is nothing for it to do."""
    assert not torch.equal(_draw(warp_strength=0.0), _draw(warp_strength=3.0))
    assert torch.equal(_draw(octaves=1.0, warp_strength=0.0), _draw(octaves=1.0, warp_strength=3.0))


def test_phase_shift_slides_the_layers_apart():
    assert not torch.equal(_draw(phase_shift=0.0), _draw(phase_shift=1.5))
    assert torch.equal(_draw(octaves=1.0, phase_shift=0.0), _draw(octaves=1.0, phase_shift=1.5))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "fractal", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "fractal", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "fractal", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    """The channel axis is drawn in one call; it must match the loop it replaces."""
    wide = generate((1, 8, 22, 39), PARAMS, "fractal", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "fractal", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_evolves_one_field():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "fractal", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.5
    assert abs(frame_correlation(False)) < 0.15
