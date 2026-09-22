"""
The interference generator: fringes, and the knobs that set them.
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


def _draw(**overrides):
    return generate((1, 8, 64, 64), dict(PARAMS, **overrides), "interference", 8888, CPU)


def _zero_crossings_per_row(noise):
    signs = noise.sign()
    return (signs[..., :, 1:] != signs[..., :, :-1]).float().mean().item()


def _lag_correlation(noise, lag):
    pairs = torch.stack([noise[..., :, :-lag].flatten(), noise[..., :, lag:].flatten()])
    return torch.corrcoef(pairs)[0, 1].item()


def test_warp_strength_is_the_fringe_density():
    soft, mid, dense = (_zero_crossings_per_row(_draw(warp_strength=w)) for w in (0.0, 2.0, 5.0))
    assert soft < mid < dense, (soft, mid, dense)


def test_the_fringes_are_finer_than_the_field_but_still_large_scale():
    """
    Cutting fringes into an FBM adds structure below the FBM's own scale, which is
    the point, but the default must stay coarse enough for a model to read: with
    the gain set twice as high the field was white within four pixels.
    """
    fringes = _draw()
    fbm = generate((1, 8, 64, 64), PARAMS, "fractal", 8888, CPU)
    assert _zero_crossings_per_row(fringes) > _zero_crossings_per_row(fbm)
    assert _lag_correlation(fringes, 4) > 0.3


def test_phase_shift_retunes_the_second_field():
    assert not torch.equal(_draw(phase_shift=0.0), _draw(phase_shift=2.0))


def test_noise_scale_zooms():
    assert _zero_crossings_per_row(_draw(scale=0.3)) < _zero_crossings_per_row(_draw(scale=3.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "interference", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "interference", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "interference", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    wide = generate((1, 8, 22, 39), PARAMS, "interference", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "interference", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_evolves_one_field():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "interference", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.4
    assert abs(frame_correlation(False)) < 0.15
