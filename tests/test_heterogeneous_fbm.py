"""
The heterogeneous_fbm generator: detail that varies across the frame.

What is pinned is the heterogeneity itself -- that fine detail is unevenly
spread, that warp_strength sets how unevenly, that phase_shift tips the frame
toward rough or smooth -- and that with the swing at zero it is the plain FBM.
"""
import pytest
import torch
import torch.nn.functional as F

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 4.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _draw(**overrides):
    return generate((1, 8, 64, 64), dict(PARAMS, **overrides), "heterogeneous_fbm", 8888, CPU)


def _detail_per_block(noise, block=8):
    """Fine-structure energy in each block: what is left after a 3x3 blur."""
    fine = (noise - F.avg_pool2d(noise, 3, stride=1, padding=1)).abs()
    return F.avg_pool2d(fine, block)


def _unevenness(noise):
    """Spread of the fine structure across blocks, relative to its mean, per channel."""
    blocks = _detail_per_block(noise).flatten(2)
    return (blocks.std(dim=2) / blocks.mean(dim=2)).mean().item()


def test_detail_is_spread_unevenly_and_warp_strength_sets_how_unevenly():
    even, uneven = _unevenness(_draw(warp_strength=0.0)), _unevenness(_draw(warp_strength=2.0))
    assert uneven > even * 1.3, (even, uneven)


def test_with_no_swing_it_is_the_plain_fbm():
    """warp_strength 0 is fractal at lacunarity 2 with no layer offsets, pixel for pixel."""
    plain = generate((1, 8, 32, 32), dict(PARAMS, warp_strength=0.0), "heterogeneous_fbm", 8888, CPU)
    fractal = generate((1, 8, 32, 32), dict(PARAMS, warp_strength=0.5, phase_shift=0.0), "fractal", 8888, CPU)
    assert torch.allclose(plain, fractal, atol=1e-6)


def test_phase_shift_tips_the_frame_toward_rough_or_smooth():
    smooth = _detail_per_block(_draw(warp_strength=1.0, phase_shift=0.0)).mean()
    rough = _detail_per_block(_draw(warp_strength=1.0, phase_shift=2.0)).mean()
    assert rough > smooth


def test_the_knobs_need_a_second_layer():
    assert torch.equal(_draw(octaves=1.0, warp_strength=0.0, phase_shift=0.0),
                       _draw(octaves=1.0, warp_strength=3.0, phase_shift=2.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "heterogeneous_fbm", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "heterogeneous_fbm", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "heterogeneous_fbm", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    wide = generate((1, 8, 22, 39), PARAMS, "heterogeneous_fbm", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "heterogeneous_fbm", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_evolves_one_field():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "heterogeneous_fbm", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.5
    assert abs(frame_correlation(False)) < 0.15
