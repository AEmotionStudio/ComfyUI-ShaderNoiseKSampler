"""
The cellular generator: Worley cells, and the knobs that shape them.
"""
import pytest
import torch
import torch.nn.functional as F

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate
from snk.shaders.cellular import worley
from snk.utils.noise_utils import create_coordinate_grid

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 1.0, "warp_strength": 0.0, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _draw(**overrides):
    return generate((1, 8, 64, 64), dict(PARAMS, **overrides), "cellular", 8888, CPU)


def _feature_points(cells):
    """Local minima of F1 over a 64x64 draw with `cells` cells across: one per feature point."""
    f1, _ = worley(create_coordinate_grid(1, 64, 64, CPU) * cells, 8888, 2.0)
    f1 = f1[0, :, :, 0]
    lowest = -F.max_pool2d(-f1[None, None], 3, stride=1, padding=1)[0, 0]
    return (f1 == lowest).sum().item()


def test_one_feature_point_per_cell():
    """
    Twice the cells along each axis is about four times the minima of F1. The
    frame's border adds a few: a pixel sloping toward a point outside the frame
    is the lowest of the neighbours it has, so the bounds allow for the rim.
    """
    four, eight = _feature_points(4.0), _feature_points(8.0)
    assert 12 <= four <= 36 and 48 <= eight <= 120 and eight > 2 * four, (four, eight)


def test_octaves_picks_one_of_four_patterns():
    patterns = [_draw(octaves=float(n)) for n in (1, 2, 3, 4)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.equal(patterns[a], patterns[b]), (a, b)
    assert torch.equal(_draw(octaves=5.0), patterns[0])


def test_phase_shift_is_the_cell_shape():
    diamond, round_, square = _draw(phase_shift=0.0), _draw(phase_shift=0.5), _draw(phase_shift=2.0)
    assert not torch.equal(diamond, round_) and not torch.equal(round_, square)


def test_warp_strength_bends_the_lattice():
    assert not torch.equal(_draw(warp_strength=0.0), _draw(warp_strength=3.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "cellular", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "cellular", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "cellular", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
@pytest.mark.parametrize("phase_shift", [0.5, 2.0])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel, phase_shift):
    """
    Exact for the Euclidean metric. The other exponents go through pow, whose
    kernel vectorises a [N,B,H,W,1] draw differently from a [B,H,W,1] one at a
    ragged shape, so those match to the last ulp rather than the bit -- the same
    allowance test_simplex makes for rotation at ragged shapes.
    """
    params = dict(PARAMS, phase_shift=phase_shift, warp_strength=0.5)
    wide = generate((1, 8, 22, 39), params, "cellular", 8888, CPU)
    alone = generate((1, 1, 22, 39), params, "cellular", 8888 + 6151 * channel, CPU)
    if phase_shift == 0.5:
        assert torch.equal(wide[:, channel:channel + 1], alone)
    else:
        assert torch.allclose(wide[:, channel:channel + 1], alone, atol=1e-6, rtol=0.0)


def test_holding_the_seed_and_advancing_time_moves_the_plane_through_the_cells():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "cellular", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.4
    assert abs(frame_correlation(False)) < 0.15
