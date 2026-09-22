"""
The projection_3d generator: a plane through a volume, and the depth knob.
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
    return generate((1, 8, 48, 48), dict(PARAMS, **overrides), "projection_3d", 8888, CPU)


def _correlation(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _neighbour_correlation(noise):
    field = noise[0]
    pairs = torch.stack([field[:, :-1, :].flatten(), field[:, 1:, :].flatten()])
    return torch.corrcoef(pairs)[0, 1].item()


def test_phase_shift_is_the_depth_of_the_slice():
    """Nearby depths are nearby pictures; a whole unit away they are unrelated."""
    here = _draw(phase_shift=0.0)
    near, far = _correlation(here, _draw(phase_shift=0.05)), _correlation(here, _draw(phase_shift=1.0))
    assert near > 0.8 and abs(far) < 0.2, (near, far)


def test_warp_strength_bends_the_plane():
    assert not torch.equal(_draw(warp_strength=0.0), _draw(warp_strength=3.0))


def test_noise_scale_zooms_and_octaves_add_detail():
    assert _neighbour_correlation(_draw(scale=0.3)) > _neighbour_correlation(_draw(scale=3.0))
    assert _neighbour_correlation(_draw(octaves=1.0)) > _neighbour_correlation(_draw(octaves=4.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "projection_3d", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "projection_3d", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "projection_3d", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    wide = generate((1, 8, 22, 39), PARAMS, "projection_3d", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "projection_3d", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_time_slides_the_plane_so_a_clip_is_coherent_by_construction():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "projection_3d", 8888, CPU, temporal_coherence=coherent)
        return _correlation(noise[0, :, 0], noise[0, :, 1])

    assert frame_correlation(True) > 0.5
    assert abs(frame_correlation(False)) < 0.15
