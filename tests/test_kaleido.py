"""
The kaleido generator: an FBM folded about the centre of the frame.

What matters is the symmetry, so that is what is pinned: which fold each order gives,
that the warp stays inside it, and that it holds on a latent that is not square.
"""
import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 1.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _draw(shape=(1, 8, 64, 64), **overrides):
    return generate(shape, dict(PARAMS, **overrides), "kaleido", 8888, CPU)


def _alike(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _mirrored(noise):
    return _alike(noise, noise.flip(-1))


def _flipped(noise):
    return _alike(noise, noise.flip(-2))


def test_order_one_mirrors_left_to_right_and_nothing_else():
    noise = _draw(octaves=1.0)
    assert _mirrored(noise) > 0.999
    assert abs(_flipped(noise)) < 0.5


def test_order_two_mirrors_both_ways():
    noise = _draw(octaves=2.0)
    assert _mirrored(noise) > 0.999 and _flipped(noise) > 0.999


@pytest.mark.parametrize("order", [3, 4, 6])
def test_higher_orders_cut_the_frame_into_that_many_wedges(order):
    """A quarter turn maps the pattern onto itself exactly when the order is a multiple of four."""
    noise = _draw(octaves=float(order))
    quarter = _alike(noise, noise.rot90(1, (-2, -1)))
    if order % 4 == 0:
        assert quarter > 0.999
    else:
        assert quarter < 0.9
    # Every order from three up is mirrored across the fold's own axis, which sits
    # level at the default phase_shift.
    assert _flipped(noise) > 0.99


def test_the_fold_holds_on_a_latent_that_is_not_square():
    assert _mirrored(_draw(shape=(1, 8, 22, 38))) > 0.999
    assert _mirrored(_draw(shape=(1, 8, 22, 38), octaves=2.0)) > 0.999


def test_warp_strength_bends_the_pattern_without_breaking_the_fold():
    still, bent = _draw(warp_strength=0.0), _draw(warp_strength=3.0)
    assert not torch.equal(still, bent)
    assert _mirrored(bent) > 0.999


def test_phase_shift_turns_the_fold():
    """A quarter of the knob's range is a quarter turn: the left-right mirror becomes top-bottom."""
    turned = _draw(phase_shift=1.0)
    assert _flipped(turned) > 0.999
    assert abs(_mirrored(turned)) < 0.5


def test_noise_scale_is_the_size_of_the_features():
    def crossings(noise):
        signs = noise.sign()
        return (signs[..., 1:] != signs[..., :-1]).float().mean().item()

    assert crossings(_draw(scale=0.5)) < crossings(_draw(scale=2.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "kaleido", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_a_higher_order_leaves_the_channels_less_room_to_differ():
    """One part in 2N of the frame is free, so the draw spans fewer channels as N grows."""
    ranks = [effective_channel_rank(_draw(shape=(1, 16, 128, 128), octaves=float(order))) for order in (1, 2, 6)]
    assert ranks[0] > ranks[1] > ranks[2], ranks


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "kaleido", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "kaleido", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    wide = generate((1, 8, 22, 39), PARAMS, "kaleido", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "kaleido", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_evolves_the_field():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "kaleido", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.7
    assert abs(frame_correlation(False)) < 0.15
