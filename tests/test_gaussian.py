"""
The gaussian generator: white noise, the control.

The properties worth pinning are the absences. It must have no spatial
structure, none of the pattern knobs may reach it, and it must still keep the
contract the other generators keep: a field per channel, channel 0 unchanged by
how many were asked for, and one field evolving under temporal coherence.
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


def test_the_field_is_white():
    noise = generate((1, 24, 22, 38), PARAMS, "gaussian", 8888, CPU)
    assert abs(_neighbour_correlation(noise)) < 0.05
    assert noise.abs().max() <= 1.0


@pytest.mark.parametrize("knob", [dict(scale=0.3), dict(scale=3.0), dict(octaves=8.0),
                                  dict(warp_strength=5.0), dict(phase_shift=2.0),
                                  dict(color_scheme="viridis")])
def test_no_pattern_knob_reaches_it(knob):
    """The tooltip promises these do nothing, so a stray use of one fails here."""
    stock = generate((1, 4, 16, 16), PARAMS, "gaussian", 8888, CPU)
    assert torch.equal(generate((1, 4, 16, 16), dict(PARAMS, **knob), "gaussian", 8888, CPU), stock)


def test_a_shape_mask_still_applies():
    stock = generate((1, 4, 16, 16), PARAMS, "gaussian", 8888, CPU)
    masked = generate((1, 4, 16, 16), dict(PARAMS, shape_type="radial"), "gaussian", 8888, CPU)
    assert not torch.equal(masked, stock)


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "gaussian", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.85


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "gaussian", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "gaussian", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


def test_holding_the_seed_and_advancing_time_turns_one_field_into_another():
    """
    A 5-frame clip steps time by a quarter, so adjacent frames sit cos(pi/8)
    apart on the great circle: 0.92. Without coherence each frame is its own seed.
    """
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "gaussian", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.8
    assert abs(frame_correlation(False)) < 0.1
    # Frame 0 is the first draw exactly, whatever time does to the frames after it.
    clip = generate((1, 4, 5, 16, 16), dict(PARAMS, use_temporal_coherence=True),
                    "gaussian", 8888, CPU, temporal_coherence=True)
    still = generate((1, 4, 16, 16), dict(PARAMS, use_temporal_coherence=True), "gaussian", 8888, CPU)
    assert torch.equal(clip[:, :, 0], still)
