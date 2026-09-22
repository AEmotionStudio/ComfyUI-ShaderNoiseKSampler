"""
The waves generator: a few plane waves, and the knobs that arrange them.
"""
import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 3.0, "warp_strength": 0.0, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _draw(**overrides):
    return generate((1, 8, 64, 64), dict(PARAMS, **overrides), "waves", 8888, CPU)


def _spectral_concentration(noise, bins):
    """Share of the power in the `bins` strongest frequencies, DC excluded, per channel."""
    power = torch.fft.rfft2(noise).abs() ** 2
    power[..., 0, 0] = 0.0
    flat = power.flatten(2)
    top = flat.topk(bins, dim=2).values.sum(dim=2)
    return (top / flat.sum(dim=2)).mean().item()


def test_a_few_waves_put_their_power_in_a_few_frequencies():
    """Three straight plane waves are a handful of spectral lines: measured 0.88."""
    assert _spectral_concentration(_draw(), 12) > 0.8


def test_warp_strength_bends_the_wavefronts():
    straight, bent = _spectral_concentration(_draw(), 12), _spectral_concentration(_draw(warp_strength=2.0), 12)
    assert bent < straight - 0.1, (straight, bent)


def test_octaves_is_the_number_of_waves():
    one, three, six = _draw(octaves=1.0), _draw(octaves=3.0), _draw(octaves=6.0)
    assert not torch.equal(one, three) and not torch.equal(three, six)
    assert _spectral_concentration(one, 4) > _spectral_concentration(six, 4)


def test_phase_shift_rearranges_the_same_waves():
    assert not torch.equal(_draw(phase_shift=0.0), _draw(phase_shift=1.0))


def test_noise_scale_is_the_wavelength():
    def crossings(noise):
        signs = noise.sign()
        return (signs[..., 1:] != signs[..., :-1]).float().mean().item()

    assert crossings(_draw(scale=0.5)) < crossings(_draw(scale=2.0))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "waves", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "waves", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "waves", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    params = dict(PARAMS, warp_strength=0.5)
    wide = generate((1, 8, 22, 39), params, "waves", 8888, CPU)
    alone = generate((1, 1, 22, 39), params, "waves", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_drifts_the_waves():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "waves", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.7
    assert abs(frame_correlation(False)) < 0.15
