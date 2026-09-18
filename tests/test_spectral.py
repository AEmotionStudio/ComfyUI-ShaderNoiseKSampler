"""
The spectral generator: what its parameters actually control.

It is the only generator that builds its field from a frequency band rather than
per pixel, so the properties worth pinning are different ones -- not "does it
look like swirls" but "is the band where the parameters say it is, and does the
field still span the channels it was asked for".
"""
import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate
from snk.shaders.spectral import SpectralNoiseGenerator

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 3.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


def _neighbour_correlation(noise):
    """How alike neighbouring pixels are: high means the field is large-scale."""
    field = noise[0]
    pairs = torch.stack([field[:, :-1, :].flatten(), field[:, 1:, :].flatten()])
    return torch.corrcoef(pairs)[0, 1].item()


def test_the_field_is_large_scale_not_white():
    """
    The whole point is to put energy where the model reads it. Gaussian noise has
    equal energy at every scale and neighbouring pixels are uncorrelated; this
    should look nothing like that -- measured 0.81 against domain_warp's 0.66 and
    curl_noise's 0.89.
    """
    noise = generate((1, 24, 22, 38), PARAMS, "spectral", 8888, CPU)
    assert _neighbour_correlation(noise) > 0.5
    assert _neighbour_correlation(torch.randn(1, 24, 22, 38)) < 0.1


def test_noise_scale_moves_the_band():
    """
    noise_scale means here what it means everywhere else in the pack: low is
    large, zoomed-in features, high is small, zoomed-out ones. For this generator
    that is literally where the band sits, so it has to show up as a change in how
    alike neighbouring pixels are.
    """
    zoomed_in = generate((1, 24, 32, 32), dict(PARAMS, scale=0.3), "spectral", 8888, CPU)
    zoomed_out = generate((1, 24, 32, 32), dict(PARAMS, scale=3.0), "spectral", 8888, CPU)
    assert _neighbour_correlation(zoomed_in) > _neighbour_correlation(zoomed_out)


def test_octaves_flattens_the_slope():
    """More octaves is more fine detail in an FBM, so here it is a shallower roll-off."""
    shallow = generate((1, 24, 32, 32), dict(PARAMS, octaves=8.0), "spectral", 8888, CPU)
    steep = generate((1, 24, 32, 32), dict(PARAMS, octaves=1.0), "spectral", 8888, CPU)
    assert _neighbour_correlation(steep) > _neighbour_correlation(shallow)


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_the_band_stays_wide_enough_to_span_the_channels(shape):
    """
    Bandwidth and channel rank are the same dial. Every channel is an independent
    draw, but they all live in whatever subspace the envelope leaves open, so a
    band narrow enough to look good can still hand the sampler a draw that spans
    far fewer directions than it has channels -- which is the failure the whole
    channel-rank effort was about. `_corner_floor` is what stops it.
    """
    noise = generate(shape, PARAMS, "spectral", 8888, CPU)
    channels = shape[1]
    assert effective_channel_rank(noise) > min(channels, CHANNEL_BASIS) * 0.6


def test_the_floor_does_not_depend_on_how_many_channels_were_asked_for():
    """
    If the envelope varied with `target_channels`, channel 0 of a wide draw would
    be drawn under a different envelope than a one-channel draw at the same seed,
    and core.shader_noise builds every travel-mode basis from one-channel draws --
    `jump` and `stamp` would stop agreeing with `walk` about what the field is.
    """
    assert SpectralNoiseGenerator._corner_floor(22, 38) == \
        SpectralNoiseGenerator._corner_floor(22, 38)
    wide = generate((1, 24, 5, 22, 38), PARAMS, "spectral", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "spectral", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


def test_holding_the_seed_and_advancing_time_evolves_one_field():
    """
    Temporal coherence should move the field rather than redraw it. Each mode
    turns at its own rate, so consecutive frames stay related -- measured 0.27,
    against 0.26 for temporal_coherent, the generator built for this.
    """
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "spectral", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.15
    assert abs(frame_correlation(False)) < 0.1


def test_it_is_far_cheaper_than_drawing_per_channel():
    """
    One inverse FFT for the whole channel stack against one procedural render per
    channel. This is the reason the generator exists, so it is asserted rather
    than left to the benchmark: a change that quietly puts it back on a per-channel
    path should fail here. Measured about 50x at H3's latent; the bar is set low
    enough to survive a loaded machine.
    """
    import time

    shape = (1, 24, 5, 22, 38)
    for shader_type in ("spectral", "domain_warp"):
        generate(shape, PARAMS, shader_type, 8888, CPU)

    timings = {}
    for shader_type in ("spectral", "domain_warp"):
        start = time.perf_counter()
        generate(shape, PARAMS, shader_type, 8888, CPU)
        timings[shader_type] = time.perf_counter() - start
    assert timings["spectral"] * 5 < timings["domain_warp"]
