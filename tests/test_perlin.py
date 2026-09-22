"""
The perlin generator: real gradient noise, and the knobs the matrix documents.
"""
import pytest
import torch

from snk.core.constants import CHANNEL_BASIS
from snk.core.shader_noise import effective_channel_rank, generate
from snk.shaders.fbm import with_time
from snk.shaders.perlin import perlin_2d, perlin_3d
from snk.utils.noise_utils import create_coordinate_grid

CPU = torch.device("cpu")
PARAMS = {
    "scale": 1.0, "octaves": 3.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}
SEEDS = torch.tensor([8888, 8888 + 6151, 4242], dtype=torch.int64).reshape(3, 1, 1, 1, 1)


def _neighbour_correlation(noise):
    field = noise[0]
    pairs = torch.stack([field[:, :-1, :].flatten(), field[:, 1:, :].flatten()])
    return torch.corrcoef(pairs)[0, 1].item()


def _draw(**overrides):
    return generate((1, 8, 32, 32), dict(PARAMS, **overrides), "perlin", 8888, CPU)


def test_gradient_noise_is_zero_on_the_lattice():
    """The cheapest identity of a real Perlin: every lattice point is a zero crossing."""
    grid = torch.stack(torch.meshgrid(torch.arange(6.0), torch.arange(6.0), indexing="ij"), dim=-1).unsqueeze(0)
    assert torch.allclose(perlin_2d(grid, 8888), torch.zeros(1, 6, 6, 1), atol=1e-6)
    assert torch.allclose(perlin_3d(with_time(grid, 2.0), 8888), torch.zeros(1, 6, 6, 1), atol=1e-6)
    off_grid = perlin_2d(grid + 0.5, 8888)
    assert off_grid.abs().max() > 0.1


@pytest.mark.parametrize("dims", [2, 3])
def test_the_primitives_batch_over_a_tensor_of_seeds_exactly(dims):
    p = create_coordinate_grid(1, 22, 39, CPU) * 3.0
    noise = perlin_2d if dims == 2 else perlin_3d
    if dims == 3:
        p = with_time(p, 0.7)
    batched = noise(p, SEEDS)
    assert tuple(batched.shape) == (3, 1, 22, 39, 1)
    for index, seed in enumerate(SEEDS.flatten().tolist()):
        assert torch.equal(batched[index], noise(p, seed))


def test_the_field_is_large_scale_not_white():
    noise = generate((1, 24, 22, 38), PARAMS, "perlin", 8888, CPU)
    assert _neighbour_correlation(noise) > 0.5
    assert noise.abs().max() <= 1.0


def test_octaves_add_detail_and_noise_scale_zooms():
    assert _neighbour_correlation(_draw(octaves=1.0)) > _neighbour_correlation(_draw(octaves=4.0))
    assert _neighbour_correlation(_draw(scale=0.3)) > _neighbour_correlation(_draw(scale=3.0))


def test_the_warp_reaches_the_detail_layers_only():
    assert not torch.equal(_draw(warp_strength=0.0), _draw(warp_strength=3.0))
    assert torch.equal(_draw(octaves=1.0, warp_strength=0.0), _draw(octaves=1.0, warp_strength=3.0))


def test_phase_shift_is_contrast():
    """Contrast 2 is the same field at twice the gain, clipped at the same clamp."""
    flat, sharp = _draw(phase_shift=0.0), _draw(phase_shift=2.0)
    assert torch.equal(torch.clamp(flat * 2.0, -1.0, 1.0), sharp)
    assert sharp.std() > flat.std()


def test_a_palette_maps_the_field_onto_three_channels_and_keeps_channel_zero():
    plain = generate((1, 16, 32, 32), PARAMS, "perlin", 8888, CPU)
    coloured = generate((1, 16, 32, 32), dict(PARAMS, color_scheme="viridis"), "perlin", 8888, CPU)
    single = generate((1, 1, 32, 32), dict(PARAMS, color_scheme="viridis"), "perlin", 8888, CPU)
    assert tuple(coloured.shape) == (1, 16, 32, 32)
    assert torch.isfinite(coloured).all()
    assert not torch.equal(coloured[:, :3], plain[:, :3])
    assert torch.equal(coloured[:, :1], single)
    assert effective_channel_rank(coloured) > 16 * 0.6


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 24, 22, 38), (1, 24, 5, 16, 16),
                                   (1, 128, 3, 8, 8), (1, 128, 3, 32, 32)])
def test_every_channel_is_its_own_field(shape):
    rank = effective_channel_rank(generate(shape, PARAMS, "perlin", 8888, CPU))
    assert rank > min(shape[1], CHANNEL_BASIS) * 0.6, rank


def test_channel_zero_does_not_depend_on_how_many_channels_were_asked_for():
    wide = generate((1, 24, 5, 22, 38), PARAMS, "perlin", 8888, CPU)
    single = generate((1, 1, 5, 22, 38), PARAMS, "perlin", 8888, CPU)
    assert torch.equal(wide[:, :1], single)


@pytest.mark.parametrize("channel", [1, 5])
def test_a_batched_channel_is_the_draw_its_seed_gives_alone(channel):
    wide = generate((1, 8, 22, 39), PARAMS, "perlin", 8888, CPU)
    alone = generate((1, 1, 22, 39), PARAMS, "perlin", 8888 + 6151 * channel, CPU)
    assert torch.equal(wide[:, channel:channel + 1], alone)


def test_holding_the_seed_and_advancing_time_evolves_one_field():
    def frame_correlation(coherent):
        noise = generate((1, 24, 5, 22, 38), dict(PARAMS, use_temporal_coherence=coherent),
                         "perlin", 8888, CPU, temporal_coherence=coherent)
        pairs = torch.stack([noise[0, :, 0].flatten(), noise[0, :, 1].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    assert frame_correlation(True) > 0.5
    assert abs(frame_correlation(False)) < 0.15
