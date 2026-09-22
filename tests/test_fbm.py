"""
The primitives the scalar-field generators share: the lattice hash, the
four-corner 3D simplex, the FBM over them, and the channel-fill skeleton.

What matters about all of them is the same thing that matters about
simplex_2d: a tensor of seeds must give exactly the fields the seeds would give
one at a time, or fill_channels' batched path drifts from the scalar path it
replaces.
"""
import pytest
import torch

from snk.core.params import ShaderParams
from snk.shaders.base import BaseNoiseGenerator
from snk.shaders.fbm import draw_channels, fbm, simplex_warp, standardise, with_time
from snk.shaders.simplex import lattice_hash, simplex_3d_full
from snk.utils.noise_utils import create_coordinate_grid

CPU = torch.device("cpu")
SEEDS = torch.tensor([8888, 8888 + 6151, 4242], dtype=torch.int64).reshape(3, 1, 1, 1, 1)


def _cells(n=24):
    ix = torch.arange(n, dtype=torch.int64).view(1, n, 1, 1).expand(1, n, n, 1)
    iy = torch.arange(n, dtype=torch.int64).view(1, 1, n, 1).expand(1, n, n, 1)
    return ix, iy


def test_lattice_hash_is_uniform_on_the_unit_interval():
    ix, iy = _cells(64)
    values = lattice_hash(ix, iy, 8888)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert abs(values.mean().item() - 0.5) < 0.02
    # The x and y jitters of a cell come from different seeds and must not line up.
    other = lattice_hash(ix, iy, 8888 + 4099)
    pairs = torch.stack([values.flatten(), other.flatten()])
    assert abs(torch.corrcoef(pairs)[0, 1].item()) < 0.1


def test_lattice_hash_has_no_repeat_at_the_cube_hash_collision_offset():
    """
    The hash's linear part repeats every (29, -10) cells mod 1013 for as long as
    the cube does not wrap int64, which at seed 20 and a 40-cell grid it does not.
    Cells that far apart would then get the same feature point.
    """
    ix, iy = _cells(64)
    values = lattice_hash(ix, iy, 20)
    shifted = lattice_hash(ix + 29, iy - 10, 20)
    pairs = torch.stack([values.flatten(), shifted.flatten()])
    assert abs(torch.corrcoef(pairs)[0, 1].item()) < 0.1


def test_lattice_hash_batches_over_a_tensor_of_seeds_exactly():
    ix, iy = _cells()
    batched = lattice_hash(ix, iy, SEEDS)
    assert tuple(batched.shape) == (3, 1, 24, 24, 1)
    for index, seed in enumerate(SEEDS.flatten().tolist()):
        assert torch.equal(batched[index], lattice_hash(ix, iy, seed))


@pytest.mark.parametrize("hw", [(16, 16), (22, 39)])
def test_the_3d_simplex_batches_over_a_tensor_of_seeds_exactly(hw):
    p = with_time(create_coordinate_grid(1, *hw, CPU) * 3.0, 0.4)
    batched = simplex_3d_full(p, SEEDS)
    assert tuple(batched.shape) == (3, 1) + hw + (1,)
    for index, seed in enumerate(SEEDS.flatten().tolist()):
        assert torch.equal(batched[index], simplex_3d_full(p, seed))


def test_the_3d_simplex_is_the_function_temporal_coherent_carries():
    from snk.shaders.temporal_coherent_noise import TemporalCoherentNoiseGenerator

    p = with_time(create_coordinate_grid(1, 16, 16, CPU) * 3.0, 0.4)
    assert torch.equal(TemporalCoherentNoiseGenerator._simplex_3d(p, 8888), simplex_3d_full(p, 8888))


@pytest.mark.parametrize("time", [None, 0.3])
def test_fbm_batches_over_a_tensor_of_seeds_exactly(time):
    p = create_coordinate_grid(1, 22, 39, CPU) * 2.0
    batched = fbm(p, 3, SEEDS, time=time)
    assert tuple(batched.shape) == (3, 1, 22, 39, 1)
    for index, seed in enumerate(SEEDS.flatten().tolist()):
        assert torch.equal(batched[index], fbm(p, 3, seed, time=time))


def test_fbm_adds_detail_with_octaves_and_is_bounded():
    p = create_coordinate_grid(1, 48, 48, CPU) * 2.0

    def neighbour_correlation(field):
        pairs = torch.stack([field[0, :-1].flatten(), field[0, 1:].flatten()])
        return torch.corrcoef(pairs)[0, 1].item()

    one = fbm(p, 1, 8888)
    four = fbm(p, 4, 8888)
    assert one.abs().max() <= 1.0 and four.abs().max() <= 1.0
    assert neighbour_correlation(one) > neighbour_correlation(four)


def test_simplex_warp_moves_points_by_at_most_its_strength():
    p = create_coordinate_grid(1, 16, 16, CPU)
    assert torch.equal(simplex_warp(p, 8888, 0.0, 0.5), p)
    warped = simplex_warp(p, 8888, 0.3, 0.5)
    assert (warped - p).abs().max() <= 0.3
    assert not torch.equal(warped, p)


def test_standardise_treats_each_seed_slice_on_its_own():
    field = torch.randn(3, 1, 8, 8, 1) * torch.tensor([1.0, 3.0, 0.2]).view(3, 1, 1, 1, 1) + 2.0
    per_slice = standardise(field, SEEDS)
    for index in range(3):
        assert torch.equal(per_slice[index], standardise(field[index], 8888))
        assert abs(per_slice[index].mean().item()) < 1e-5
        assert abs(per_slice[index].std().item() - 1.0) < 1e-3


def _params(**overrides):
    return ShaderParams({
        "scale": 1.0, "octaves": 2.0, "warp_strength": 0.5, "phase_shift": 0.5,
        "shape_type": "none", "shape_strength": 1.0, "color_scheme": "none",
        "color_intensity": 0.8, "time": 0.0, "base_seed": 8888, **overrides,
    }).validate()


def _field(coords):
    def field(seed):
        return fbm(coords * 2.0, 2, seed)
    return field


@pytest.mark.parametrize("shape_type", ["none", "radial"])
def test_draw_channels_keeps_channel_zero_on_the_scalar_path(shape_type):
    coords = create_coordinate_grid(1, 22, 39, CPU)
    params = _params(shape_type=shape_type)
    wide = draw_channels(_field(coords), coords, params, 8888, 24)
    single = draw_channels(_field(coords), coords, params, 8888, 1)
    assert tuple(wide.shape) == (1, 24, 22, 39)
    assert torch.equal(wide[:, :1], single)
    assert wide.abs().max() <= 1.0
    # Every other channel is the draw its own seed would give, drawn alone.
    for channel in (1, 7):
        alone = draw_channels(_field(coords), coords, params, 8888 + 6151 * channel, 1)
        assert torch.equal(wide[:, channel:channel + 1], alone)


def test_draw_channels_does_not_touch_the_global_rng(monkeypatch):
    coords = create_coordinate_grid(1, 16, 16, CPU)
    monkeypatch.setattr(torch, "manual_seed", lambda *a, **k: pytest.fail("reseeded the global RNG"))
    draw_channels(_field(coords), coords, _params(), 8888, 8)


def test_palette_channels_maps_one_field_to_three():
    field = torch.linspace(-1.0, 1.0, 64).view(1, 1, 8, 8)
    assert torch.equal(BaseNoiseGenerator.palette_channels(field, _params()), field)
    coloured = BaseNoiseGenerator.palette_channels(field, _params(color_scheme="viridis"))
    assert tuple(coloured.shape) == (1, 3, 8, 8)
    assert coloured.abs().max() <= 1.0
    # The three are a palette of one field, so they are not copies of each other.
    assert not torch.equal(coloured[:, 0], coloured[:, 1])
    rainbow = BaseNoiseGenerator.palette_channels(field, _params(color_scheme="rainbow"))
    assert tuple(rainbow.shape) == (1, 3, 8, 8)
