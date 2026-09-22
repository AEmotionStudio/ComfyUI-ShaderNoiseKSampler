"""
Worley cells.

Every lattice cell holds one feature point, placed by lattice_hash, and each
pixel measures its distance to the nearest (F1) and second-nearest (F2) point
over the surrounding cells. That is Worley noise, and the classic patterns
are combinations of the two: F1 is cell interiors, F2 - F1 is cell edges,
F1 * F2 a finer, bubblier look. octaves picks the pattern, so the node's
default of 1 is F1 and a fractional value blends two patterns, the way
domain_warp's warp type already does.

phase_shift is the distance metric, Minkowski with exponent 1 + 2*phase: 0 is
Manhattan and the cells are diamonds, 0.5 is Euclidean and they are round, 2
is close to Chebyshev and they are square. warp_strength bends the lattice
with a low-frequency simplex field before the cells are measured. On a clip
the cells live in 3D and time moves the plane through them.

F1 and F2 - F1 are one-sided and right-skewed; standardising centres them and
keeps the skew and the sharp edges, which is the point of the type.
"""
import itertools

import torch

from .base import BaseNoiseGenerator
from .fbm import draw_channels, simplex_warp, time_is_an_axis, with_time
from .registry import shader_generator
from .simplex import lattice_hash
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Cells across the frame at noise_scale 1: four, so that scale 1 is a few
# cells rather than a single radial gradient.
CELLS_PER_UNIT = 4.0
# The warp field's frequency relative to the cells, and how much of
# warp_strength reaches it.
WARP_FREQUENCY = 0.5
WARP_GAIN = 0.5
# Cells the plane moves through over a clip.
DEPTH_PER_TIME = 1.0
# Seed offsets for the feature point's jitter along each axis.
JITTER_SEEDS = (0, 4099, 8191)


def worley(p, seed, exponent):
    """
    F1 and F2 over `p[..., :2]` or `p[..., :3]`, one feature point per cell.

    `exponent` is the Minkowski exponent as a Python float, so the batched and
    scalar paths run the same kernel. The Euclidean case skips the powers.
    """
    dims = min(p.shape[-1], 3)
    cell = torch.floor(p[..., :dims])
    index = [cell[..., k:k + 1].long() for k in range(dims)]
    frac = [p[..., k:k + 1] - cell[..., k:k + 1] for k in range(dims)]

    def jitter(neighbour, axis):
        return lattice_hash(neighbour[0], neighbour[1], seed + JITTER_SEEDS[axis],
                            neighbour[2] if dims == 3 else None)

    f1 = f2 = None
    for offset in itertools.product((-1, 0, 1), repeat=dims):
        neighbour = [i + o for i, o in zip(index, offset)]
        distance = None
        for axis, o in enumerate(offset):
            delta = o + jitter(neighbour, axis) - frac[axis]
            term = delta * delta if exponent == 2.0 else delta.abs() ** exponent
            distance = term if distance is None else distance + term
        if f1 is None:
            f1, f2 = distance, torch.full_like(distance, 64.0)
        else:
            f2 = torch.where(distance < f1, f1, torch.minimum(f2, distance))
            f1 = torch.minimum(f1, distance)
    if exponent == 2.0:
        return f1.sqrt(), f2.sqrt()
    return f1 ** (1.0 / exponent), f2 ** (1.0 / exponent)


@shader_generator("cellular", metadata={
    "description": "Worley cells; octaves picks F1, F2, edges or product, phase_shift the cell shape",
    "supports_temporal": True,
})
class CellularNoiseGenerator(BaseNoiseGenerator):
    """Worley noise in its four classic patterns."""

    @staticmethod
    def generate(
        batch_size: int,
        height: int,
        width: int,
        params: ShaderParams,
        device: torch.device,
        seed: int = 0,
        target_channels: int = DEFAULT_CHANNELS,
    ) -> torch.Tensor:
        scale = float(params.scale)
        pattern = (max(int(params.octaves), 1) - 1) % 4
        warp_strength = float(params.warp_strength)
        exponent = 1.0 + 2.0 * float(params.phase_shift)
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)
        z = time * DEPTH_PER_TIME if time_is_an_axis(params.get("time_axis", None), time) else None

        def field(channel_seed):
            q = simplex_warp(p, channel_seed, warp_strength * WARP_GAIN, WARP_FREQUENCY)
            if z is not None:
                q = with_time(q, z)
            f1, f2 = worley(q, channel_seed, exponent)
            return (f1, f2, f2 - f1, f1 * f2)[pattern]

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_cellular_tensor(batch_size, height, width, params, device, seed=0,
                             target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return CellularNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
