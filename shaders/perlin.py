"""
Classic gradient noise, layered.

Perlin noise puts a random gradient at every lattice point and blends the dot
products with a quintic fade, which gives a smoother, more lattice-like field
than simplex: features line up with the grid a little, and the zero crossings
sit on it. The gradients come from lattice_hash, so a tensor of seeds draws the
whole channel axis in one call like every other hash-based type.

warp_strength is a low-frequency simplex displacement applied to the detail
layers only, the design the matrix documents: the base layer keeps its shape
and the detail swirls around it. phase_shift is contrast, as it is in
domain_warp. On a clip the layers are evaluated in 3D with time as the third
axis, from a 3D Perlin over the same gradient table the simplex uses.
"""
import math

import torch
import torch.nn.functional as F

from .base import BaseNoiseGenerator
from .fbm import MAX_OCTAVES, draw_channels, simplex_warp, time_is_an_axis, with_time
from .registry import shader_generator
from .simplex import SIMPLEX_GRADIENTS, lattice_hash
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Lattice cells across the frame at noise_scale 1. Three rather than the two the
# simplex types use: a 3x3 lattice of gradients is too little variety per
# channel, and 24 channels spanned only 12 directions at 22x38 with it.
CELLS_PER_UNIT = 3.0
# The warp field's frequency relative to the base layer, and how much of
# warp_strength reaches it.
WARP_FREQUENCY = 0.5
WARP_GAIN = 0.5


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin_2d(p, seed):
    """2D gradient noise over `p[..., :2]`, zero on the lattice, roughly in [-1, 1]."""
    cell = torch.floor(p[..., :2])
    fx, fy = p[..., 0:1] - cell[..., 0:1], p[..., 1:2] - cell[..., 1:2]
    ix, iy = cell[..., 0:1].long(), cell[..., 1:2].long()
    u, v = _fade(fx), _fade(fy)

    def corner(dx, dy):
        angle = lattice_hash(ix + dx, iy + dy, seed) * (2.0 * math.pi)
        return torch.cos(angle) * (fx - dx) + torch.sin(angle) * (fy - dy)

    n00, n10, n01, n11 = corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * math.sqrt(2.0)


def perlin_3d(p, seed):
    """3D gradient noise over `p[..., :3]`, gradients from the simplex table."""
    cell = torch.floor(p[..., :3])
    fx, fy, fz = (p[..., k:k + 1] - cell[..., k:k + 1] for k in range(3))
    ix, iy, iz = (cell[..., k:k + 1].long() for k in range(3))
    u, v, w = _fade(fx), _fade(fy), _fade(fz)
    gradients = SIMPLEX_GRADIENTS.to(p.device)

    def corner(dx, dy, dz):
        index = (lattice_hash(ix + dx, iy + dy, seed, iz + dz) * 12.0).long() % 12
        grads = F.embedding(index, gradients)
        return grads[..., 0] * (fx - dx) + grads[..., 1] * (fy - dy) + grads[..., 2] * (fz - dz)

    def along_x(dy, dz):
        a, b = corner(0, dy, dz), corner(1, dy, dz)
        return a + u * (b - a)

    ny0 = along_x(0, 0) + v * (along_x(1, 0) - along_x(0, 0))
    ny1 = along_x(0, 1) + v * (along_x(1, 1) - along_x(0, 1))
    return ny0 + w * (ny1 - ny0)


@shader_generator("perlin", metadata={
    "description": "Classic gradient noise, layered, with a warp on the detail layers",
    "supports_temporal": True,
})
class PerlinNoiseGenerator(BaseNoiseGenerator):
    """Layered Perlin gradient noise."""

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
        octaves = int(params.octaves)
        warp_strength = float(params.warp_strength)
        contrast = 1.0 + float(params.phase_shift) * 0.5
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)
        z = time if time_is_an_axis(params.get("time_axis", None), time) else None

        def field(channel_seed):
            warped = simplex_warp(p, channel_seed + 1000, warp_strength * WARP_GAIN, WARP_FREQUENCY)
            total = None
            amp, freq, norm = 1.0, 1.0, 0.0
            for i in range(min(max(octaves, 1), MAX_OCTAVES)):
                q = (p if i == 0 else warped) * freq
                if z is None:
                    layer = perlin_2d(q, channel_seed + i)
                else:
                    layer = perlin_3d(with_time(q, z * (0.2 + 0.05 * i)), channel_seed + i)
                total = amp * layer if total is None else total + amp * layer
                norm += amp
                amp *= 0.5
                freq *= 2.0
            return total / norm

        return draw_channels(field, coords, params, current_seed, target_channels, contrast=contrast)


def generate_perlin_tensor(batch_size, height, width, params, device, seed=0,
                           target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return PerlinNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
