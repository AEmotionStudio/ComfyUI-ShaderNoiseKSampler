"""
A field folded onto itself: mirrored, or turned N ways round the centre.

The coordinates are folded before the noise is drawn, so the symmetry is the grid's
own, good to float rounding: a field and its mirror image correlate 1.0000.
Underneath is a three-layer FBM, the same one fractal draws.

octaves is the order of the fold rather than a layer count, the way cellular
spends it on a pattern: 1 mirrors left to right, 2 mirrors both ways, and from 3
up the frame is cut into that many wedges, each a mirror of its neighbour.
phase_shift turns the fold about the centre; 0.5 is upright. warp_strength bends
the pattern inside the fold, which keeps the symmetry, and noise_scale is the size
of its features. On a clip the FBM is evaluated in 3D with time as the third axis.

Two things to know before reaching for a high order. A fold of order N leaves one
part in 2N of the frame free, so the channels have less room to differ and the draw
spans fewer of them: 12.7 of 16 at order 1, 10.6 at 2 and 7.9 at 6, on a 128x128
latent. That is the direction that makes a shader push harder. And stage_progression
shifts octaves by one between stages, so it changes the order mid-run.

Only the shader's share of the noise is symmetric; the base noise it is mixed into
is not. Measured on Krea 2 at 1024x1024, two seeds, multiply: under 0.2 nothing
shows, and the picture correlates with its own mirror image no more than an unshaded
one does (-0.09). From about 0.3 it comes back symmetric at orders 2 and 6 on both
seeds, a subject centred on a mirrored scene, with that correlation at 0.23 to 0.84.
A single mirror, order 1, took on one seed and on the other only at 0.3.

It is an FBM underneath and it fails the way that family does. One of the two seeds
laid rainbow bands across the background at orders 1 and 2 from 0.2 up, symmetric
ones, with the subject intact in front of them. Order 6 did not.
"""
import math

import torch

from .base import BaseNoiseGenerator
from .fbm import MAX_OCTAVES, draw_channels, fbm, simplex_warp, time_is_an_axis
from .registry import shader_generator
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Simplex cells across the short side of the frame at noise_scale 1, and the FBM's layers.
CELLS_PER_UNIT = 3.0
LAYERS = 3
# The warp field's frequency and how much of warp_strength reaches it.
WARP_FREQUENCY = 1.0
WARP_GAIN = 0.15


def fold(coords, order, turn):
    """
    Fold [B, H, W, 2] unit coordinates about the centre of the frame.

    The short side spans one unit, so a wedge is the same angle on a wide latent as
    on a square one.
    """
    height, width = coords.shape[1:3]
    short = min(height, width)
    x = (coords[..., 0:1] - 0.5) * (width / short)
    y = (coords[..., 1:2] - 0.5) * (height / short)
    x, y = x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn)
    if order == 1:
        return torch.cat([x.abs(), y], dim=-1)
    if order == 2:
        return torch.cat([x.abs(), y.abs()], dim=-1)
    wedge = 2.0 * math.pi / order
    angle = (torch.remainder(torch.atan2(y, x), wedge) - wedge / 2.0).abs()
    radius = (x ** 2 + y ** 2).sqrt()
    return torch.cat([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


@shader_generator("kaleido", metadata={
    "description": "An FBM folded about the centre; octaves is the order of the fold",
    "supports_temporal": True,
})
class KaleidoNoiseGenerator(BaseNoiseGenerator):
    """Fractal noise over mirrored or N-fold coordinates."""

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
        order = min(max(int(params.octaves), 1), MAX_OCTAVES)
        warp_strength = float(params.warp_strength)
        turn = (float(params.phase_shift) - 0.5) * math.pi
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = fold(coords, order, turn) * (scale * CELLS_PER_UNIT)
        z = time if time_is_an_axis(params.get("time_axis", None), time) else None

        def field(channel_seed):
            q = simplex_warp(p, channel_seed, warp_strength * WARP_GAIN, WARP_FREQUENCY)
            return fbm(q, LAYERS, channel_seed, time=z)

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_kaleido_tensor(batch_size, height, width, params, device, seed=0,
                            target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return KaleidoNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
