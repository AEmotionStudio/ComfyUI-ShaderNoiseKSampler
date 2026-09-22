"""
A slice through a 3D field.

The field is a four-corner 3D simplex FBM; the picture is a plane through it.
phase_shift is the depth of that plane, so it is the "different facets of the
same field" knob literally: nearby depths give nearby pictures, distant ones
unrelated pictures. time slides the plane through the volume, which makes the
type coherent across frames by construction -- there is no 2D path to fall
back to, and a single image is simply the slice at depth phase_shift.
warp_strength bends the plane with a low-frequency simplex field before it
cuts the volume, the legacy implementation's pre-warp.
"""
import torch

from .base import BaseNoiseGenerator
from .fbm import MAX_OCTAVES, draw_channels, simplex_warp, with_time
from .registry import shader_generator
from .simplex import simplex_3d_full
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Simplex cells across the frame at noise_scale 1.
CELLS_PER_UNIT = 2.0
# The pre-warp's frequency relative to the base layer, and how much of
# warp_strength reaches it.
WARP_FREQUENCY = 0.3
WARP_GAIN = 0.5
# Depth per unit of phase_shift: the picture is unrelated half a cell away, so
# the slider passes through about four facets over its range. Depth per unit
# of time: adjacent frames of an 8-frame clip correlate at 0.84 with the seed
# held, between spectral's 0.68 and domain_warp's 0.96; at twice this they
# fell to 0.63. And the extra depth each layer sits at, so the layers do not
# share their zero crossings.
DEPTH_PER_PHASE = 1.0
DEPTH_PER_TIME = 0.5
LAYER_DEPTH_STEP = 1.5


@shader_generator("projection_3d", metadata={
    "description": "A plane through a 3D simplex FBM; phase_shift is the depth, time slides it",
    "supports_temporal": True,
})
class Projection3DNoiseGenerator(BaseNoiseGenerator):
    """A 2D slice of a 3D fractal field."""

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
        depth = float(params.phase_shift) * DEPTH_PER_PHASE + float(params.time) * DEPTH_PER_TIME
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)

        def field(channel_seed):
            plane = simplex_warp(p, channel_seed + 1234, warp_strength * WARP_GAIN, WARP_FREQUENCY)
            total = None
            amp, freq, norm = 1.0, 1.0, 0.0
            for i in range(min(max(octaves, 1), MAX_OCTAVES)):
                layer = simplex_3d_full(with_time(plane * freq, depth * freq + LAYER_DEPTH_STEP * i),
                                        channel_seed + i)
                total = amp * layer if total is None else total + amp * layer
                norm += amp
                amp *= 0.5
                freq *= 2.0
            return total / norm

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_projection_3d_tensor(batch_size, height, width, params, device, seed=0,
                                  target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return Projection3DNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
