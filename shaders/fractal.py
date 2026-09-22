"""
The reference FBM: layered simplex noise and nothing else.

Every knob maps to one parameter of the FBM, which makes this the type to read
the others against. noise_scale is the base frequency; octaves the number of
layers; warp_strength the spacing between them -- a lacunarity dial is what an
FBM has where the others have a warp, from layers packed at 1.5x apart to the
classic doubling at the default and 4x at the top; phase_shift slides each
layer across the field by its own offset, so the layers come from different
parts of the same noise rather than sitting on top of each other. On a clip the
layers are evaluated in 3D with time as the third axis.
"""
import torch

from .base import BaseNoiseGenerator
from .fbm import draw_channels, fbm, time_is_an_axis
from .registry import shader_generator
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Simplex cells across the frame at noise_scale 1.
CELLS_PER_UNIT = 2.0
# Layer spacing at warp_strength 0, and its ceiling.
BASE_LACUNARITY = 1.5
MAX_LACUNARITY = 4.0
# How far phase_shift slides each successive layer, in cells.
LAYER_OFFSET = (0.37, 0.61)


@shader_generator("fractal", metadata={
    "description": "Layered simplex noise, the reference FBM",
    "supports_temporal": True,
})
class FractalNoiseGenerator(BaseNoiseGenerator):
    """Plain fractal Brownian motion over the shared simplex."""

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
        lacunarity = min(BASE_LACUNARITY + float(params.warp_strength), MAX_LACUNARITY)
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)
        z = time if time_is_an_axis(params.get("time_axis", None), time) else None
        offset = torch.tensor(LAYER_OFFSET, device=device) * float(params.phase_shift)

        def field(channel_seed):
            return fbm(p, octaves, channel_seed, lacunarity=lacunarity, time=z, octave_offset=offset)

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_fractal_tensor(batch_size, height, width, params, device, seed=0,
                            target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return FractalNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
