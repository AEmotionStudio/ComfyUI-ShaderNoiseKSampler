"""
Two FBMs turned into fringes.

Wave interference is the superposition of two fields: where they agree the
result reinforces, where they disagree it cancels. Here the two are FBMs at
different scales and seeds, each passed through a cosine or a sine so that its
level sets become fringes, and the fringes of one cross the fringes of the
other. The result is banded and moire-like, with a bimodal value distribution
instead of the bell of a plain FBM, and standardising keeps that shape -- it is
the character of the type.

The matrix's formula reads cos(pi*f1) + sin(pi*f2); an FBM normalised to
[-1, 1] rarely leaves [-0.5, 0.5], so straight into a cosine it stays near 1
and shows no fringes. A gain in front of it fixes that, and warp_strength is
that gain: fringe density, from soft folds to dense moire. phase_shift is the
relative frequency of the second field. On a clip both FBMs are evaluated in 3D.
"""
import math

import torch

from .base import BaseNoiseGenerator
from .fbm import draw_channels, fbm, time_is_an_axis
from .registry import shader_generator
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Simplex cells across the frame at noise_scale 1.
CELLS_PER_UNIT = 2.0
# The second field: its scale against the first, and where in the noise it sits.
SECOND_SCALE = 1.5
SECOND_OFFSET = 17.3
# Half-turns of fringe per unit of field at warp_strength 0, and per unit of
# warp_strength. Calibrated on a 64x64 draw: at 0 the fringes are still coarser
# than four pixels (lag-4 correlation 0.51 against fractal's 0.69), at 5 they
# are dense moire. Twice this made the default field white within four pixels.
BASE_GAIN = 0.5
GAIN_PER_WARP = 0.5


@shader_generator("interference", metadata={
    "description": "Two FBMs turned into crossing cos/sin fringes: banded, moire-like",
    "supports_temporal": True,
})
class InterferenceNoiseGenerator(BaseNoiseGenerator):
    """Crossing fringes from two fractal fields."""

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
        gain = math.pi * (BASE_GAIN + GAIN_PER_WARP * float(params.warp_strength))
        second_rate = 1.0 + 0.2 * float(params.phase_shift)
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)
        z = time if time_is_an_axis(params.get("time_axis", None), time) else None

        def field(channel_seed):
            first = fbm(p, octaves, channel_seed, time=z)
            second = fbm(p * SECOND_SCALE + SECOND_OFFSET, max(1, octaves - 1), channel_seed + 10,
                         persistence=0.6, lacunarity=1.8, time=z)
            return torch.cos(gain * first) + torch.sin(gain * second_rate * second)

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_interference_tensor(batch_size, height, width, params, device, seed=0,
                                 target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return InterferenceNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
