"""
An FBM whose detail varies across the frame.

A plain FBM has one persistence everywhere, so every part of the frame is
equally rough. Here a slow control field sets the persistence per pixel, so
some regions keep their fine layers and others lose them: rough patches and
smooth ones in the same draw. That is the matrix's design, and the legacy
implementation's, which weighted each layer by a low-frequency field.

warp_strength is how far the rough and smooth patches sit from the middle
(0 is plain FBM at persistence 0.5); phase_shift is how much of the frame is
rough (0 mostly smooth, 2 mostly rough). Both need a second layer to act on,
so at octaves 1 the type is plain simplex noise. On a clip the control and the
layers are evaluated in 3D with time as the third axis.
"""
import torch

from .base import BaseNoiseGenerator
from .fbm import MAX_OCTAVES, draw_channels, time_is_an_axis, with_time
from .registry import shader_generator
from .simplex import simplex_2d, simplex_3d_full
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Simplex cells across the frame at noise_scale 1.
CELLS_PER_UNIT = 2.0
# The control field's frequency relative to the base layer.
CONTROL_FREQUENCY = 0.5
# How much of warp_strength becomes persistence swing, and its ceiling.
HETERO_GAIN = 0.7
MAX_HETERO = 1.0
MIN_PERSISTENCE, MAX_PERSISTENCE = 0.15, 0.9


@shader_generator("heterogeneous_fbm", metadata={
    "description": "FBM whose persistence varies across the frame: rough patches and smooth ones",
    "supports_temporal": True,
})
class HeterogeneousFBMGenerator(BaseNoiseGenerator):
    """Fractal Brownian motion with a per-pixel persistence."""

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
        hetero = min(HETERO_GAIN * float(params.warp_strength), MAX_HETERO)
        bias = float(params.phase_shift) - 0.5
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * (scale * CELLS_PER_UNIT)
        z = time if time_is_an_axis(params.get("time_axis", None), time) else None

        def layer(q, layer_seed, depth):
            if z is None:
                return simplex_2d(q, layer_seed)
            return simplex_3d_full(with_time(q, z * depth), layer_seed)

        def field(channel_seed):
            control = layer(p * CONTROL_FREQUENCY, channel_seed + 500, 0.2)
            persistence = torch.clamp(0.5 + hetero * (control + bias), MIN_PERSISTENCE, MAX_PERSISTENCE)
            total = None
            amp, freq, norm = 1.0, 1.0, 0.0
            for i in range(min(max(octaves, 1), MAX_OCTAVES)):
                current = layer(p * freq, channel_seed + i, 0.2 + 0.05 * i)
                total = amp * current if total is None else total + amp * current
                norm = norm + amp
                # A tensor from the second layer on: each pixel keeps its own share.
                amp = amp * persistence
                freq *= 2.0
            return total / norm

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_heterogeneous_fbm_tensor(batch_size, height, width, params, device, seed=0,
                                      target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return HeterogeneousFBMGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
