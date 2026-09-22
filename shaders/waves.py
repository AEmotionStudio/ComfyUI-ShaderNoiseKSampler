"""
Plane waves, summed.

octaves seeded sine waves, each with its own direction, frequency, phase and
drift rate drawn from lattice_hash, so there is no RNG and a tensor of seeds
draws the whole channel axis in one call. Their sum is an interference pattern
of straight wavefronts: ripples at one wave, moire at several. A fractional
octave count blends the sum with and without the next wave, which fades it in.

warp_strength bends the wavefronts with a low-frequency simplex field; at 0
they are straight. phase_shift shifts each wave's phase by a different amount,
so it is a different interference pattern from the same waves. time advances
each wave at its own rate, so holding the seed and advancing it moves the
pattern rather than redrawing it, the way spectral's modes drift.
"""
import math

import torch

from .base import BaseNoiseGenerator
from .fbm import MAX_OCTAVES, draw_channels, simplex_warp
from .registry import shader_generator
from .simplex import lattice_hash
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid

# Cycles of the first wave across the frame at noise_scale 1.
WAVES_PER_UNIT = 3.0
# The warp field's frequency and how much of warp_strength reaches it: a
# little bends straight wavefronts a lot.
WARP_FREQUENCY = 1.0
WARP_GAIN = 0.15
# Cycles each wave drifts over a clip, before its own rate scales it.
DRIFT_PER_TIME = 0.25


@shader_generator("waves", metadata={
    "description": "octaves seeded plane waves summed; straight at warp 0, bent above it",
    "supports_temporal": True,
})
class WavesNoiseGenerator(BaseNoiseGenerator):
    """A sum of seeded plane waves."""

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
        count = min(max(int(params.octaves), 1), MAX_OCTAVES)
        warp_strength = float(params.warp_strength)
        phase_shift = float(params.phase_shift)
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        coords = create_coordinate_grid(batch_size, height, width, device)
        p = coords * scale

        def field(channel_seed):
            q = simplex_warp(p, channel_seed, warp_strength * WARP_GAIN, WARP_FREQUENCY)
            total = None
            for i in range(count):
                # One hash per wave parameter, at lattice point (wave, slot).
                def draw(slot):
                    return lattice_hash(torch.tensor(i, device=device), torch.tensor(slot, device=device),
                                        channel_seed)
                angle = draw(1) * math.pi
                frequency = WAVES_PER_UNIT * (1.0 + 0.5 * i + 0.5 * draw(2))
                phase = draw(3) * (2.0 * math.pi) + phase_shift * math.pi * (i + 1)
                rate = DRIFT_PER_TIME * (0.5 + draw(4))
                along = q[..., 0:1] * torch.cos(angle) + q[..., 1:2] * torch.sin(angle)
                wave = torch.sin(2.0 * math.pi * (frequency * along + rate * time) + phase)
                total = wave if total is None else total + wave
            return total / math.sqrt(count)

        return draw_channels(field, coords, params, current_seed, target_channels)


def generate_waves_tensor(batch_size, height, width, params, device, seed=0,
                          target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return WavesNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
