"""
White noise: the control.

Every other type draws a structured field. This one draws exactly what the
sampler already starts from, an independent Gaussian field per channel, so
blending it in moves the run toward another seed's neighbourhood without adding
any structure of its own. That isolates what shader_strength and the blend mode
do by themselves, which is what a control is for.

Nothing spatial reaches it: noise_scale, octaves, warp_strength, phase_shift and
colour schemes do nothing, and the tooltip says so. Shape masks still apply,
since a mask is a spatial shape imposed on whatever field it is given.

Time turns one white field into a second along a great circle, so holding the
seed and advancing it evolves the noise smoothly instead of redrawing it: at
time 0 the field is the first draw exactly, at time 1 the second.
"""
import math

import torch

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..core.constants import DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid
from ..utils.shape_masks import apply_shape_mask

# Channel seed stride, shared with shaders/base.py so the conventions agree.
_CHANNEL_SEED_STRIDE = 6151


@shader_generator("gaussian", metadata={
    "description": "White noise, an independent Gaussian field per channel: the control",
    "supports_temporal": True,
})
class GaussianNoiseGenerator(BaseNoiseGenerator):
    """An independent Gaussian field per channel, with no spatial structure."""

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
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        current_seed = base_seed if params.use_temporal_coherence else seed

        # One CPU generator per channel, seeded the way shaders/base.py seeds its
        # extra channels, so channel 0 of a wide draw is the one-channel draw at
        # the same seed; shaders/spectral.py records why one wide randn is not.
        angle = time * math.pi / 2.0
        fields = []
        for channel in range(target_channels):
            channel_generator = torch.Generator(device="cpu").manual_seed(
                int(current_seed) + _CHANNEL_SEED_STRIDE * channel)
            first = torch.randn(batch_size, 1, height, width, generator=channel_generator)
            second = torch.randn(batch_size, 1, height, width, generator=channel_generator)
            fields.append(first * math.cos(angle) + second * math.sin(angle))
        field = torch.cat(fields, dim=1).to(device)
        mean = field.mean(dim=(-2, -1), keepdim=True)
        std = field.std(dim=(-2, -1), keepdim=True)
        field = (field - mean) / (std + 1e-8)

        shape_type = params.shape_type
        shape_strength = params.shape_strength
        if shape_type not in ["none", "0"] and shape_strength > 0:
            coords = create_coordinate_grid(batch_size, height, width, device)
            mask = apply_shape_mask(coords, shape_type, time, current_seed, shape_strength)
            field = torch.lerp(field, field * mask.permute(0, 3, 1, 2), shape_strength)

        # Same convention as the other generators: standardised, halved, clamped.
        return torch.clamp(field * 0.5, -1.0, 1.0)


def generate_gaussian_tensor(batch_size, height, width, params, device, seed=0,
                             target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return GaussianNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
