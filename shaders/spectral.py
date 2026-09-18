"""
Noise built in the frequency domain instead of pixel by pixel.

The other four generators evaluate a procedural field per pixel per channel, so
a wide latent costs one render per channel. This one draws the whole channel
stack's Fourier coefficients at once, shapes their amplitude by a radial
envelope, and takes a single inverse real FFT: O(N log N) for the lot, and the
channel axis is nearly free. At MiniMax H3's default latent that is about 90 ms
against domain_warp's 5 to 8 seconds.

**It is a different instrument, not a faster domain_warp.** A shaped-Gaussian
field is a cloud: no filaments, no swirls, and colour schemes mean nothing to it,
because there is no vector field to map onto a palette. What it does have is
direct control over the one property the model actually reads. The measurements
in HANDOFF found that the model settles composition from the noise's large-scale
structure and that the shader's effect is mostly large-scale structure; the
video-diffusion literature reaches the same place from the other side (FreeInit,
ECCV 2024, and FreqPrior, 2025, both find the low-frequency band of the initial
noise determines the outcome, and PYoCo, ICCV 2023, builds noise correlated
across frames). Here that band is a parameter rather than a side effect.

The envelope is `(1 + (|k|/corner)^2)^(-beta/2)`, a soft shelf rather than a
`|k|^-beta` power law, because a power law has a pole at DC: the zero-frequency
coefficient would swamp everything and every channel would come out a flat wash.

Bandwidth and channel rank are the same dial, which is worth understanding before
reaching for a very low corner. Every channel is an independent draw, but they
all live in the subspace the envelope leaves open, so once the latent has more
channels than the envelope has usable modes the rank is capped by the bandwidth
rather than by the channel count. Measured at 128 channels over 32x32: corner
0.30 spans about 118 of 128, corner 0.05 with beta 2.0 only 44. The defaults sit
where the field is still clearly large-scale and the rank still holds up.
"""
import logging
import math

import torch

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..core.constants import CHANNEL_BASIS, DEFAULT_CHANNELS
from ..core.params import ShaderParams
from ..utils.noise_utils import create_coordinate_grid
from ..utils.shape_masks import apply_shape_mask

logger = logging.getLogger(__name__)

# The band the defaults sit in, as a fraction of Nyquist, before noise_scale moves it.
BASE_CORNER = 0.15
# Slope at octaves 1, and how much each further octave flattens it. More octaves
# means more fine detail in an FBM, so here it means a shallower roll-off.
BASE_SLOPE = 2.5
SLOPE_PER_OCTAVE = 0.25
MIN_SLOPE, MAX_SLOPE = 0.5, 3.0
MIN_CORNER, MAX_CORNER = 0.02, 0.5
# Channel seed stride, shared with shaders/base.py so the conventions agree.
_CHANNEL_SEED_STRIDE = 6151


@shader_generator("spectral", metadata={
    "description": "Frequency-domain noise: one inverse FFT for the whole channel stack",
    "supports_temporal": True,
})
class SpectralNoiseGenerator(BaseNoiseGenerator):
    """Shaped Gaussian noise synthesised from its spectrum."""

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
        octaves = float(params.octaves)
        warp_strength = float(params.warp_strength)
        phase_shift = float(params.phase_shift)
        time = float(params.time)
        base_seed = params.get("base_seed", seed)
        use_temporal_coherence = params.use_temporal_coherence

        # noise_scale means the same thing as everywhere else in this pack: low is
        # large, zoomed-in features, high is small, zoomed-out ones. Here that is
        # literally where the band sits.
        corner = min(max(BASE_CORNER * scale, MIN_CORNER), MAX_CORNER)
        slope = min(max(BASE_SLOPE - SLOPE_PER_OCTAVE * (octaves - 1.0), MIN_SLOPE), MAX_SLOPE)
        corner = max(corner, SpectralNoiseGenerator._corner_floor(height, width))

        current_seed = base_seed if use_temporal_coherence else seed
        field = SpectralNoiseGenerator._synthesise(
            batch_size, target_channels, height, width, device,
            int(current_seed), corner, slope, warp_strength, phase_shift, time)

        shape_type = params.shape_type
        shape_strength = params.shape_strength
        if shape_type not in ["none", "0"] and shape_strength > 0:
            try:
                coords = create_coordinate_grid(batch_size, height, width, device)
                mask_seed = base_seed if use_temporal_coherence else seed
                mask = apply_shape_mask(coords, shape_type, time, mask_seed, shape_strength)
                mask = mask.permute(0, 3, 1, 2) if mask.dim() == 4 else mask.unsqueeze(1)
                field = torch.lerp(field, field * mask, shape_strength)
            except Exception as exc:
                logger.warning(f"Error applying shape mask: {exc}")

        # Same convention as the other generators: standardised, halved, clamped,
        # so shader_strength means what it means everywhere else.
        return torch.clamp(field * 0.5, -1.0, 1.0)

    @staticmethod
    def _corner_floor(height, width):
        """
        The narrowest band that can still carry the widest basis the pack asks for.

        Every channel is an independent draw but they all live in whatever subspace
        the envelope leaves open, so a latent with more channels than the band has
        modes comes back narrower than it looks -- the failure mode the whole
        channel-rank effort was about. The number of modes inside a disc of radius r
        is about `pi * r^2 * height * width`, so requiring that to reach
        CHANNEL_BASIS gives the floor below.

        It is deliberately CHANNEL_BASIS and not `target_channels`: making the
        envelope depend on how many channels this particular call asked for would
        mean channel 0 of a wide draw was drawn under a different envelope than a
        one-channel draw at the same seed, and core.shader_noise builds every
        travel-mode basis from one-channel draws. `jump` and `stamp` would stop
        agreeing with `walk` about what the field is.

        A no-op at most shapes -- 64x64 wants 0.070, against the 0.15 default -- and
        it opens the band only where the grid is small enough to be the binding
        constraint: 22x38 wants 0.156, 8x8 wants more than Nyquist and gets it.
        """
        modes = math.pi * max(height * width, 1)
        return min(math.sqrt(CHANNEL_BASIS / modes), 0.5)

    @staticmethod
    def _synthesise(batch, channels, height, width, device, seed,
                    corner, slope, anisotropy, phase_shift, time):
        """Draw the spectrum, shape it, and bring it back with one inverse FFT."""
        bins = width // 2 + 1

        # One draw holding real and imaginary parts together, so that channel 0 of a
        # wide request is the same field as a one-channel request at the same seed.
        # Two separate randn calls would not be: the second would start at a
        # different offset in the stream. core.shader_noise builds every travel-mode
        # basis from one-channel draws, and a test pins the identity.
        # One generator per channel, seeded the way shaders/base.py seeds its extra
        # channels. The obvious version -- one draw of shape [B, C, H, bins, 2] --
        # looks like it would give channel 0 the same values whatever C is, and it
        # does not: torch's CPU normal fill works in blocks of 16, so the first
        # channel of a wide draw only matches a one-channel draw when the per-channel
        # element count happens to be a multiple of 16. It was for the coefficients
        # at one shape and not for the drift rates, which made channel 0 agree on
        # frame 0 and diverge on every frame after it. Seeding per channel does not
        # depend on any of that, and C small draws cost nothing beside the FFT.
        parts, rates = [], []
        for channel in range(channels):
            channel_generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + _CHANNEL_SEED_STRIDE * channel)
            parts.append(torch.randn(batch, 1, height, bins, 2, generator=channel_generator))
            rates.append(torch.randn(1, 1, height, bins, generator=channel_generator))
        parts = torch.cat(parts, dim=1).to(device)
        rates = torch.cat(rates, dim=1).to(device)

        fy = torch.fft.fftfreq(height, device=device).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device).view(1, bins)
        # warp_strength stretches the band along x against y, so the field grows a
        # direction the way the other generators' warp does.
        stretch = 1.0 + anisotropy
        radius = ((fy * stretch) ** 2 + (fx / stretch) ** 2).sqrt()
        envelope = (1.0 + (radius / corner) ** 2) ** (-slope / 2.0)

        # phase_shift turns every mode by a fixed angle and time turns each one at
        # its own rate, so holding the seed and advancing time evolves the same
        # field smoothly instead of redrawing it -- which is what temporal
        # coherence asks for.
        angle = phase_shift * math.pi + 2.0 * math.pi * time * rates
        rotation = torch.polar(torch.ones_like(angle), angle)
        spectrum = torch.complex(parts[..., 0], parts[..., 1]) * rotation * envelope

        field = torch.fft.irfft2(spectrum, s=(height, width))
        mean = field.mean(dim=(-2, -1), keepdim=True)
        std = field.std(dim=(-2, -1), keepdim=True)
        return (field - mean) / (std + 1e-8)


def generate_spectral_tensor(batch_size, height, width, params, device, seed=0,
                             target_channels=DEFAULT_CHANNELS):
    """Function form, matching the other generators' module-level entry points."""
    return SpectralNoiseGenerator.generate(
        batch_size, height, width, params, device, seed, target_channels)
