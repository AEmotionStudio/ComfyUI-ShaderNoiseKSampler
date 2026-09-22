"""
Base class for shader noise generators.

This module provides the abstract base class that all shader generators
must inherit from, ensuring consistent interface and shared functionality.
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional, Tuple

from ..utils.color_utils import apply_color_scheme, hsv_to_rgb, interpolate_colors, COLOR_SCHEMES
from ..utils.shape_masks import apply_shape_mask, apply_mask_to_tensor
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from ..core.constants import CHANNEL_BASIS, DEFAULT_CHANNELS

# Seed step between a draw's channels. Prime and unrelated to
# core.shader_noise._MIX_SEED_STRIDE, and checked against the mod-10000 seed
# hashing curl_noise does internally: no two of the first 64 channels land on
# the same internal seed.
_CHANNEL_SEED_STRIDE = 6151


class BaseNoiseGenerator(ABC):
    """
    Abstract base class for all shader noise generators.
    
    Provides common functionality for coordinate grid creation,
    shape mask application, and color scheme handling.
    """
    
    @staticmethod
    @abstractmethod
    def generate(
        batch_size: int,
        height: int,
        width: int,
        params: ShaderParams,
        device: torch.device,
        seed: int = 0,
        target_channels: int = DEFAULT_CHANNELS
    ) -> torch.Tensor:
        """
        Generate noise tensor.
        
        Args:
            batch_size: Number of images in batch
            height: Height of tensor
            width: Width of tensor
            params: Shader parameters
            device: Device to create tensor on
            seed: Random seed for deterministic results
            target_channels: Number of output channels
            
        Returns:
            Tensor with shape [batch_size, target_channels, height, width]
        """
        pass
    
    @staticmethod
    def create_coordinate_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        range_type: str = "unit"
    ) -> torch.Tensor:
        """
        Create a coordinate grid for noise generation.
        
        Args:
            batch_size: Number of batches
            height: Grid height
            width: Grid width
            device: Target device
            dtype: Data type
            range_type: "unit" for [0, 1], "centered" for [-0.5, 0.5], "symmetric" for [-1, 1]
            
        Returns:
            Coordinate tensor [B, H, W, 2] with (x, y) coordinates
        """
        return create_coordinate_grid(batch_size, height, width, device, dtype, range_type)
    
    @staticmethod
    def apply_shape_mask(
        noise: torch.Tensor,
        coords: torch.Tensor,
        params: ShaderParams
    ) -> torch.Tensor:
        """
        Apply shape mask to noise tensor.
        
        Args:
            noise: Input noise tensor [B, H, W, C] or [B, C, H, W]
            coords: Coordinate grid [B, H, W, 2]
            params: Shader parameters containing shape_type and shape_strength
            
        Returns:
            Masked noise tensor
        """
        shape_type = params.shape_type
        shape_strength = params.shape_strength
        time = params.time
        base_seed = params.get("base_seed", 0)
        
        if shape_type in ["none", "0"] or shape_strength <= 0:
            return noise
        
        # Generate shape mask
        mask = apply_shape_mask(coords, shape_type, time, base_seed, shape_strength)
        
        # Apply mask to noise
        return apply_mask_to_tensor(noise, mask, shape_strength)
    
    @staticmethod
    def apply_color_scheme(
        noise: torch.Tensor,
        params: ShaderParams,
        velocity_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply color scheme to noise tensor.
        
        Args:
            noise: Input noise tensor [B, C, H, W]
            params: Shader parameters containing color_scheme and color_intensity
            velocity_field: Optional velocity field for direction-based coloring [B, 2, H, W]
            
        Returns:
            Color-modified noise tensor
        """
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        time = params.time
        
        if color_scheme in ["none", "0"] or color_intensity <= 0:
            return noise
        
        return apply_color_scheme(noise, color_scheme, color_intensity, velocity_field, time)
    
    @staticmethod
    def palette_channels(field: torch.Tensor, params: ShaderParams) -> torch.Tensor:
        """
        Map one field [B, 1, H, W] onto the chosen colour scheme, as three channels.

        The palette half of domain_warp's _apply_color_variations, for the
        generators that draw a scalar field, without the alpha it also returned
        (the field itself again). Channel 0 comes back the same whether one channel
        or many were asked for, which is the identity every travel-mode basis rests
        on; the other two are correlated with it, exactly as domain_warp's are.
        """
        scheme = params.color_scheme
        intensity = params.color_intensity
        if scheme in ["none", "0"] or intensity <= 0:
            return field

        t = (field + 1.0) * 0.5
        if scheme in COLOR_SCHEMES:
            r, g, b = interpolate_colors([(s[0], s[1]) for s in COLOR_SCHEMES[scheme]], t, field.device)
        elif scheme in ("rainbow", "hsv"):
            r, g, b = hsv_to_rgb(t, torch.full_like(t, 0.8), torch.clamp(t + 0.2, 0.0, 1.0))
        else:
            return field
        colours = torch.cat([r, g, b], dim=1) * 2.0 - 1.0
        return torch.lerp(field.expand(-1, 3, -1, -1), colours, intensity)

    @staticmethod
    def apply_common_postprocessing(
        noise: torch.Tensor,
        params: ShaderParams,
        coords: torch.Tensor,
        velocity_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply common postprocessing steps (shape mask, color scheme).
        
        Args:
            noise: Input noise tensor [B, C, H, W]
            params: Shader parameters
            coords: Coordinate grid [B, H, W, 2]
            velocity_field: Optional velocity field for color schemes
            
        Returns:
            Postprocessed noise tensor
        """
        # Apply color scheme first (operates on channels)
        noise = BaseNoiseGenerator.apply_color_scheme(noise, params, velocity_field)
        
        # Apply shape mask using the coords parameter
        noise = BaseNoiseGenerator.apply_shape_mask(noise, coords, params)
        
        return noise
    
    @staticmethod
    def normalize_to_range(
        tensor: torch.Tensor,
        target_min: float = -1.0,
        target_max: float = 1.0
    ) -> torch.Tensor:
        """
        Normalize tensor to target range.
        
        Args:
            tensor: Input tensor
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized tensor
        """
        t_min = tensor.min()
        t_max = tensor.max()
        
        if t_max - t_min < 1e-8:
            # Avoid division by zero for constant tensors
            return torch.full_like(tensor, (target_min + target_max) / 2)
        
        normalized = (tensor - t_min) / (t_max - t_min)
        return normalized * (target_max - target_min) + target_min
    
    @staticmethod
    def fill_channels(
        render: Callable[[int], torch.Tensor],
        base: torch.Tensor,
        target_channels: int,
        seed: int,
        render_many=None,
    ) -> torch.Tensor:
        """
        Fill the channel axis with independent draws of the generator's field.

        A sampler expects every latent channel to carry its own noise. Building the
        extra channels out of the first one or two -- copying them, or passing them
        through sin and abs -- leaves the draw spanning about one channel however
        many it has: rank 1.00 for domain_warp at four. The shader then stops
        steering the sample and starts overwriting it.

        Args:
            render: draws one field [B, 1, H, W] at the seed it is given
            base: channels the generator has already drawn, kept as the first ones.
                Channel 0 is the generator's own draw at `seed`, so a one-channel
                request comes back exactly as it did before this existed -- which
                matters, because core.shader_noise builds the travel-mode basis
                from one-channel draws.
            target_channels: channels to return
            seed: the seed channel 0 was drawn with

        Returns:
            [B, target_channels, H, W]

        Channels up to CHANNEL_BASIS are each rendered. Past it, the rest are
        mixtures of those renders through a seeded orthogonal matrix, which keeps
        the mixtures from re-correlating what the renders kept apart.
        """
        if base.shape[1] >= target_channels:
            return base[:, :target_channels]

        seed = int(seed.item() if isinstance(seed, torch.Tensor) else seed)
        rendered = min(target_channels, max(CHANNEL_BASIS, base.shape[1]))

        # Generators reseed the global RNG inside each render. Forking leaves the
        # caller's RNG exactly where channel 0 left it.
        devices = [base.device.index if base.device.index is not None else torch.cuda.current_device()] \
            if base.device.type == "cuda" else []
        channel_seeds = [seed + _CHANNEL_SEED_STRIDE * c
                         for c in range(base.shape[1], rendered)]
        with torch.random.fork_rng(devices=devices):
            if render_many is not None and len(channel_seeds) > 1:
                # One call for the whole remaining channel axis. Generators that
                # offer this draw every field in one pass instead of one per
                # channel, which is most of what a wide latent costs.
                extra = [render_many(torch.tensor(channel_seeds, dtype=torch.int64,
                                                  device=base.device)).to(base)]
            else:
                # A single extra channel is not worth batching, and channel 0 never
                # comes through here at all -- it is `base`, drawn on the scalar
                # path, which is what keeps a one-channel draw and the travel-mode
                # bases they are built from unchanged.
                extra = [render(channel_seed).to(base) for channel_seed in channel_seeds]
        channels = torch.cat([base, *extra], dim=1)

        remaining = target_channels - rendered
        if remaining <= 0:
            return channels

        mixer = torch.Generator(device="cpu").manual_seed(seed)
        if remaining >= rendered:
            weights = torch.linalg.qr(
                torch.randn(remaining, rendered, generator=mixer, dtype=torch.float64)).Q
        else:
            weights = torch.linalg.qr(
                torch.randn(rendered, remaining, generator=mixer, dtype=torch.float64)).Q.T
        weights = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
        mixed = torch.einsum("mc,bchw->bmhw", weights.to(device=base.device, dtype=base.dtype), channels)
        return torch.cat([channels, mixed], dim=1)
    
    @staticmethod
    def get_target_channels(
        params: ShaderParams,
        default: int = DEFAULT_CHANNELS
    ) -> int:
        """
        Get target channel count from parameters.
        
        Args:
            params: Shader parameters
            default: Default channel count
            
        Returns:
            Target number of channels
        """
        return int(params.get("target_channels", default))
