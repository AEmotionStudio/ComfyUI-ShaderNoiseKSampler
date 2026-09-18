"""
Domain Warp Noise Generator.

This module implements domain warping noise where one noise function distorts
the input coordinates for another noise function, creating complex, swirling
patterns used to influence the sampling process in image generation.
"""

import torch
import math
import logging
from typing import Dict, Any, Optional, Union

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..utils.color_utils import apply_color_scheme, hsv_to_rgb, interpolate_colors, COLOR_SCHEMES
from ..utils.shape_masks import apply_shape_mask, apply_mask_to_tensor
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from .simplex import simplex_2d, simplex_3d
from ..core.constants import DEFAULT_CHANNELS

logger = logging.getLogger(__name__)


def _time_is_an_axis(time_axis, time):
    """
    Whether this draw should evaluate its field in 3D with time as the third axis.

    `time_axis` is set by core.shader_noise from the latent's frame count. When it
    is absent -- a direct caller, or an older saved path -- fall back to the old
    test, which is what the golden fixtures for single images pin.
    """
    return (time != 0) if time_axis is None else time_axis


@shader_generator("domain_warp", metadata={"description": "Domain warping noise for swirling patterns"})
class DomainWarpGenerator(BaseNoiseGenerator):
    """
    PyTorch implementation of domain warp animation.
    
    This class generates domain warping noise, where one noise function distorts
    the input coordinates for another noise function, creating complex, swirling 
    patterns used to influence the sampling process in image generation.
    """
    
    @staticmethod
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
        Generate domain warp noise tensor.
        
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
        # Extract parameters
        scale = params.scale
        warp_strength = params.warp_strength
        phase_shift = params.phase_shift
        octaves = params.octaves
        time = params.time
        
        # Temporal coherence settings
        base_seed = params.get("base_seed", seed)
        use_temporal_coherence = params.use_temporal_coherence
        
        # Get target channels from params or use default
        target_channels = params.get("target_channels", target_channels)
        
        # Handle model-specific channel overrides
        model_class = params.get("model_class", "")
        inner_model_class = params.get("inner_model_class", "")
        
        if inner_model_class == "CosmosVideo" or model_class == "CosmosVideo":
            target_channels = 16
        elif inner_model_class == "ACEStep":
            target_channels = 8
        
        # Determine which seed to use
        current_seed = base_seed if use_temporal_coherence else seed
        
        # Create coordinate grid [B, H, W, 2] in [0, 1] range
        coords = create_coordinate_grid(batch_size, height, width, device)
        
        # Generate domain warp noise
        time_axis = params.get("time_axis", None)
        warp_type = int(octaves % 4)
        
        try:
            result = DomainWarpGenerator._domain_warp_with_phase(
                coords, device, octaves, current_seed, 0, warp_type, 
                scale, warp_strength, phase_shift, time, time_axis
            )
        except Exception as e:
            logger.warning(f"Error generating domain warp: {e}, using fallback noise")
            result = torch.randn((batch_size, height, width, 1), device=device)
        
        # Apply contrast adjustment
        contrast = 1.0 + phase_shift * 0.5
        result = result * contrast
        
        # Apply shape mask if requested
        shape_type = params.shape_type
        shape_strength = params.shape_strength
        
        applied_mask = None
        if shape_type not in ["none", "0"] and shape_strength > 0:
            try:
                mask_seed = base_seed if use_temporal_coherence else seed
                shape_mask = apply_shape_mask(coords, shape_type, time, mask_seed, shape_strength)
                result = torch.lerp(result, result * shape_mask, shape_strength)
                applied_mask = shape_mask
            except Exception as e:
                logger.warning(f"Error applying shape mask: {e}")
        
        # Clamp and convert to BCHW format
        result = torch.clamp(result, -1.0, 1.0)
        result = result.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        # Apply color scheme and expand to target channels
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        
        if color_scheme not in ["none", "0"] and color_intensity > 0:
            # A palette maps one field to three colours that cannot help but
            # correlate. Those three are kept; the alpha it also returns is the
            # field itself again, so it is dropped rather than spent on a channel.
            result = DomainWarpGenerator._apply_color_variations(
                result, coords, params, device, seed, warp_type
            )[:, :3]

        # The same shape mask applies to every channel: it is a spatial shape, and
        # apply_shape_mask reseeds the global RNG, so it is drawn once above.
        def draw(channel_seed):
            field = DomainWarpGenerator._domain_warp_with_phase(
                coords, device, octaves, channel_seed, 0, warp_type,
                scale, warp_strength, phase_shift, time, time_axis
            ) * contrast
            if applied_mask is not None:
                field = torch.lerp(field, field * applied_mask, shape_strength)
            return torch.clamp(field, -1.0, 1.0).permute(0, 3, 1, 2)

        def draw_many(channel_seeds):
            fields = DomainWarpGenerator._domain_warp_with_phase(
                coords, device, octaves, channel_seeds.reshape(-1, 1, 1, 1, 1),
                0, warp_type, scale, warp_strength, phase_shift, time, time_axis
            ) * contrast
            if applied_mask is not None:
                fields = torch.lerp(fields, fields * applied_mask, shape_strength)
            # [N, B, H, W, 1] -> [B, N, H, W]
            return torch.clamp(fields, -1.0, 1.0).squeeze(-1).permute(1, 0, 2, 3)

        return BaseNoiseGenerator.fill_channels(
            draw, result, target_channels, current_seed, render_many=draw_many)
    
    @staticmethod
    def get_domain_warp(batch_size, height, width, shader_params, device="cuda", seed=0):
        """
        Legacy interface for domain warp generation.
        
        Maintained for backward compatibility with existing code.
        """
        # Convert dict to ShaderParams if needed
        if isinstance(shader_params, dict):
            params = ShaderParams(shader_params).validate()
        else:
            params = shader_params
        
        target_channels = shader_params.get("target_channels", 9) if isinstance(shader_params, dict) else params.get("target_channels", 9)
        
        return DomainWarpGenerator.generate(
            batch_size, height, width, params, device, seed, target_channels
        )
    
    @staticmethod
    def _apply_color_variations(
        result: torch.Tensor,
        coords: torch.Tensor,
        params: ShaderParams,
        device: torch.device,
        seed: int,
        warp_type: int
    ) -> torch.Tensor:
        """
        Apply color scheme variations to single-channel noise.
        
        Args:
            result: Single channel noise [B, 1, H, W]
            coords: Coordinate grid [B, H, W, 2]
            params: Shader parameters
            device: Target device
            seed: Random seed
            warp_type: Warp type for variations
            
        Returns:
            RGBA tensor [B, 4, H, W]
        """
        batch, _, height, width = result.shape
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        intensity_factor = 0.5 + color_intensity * 0.5
        
        # Normalize result to [0, 1] for color interpolation
        normalized = (result + 1.0) / 2.0
        
        # Handle different color schemes
        if color_scheme in COLOR_SCHEMES:
            # Use centralized color schemes
            stops = [(s[0], s[1]) for s in COLOR_SCHEMES[color_scheme]]
            r, g, b = interpolate_colors(stops, normalized, device)
        elif color_scheme == "rainbow":
            # Rainbow using HSV
            hue = normalized
            sat = torch.ones_like(hue) * 0.8
            val = torch.clamp(normalized + 0.2, 0, 1)
            r, g, b = hsv_to_rgb(hue, sat, val)
        elif color_scheme == "hsv":
            r, g, b = hsv_to_rgb(normalized, torch.ones_like(normalized) * 0.95, torch.ones_like(normalized) * 0.95)
        elif color_scheme == "plasma":
            # Dynamic plasma with proxy velocity
            r, g, b = DomainWarpGenerator._plasma_color(result, coords, params, device, seed, warp_type)
        elif color_scheme == "complementary":
            # Contrasting pattern variations
            r, g, b = DomainWarpGenerator._complementary_color(result, coords, params, device, seed, warp_type, intensity_factor)
        else:
            # Default: simple channel replication with slight variation
            r = result
            g = torch.roll(result, shifts=int(height * 0.05 * intensity_factor), dims=2)
            b = torch.roll(result, shifts=int(width * 0.05 * intensity_factor), dims=3)
        
        # Alpha channel is original noise
        a = result
        
        return torch.cat([r, g, b, a], dim=1)
    
    @staticmethod
    def _plasma_color(
        result: torch.Tensor,
        coords: torch.Tensor,
        params: ShaderParams,
        device: torch.device,
        seed: int,
        warp_type: int
    ) -> tuple:
        """Generate plasma color scheme."""
        scale = params.scale
        warp_strength = params.warp_strength
        phase_shift = params.phase_shift
        octaves = params.octaves
        time = params.time
        color_intensity = params.color_intensity
        intensity_factor = 0.5 + color_intensity * 0.5
        
        # Generate proxy velocity components
        p_vx = coords * (scale * 0.92)
        p_vy = coords * (scale * 1.08)
        
        vx_proxy = DomainWarpGenerator._domain_warp_with_phase(
            p_vx, device, octaves, seed + 10, 0, warp_type, scale * 0.92,
            warp_strength * (1.0 + 0.15 * intensity_factor),
            phase_shift + 0.20 * intensity_factor, time
        )
        vy_proxy = DomainWarpGenerator._domain_warp_with_phase(
            p_vy, device, octaves, seed + 20, 0, warp_type, scale * 1.08,
            warp_strength * (1.0 - 0.15 * intensity_factor),
            phase_shift - 0.20 * intensity_factor, time
        )
        
        vx_proxy = torch.clamp(vx_proxy, 0.0, 1.0).permute(0, 3, 1, 2)
        vy_proxy = torch.clamp(vy_proxy, 0.0, 1.0).permute(0, 3, 1, 2)
        
        vangle = torch.atan2(vy_proxy, vx_proxy)
        vangle = (vangle + math.pi) / (2 * math.pi)
        
        normalized = (result + 1.0) / 2.0
        time_t = torch.tensor(time, device=device, dtype=result.dtype)
        
        r = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + time_t)
        g = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + normalized * 3.14159 + time_t * 2.0)
        b = 0.5 + 0.5 * torch.cos(vangle * 3.14159 + normalized * 6.28318 + time_t * 3.0)
        
        return r, g, b
    
    @staticmethod
    def _complementary_color(
        result: torch.Tensor,
        coords: torch.Tensor,
        params: ShaderParams,
        device: torch.device,
        seed: int,
        warp_type: int,
        intensity_factor: float
    ) -> tuple:
        """Generate complementary color scheme."""
        batch, _, height, width = result.shape
        scale = params.scale
        warp_strength = params.warp_strength
        phase_shift = params.phase_shift
        octaves = params.octaves
        time = params.time
        contrast = 1.0 + phase_shift * 0.5
        
        p_g = torch.roll(coords, shifts=int(width * 0.1 * intensity_factor), dims=2)
        p_b = torch.roll(coords, shifts=int(height * 0.1 * intensity_factor), dims=1)
        
        g_result = DomainWarpGenerator._domain_warp_with_phase(
                            p_g, device, octaves, seed, (warp_type + 1) % 4, 
            scale * 1.1, warp_strength * (1.0 - 0.3 * intensity_factor),
                            phase_shift + 0.5, time
                        )
        b_result = DomainWarpGenerator._domain_warp_with_phase(
                            p_b, device, octaves, seed, (warp_type + 2) % 4, 
            scale * 0.9, warp_strength * (1.0 + 0.3 * intensity_factor),
                            phase_shift + 0.5, time
                        )
                        
        g = torch.clamp(g_result * contrast * 0.9, -1.0, 1.0).permute(0, 3, 1, 2)
        b = torch.clamp(b_result * contrast * 1.1, -1.0, 1.0).permute(0, 3, 1, 2)
        
        return result, g, b
    
    @staticmethod
    def _domain_warp_with_phase(p, device, octaves, seed, warp_layer, warp_type, scale,
                                warp_strength, phase_shift, time, time_axis=None):
        """
        Generate domain warp noise with phase shift parameter.
        
        This is the core domain warping algorithm that generates swirling patterns.
        """
        # No reseed here: everything below is a coordinate hash, so this ran once
        # per channel render -- 888 times per draw at H3's default latent -- and
        # reseeded every CUDA device for nothing. Verified byte-identical without it.
        #
        # `seed` may also be a tensor holding one seed per channel. The coordinates
        # stay shared until the first simplex call, which grows the leading axis,
        # and everything after it broadcasts.
        scaled_p = p * scale
        
        # Generate base noise for warping
        warp_noise_x = DomainWarpGenerator.simplex_noise(scaled_p, seed + 100)
        warp_noise_y = DomainWarpGenerator.simplex_noise(scaled_p, seed + 200)
        
        # Built by concatenation rather than in-place writes into a clone, so that
        # the same code serves a shared [B,H,W,2] and a per-channel [N,B,H,W,2].
        # The order matters and is the order the in-place version ran in: the y
        # update reads the x that the line above it has already shifted.
        warped_x = scaled_p[..., 0:1] + warp_noise_x * warp_strength
        warped_y = scaled_p[..., 1:2] + warp_noise_y * warp_strength
        
        phase_effect = phase_shift * math.pi
        warped_x = warped_x + torch.sin(warped_y * 2.0 + phase_effect) * 0.1 * warp_strength
        warped_y = warped_y + torch.cos(warped_x * 2.0 + phase_effect) * 0.1 * warp_strength
        warped_p = torch.cat([warped_x, warped_y], dim=-1)
        
        # Generate final noise based on warp type
        octaves_int = max(1, int(octaves))
        
        if warp_type == 0:
            # Standard FBM
            result = DomainWarpGenerator.fbm_noise(warped_p, octaves_int, time, device, seed,
                                                  time_axis=time_axis)
        elif warp_type == 1:
            # Ridged FBM
            result = DomainWarpGenerator.fbm_noise(warped_p, octaves_int, time, device, seed,
                                                  time_axis=time_axis)
            result = 1.0 - torch.abs(result)
        elif warp_type == 2:
            # Turbulent FBM
            result = torch.abs(DomainWarpGenerator.fbm_noise(
                warped_p, octaves_int, time, device, seed, time_axis=time_axis))
        else:
            # Domain warp FBM
            result = DomainWarpGenerator.fbm_noise_domain_warp(
                warped_p, octaves_int, time, device, seed, time_axis=time_axis)
        
        # Normalize result. Per draw when the seed is a tensor: each channel has to
        # be standardised against itself, exactly as it would be on its own, or a
        # batched draw stops matching the loop it replaces.
        if torch.is_tensor(seed):
            dims = tuple(range(1, result.dim()))
            result = (result - result.mean(dim=dims, keepdim=True)) \
                / (result.std(dim=dims, keepdim=True) + 1e-8)
        else:
            result = (result - result.mean()) / (result.std() + 1e-8)
        result = torch.clamp(result * 0.5, -1.0, 1.0)
        
        return result
    
    @staticmethod
    def simplex_noise(p, seed):
        """
        2D simplex noise, rotated by the seed.

        Delegates to shaders/simplex.py, which holds the one copy of this and takes
        a tensor of seeds as well as an int. Verified bit-identical to the copy that
        used to live here, across shapes and seeds.
        """
        return simplex_2d(p, seed, rotate=True)

    @staticmethod
    def simplex_noise_3d(coords, seed=0):
        """
        3D simplex noise, first corner only.

        Delegates to shaders/simplex.py. Verified bit-identical to the copy that
        used to live here, which also computed three more corner hashes and threw
        them away.
        """
        return simplex_3d(coords, seed, corners=1)

    @staticmethod
    def fbm_noise(p, octaves, time, device, seed, use_temporal_coherence=True, time_axis=None):
        """
        Generate FBM (Fractal Brownian Motion) noise.
    
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            octaves: Number of octaves
            time: Animation time
            device: Target device
            seed: Random seed
            use_temporal_coherence: Whether to use 3D noise for temporal coherence
        
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        # `seed` may be a tensor, one per channel, in which case `p` carries a
        # leading axis of that many draws and everything below broadcasts over it.
        result = torch.zeros_like(p[..., 0:1])
        
        amp = 1.0
        freq = 1.0
        max_amp = 0.0
        
        for i in range(min(octaves, 8)):
            current_p = p * freq
            
            if use_temporal_coherence and _time_is_an_axis(time_axis, time):
                time_offset = time * (0.2 + i * 0.05)
                current_p_3d = torch.cat([
                    current_p,
                    torch.ones_like(current_p[..., 0:1]) * time_offset
                ], dim=-1)
                noise = DomainWarpGenerator.simplex_noise_3d(current_p_3d, seed + i)
            else:
                noise = DomainWarpGenerator.simplex_noise(current_p, seed + i)
            
            result = result + amp * noise
            max_amp += amp
            
            freq *= 2.0
            amp *= 0.5
        
        return result / max_amp
    
    @staticmethod
    def fbm_noise_domain_warp(p, octaves, time, device, seed, time_axis=None):
        """
        Generate FBM noise with domain warping applied at each octave.
        """
        result = torch.zeros_like(p[..., 0:1])
        
        amp = 1.0
        freq = 1.0
        max_amp = 0.0
        
        # Initial warp
        warp = DomainWarpGenerator.simplex_noise(p * 1.5, seed + 1234) * 0.5
        p_original = p.clone()
        
        for i in range(min(octaves, 8)):
            # Progressive warping
            if i > 0:
                warp_amt = 0.1 * (i / octaves) * warp
                p_warped = p_original + warp_amt
            else:
                p_warped = p
            
            current_p = p_warped * freq
            
            if _time_is_an_axis(time_axis, time):
                time_offset = time * (0.2 + i * 0.05)
                current_p_3d = torch.cat([
                    current_p,
                    torch.ones_like(current_p[..., 0:1]) * time_offset
                ], dim=-1)
                noise = DomainWarpGenerator.simplex_noise_3d(current_p_3d, seed + i)
            else:
                noise = DomainWarpGenerator.simplex_noise(current_p, seed + i)
            
            # Octave-specific transformations
            if i == 0:
                noise = torch.abs(noise) * 2.0 - 1.0
            
            result = result + amp * noise
            max_amp += amp
            
            freq *= 2.0
            amp *= 0.5
        
        return result / max_amp


# Backward compatibility functions
def add_domain_warp_to_tensor(tensor_class):
    """Legacy function for backward compatibility."""
    pass


def register_shader_generator(generators_dict):
    """Legacy function for backward compatibility."""
    generators_dict["domain_warp"] = generate_domain_warp_tensor


def generate_domain_warp_tensor(
    shader_params: Dict[str, Any],
    height: int,
    width: int,
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 0,
    target_channels: int = DEFAULT_CHANNELS,
    **kwargs
) -> torch.Tensor:
    """
    Generate domain warp noise tensor.
    
    This is a wrapper function for backward compatibility with the old API.
    
    Args:
        shader_params: Dictionary of shader parameters
        height: Height of the output tensor
        width: Width of the output tensor
        batch_size: Number of images in batch
        device: Device to create tensor on
        seed: Random seed
        target_channels: Number of output channels
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Tensor with shape [batch_size, target_channels, height, width]
    """
    # Convert dict to ShaderParams if needed
    if isinstance(shader_params, dict):
        params = ShaderParams(shader_params)
    else:
        params = shader_params
    
    # Override target_channels if provided in params
    if "target_channels" in shader_params:
        target_channels = shader_params["target_channels"]
    
    return DomainWarpGenerator.generate(
        batch_size=batch_size,
        height=height,
        width=width,
        params=params,
        device=torch.device(device),
        seed=seed,
        target_channels=target_channels
    )
