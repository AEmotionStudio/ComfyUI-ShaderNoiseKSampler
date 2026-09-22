"""
Temporal Coherent Noise Generator.

This module implements temporally coherent noise that maintains consistency
between animation frames by treating time as a proper 4th dimension.
"""

import torch
import math
import logging
from typing import Dict, Any, Optional

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..utils.shape_masks import apply_shape_mask
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from ..core.constants import DEFAULT_CHANNELS
from .simplex import simplex_3d_full

logger = logging.getLogger(__name__)


@shader_generator("temporal_coherent", metadata={"description": "Temporally coherent noise for smooth animations"})
class TemporalCoherentNoiseGenerator(BaseNoiseGenerator):
    """
    Implementation of temporally coherent noise.
    
    Generates noise that smoothly transitions between frames by implementing
    true 4D (x,y,z,time) noise functions.
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
        Generate temporally coherent noise tensor.
        
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
        octaves = int(params.octaves)
        time = params.time
        frequency_range = params.get("frequency_range", 0)
        
        base_seed = params.get("base_seed", seed)
        shape_type = params.shape_type
        shape_strength = params.shape_strength

        # Hold the seed only when temporal coherence asks for it, the way
        # domain_warp does. Reading base_seed unconditionally ignored the seed
        # argument entirely: the node always sets base_seed, so every stage and
        # every channel drew the same field, and only `time` still varied.
        use_temporal_coherence = params.use_temporal_coherence
        current_seed = base_seed if use_temporal_coherence else seed

        # Create coordinate grid in [-1, 1] range
        coords = create_coordinate_grid(batch_size, height, width, device, range_type="symmetric")

        # No reseed: temporal_spectral_noise hashes current_seed directly.

        # Generate temporal coherent noise
        result = TemporalCoherentNoiseGenerator.temporal_spectral_noise(
            coords, scale, warp_strength, phase_shift, octaves,
            frequency_range, time, device, current_seed
        )
        
        # Apply shape mask
        mask = None
        if shape_type not in ["none", "0"] and shape_strength > 0:
            # Convert coords to [0, 1] range for shape mask
            coords_01 = (coords + 1.0) / 2.0
            mask = apply_shape_mask(coords_01, shape_type, time, base_seed, shape_strength)
            result = torch.lerp(result, result * mask, shape_strength)
        
        # Clamp and convert to BCHW
        result = torch.clamp(result, -1.0, 1.0)
        result = result.permute(0, 3, 1, 2)  # [B, 1, H, W]

        # Every channel is a field of its own. This used to broadcast the one
        # field above to all of them, which is rank 1.00 at any channel count.
        def draw(channel_seed):
            field = TemporalCoherentNoiseGenerator.temporal_spectral_noise(
                coords, scale, warp_strength, phase_shift, octaves,
                frequency_range, time, device, channel_seed
            )
            if mask is not None:
                field = torch.lerp(field, field * mask, shape_strength)
            return torch.clamp(field, -1.0, 1.0).permute(0, 3, 1, 2)

        def draw_many(channel_seeds):
            fields = TemporalCoherentNoiseGenerator.temporal_spectral_noise(
                coords, scale, warp_strength, phase_shift, octaves,
                frequency_range, time, device, channel_seeds.reshape(-1, 1, 1, 1, 1)
            )
            if mask is not None:
                fields = torch.lerp(fields, fields * mask, shape_strength)
            # [N, B, H, W, 1] -> [B, N, H, W]
            return torch.clamp(fields, -1.0, 1.0).squeeze(-1).permute(1, 0, 2, 3)

        return BaseNoiseGenerator.fill_channels(draw, result, target_channels, current_seed,
                                                render_many=draw_many)
    
    @staticmethod
    def get_temporal_noise(batch_size, height, width, shader_params, device="cuda", base_seed=0):
        """
        Legacy interface for temporal noise generation.
        """
        if isinstance(shader_params, dict):
            params = ShaderParams(shader_params).validate()
        else:
            params = shader_params
        
        return TemporalCoherentNoiseGenerator.generate(
            batch_size, height, width, params, device, base_seed, 4
        )
    
    @staticmethod
    def temporal_spectral_noise(p, scale, warp_strength, phase_shift, octaves, frequency_range, time, device, base_seed):
        """
        Generate spectral noise with true temporal coherence.
        
        Args:
            p: Coordinate tensor [B, H, W, 2]
            scale: Scale factor
            warp_strength: Warping strength
            phase_shift: Phase adjustment
            octaves: Number of octaves
            frequency_range: Frequency filtering type
            time: Animation time
            device: Target device
            base_seed: Random seed
            
        Returns:
            Noise tensor [B, H, W, 1]
        """
        batch, height, width, _ = p.shape
        
        # Scale coordinates
        p = p * scale
        
        # Extend to 3D with time dimension
        p_temporal = torch.cat([
            p,
            torch.ones_like(p[..., 0:1]) * time
        ], dim=-1)
        
        # Apply 3D warp
        if warp_strength > 0.0:
            warp_p = p_temporal * 0.4
            
            warp_noise1 = TemporalCoherentNoiseGenerator._simplex_3d(warp_p, base_seed)
            warp_noise2 = TemporalCoherentNoiseGenerator._simplex_3d(warp_p + 5.0, base_seed + 1)
            
            warped_x = p_temporal[..., 0:1] + warp_noise1 * warp_strength
            warped_y = p_temporal[..., 1:2] + warp_noise2 * warp_strength
            # Time is unchanged, but it has to match the rank the warp just grew:
            # with a tensor of seeds the two lines above carry one slice per channel
            # and this one still carries the shared coordinates.
            p_temporal = torch.cat([
                warped_x, warped_y, p_temporal[..., 2:3].expand_as(warped_x)
            ], dim=-1)
        
        # Frequency domain processing
        freq = (p_temporal[..., :2] - 0.5) * 2.0
        radius = torch.sqrt(freq[..., 0]**2 + freq[..., 1]**2)
        angle = torch.atan2(freq[..., 1], freq[..., 0])
        
        # Apply phase shift with time
        angle = angle + (phase_shift + time * 0.1) * math.pi
        
        # Initialize filter
        filter_tensor = torch.ones_like(radius)
        
        # Apply frequency filtering
        if frequency_range == 1:  # Low-pass
            cutoff = 0.25 + 0.05 * torch.sin(torch.tensor(time * 0.2, device=device))
            filter_tensor = torch.sigmoid((1.0 - radius - cutoff) * 10.0)
        elif frequency_range == 2:  # Band-pass
            center = 0.5 + 0.1 * torch.sin(torch.tensor(time * 0.3, device=device))
            width = 0.2 + 0.05 * torch.cos(torch.tensor(time * 0.25, device=device))
            low_pass = torch.sigmoid((radius - (center - width/2)) * 10.0)
            high_pass = torch.sigmoid(((center + width/2) - radius) * 10.0)
            filter_tensor = low_pass * high_pass
        elif frequency_range == 3:  # High-pass
            cutoff = 0.6 + 0.05 * torch.sin(torch.tensor(time * 0.15, device=device))
            filter_tensor = torch.sigmoid((radius - cutoff) * 10.0)
        elif frequency_range == 4:  # Directional
            num_dir = 4
            angle_mod = (angle + time * 0.2) % (2.0 * math.pi)
            dir_filter = 0.5 + 0.5 * torch.cos(torch.tensor(float(num_dir), device=device) * angle_mod)
            filter_tensor = torch.lerp(torch.ones_like(dir_filter), dir_filter, 0.8)
        
        # Generate noise with octaves
        noise = torch.zeros_like(p_temporal[..., 0:1])
        max_octaves = min(octaves, 8)
        
        for i in range(max_octaves):
            freq_scale = 2.0 ** i
            amp = 1.0 / freq_scale
            
            octave_p = p_temporal * freq_scale + torch.tensor([0.0, 0.0, i * 1.5], device=device)
            noise_val = TemporalCoherentNoiseGenerator._simplex_3d(octave_p, base_seed + i)
            
            # Apply frequency filtering
            freq_factor = i / max(max_octaves - 1, 1)
            freq_filter = 1.0
            
            if frequency_range == 1:
                freq_filter = 1.0 - freq_factor
            elif frequency_range == 2:
                freq_filter = 1.0 - abs(freq_factor - 0.5) * 2.0
            elif frequency_range == 3:
                freq_filter = freq_factor
            
            noise = noise + noise_val * amp * freq_filter
        
        # Apply filter and normalize
        noise = noise * filter_tensor.unsqueeze(-1)
        
        # Temporal modulation
        time_factor = torch.sin(torch.tensor(time * 0.3, device=device))
        noise = noise * (1.0 + 0.1 * time_factor)
        
        return torch.clamp(noise * 1.5, -1.0, 1.0)
    
    @staticmethod
    def _simplex_3d(coords, seed=0):
        """
        Four-corner 3D simplex noise, [B, H, W, 3] -> [B, H, W, 1].

        Lives in shaders/simplex.py now, so the generators added later can share
        it; the golden fixture video_temporal_coherent pins this one across the move.
        """
        return simplex_3d_full(coords, seed)


# Backward compatibility functions
def add_temporal_coherent_to_tensor(tensor_class):
    """Legacy function for backward compatibility."""
    pass


def integrate_temporal_coherent_noise():
    """Legacy function for backward compatibility."""
    pass


def register_shader_generator(generators_dict):
    """Legacy function for backward compatibility."""
    generators_dict["temporal_coherent"] = generate_temporal_coherent_noise_tensor


def generate_temporal_coherent_noise_tensor(
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
    Generate temporal coherent noise tensor.
    
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
    
    return TemporalCoherentNoiseGenerator.generate(
        batch_size=batch_size,
        height=height,
        width=width,
        params=params,
        device=torch.device(device),
        seed=seed,
        target_channels=target_channels
    )
