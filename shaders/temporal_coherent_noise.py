"""
Temporal Coherent Noise Generator.

This module implements temporally coherent noise that maintains consistency
between animation frames by treating time as a proper 4th dimension.
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..utils.shape_masks import apply_shape_mask
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from ..core.constants import DEFAULT_CHANNELS

logger = logging.getLogger(__name__)


# Precomputed gradients for 3D Simplex noise to avoid runtime branching
SIMPLEX_GRADIENTS = torch.tensor([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
], dtype=torch.float32)


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
        
        # Create coordinate grid in [-1, 1] range
        coords = create_coordinate_grid(batch_size, height, width, device, range_type="symmetric")
        
        # Set consistent seed
        torch.manual_seed(base_seed)
        
        # Generate temporal coherent noise
        result = TemporalCoherentNoiseGenerator.temporal_spectral_noise(
            coords, scale, warp_strength, phase_shift, octaves,
            frequency_range, time, device, base_seed
        )
        
        # Apply shape mask
        if shape_type not in ["none", "0"] and shape_strength > 0:
            # Convert coords to [0, 1] range for shape mask
            coords_01 = (coords + 1.0) / 2.0
            mask = apply_shape_mask(coords_01, shape_type, time, base_seed, shape_strength)
            result = torch.lerp(result, result * mask, shape_strength)
        
        # Clamp and convert to BCHW
        result = torch.clamp(result, -1.0, 1.0)
        result = result.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        # Expand to target channels
        result = result.expand(-1, target_channels, -1, -1).clone()
        
        return result
    
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
            
            p_temporal = torch.cat([
                p_temporal[..., 0:1] + warp_noise1 * warp_strength,
                p_temporal[..., 1:2] + warp_noise2 * warp_strength,
                p_temporal[..., 2:3]  # Time unchanged
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
        noise = torch.zeros(batch, height, width, 1, device=device)
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
        Generate 3D simplex noise.
        
        Args:
            coords: Coordinate tensor [B, H, W, 3]
            seed: Random seed
            
        Returns:
            Noise tensor [B, H, W, 1]
        """
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        batch, height, width, dim = coords.shape
        device = coords.device
        
        # Ensure gradients are on the correct device
        gradients = SIMPLEX_GRADIENTS.to(device)

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2] if dim > 2 else torch.zeros_like(x)
        
        F3 = 1.0 / 3.0
        G3 = 1.0 / 6.0
        
        s = (x + y + z) * F3
        i = torch.floor(x + s)
        j = torch.floor(y + s)
        k = torch.floor(z + s)
        
        t = (i + j + k) * G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)
        
        # Determine simplex
        x_ge_y = (x0 >= y0).float()
        y_ge_z = (y0 >= z0).float()
        x_ge_z = (x0 >= z0).float()
        
        i1 = x_ge_y * x_ge_z
        j1 = (1 - x_ge_y) * y_ge_z
        k1 = (1 - x_ge_z) * (1 - y_ge_z)
        
        i2 = x_ge_y + (1 - x_ge_y) * x_ge_z
        j2 = x_ge_y * (1 - x_ge_z) + (1 - x_ge_y)
        k2 = (1 - x_ge_z) + x_ge_z * (1 - x_ge_y)
        
        # Optimized gradient calculation using embedding lookup
        def grad3d_optimized(ix, iy, iz, gx, gy, gz):
            h = (ix * 1619 + iy * 31337 + iz * 6971 + seed * 2459)
            h = torch.fmod(h * h * h, 1013)
            h_int = h.long() % 12
            
            # Lookup gradients from precomputed table
            grads = F.embedding(h_int, gradients)

            # Dot product
            return grads[..., 0] * gx + grads[..., 1] * gy + grads[..., 2] * gz
        
        noise = torch.zeros_like(x0)
        
        # Corner 0
        t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
        mask0 = (t0 >= 0).float()
        t0 = t0 * t0
        noise = noise + mask0 * t0 * t0 * grad3d_optimized(i, j, k, x0, y0, z0)
        
        # Corner 1
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
        mask1 = (t1 >= 0).float()
        t1 = t1 * t1
        noise = noise + mask1 * t1 * t1 * grad3d_optimized(i + i1, j + j1, k + k1, x1, y1, z1)
        
        # Corner 2
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
        mask2 = (t2 >= 0).float()
        t2 = t2 * t2
        noise = noise + mask2 * t2 * t2 * grad3d_optimized(i + i2, j + j2, k + k2, x2, y2, z2)
        
        # Corner 3
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3
        t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
        mask3 = (t3 >= 0).float()
        t3 = t3 * t3
        noise = noise + mask3 * t3 * t3 * grad3d_optimized(i + 1, j + 1, k + 1, x3, y3, z3)
        
        result = noise * 32.0
        return result.unsqueeze(-1)


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
