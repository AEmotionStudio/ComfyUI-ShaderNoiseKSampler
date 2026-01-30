"""
Tensor Field Noise Generator.

This module implements tensor field noise patterns that can be used to 
influence the sampling process in image generation.
"""

import torch
import math
import logging
from typing import Dict, Any, Optional, Tuple

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..utils.color_utils import apply_color_scheme, interpolate_colors, COLOR_SCHEMES
from ..utils.shape_masks import apply_shape_mask, apply_mask_to_tensor
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from ..core.constants import DEFAULT_CHANNELS

logger = logging.getLogger(__name__)


@shader_generator("tensor_field", metadata={"description": "Tensor field noise for directional patterns"})
class TensorFieldGenerator(BaseNoiseGenerator):
    """
    PyTorch implementation of tensor field animation.
    
    This class generates tensor field patterns that can be used to influence the
    sampling process in image generation.
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
        Generate tensor field noise tensor.
        
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
        
        base_seed = params.get("base_seed", seed)
        use_temporal_coherence = params.use_temporal_coherence
        
        shape_type = params.shape_type
        shape_strength = params.shape_strength
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        
        # Get target channels from params or use default
        target_channels = params.get("target_channels", target_channels)
        
        # Handle model-specific channel overrides
        model_class = params.get("model_class", "")
        inner_model_class = params.get("inner_model_class", "")

        if inner_model_class == "CosmosVideo" or model_class == "CosmosVideo":
            target_channels = 16
        elif inner_model_class == "ACEStep":
            target_channels = 8
        elif inner_model_class == "WAN21" or model_class == "WAN21":
            target_channels = 16
        
        # Create coordinate grid [B, H, W, 2] in [0, 1] range
        coords = create_coordinate_grid(batch_size, height, width, device)
        
        # Seed management
        loop_seed = base_seed if use_temporal_coherence else seed
        viz_type = int(octaves % 4)
        
        # Generate all channels
        all_channels = []
        
        # Get color channel variations if color scheme is active
        color_variations = TensorFieldGenerator._get_color_variations(
            color_scheme, color_intensity, octaves, scale, warp_strength, time
        )

        for i in range(target_channels):
            current_seed = loop_seed + 700 + (i * 130)
            torch.manual_seed(current_seed)
            
            current_coords = coords.clone()
            current_time = time
            current_viz_type = viz_type
            current_scale = scale
            current_warp = warp_strength
            
            is_rgb_color = color_scheme != "none" and color_intensity > 0 and i < 3
            
            if is_rgb_color:
                if i == 0:  # Red
                    current_seed = base_seed if use_temporal_coherence else seed
                elif i == 1:  # Green
                    current_time = time + color_variations["g_time"]
                    current_viz_type = color_variations["g_viz"]
                    current_scale = color_variations["g_scale"]
                    current_warp = color_variations["g_warp"]
                    current_seed = (base_seed + 42) if use_temporal_coherence else (seed + 42)
                elif i == 2:  # Blue
                    current_time = time + color_variations["b_time"]
                    current_viz_type = color_variations["b_viz"]
                    current_scale = color_variations["b_scale"]
                    current_warp = color_variations["b_warp"]
                    current_seed = (base_seed + 123) if use_temporal_coherence else (seed + 123)
            else:
                # Structured variation for non-RGB channels
                pert_scale = 0.005 + (i * 0.001)
                current_coords = current_coords + (torch.randn_like(coords) * pert_scale)
                current_coords = torch.clamp(current_coords, 0.0, 1.0)
                current_time = time + (i * 0.02)
                current_viz_type = (viz_type + i) % 4
                current_scale = scale * (1.0 + ((i % 5 - 2) * 0.03))
                current_warp = warp_strength * (1.0 + ((i % 7 - 3) * 0.03))
            
            # Generate single channel
            channel = TensorFieldGenerator.tensor_field(
                current_coords, current_viz_type, current_scale, current_warp,
                current_time, device, current_seed, use_temporal_coherence
            )
            
            if len(channel.shape) == 3:
                channel = channel.unsqueeze(-1)
            
            # Apply contrast and clamp
            contrast = 1.0 + phase_shift
            channel = torch.clamp(channel * contrast, -1.0, 1.0)
            
            # Special handling for blue_red G channel
            if is_rgb_color and i == 1 and color_scheme.lower() == "blue_red":
                channel = torch.full_like(channel, -1.0)
            
            # Apply shape mask
            if shape_type not in ["none", "0"] and shape_strength > 0:
                mask = apply_shape_mask(coords, shape_type, time, base_seed, shape_strength)
                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(-1)
                channel = torch.lerp(channel, channel * mask, shape_strength)
                channel = torch.clamp(channel, -1.0, 1.0)
            
            # Convert to BCHW
            channel = channel.permute(0, 3, 1, 2)
            all_channels.append(channel)
        
        # Concatenate all channels
        result = torch.cat(all_channels, dim=1)
        
        # Ensure correct channel count
        if result.shape[1] != target_channels:
            corrected = torch.zeros((batch_size, target_channels, height, width), device=device)
            min_ch = min(result.shape[1], target_channels)
            corrected[:, :min_ch] = result[:, :min_ch]
            result = corrected
        
        return result
    
    @staticmethod
    def get_tensor_field(batch_size, height, width, shader_params, device="cuda", seed=0):
        """
        Legacy interface for tensor field generation.
        
        Maintained for backward compatibility with existing code.
        """
        if isinstance(shader_params, dict):
            params = ShaderParams(shader_params).validate()
        else:
            params = shader_params
        
        target_channels = shader_params.get("target_channels", 4) if isinstance(shader_params, dict) else params.get("target_channels", 4)
        
        return TensorFieldGenerator.generate(
            batch_size, height, width, params, device, seed, target_channels
        )
    
    @staticmethod
    def _get_color_variations(
        color_scheme: str,
        color_intensity: float,
        octaves: int,
        scale: float,
        warp_strength: float,
        time: float
    ) -> Dict[str, Any]:
        """Get color channel variations based on scheme."""
        viz_type = int(octaves % 4)
        intensity = 0.5 + color_intensity * 0.5
        
        # Default values
        result = {
            "g_time": 0.2 * intensity,
            "b_time": 0.4 * intensity,
            "g_viz": viz_type,
            "b_viz": viz_type,
            "g_scale": scale * (1.0 + 0.05 * intensity),
            "b_scale": scale * (1.0 - 0.05 * intensity),
            "g_warp": warp_strength * (1.0 - 0.1 * intensity),
            "b_warp": warp_strength * (1.0 + 0.1 * intensity),
        }
        
        scheme = color_scheme.lower()
        
        if scheme == "rainbow":
            result["g_time"] = 0.33 * intensity
            result["b_time"] = 0.66 * intensity
            result["g_viz"] = (viz_type + 1) % 4
            result["b_viz"] = (viz_type + 2) % 4
        elif scheme == "plasma":
            result["g_time"] = 0.25 * intensity
            result["b_time"] = 0.55 * intensity
            result["g_viz"] = (viz_type + 2) % 4
        elif scheme == "viridis":
            result["g_time"] = 0.20 * intensity
            result["b_time"] = 0.40 * intensity
            result["g_viz"] = (viz_type + 1) % 4
            result["b_viz"] = (viz_type + 3) % 4
        elif scheme == "inferno":
            result["g_time"] = 0.15 * intensity
            result["b_time"] = 0.35 * intensity
            result["g_viz"] = (viz_type + 3) % 4
            result["b_viz"] = (viz_type + 1) % 4
        elif scheme == "magma":
            result["g_time"] = 0.40 * intensity
            result["b_time"] = 0.70 * intensity
            result["g_viz"] = (viz_type + 2) % 4
        elif scheme == "jet":
            result["g_time"] = 0.30 * intensity
            result["b_time"] = 0.60 * intensity
            result["g_viz"] = (viz_type + 1) % 4
            result["b_viz"] = (viz_type + 3) % 4
        elif scheme == "hot":
            result["g_time"] = 0.1 * intensity
            result["b_time"] = 0.7 * intensity
        elif scheme == "cool":
            result["g_time"] = 0.6 * intensity
            result["b_time"] = 0.2 * intensity
        elif scheme == "blue_red":
            result["g_time"] = 0.0
            result["g_viz"] = 0
            result["g_scale"] = scale * 0.01
            result["g_warp"] = 0.0
            result["b_time"] = 0.5 * intensity if viz_type != 2 else math.pi
        
        return result
    
    @staticmethod
    def tensor_field(p, viz_type, scale, warp_strength, time, device, seed, use_temporal_coherence=False):
        """
        Generate tensor field noise for given coordinates.
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            viz_type: Visualization type (0-3)
            scale: Scale factor
            warp_strength: Warping strength
            time: Animation time
            device: Target device
            seed: Random seed
            use_temporal_coherence: Whether to use temporal coherence
            
        Returns:
            Noise tensor [batch, height, width, 1]
        """
        batch, height, width, _ = p.shape
        
        # Compute tensor properties (eigenvalues and eigenvectors)
        lambda1, lambda2, v1, v2 = TensorFieldGenerator.compute_tensor_properties(
            p, scale, warp_strength, time, device, seed, use_temporal_coherence
        )
        
        # Visualize based on type
        if viz_type == 0:
            # Eigenvalue magnitude
            result = (torch.abs(lambda1) + torch.abs(lambda2)) * 0.5
        elif viz_type == 1:
            # Eigenvalue difference
            result = (lambda1 - lambda2).unsqueeze(-1)
        elif viz_type == 2:
            # Hyperstreamlines
            angle = torch.atan2(v1[..., 1:2], v1[..., 0:1])
            result = torch.sin(angle * 4.0 + time)
        else:
            # Eigenvector direction
            result = v1[..., 0:1] * v1[..., 1:2] * 2.0
        
        # Ensure result has correct shape
        if len(result.shape) == 3:
            result = result.unsqueeze(-1)
        
        # Normalize to [-1, 1]
        result = (result - result.mean()) / (result.std() + 1e-8)
        result = torch.clamp(result * 0.5, -1.0, 1.0)
        
        return result
            
    @staticmethod
    def compute_tensor_properties(p, scale, warp_strength, time, device, seed, use_temporal_coherence=False):
        """
        Compute tensor field properties (eigenvalues and eigenvectors).
        
        Args:
            p: Coordinate tensor [batch, height, width, 2]
            scale: Scale factor
            warp_strength: Warping strength
            time: Animation time
            device: Target device
            seed: Random seed
            use_temporal_coherence: Whether to use temporal coherence
            
        Returns:
            Tuple of (lambda1, lambda2, v1, v2)
        """
        batch, height, width, _ = p.shape
        
        # Offset based on time
        offset = torch.tensor([[[[time * 0.05, 0.0]]]], device=device, dtype=p.dtype)
        p1 = p * scale + offset
        
        # Apply warp
        if warp_strength > 0.0:
            if use_temporal_coherence:
                warp_noise1 = TensorFieldGenerator.simplex_noise_3d(p1 * 0.3, seed, time * 0.2)
                warp_noise2 = TensorFieldGenerator.simplex_noise_3d(p1 * 0.3, seed + 1, time * 0.2 + 3.33)
            else:
                warp_noise1 = TensorFieldGenerator.simplex_noise(p1 * 0.3, seed)
                warp_noise2 = TensorFieldGenerator.simplex_noise(p1 * 0.3, seed + 1)
            
            p1 = p1 + torch.cat([warp_noise1, warp_noise2], dim=-1) * warp_strength
        
        # Compute tensor field derivatives
        eps = 0.01
        
        if use_temporal_coherence:
            n00 = TensorFieldGenerator.simplex_noise_3d(p1, seed + 2, time * 0.1)
            n10 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[eps, 0]]]], device=device), seed + 2, time * 0.1)
            n01 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[0, eps]]]], device=device), seed + 2, time * 0.1)
        else:
            n00 = TensorFieldGenerator.simplex_noise(p1, seed + 2)
            n10 = TensorFieldGenerator.simplex_noise(p1 + torch.tensor([[[[eps, 0]]]], device=device), seed + 2)
            n01 = TensorFieldGenerator.simplex_noise(p1 + torch.tensor([[[[0, eps]]]], device=device), seed + 2)
        
        # Compute gradients
        dx = (n10 - n00) / eps
        dy = (n01 - n00) / eps
        
        # Build tensor
        a = dx * dx
        b = dx * dy
        c = dy * dy
        
        # Compute eigenvalues
        trace = a + c
        det = a * c - b * b
        
        discriminant = torch.sqrt(torch.clamp(trace * trace * 0.25 - det, min=0))
        lambda1 = trace * 0.5 + discriminant
        lambda2 = trace * 0.5 - discriminant
        
        # Compute eigenvectors
        angle = torch.atan2(2.0 * b.squeeze(-1), (a - c).squeeze(-1)) * 0.5
        v1 = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        v2 = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=-1)
        
        return lambda1, lambda2, v1, v2
    
    @staticmethod
    def simplex_noise(p, seed):
        """Generate 2D simplex noise."""
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed) % 10000
        
        F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        G2 = (3.0 - math.sqrt(3.0)) / 6.0
        
        x = p[..., 0:1]
        y = p[..., 1:2]
        
        s = (x + y) * F2
        i = torch.floor(x + s)
        j = torch.floor(y + s)
        
        t = (i + j) * G2
        x0 = x - (i - t)
        y0 = y - (j - t)
        
        i1 = (x0 > y0).float()
        j1 = 1.0 - i1
        
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        
        def hash_coord(ix, iy):
            h = ix * 1619 + iy * 31337 + seed * 2459
            return torch.fmod(h * h * h, 1013)
        
        def grad(h, gx, gy):
            h_int = h.long() % 8
            u = torch.where(h_int < 4, gx, gy)
            v = torch.where(h_int < 4, gy, gx)
            return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)
        
        i0 = i.long()
        j0 = j.long()
        
        h0 = hash_coord(i0, j0)
        h1 = hash_coord(i0 + i1.long(), j0 + j1.long())
        h2 = hash_coord(i0 + 1, j0 + 1)
        
        t0 = torch.maximum(0.5 - x0*x0 - y0*y0, torch.zeros_like(x0))
        t1 = torch.maximum(0.5 - x1*x1 - y1*y1, torch.zeros_like(x1))
        t2 = torch.maximum(0.5 - x2*x2 - y2*y2, torch.zeros_like(x2))
        
        n0 = t0**4 * grad(h0, x0, y0)
        n1 = t1**4 * grad(h1, x1, y1)
        n2 = t2**4 * grad(h2, x2, y2)
        
        return 70.0 * (n0 + n1 + n2)
    
    @staticmethod
    def simplex_noise_3d(coords, seed=0, time_offset=0.0):
        """Generate 3D simplex noise with time."""
        if isinstance(seed, torch.Tensor):
            seed = seed.item()
        seed = int(seed)
        
        # Add time dimension
        if coords.shape[-1] == 2:
            time_dim = torch.ones_like(coords[..., 0:1]) * time_offset
            coords = torch.cat([coords, time_dim], dim=-1)
        
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        z = coords[..., 2:3] if coords.shape[-1] > 2 else torch.zeros_like(x)
        
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
        
        def hash3(ix, iy, iz):
            h = ix * 1619 + iy * 31337 + iz * 6971 + seed * 2459
            return torch.fmod(h * h * h, 1013)
        
        def grad3(h, gx, gy, gz):
            h_int = h.long() % 12
            u = torch.where(h_int < 8, gx, gy)
            v = torch.where(h_int < 4, gy, torch.where((h_int == 12) | (h_int == 14), gx, gz))
            return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)
        
        i0, j0, k0 = i.long(), j.long(), k.long()
        h0 = hash3(i0, j0, k0)
        
        t0 = torch.maximum(0.6 - x0*x0 - y0*y0 - z0*z0, torch.zeros_like(x0))
        n = t0**4 * grad3(h0, x0, y0, z0)
        
        return 32.0 * n


# Backward compatibility functions
def add_tensor_field_to_tensor(tensor_class):
    """Legacy function for backward compatibility."""
    pass


def register_shader_generator(generators_dict):
    """Legacy function for backward compatibility."""
    generators_dict["tensor_field"] = generate_tensor_field_tensor


def generate_tensor_field_tensor(
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
    Generate tensor field noise tensor.
    
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
    
    return TensorFieldGenerator.generate(
        batch_size=batch_size,
        height=height,
        width=width,
        params=params,
        device=torch.device(device),
        seed=seed,
        target_channels=target_channels
    )
