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
from .simplex import simplex_2d, simplex_3d
from ..core.constants import DEFAULT_CHANNELS

logger = logging.getLogger(__name__)


def _as_field(value, reference):
    """Round a per-channel scalar into the field's dtype, no earlier than needed."""
    return value.to(reference.dtype) if torch.is_tensor(value) else value


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

        # Drawn once rather than per channel: every argument is loop-invariant, so
        # the other three generators hoist it too. At LTXV's 128 channels this was
        # 127 identical masks. Safe despite apply_shape_mask reseeding the global
        # RNG, because the reseed at the top of each iteration overwrites it before
        # the randn_like below ever reads it.
        hoisted_mask = None
        if shape_type not in ["none", "0"] and shape_strength > 0:
            hoisted_mask = apply_shape_mask(coords, shape_type, time, base_seed, shape_strength)
            if len(hoisted_mask.shape) == 3:
                hoisted_mask = hoisted_mask.unsqueeze(-1)

        # Channels 0-2 are the palette's own when a colour scheme is active, each
        # with its own bespoke parameters; everything past them is formulaic and
        # goes through in a single batched draw below.
        # Channel 0 always stays on the scalar path, the way the other three
        # generators keep theirs outside fill_channels. Every travel-mode basis is
        # built from one-channel draws, and a test pins channel 0 of a wide draw to
        # be that same draw; batching it would break the identity at shapes whose
        # per-slice element count is not vector-aligned.
        scalar_channels = 3 if (color_scheme != "none" and color_intensity > 0) else 1
        scalar_channels = min(scalar_channels, target_channels)

        for i in range(scalar_channels):
            current_seed = loop_seed + 700 + (i * 130)
            torch.manual_seed(current_seed)
            
            # No clone: the RGB path never writes to this, and the other path
            # rebinds it to a fresh tensor on the next line.
            current_coords = coords
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
            if hoisted_mask is not None:
                channel = torch.lerp(channel, channel * hoisted_mask, shape_strength)
                channel = torch.clamp(channel, -1.0, 1.0)
            
            # Convert to BCHW
            channel = channel.permute(0, 3, 1, 2)
            all_channels.append(channel)
        
        if target_channels > scalar_channels:
            indices = range(scalar_channels, target_channels)
            perturbed, seeds, times, vizzes, scales, warps = [], [], [], [], [], []
            for i in indices:
                channel_seed = loop_seed + 700 + (i * 130)
                # A generator of its own on the coordinates' own device, which is
                # bit-identical to the global manual_seed + randn_like this replaces
                # and does not disturb the caller's stream.
                channel_generator = torch.Generator(device=coords.device).manual_seed(channel_seed)
                perturbation = torch.randn(coords.shape, generator=channel_generator,
                                           device=coords.device, dtype=coords.dtype)
                perturbed.append(torch.clamp(
                    coords + perturbation * (0.005 + i * 0.001), 0.0, 1.0).unsqueeze(0))
                seeds.append(channel_seed)
                times.append(time + (i * 0.02))
                vizzes.append((viz_type + i) % 4)
                scales.append(scale * (1.0 + ((i % 5 - 2) * 0.03)))
                warps.append(warp_strength * (1.0 + ((i % 7 - 3) * 0.03)))

            def column(values, dtype=torch.float32):
                return torch.tensor(values, device=device, dtype=dtype).reshape(-1, 1, 1, 1, 1)

            fields = TensorFieldGenerator.tensor_field_many(
                torch.cat(perturbed, dim=0), vizzes, column(scales), column(warps),
                column(times, torch.float64), device, column(seeds, torch.int64),
                use_temporal_coherence
            )
            fields = torch.clamp(fields * (1.0 + phase_shift), -1.0, 1.0)
            if hoisted_mask is not None:
                fields = torch.clamp(
                    torch.lerp(fields, fields * hoisted_mask, shape_strength), -1.0, 1.0)
            # [N, B, H, W, 1] -> [B, N, H, W]
            all_channels.append(fields.squeeze(-1).permute(1, 0, 2, 3))

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
            # Eigenvalue difference. compute_tensor_properties returns [B, H, W, 1]
            # tensors, so unsqueezing here made it 5D and broke the permute below.
            result = lambda1 - lambda2
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
    def tensor_field_many(p, viz_types, scale, warp_strength, time, device, seed,
                          use_temporal_coherence=False):
        """
        `tensor_field` for a whole channel axis at once.

        The expensive part -- five simplex evaluations inside
        compute_tensor_properties -- does not depend on the visualisation, so it
        runs once for every channel together. The four visualisations are then
        cheap elementwise reads of the same eigen-decomposition, so all four are
        computed and each channel takes the one its index asks for. That is less
        work than grouping the channels by visualisation and making four calls.

        `p` is [N, B, H, W, 2]; `scale`, `warp_strength`, `time` and `seed` are
        [N, 1, 1, 1, 1]; `viz_types` is a sequence of N integers in [0, 4).
        """
        lambda1, lambda2, v1, v2 = TensorFieldGenerator.compute_tensor_properties(
            p, scale, warp_strength, time, device, seed, use_temporal_coherence
        )
        angle = torch.atan2(v1[..., 1:2], v1[..., 0:1])
        options = torch.stack([
            (torch.abs(lambda1) + torch.abs(lambda2)) * 0.5,
            lambda1 - lambda2,
            torch.sin(angle * 4.0 + _as_field(time, angle)),
            v1[..., 0:1] * v1[..., 1:2] * 2.0,
        ])                                              # [4, N, B, H, W, 1]

        chooser = torch.tensor(list(viz_types), device=options.device, dtype=torch.int64)
        chooser = chooser.reshape(1, -1, *([1] * (options.dim() - 2))).expand_as(options[:1])
        result = options.gather(0, chooser).squeeze(0)

        # Per channel, exactly as each would be normalised on its own.
        dims = tuple(range(1, result.dim()))
        result = (result - result.mean(dim=dims, keepdim=True)) \
            / (result.std(dim=dims, keepdim=True) + 1e-8)
        return torch.clamp(result * 0.5, -1.0, 1.0)

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
        # `scale`, `warp_strength`, `time` and `seed` may each be a tensor holding
        # one value per channel, shaped to broadcast against `p`'s leading axes.
        if torch.is_tensor(time):
            # Held in float64 and cast here, not before. The scalar path computes
            # `time * 0.05` as a Python float and only then rounds it into the
            # field's dtype; rounding `time` first instead shifts the coordinate by
            # an ulp, and simplex noise turns that into an O(1) change wherever a
            # point crosses a cell boundary -- 5.6e-04 in the draw, not 1e-07.
            shifted = (time * 0.05).to(p.dtype)
            offset = torch.cat([shifted, torch.zeros_like(shifted)], dim=-1)
        else:
            offset = torch.tensor([[[[time * 0.05, 0.0]]]], device=device, dtype=p.dtype)
        p1 = p * scale + offset
        
        # Every channel's warp is the same sign as the generator's, so whether the
        # warp runs is still a scalar decision even when its strength is per channel.
        warp_active = bool((warp_strength > 0.0).all()) if torch.is_tensor(warp_strength) \
            else warp_strength > 0.0
        if warp_active:
            if use_temporal_coherence:
                warp_noise1 = TensorFieldGenerator.simplex_noise_3d(p1 * 0.3, seed, _as_field(time * 0.2, p))
                warp_noise2 = TensorFieldGenerator.simplex_noise_3d(p1 * 0.3, seed + 1, _as_field(time * 0.2 + 3.33, p))
            else:
                warp_noise1 = TensorFieldGenerator.simplex_noise(p1 * 0.3, seed)
                warp_noise2 = TensorFieldGenerator.simplex_noise(p1 * 0.3, seed + 1)
            
            p1 = p1 + torch.cat([warp_noise1, warp_noise2], dim=-1) * warp_strength
        
        # Compute tensor field derivatives
        eps = 0.01
        
        if use_temporal_coherence:
            n00 = TensorFieldGenerator.simplex_noise_3d(p1, seed + 2, _as_field(time * 0.1, p))
            n10 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[eps, 0]]]], device=device), seed + 2, _as_field(time * 0.1, p))
            n01 = TensorFieldGenerator.simplex_noise_3d(p1 + torch.tensor([[[[0, eps]]]], device=device), seed + 2, _as_field(time * 0.1, p))
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
        """2D simplex noise, unrotated. See shaders/simplex.py."""
        return simplex_2d(p, seed, rotate=False)

    @staticmethod
    def simplex_noise_3d(coords, seed=0, time_offset=0.0):
        """3D simplex noise with time as the third axis, first corner only."""
        z = torch.ones_like(coords[..., 0:1]) * time_offset
        return simplex_3d(torch.cat([coords[..., 0:1], coords[..., 1:2], z], dim=-1),
                          seed, corners=1)

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
