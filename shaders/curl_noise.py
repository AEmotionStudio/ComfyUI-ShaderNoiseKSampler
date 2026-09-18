"""
Curl Noise Generator.

This module implements curl noise (divergence-free vector fields) that generates
fluid-like patterns used to influence the sampling process in image generation.
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional, Tuple

from .base import BaseNoiseGenerator
from .registry import shader_generator
from ..utils.color_utils import apply_color_scheme, hsv_to_rgb, interpolate_colors, COLOR_SCHEMES
from ..utils.shape_masks import apply_shape_mask, apply_mask_to_tensor, smoothstep
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from .simplex import simplex_2d, simplex_3d
from ..core.constants import DEFAULT_CHANNELS, HIGH_CHANNEL_THRESHOLD

logger = logging.getLogger(__name__)


@shader_generator("curl_noise", metadata={"description": "Curl/fluid noise for flowing patterns"})
class CurlNoiseGenerator(BaseNoiseGenerator):
    """
    PyTorch implementation of Curl Noise.
    
    This class generates fluid-like curl noise by computing divergence-free
    vector fields and advecting properties along them.
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
        Generate curl noise tensor.
        
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
        
        base_seed = params.get("base_seed", seed)
        use_temporal_coherence = params.use_temporal_coherence
        
        shape_type = params.shape_type
        shape_strength = params.shape_strength
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        
        # Get target channels from params
        target_channels = params.get("target_channels", target_channels)
        
        # Handle model-specific channel overrides
        model_class = params.get("model_class", "")
        inner_model_class = params.get("inner_model_class", "")
        
        if inner_model_class == "CosmosVideo" or model_class == "CosmosVideo":
            target_channels = 16
        elif inner_model_class == "ACEStep":
            target_channels = 8
        
        # Create coordinate grid
        coords = create_coordinate_grid(batch_size, height, width, device)
        
        # No reseed: the fields below are coordinate hashes of current_seed.
        current_seed = base_seed if use_temporal_coherence else seed
        
        # Get velocity field
        velocity = CurlNoiseGenerator.get_velocity_field(
            coords * scale, time, octaves, device, current_seed, use_temporal_coherence
        )
        
        # Apply warp intensity
        velocity = CurlNoiseGenerator.apply_warp_intensity(velocity, warp_strength)
        
        # Apply shape mask to velocity
        mask = None
        if shape_type not in ["none", "0"] and shape_strength > 0:
            mask = apply_shape_mask(coords, shape_type, time, base_seed, shape_strength)
            velocity = torch.lerp(velocity, velocity * mask, shape_strength)
        
        # Convert velocity to BCHW
        velocity_bchw = velocity.permute(0, 3, 1, 2)
        
        # Ensure 2 channels
        if velocity_bchw.shape[1] == 1:
            velocity_bchw = velocity_bchw.repeat(1, 2, 1, 1)
            velocity_bchw[:, 1] = velocity_bchw[:, 0] * 0.8
        
        vx = velocity_bchw[:, 0:1]
        vy = velocity_bchw[:, 1:2]
        
        # Generate advected noise
        dt = 0.2 + phase_shift * 1.8
        advected = CurlNoiseGenerator.advect(
            coords * scale, velocity, time, dt, octaves, scale, device, current_seed, use_temporal_coherence
        )
        advected_bchw = advected.permute(0, 3, 1, 2)
        
        # Every extra channel draws its own velocity field, under the same warp and
        # shape mask as the one above.
        def velocity_at(channel_seed):
            field = CurlNoiseGenerator.apply_warp_intensity(
                CurlNoiseGenerator.get_velocity_field(
                    coords * scale, time, octaves, device, channel_seed, use_temporal_coherence
                ), warp_strength)
            if mask is not None:
                field = torch.lerp(field, field * mask, shape_strength)
            return field

        # Generate channels
        if color_scheme not in ["none", "0"] and color_intensity > 0 and target_channels >= 3:
            def magnitude_at(channel_seed):
                v = velocity_at(channel_seed).permute(0, 3, 1, 2)
                m = torch.sqrt(v[:, 0:1] ** 2 + v[:, 1:2] ** 2)
                return m / (m.max() + 1e-8)

            result = CurlNoiseGenerator._apply_color_scheme(
                vx, vy, color_scheme, color_intensity, target_channels, device, time,
                magnitude_at, current_seed
            )
        else:
            def advected_at(channel_seed):
                return CurlNoiseGenerator.advect(
                    coords * scale, velocity_at(channel_seed), time, dt, octaves, scale,
                    device, channel_seed, use_temporal_coherence
                ).permute(0, 3, 1, 2)

            def advected_many(channel_seeds):
                seeds = channel_seeds.reshape(-1, 1, 1, 1, 1)
                fields = CurlNoiseGenerator.advect(
                    coords * scale, velocity_at(seeds), time, dt, octaves, scale,
                    device, seeds, use_temporal_coherence)
                # [N, B, H, W, 1] -> [B, N, H, W]
                return fields.squeeze(-1).permute(1, 0, 2, 3)

            result = CurlNoiseGenerator._generate_channels(
                advected_bchw, vx, vy, advected_at, target_channels, params, current_seed,
                advected_many
            )
        
        # Scale to [-1, 1]
        result = result * 2.0 - 1.0
        
        # Ensure correct channel count
        if result.shape[1] != target_channels:
            corrected = torch.zeros((batch_size, target_channels, height, width), device=device)
            min_ch = min(result.shape[1], target_channels)
            corrected[:, :min_ch] = result[:, :min_ch]
            result = corrected
        
        return result
    
    @staticmethod
    def get_curl_noise(batch_size, height, width, shader_params, device="cuda", seed=0, target_channels=4):
        """
        Legacy interface for curl noise generation.
        """
        if isinstance(shader_params, dict):
            params = ShaderParams(shader_params).validate()
        else:
            params = shader_params
        
        target_ch = shader_params.get("target_channels", target_channels) if isinstance(shader_params, dict) else params.get("target_channels", target_channels)
        
        return CurlNoiseGenerator.generate(
            batch_size, height, width, params, device, seed, target_ch
        )
    
    @staticmethod
    def _apply_color_scheme(
        vx: torch.Tensor,
        vy: torch.Tensor,
        color_scheme: str,
        color_intensity: float,
        target_channels: int,
        device: torch.device,
        time: float,
        render,
        seed: int,
    ) -> torch.Tensor:
        """Apply color scheme based on velocity field."""
        vmag = torch.sqrt(vx**2 + vy**2)
        vmag = vmag / (vmag.max() + 1e-8)
        
        vangle = torch.atan2(vy, vx)
        vangle = (vangle + math.pi) / (2 * math.pi)
        
        # Use first channel as normalized value for color mapping
        normalized = vmag
        
        if color_scheme in COLOR_SCHEMES:
            stops = [(s[0], s[1]) for s in COLOR_SCHEMES[color_scheme]]
            r, g, b = interpolate_colors(stops, normalized, device)
        elif color_scheme == "rainbow":
            r, g, b = hsv_to_rgb(vangle, torch.ones_like(vangle) * 0.8, vmag)
        elif color_scheme == "heatmap":
            r = torch.pow(normalized, 0.5)
            g = torch.pow(normalized, 1.5)
            b = torch.pow(normalized, 3.0)
        elif color_scheme == "vorticity":
            curl_mag = torch.abs(vx - vy)
            curl_norm = curl_mag / (curl_mag.max() + 1e-8)
            pos_mask = (vx > vy).float()
            neg_mask = (vx <= vy).float()
            r = pos_mask * curl_norm
            g = (pos_mask + neg_mask) * (1.0 - curl_norm)
            b = neg_mask * curl_norm
        elif color_scheme == "plasma":
            time_t = torch.tensor(time, device=device, dtype=vx.dtype)
            r = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + time_t)
            g = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + normalized * 3.14159 + time_t * 2.0)
            b = 0.5 + 0.5 * torch.cos(vangle * 3.14159 + normalized * 6.28318 + time_t * 3.0)
        else:
            r, g, b = vx, vy, vmag
        
        # Apply intensity
        if color_intensity < 1.0:
            grayscale = (r + g + b) / 3.0
            r = r * color_intensity + grayscale * (1 - color_intensity)
            g = g * color_intensity + grayscale * (1 - color_intensity)
            b = b * color_intensity + grayscale * (1 - color_intensity)
        
        result = torch.cat([r, g, b], dim=1)
        
        # Channels past the palette's three are each the magnitude of a velocity
        # field of their own, instead of copies of this one's.
        return BaseNoiseGenerator.fill_channels(render, result, target_channels, seed)
    
    @staticmethod
    def _generate_channels(
        advected: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        render,
        target_channels: int,
        params: ShaderParams,
        seed: int,
        render_many=None,
    ) -> torch.Tensor:
        """Generate channels without color scheme."""
        vmag = torch.sqrt(vx**2 + vy**2)
        vmag = vmag / (vmag.max() + 1e-8)
        
        fast_mode = params.get("fast_high_channel_noise", False)
        
        if fast_mode and target_channels > HIGH_CHANNEL_THRESHOLD:
            # Fast mode: tile base channels
            base = torch.cat([vx, vy, vmag, advected], dim=1)
            num_repeats = math.ceil(target_channels / base.shape[1])
            return torch.cat([base] * num_repeats, dim=1)[:, :target_channels]

        # The four native channels are different fields already. Every channel past
        # them is an advected field drawn from a velocity field of its own -- not one
        # component of a reseeded field put through sin or abs on a repeating cycle,
        # which held this generator to 25 of 128 channels.
        native = torch.cat([advected, vx, vy, vmag], dim=1)
        return BaseNoiseGenerator.fill_channels(render, native, target_channels, seed,
                                                render_many=render_many)
    
    @staticmethod
    def get_velocity_field(p, time, octaves, device, seed, use_temporal_coherence=False):
        """
        Get curl noise velocity field.
        
        Args:
            p: Coordinate tensor [B, H, W, 2]
            time: Animation time
            octaves: Number of noise octaves
            device: Target device
            seed: Random seed
            use_temporal_coherence: Whether to use temporal coherence
            
        Returns:
            Velocity field [B, H, W, 2]
        """
        # Accumulated as two tensors and joined at the end rather than written in
        # place into a preallocated [B,H,W,2]. `seed` may be a tensor holding one
        # seed per channel, in which case the simplex calls below return a leading
        # axis of that many fields and there is no fixed shape to preallocate.
        # The adds happen in the same order, so the values are unchanged.
        acc_x = acc_y = None
        amp = 1.0
        freq = 1.0
        max_amp = 0.0
        
        for i in range(min(octaves, 8)):
            current_p = p * freq
            
            if use_temporal_coherence:
                time_offset = time * (0.2 + i * 0.05)
                noise_x = CurlNoiseGenerator._simplex_3d(current_p, seed + i * 100, time_offset)
                noise_y = CurlNoiseGenerator._simplex_3d(current_p, seed + i * 100 + 50, time_offset + 1.0)
            else:
                noise_x = CurlNoiseGenerator._simplex_2d(current_p, seed + i * 100)
                noise_y = CurlNoiseGenerator._simplex_2d(current_p, seed + i * 100 + 50)
            
            acc_x = amp * noise_x if acc_x is None else acc_x + amp * noise_x
            acc_y = amp * noise_y if acc_y is None else acc_y + amp * noise_y
            max_amp += amp
            
            freq *= 2.0
            amp *= 0.5
        
        return torch.cat([acc_x, acc_y], dim=-1) / max_amp
    
    @staticmethod
    def apply_warp_intensity(velocity, warp_strength):
        """Apply warp intensity to velocity field."""
        return velocity * warp_strength
    
    @staticmethod
    def advect(p, velocity, time, dt, octaves, scale, device, seed, use_temporal_coherence=False):
        """
        Advect noise along velocity field.
        
        Args:
            p: Coordinate tensor [B, H, W, 2]
            velocity: Velocity field [B, H, W, 2]
            time: Animation time
            dt: Time step
            octaves: Number of octaves
            scale: Scale factor
            device: Target device
            seed: Random seed
            use_temporal_coherence: Whether to use temporal coherence
            
        Returns:
            Advected noise [B, H, W, 1]
        """
        # Simple advection: sample noise at displaced position
        advected_p = p + velocity * dt
        
        if use_temporal_coherence:
            result = CurlNoiseGenerator._simplex_3d(advected_p * scale, seed + 999, time * 0.3)
        else:
            result = CurlNoiseGenerator._simplex_2d(advected_p * scale, seed + 999)
        
        return result
    
    @staticmethod
    def _simplex_2d(p, seed):
        """2D simplex noise, unrotated. See shaders/simplex.py."""
        return simplex_2d(p, seed, rotate=False)

    @staticmethod
    def _simplex_3d(coords, seed=0, time_offset=0.0):
        """3D simplex noise with time as the third axis, first corner only."""
        z = torch.ones_like(coords[..., 0:1]) * time_offset
        return simplex_3d(torch.cat([coords[..., 0:1], coords[..., 1:2], z], dim=-1),
                          seed, corners=1)

def add_curl_noise_to_tensor(tensor_class):
    """Legacy function for backward compatibility."""
    pass


def register_shader_generator(generators_dict):
    """Legacy function for backward compatibility."""
    generators_dict["curl_noise"] = generate_curl_noise_tensor


def generate_curl_noise_tensor(
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
    Generate curl noise tensor.
    
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
    
    return CurlNoiseGenerator.generate(
        batch_size=batch_size,
        height=height,
        width=width,
        params=params,
        device=torch.device(device),
        seed=seed,
        target_channels=target_channels
    )