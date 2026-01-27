"""
Shape mask utilities for shader noise generation.

This module provides centralized shape mask generation functions
used by all shader generators.
"""

import torch
import math
from typing import Optional, Union

# List of all supported shape types
SHAPE_TYPES = [
    "none",
    "circle",
    "square",
    "radial",
    "linear",
    "spiral",
    "checkerboard",
    "spots",
    "hexgrid",
    "stripes",
    "gradient",
    "gradient_x",
    "gradient_y",
    "vignette",
    "cross",
    "stars",
    "triangles",
    "concentric",
    "rays",
    "zigzag",
]


def smoothstep(edge0: Union[float, torch.Tensor], edge1: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Perform smooth Hermite interpolation between 0 and 1.
    
    Args:
        edge0: Lower edge of the Hermite function
        edge1: Upper edge of the Hermite function
        x: Source value for interpolation
        
    Returns:
        Smoothly interpolated value clamped to [0, 1]
    """
    if not isinstance(edge0, torch.Tensor):
        edge0 = torch.full_like(x, float(edge0), device=x.device, dtype=x.dtype)
    if not isinstance(edge1, torch.Tensor):
        edge1 = torch.full_like(x, float(edge1), device=x.device, dtype=x.dtype)
    
    delta = edge1 - edge0
    # Avoid division by zero
    safe_delta = torch.where(
        torch.abs(delta) < 1e-8,
        torch.sign(delta) * 1e-8 + 1e-8 * (1 - torch.abs(torch.sign(delta))),
        delta
    )
    t = torch.clamp((x - edge0) / safe_delta, 0.0, 1.0)
    
    return t * t * (3.0 - 2.0 * t)


def _random_val(coords: torch.Tensor, seed_offset: int, base_seed: int = 0) -> torch.Tensor:
    """
    Generate pseudo-random values based on coordinates.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        seed_offset: Offset added to base seed
        base_seed: Base random seed
        
    Returns:
        Random values tensor [B, H, W]
    """
    torch.manual_seed(base_seed + seed_offset)
    hash_val = torch.sin(
        coords[:, :, :, 0] * (12.9898 + seed_offset) + 
        coords[:, :, :, 1] * (78.233 + seed_offset)
    ) * 43758.5453
    return torch.frac(hash_val)


def apply_shape_mask(
    coords: torch.Tensor,
    shape_type: str,
    time: float = 0.0,
    base_seed: int = 0,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Generate and apply a shape mask to coordinates.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2] in [0, 1] range
        shape_type: Type of shape mask to apply
        time: Animation time for animated masks
        base_seed: Base seed for random-based masks
        strength: Mask strength [0.0 to 1.0]
        
    Returns:
        Shape mask tensor [B, H, W, 1]
    """
    if shape_type in ["none", "0"] or strength <= 0:
        batch, height, width, _ = coords.shape
        return torch.ones((batch, height, width, 1), device=coords.device, dtype=coords.dtype)
    
    device = coords.device
    dtype = coords.dtype
    batch, height, width, _ = coords.shape
    
    # Compute common values
    centered = coords - 0.5  # Center coordinates at (0, 0)
    center_dist = torch.sqrt(centered[:, :, :, 0]**2 + centered[:, :, :, 1]**2)
    
    mask = None
    
    if shape_type == "circle":
        center_x, center_y = 0.5, 0.5
        y_diff = coords[:, :, :, 1] - center_y
        x_diff = coords[:, :, :, 0] - center_x
        dist = torch.sqrt(x_diff**2 + y_diff**2)
        mask = 1.0 - torch.clamp(dist * 2, 0, 1)
    
    elif shape_type == "square":
        x_mask = torch.abs(coords[:, :, :, 0] - 0.5) * 2
        y_mask = torch.abs(coords[:, :, :, 1] - 0.5) * 2
        dist = torch.max(x_mask, y_mask)
        mask = 1.0 - torch.clamp(dist, 0, 1)
    
    elif shape_type == "radial":
        time_t = torch.tensor(time, device=device, dtype=dtype)
        center_x = 0.5 + 0.2 * torch.cos(time_t)
        center_y = 0.5 + 0.2 * torch.sin(time_t)
        y_diff = coords[:, :, :, 1] - center_y
        x_diff = coords[:, :, :, 0] - center_x
        dist = torch.sqrt(x_diff**2 + y_diff**2) * 2.0
        mask = torch.clamp(1.0 - dist, 0.0, 1.0)
    
    elif shape_type == "linear":
        time_t = torch.tensor(time * 0.2, device=device, dtype=dtype)
        x_offset = torch.fmod(time_t, 1.0) * 2.0
        shifted_x = torch.fmod(coords[:, :, :, 0] + x_offset, 1.0)
        mask = shifted_x
    
    elif shape_type == "spiral":
        theta = torch.atan2(centered[:, :, :, 1], centered[:, :, :, 0])
        r = torch.norm(centered, dim=-1) * 2.0
        time_t = torch.tensor(time, device=device, dtype=dtype)
        theta = theta + time_t
        mask = torch.fmod((theta / (2.0 * math.pi) + r), 1.0)
    
    elif shape_type == "checkerboard":
        grid_size = 8.0
        time_gs_02 = torch.tensor(time * grid_size * 0.2, device=device, dtype=dtype)
        time_gs_01 = torch.tensor(time * grid_size * 0.1, device=device, dtype=dtype)
        x_offset = time_gs_02
        y_offset = time_gs_01
        x_grid = torch.floor((coords[:, :, :, 0] + x_offset / grid_size) * grid_size) * 0.5
        y_grid = torch.floor((coords[:, :, :, 1] + y_offset / grid_size) * grid_size) * 0.5
        mask = torch.fmod(x_grid + y_grid, 1.0)
    
    elif shape_type == "spots":
        mask = torch.zeros((batch, height, width), device=device, dtype=dtype)
        num_spots = 10
        time_t = torch.tensor(time, device=device, dtype=dtype)
        
        for i in range(num_spots):
            rand_x = _random_val(coords, i * 78, base_seed)
            rand_y = _random_val(coords, i * 12, base_seed)
            size = (_random_val(coords, i * 93, base_seed) * 0.3 + 0.1)
            
            angle_t = torch.tensor(time + float(i), device=device, dtype=dtype)
            spot_pos_x = 0.5 + torch.cos(angle_t) * 0.4 * rand_x
            spot_pos_y = 0.5 + torch.sin(angle_t) * 0.4 * rand_y
            
            size_angle = torch.tensor(time * 2.0 + float(i), device=device, dtype=dtype)
            size = size * (1.0 + 0.2 * torch.sin(size_angle))
            
            dist = torch.sqrt((coords[:, :, :, 0] - spot_pos_x)**2 + (coords[:, :, :, 1] - spot_pos_y)**2)
            spot_mask = torch.clamp(1.0 - dist / size, 0.0, 1.0)
            mask = torch.maximum(mask, spot_mask)
    
    elif shape_type == "hexgrid":
        hex_uv = coords.clone() * 6.0
        time_05 = torch.tensor(time * 0.5, device=device, dtype=dtype)
        time_03 = torch.tensor(time * 0.3, device=device, dtype=dtype)
        time_t = torch.tensor(time, device=device, dtype=dtype)
        
        hex_uv[:, :, :, 0] = hex_uv[:, :, :, 0] + torch.sin(time_05) * 0.5
        hex_uv[:, :, :, 1] = hex_uv[:, :, :, 1] + torch.cos(time_03) * 0.5
        
        r = torch.tensor([1.0, 1.73], device=device, dtype=dtype).reshape(1, 1, 1, 2)
        h = r * 0.5
        a = torch.fmod(hex_uv, r) - h
        b = torch.fmod(hex_uv + h, r) - h
        
        dist = torch.minimum(torch.norm(a, dim=-1), torch.norm(b, dim=-1))
        cell_size = 0.3 + 0.1 * torch.sin(time_t)
        mask = smoothstep(cell_size + 0.05, cell_size - 0.05, dist)
    
    elif shape_type == "stripes":
        freq = 10.0
        time_t = torch.tensor(time, device=device, dtype=dtype)
        time_02 = torch.tensor(time * 0.2, device=device, dtype=dtype)
        angle = 0.5 * torch.sin(time_02)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotated_x = coords[:, :, :, 0] * cos_a - coords[:, :, :, 1] * sin_a
        stripes = torch.sin(rotated_x * freq + time_t)
        mask = smoothstep(0.0, 0.1, stripes) * smoothstep(0.0, -0.1, -stripes)
    
    elif shape_type == "gradient":
        time_02 = torch.tensor(time * 0.2, device=device, dtype=dtype)
        angle = time_02
        dir_x = torch.cos(angle)
        dir_y = torch.sin(angle)
        proj = (coords[:, :, :, 0] - 0.5) * dir_x + (coords[:, :, :, 1] - 0.5) * dir_y + 0.5
        mask = proj
    
    elif shape_type == "gradient_x":
        mask = coords[:, :, :, 0]
    
    elif shape_type == "gradient_y":
        mask = coords[:, :, :, 1]
    
    elif shape_type == "vignette":
        time_03 = torch.tensor(time * 0.3, device=device, dtype=dtype)
        time_04 = torch.tensor(time * 0.4, device=device, dtype=dtype)
        time_05 = torch.tensor(time * 0.5, device=device, dtype=dtype)
        center_x = 0.5 + 0.2 * torch.sin(time_03)
        center_y = 0.5 + 0.2 * torch.cos(time_04)
        dist = torch.sqrt((coords[:, :, :, 0] - center_x)**2 + (coords[:, :, :, 1] - center_y)**2)
        radius = 0.6 + 0.2 * torch.sin(time_05)
        smoothness = 0.3
        mask = 1.0 - smoothstep(radius - smoothness, radius, dist)
    
    elif shape_type == "cross":
        time_t = torch.tensor(time, device=device, dtype=dtype)
        time_02 = torch.tensor(time * 0.2, device=device, dtype=dtype)
        thickness = 0.1 + 0.05 * torch.sin(time_t)
        rotation = time_02
        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)
        centered_x = coords[:, :, :, 0] - 0.5
        centered_y = coords[:, :, :, 1] - 0.5
        rotated_x = centered_x * cos_r - centered_y * sin_r + 0.5
        rotated_y = centered_x * sin_r + centered_y * cos_r + 0.5
        
        h_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated_y) * \
                smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated_y)
        v_bar = smoothstep(0.5 - thickness, 0.5 - thickness + 0.02, rotated_x) * \
                smoothstep(0.5 + thickness, 0.5 + thickness - 0.02, rotated_x)
        mask = torch.maximum(h_bar, v_bar)
    
    elif shape_type == "stars":
        mask = torch.zeros((batch, height, width), device=device, dtype=dtype)
        num_stars = 20
        time_t = torch.tensor(time, device=device, dtype=dtype)
        time_01 = torch.tensor(time * 0.1, device=device, dtype=dtype)
        time_015 = torch.tensor(time * 0.15, device=device, dtype=dtype)
        
        for i in range(num_stars):
            rand_x = _random_val(coords, i * 78 + 10, base_seed)
            rand_y = _random_val(coords, i * 12 + 20, base_seed)
            
            time_sin_arg = torch.tensor(float(i), device=device, dtype=dtype) + time_01
            time_cos_arg = torch.tensor(float(i) * 1.5, device=device, dtype=dtype) + time_015
            star_pos_x = torch.fmod(rand_x + 0.05 * torch.sin(time_sin_arg), 1.0)
            star_pos_y = torch.fmod(rand_y + 0.05 * torch.cos(time_cos_arg), 1.0)
            
            brightness_arg = torch.tensor(float(i), device=device, dtype=dtype) + time_t * (0.5 + rand_x * 0.5)
            brightness = 0.5 + 0.5 * torch.sin(brightness_arg)
            size = 0.01 + 0.015 * rand_y * brightness
            
            dist = torch.sqrt((coords[:, :, :, 0] - star_pos_x)**2 + (coords[:, :, :, 1] - star_pos_y)**2)
            star_mask = smoothstep(size, size * 0.5, dist) * brightness
            mask = torch.maximum(mask, star_mask)
    
    elif shape_type == "triangles":
        t_sin = torch.tensor(time * 0.2, device=device, dtype=dtype)
        t_cos = torch.tensor(time * 0.7, device=device, dtype=dtype)
        t_border = torch.tensor(time * 1.5, device=device, dtype=dtype)
        scale_factor = 5.0
        uv = coords.clone() * scale_factor
        uv[:, :, :, 0] = uv[:, :, :, 0] + torch.sin(t_sin) * 0.5
        uv[:, :, :, 1] = uv[:, :, :, 1] + torch.cos(t_cos) * 0.5
        
        gv = torch.fmod(uv, 1.0) - 0.5
        
        d1 = torch.abs(gv[:, :, :, 0] + gv[:, :, :, 1])
        d2 = torch.abs(gv[:, :, :, 0] - gv[:, :, :, 1])
        d3 = torch.abs(gv[:, :, :, 0]) * 0.866 + torch.abs(gv[:, :, :, 1]) * 0.5
        
        d = torch.minimum(torch.minimum(d1, d2), d3) * 0.7
        border_width = 0.05 + 0.03 * torch.sin(t_border)
        mask = smoothstep(border_width, border_width - 0.02, d)
    
    elif shape_type == "concentric":
        time_03 = torch.tensor(time * 0.3, device=device, dtype=dtype)
        time_04 = torch.tensor(time * 0.4, device=device, dtype=dtype)
        time_01 = torch.tensor(time * 0.1, device=device, dtype=dtype)
        time_05 = torch.tensor(time * 0.5, device=device, dtype=dtype)
        
        center_x = 0.5 + 0.2 * torch.sin(time_03)
        center_y = 0.5 + 0.2 * torch.cos(time_04)
        dist = torch.sqrt((coords[:, :, :, 0] - center_x)**2 + (coords[:, :, :, 1] - center_y)**2)
        freq = 10.0 + 5.0 * torch.sin(time_01)
        phase = time_05
        rings = torch.sin(dist * freq + phase)
        mask = smoothstep(0.0, 0.1, rings) * smoothstep(0.0, -0.1, -rings)
    
    elif shape_type == "rays":
        time_03 = torch.tensor(time * 0.3, device=device, dtype=dtype)
        time_04 = torch.tensor(time * 0.4, device=device, dtype=dtype)
        time_05 = torch.tensor(time * 0.5, device=device, dtype=dtype)
        
        center_x = 0.5 + 0.1 * torch.sin(time_03)
        center_y = 0.5 + 0.1 * torch.cos(time_04)
        to_center_x = coords[:, :, :, 0] - center_x
        to_center_y = coords[:, :, :, 1] - center_y
        angle = torch.atan2(to_center_y, to_center_x)
        freq = 8.0
        phase = time_05
        rays_val = torch.sin(angle * freq + phase)
        dist = torch.sqrt(to_center_x**2 + to_center_y**2)
        falloff = 1.0 - smoothstep(0.0, 0.8, dist)
        mask = smoothstep(0.0, 0.3, rays_val) * falloff
    
    elif shape_type == "zigzag":
        freq = 10.0
        time_t = torch.tensor(time, device=device, dtype=dtype)
        time_02 = torch.tensor(time * 0.2, device=device, dtype=dtype)
        time_05 = torch.tensor(time * 0.5, device=device, dtype=dtype)
        time_03 = torch.tensor(time * 0.3, device=device, dtype=dtype)
        
        angle = 0.5 * torch.sin(time_02)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotated_x = coords[:, :, :, 0] * cos_a - coords[:, :, :, 1] * sin_a
        rotated_y = coords[:, :, :, 0] * sin_a + coords[:, :, :, 1] * cos_a
        
        zigzag1 = torch.abs(2.0 * torch.fmod(rotated_x * freq - time_05, 1.0) - 1.0)
        zigzag2 = torch.abs(2.0 * torch.fmod(rotated_y * freq + time_03, 1.0) - 1.0)
        
        zigzag = torch.minimum(zigzag1, zigzag2)
        thickness = 0.3 + 0.1 * torch.sin(time_t)
        mask = torch.heaviside(zigzag - thickness, torch.tensor(0.5, device=device, dtype=dtype))
    
    else:
        # Unknown shape type - return ones
        mask = torch.ones((batch, height, width), device=device, dtype=dtype)
    
    # Ensure mask has shape [B, H, W, 1]
    if mask is not None:
        if len(mask.shape) == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(-1)
            if batch > 1:
                mask = mask.expand(batch, height, width, 1)
        elif len(mask.shape) == 3:  # [B, H, W]
            mask = mask.unsqueeze(-1)
    
    return mask


def apply_mask_to_tensor(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Apply a shape mask to a tensor using linear interpolation.
    
    Args:
        tensor: Input tensor [B, H, W, C] or [B, C, H, W]
        mask: Shape mask [B, H, W, 1]
        strength: Mask strength [0.0 to 1.0]
        
    Returns:
        Masked tensor with same shape as input
    """
    if strength <= 0:
        return tensor
    
    # Determine if tensor is in BCHW or BHWC format
    is_bchw = len(tensor.shape) == 4 and tensor.shape[1] < tensor.shape[2]
    
    if is_bchw:
        # Convert mask from [B, H, W, 1] to [B, 1, H, W]
        mask_bchw = mask.permute(0, 3, 1, 2)
        return torch.lerp(tensor, tensor * mask_bchw, strength)
    else:
        # Tensor is BHWC, mask is already BHWC compatible
        return torch.lerp(tensor, tensor * mask, strength)
