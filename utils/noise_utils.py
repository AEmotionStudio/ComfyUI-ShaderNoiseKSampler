"""
Noise generation utilities for shader noise.

This module provides centralized noise generation functions including
simplex noise, FBM, and gradient noise used by all shader generators.
"""

import torch

# Hash constant used in noise functions
HASH_CONSTANT = 43758.5453


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
        dtype: Data type for the tensor
        range_type: "unit" for [0, 1], "centered" for [-0.5, 0.5], "symmetric" for [-1, 1]
        
    Returns:
        Coordinate tensor [B, H, W, 2] with (x, y) coordinates
    """
    y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype)
    x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype)
    
    if range_type == "centered":
        y_coords = y_coords - 0.5
        x_coords = x_coords - 0.5
    elif range_type == "symmetric":
        y_coords = y_coords * 2.0 - 1.0
        x_coords = x_coords * 2.0 - 1.0
    
    # Create meshgrid and stack
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack to [H, W, 2] then expand to [B, H, W, 2]
    coords = torch.stack([xx, yy], dim=-1)
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    return coords
