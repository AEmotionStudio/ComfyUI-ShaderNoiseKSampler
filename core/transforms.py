"""
Noise transformation operations.
"""
import torch
import math
from typing import Optional

from .constants import SUPPORTED_TRANSFORMS


def apply_noise_transform(noise: torch.Tensor, transform: str) -> torch.Tensor:
    """
    Apply mathematical transformations to noise.
    
    Args:
        noise: Input noise tensor
        transform: Transform type (one of SUPPORTED_TRANSFORMS)
        
    Returns:
        Transformed noise tensor
    """
    if transform == "none":
        return noise
    elif transform == "reverse":
        return -noise
    elif transform == "inverse":
        # Add small epsilon to avoid division by zero
        return 1.0 / (noise + 1e-8)
    elif transform == "absolute":
        return torch.abs(noise)
    elif transform == "square":
        return noise ** 2
    elif transform == "sqrt":
        return torch.sqrt(torch.abs(noise))
    elif transform == "log":
        return torch.log(torch.abs(noise) + 1.0)
    elif transform == "sin":
        return torch.sin(noise * math.pi)
    elif transform == "cos":
        return torch.cos(noise * math.pi)
    else:
        return noise  # Default to no transform


def normalize_noise(noise: torch.Tensor) -> torch.Tensor:
    """
    Normalize noise tensor to have standard deviation of 1.0 and mean of 0.0.
    This ensures consistent blending behavior regardless of the underlying distribution.
    
    Args:
        noise: Input noise tensor
        
    Returns:
        Normalized noise tensor
    """
    if noise.numel() == 0:
        return noise
        
    # Calculate current mean and standard deviation
    current_mean = noise.mean()
    current_std = noise.std()
    
    # Only normalize if standard deviation is not very close to zero
    if current_std > 1e-6:
        # Normalize: (x - mean) / std
        noise = (noise - current_mean) / current_std
    
    return noise


def resize_noise_spatial(
    noise: torch.Tensor,
    target_size: tuple,
    mode: Optional[str] = None
) -> torch.Tensor:
    """
    Resize noise tensor spatial dimensions using interpolation.
    
    Args:
        noise: Input noise tensor
        target_size: Target spatial dimensions (H, W) or (D, H, W)
        mode: Interpolation mode (auto-detected if None)
        
    Returns:
        Resized noise tensor
    """
    num_spatial_dims = len(target_size)
    
    if mode is None:
        # Auto-detect interpolation mode based on dimensions
        if num_spatial_dims == 1:
            mode = 'linear'
        elif num_spatial_dims == 2:
            mode = 'bilinear'
        elif num_spatial_dims == 3:
            mode = 'trilinear'
        else:
            mode = 'nearest'
    
    align_corners = False if mode in ['linear', 'bilinear', 'trilinear'] else None
    
    return torch.nn.functional.interpolate(
        noise,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )


def resize_noise_channels(
    noise: torch.Tensor,
    target_channels: int,
    channel_dim: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Resize noise tensor to have the target number of channels.
    
    Args:
        noise: Input noise tensor
        target_channels: Target number of channels
        channel_dim: Index of the channel dimension
        device: Device for the output tensor
        dtype: Data type for the output tensor
        
    Returns:
        Resized noise tensor
    """
    if device is None:
        device = noise.device
    if dtype is None:
        dtype = noise.dtype
        
    current_channels = noise.shape[channel_dim]
    
    if current_channels == target_channels:
        return noise
    
    # Build new shape
    new_shape = list(noise.shape)
    new_shape[channel_dim] = target_channels
    
    # Create new tensor
    new_noise = torch.randn(new_shape, device=device, dtype=dtype)
    
    # Copy existing channels
    min_channels = min(current_channels, target_channels)
    
    # Build slicing tuples dynamically
    src_slices = [slice(None)] * len(noise.shape)
    dst_slices = [slice(None)] * len(noise.shape)
    src_slices[channel_dim] = slice(0, min_channels)
    dst_slices[channel_dim] = slice(0, min_channels)
    
    new_noise[tuple(dst_slices)] = noise[tuple(src_slices)]
    
    return new_noise


def match_noise_shape(
    noise: torch.Tensor,
    target_shape: tuple,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    is_video: bool = False
) -> torch.Tensor:
    """
    Match noise tensor to target shape, handling both channel and spatial mismatches.
    
    Args:
        noise: Input noise tensor
        target_shape: Target shape to match
        device: Device for the output tensor
        dtype: Data type for the output tensor
        is_video: Whether this is a video tensor (5D)
        
    Returns:
        Reshaped noise tensor
    """
    if noise.shape == target_shape:
        return noise
    
    if device is None:
        device = noise.device
    if dtype is None:
        dtype = noise.dtype
    
    channel_dim = 2 if is_video else 1
    spatial_dims_start = 3 if is_video else 2
    
    result = noise
    
    # Handle channel mismatch
    if noise.shape[channel_dim] != target_shape[channel_dim]:
        result = resize_noise_channels(
            result,
            target_shape[channel_dim],
            channel_dim=channel_dim,
            device=device,
            dtype=dtype
        )
    
    # Handle spatial dimension mismatch
    target_spatial = target_shape[spatial_dims_start:]
    if result.shape[spatial_dims_start:] != target_spatial:
        result = resize_noise_spatial(result, target_spatial)
    
    return result
