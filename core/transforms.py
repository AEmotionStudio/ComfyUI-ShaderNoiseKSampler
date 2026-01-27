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
    import torch.nn.functional as F
    
    input_dims = len(noise.shape)
    num_spatial_dims = len(target_size)
    
    if mode is None:
        # Auto-detect interpolation mode based on INPUT tensor dimensionality
        # This is critical: a 5D tensor needs trilinear or special handling,
        # even if target_size is only (H, W)
        if input_dims == 5:
            # 5D video tensor - need trilinear or frame-by-frame handling
            if num_spatial_dims == 3:
                mode = 'trilinear'
            else:
                # target_size is (H, W) but input is 5D
                # We need to handle this specially - resize frame by frame
                return _resize_5d_tensor_spatial(noise, target_size)
        elif input_dims == 4:
            mode = 'bilinear'
        elif input_dims == 3:
            mode = 'linear'
        else:
            mode = 'nearest'
    
    align_corners = False if mode in ['linear', 'bilinear', 'trilinear'] else None
    
    return F.interpolate(
        noise,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )


def _resize_5d_tensor_spatial(
    noise: torch.Tensor,
    target_size: tuple
) -> torch.Tensor:
    """
    Resize a 5D tensor's spatial dimensions (H, W) using frame-by-frame bilinear.
    
    Args:
        noise: 5D input tensor (B,C,F,H,W or B,F,C,H,W format)
        target_size: Target spatial dimensions (H, W)
        
    Returns:
        Resized 5D tensor
    """
    import torch.nn.functional as F
    
    # Detect format using channel heuristics
    channel_dim = _detect_video_channel_dim(noise)
    frame_dim = 2 if channel_dim == 1 else 1
    
    num_frames = noise.shape[frame_dim]
    frames_list = []
    
    for i in range(num_frames):
        if frame_dim == 1:
            frame = noise[:, i]  # B,C,H,W
        else:
            frame = noise[:, :, i]  # B,C,H,W
        
        frame_resized = F.interpolate(
            frame,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        frames_list.append(frame_resized)
    
    if frame_dim == 1:
        return torch.stack(frames_list, dim=1)
    else:
        return torch.stack(frames_list, dim=2)


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


def _detect_video_channel_dim(tensor: torch.Tensor) -> int:
    """
    Detect the channel dimension index for a 5D video tensor.

    Uses heuristics based on common latent channel counts to determine
    whether the format is B,C,F,H,W (channel_dim=1) or B,F,C,H,W (channel_dim=2).

    Args:
        tensor: 5D video tensor

    Returns:
        Channel dimension index (1 or 2)
    """
    if len(tensor.shape) != 5:
        return 1  # Default for non-5D tensors

    # Common latent channel counts
    common_channels = {4, 8, 12, 16, 32, 64, 128}

    dim1 = tensor.shape[1]
    dim2 = tensor.shape[2]

    # If dim1 looks like channels and dim2 doesn't -> B,C,F,H,W
    if dim1 in common_channels and dim2 not in common_channels:
        return 1
    # If dim2 looks like channels and dim1 doesn't -> B,F,C,H,W
    elif dim2 in common_channels and dim1 not in common_channels:
        return 2
    # Default to B,F,C,H,W format (most common in video models)
    # This must match the default in blending.py and sampler.py
    else:
        return 2


def match_noise_shape(
    noise: torch.Tensor,
    target_shape: tuple,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    is_video: bool = False,
    channel_dim: Optional[int] = None
) -> torch.Tensor:
    """
    Match noise tensor to target shape, handling both channel and spatial mismatches.

    Args:
        noise: Input noise tensor
        target_shape: Target shape to match
        device: Device for the output tensor
        dtype: Data type for the output tensor
        is_video: Whether this is a video tensor (5D)
        channel_dim: Optional explicit channel dimension index. If None, will be
                     auto-detected for video tensors using heuristics.

    Returns:
        Reshaped noise tensor
    """
    if noise.shape == target_shape:
        return noise

    if device is None:
        device = noise.device
    if dtype is None:
        dtype = noise.dtype

    # Determine channel dimension
    if channel_dim is not None:
        # Use explicitly provided channel_dim
        pass
    elif is_video and len(noise.shape) == 5:
        # Auto-detect for video tensors
        channel_dim = _detect_video_channel_dim(noise)
    else:
        # Default for image tensors
        channel_dim = 1

    spatial_dims_start = 3 if (is_video and len(noise.shape) == 5) else 2
    
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
