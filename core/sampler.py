"""
Core sampling functionality for shader noise KSampler.

This module contains the main sampling logic, stage calculations,
and shader noise generation coordination.
"""
import torch
import math
import contextlib
from typing import Optional, Dict, Any, List, Tuple, Callable

import torch.nn.functional as F

from .constants import SUPPORTED_DISTRIBUTIONS
from .transforms import normalize_noise
from .params import ShaderParams


def calculate_stage_strengths(
    base_strength: float,
    num_stages: int,
    distribution: str
) -> List[float]:
    """
    Calculate strength for each stage based on distribution type.
    
    Args:
        base_strength: Base shader strength
        num_stages: Number of stages
        distribution: Distribution type (one of SUPPORTED_DISTRIBUTIONS)
        
    Returns:
        List of strength values for each stage
    """
    if num_stages <= 0:
        return []
    
    strengths = []
    
    if distribution == "uniform":
        strengths = [base_strength] * num_stages
    
    elif distribution == "linear_decrease":
        min_factor = 0.25  # Ensure a minimum shader strength in each stage
        for i in range(num_stages):
            factor = 1.0 - (i / (num_stages - 1 if num_stages > 1 else 1))
            factor = max(factor, min_factor)
            strengths.append(base_strength * factor)
    
    elif distribution == "linear_increase":
        min_factor = 0.25  # Ensure a minimum shader strength (same as linear_decrease)
        for i in range(num_stages):
            factor = i / (num_stages - 1 if num_stages > 1 else 1)
            # For single stage, use full strength instead of zero
            if num_stages == 1:
                factor = 1.0
            factor = max(factor, min_factor)
            strengths.append(base_strength * factor)
    
    elif distribution == "gaussian":
        # Create a bell curve with peak in the middle
        mid_point = (num_stages - 1) / 2
        for i in range(num_stages):
            # Standard deviation as 1/3 of the range
            std_dev = num_stages / 3
            factor = math.exp(-((i - mid_point) ** 2) / (2 * std_dev ** 2))
            strengths.append(base_strength * factor)
    
    elif distribution == "first_stronger":
        # First stage strongest, rapid falloff
        for i in range(num_stages):
            factor = math.exp(-i)  # Exponential decay
            strengths.append(base_strength * factor)
    
    elif distribution == "last_stronger":
        # Last stage strongest, exponential increase
        for i in range(num_stages):
            factor = math.exp(i - num_stages + 1)  # Exponential increase
            strengths.append(base_strength * factor)
    
    else:
        # Default to uniform if unknown distribution
        strengths = [base_strength] * num_stages
        
    return strengths


def calculate_step_ranges(total_steps: int, num_stages: int) -> List[Tuple[int, int]]:
    """
    Calculate step ranges for sequential stages.
    
    Args:
        total_steps: Total number of sampling steps
        num_stages: Number of stages
        
    Returns:
        List of (start_step, end_step) tuples
    """
    if num_stages <= 0:
        return []
        
    step_ranges = []
    step_size = total_steps / num_stages
    
    for i in range(num_stages):
        start_step = int(i * step_size)
        end_step = int((i + 1) * step_size) if i < num_stages - 1 else total_steps
        step_ranges.append((start_step, end_step))
        
    return step_ranges


def calculate_step_points(total_steps: int, num_stages: int) -> List[int]:
    """
    Calculate at which steps to apply different shader noises for injection stages.
    
    Args:
        total_steps: Total number of sampling steps
        num_stages: Number of injection stages
        
    Returns:
        List of step points
    """
    # Match behavior of calculate_stage_strengths and calculate_step_ranges
    if num_stages <= 0:
        return []
    if num_stages == 1:
        return [0]
        
    step_points = []
    for i in range(num_stages):
        step_point = int(i * total_steps / (num_stages - 1)) if num_stages > 1 else 0
        step_points.append(min(step_point, total_steps - 1))
        
    return step_points


def generate_shader_noise(
    latent_samples: torch.Tensor,
    target_noise_shape: Tuple[int, ...],
    shader_params: Dict[str, Any],
    shader_type: str,
    seed: int,
    device: str = "cuda",
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    frame_count: int = 1,
    frame_dim_idx: int = -1,
    generator_func: Optional[Callable] = None,
    debugger: Optional[object] = None
) -> torch.Tensor:
    """
    Generate noise using the specified shader.
    
    Args:
        latent_samples: The input latent samples (for context/shape)
        target_noise_shape: Expected shape of the final noise tensor
        shader_params: Parameters for shader generation
        shader_type: Type of shader generator to use
        seed: Random seed
        device: Device to create tensor on
        model: The model object (for channel detection)
        model_name: Optional model name for customized generation
        frame_count: Number of frames to generate
        frame_dim_idx: Index of the frame dimension (-1 if not video)
        generator_func: Shader generator function to use
        debugger: Optional debugger for logging
        
    Returns:
        Generated noise tensor with shape matching target_noise_shape
    """
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Extract shape information from the target_noise_shape
    is_video = len(target_noise_shape) == 5
    batch_size = target_noise_shape[0]
    height = target_noise_shape[-2]
    width = target_noise_shape[-1]
    
    # Get channels from target_noise_shape
    if is_video:
        # Handle default frame_dim_idx of -1 by detecting format from shape
        if frame_dim_idx == -1:
            # Common latent channel counts vs frame counts
            common_channels = {4, 8, 12, 16, 32, 64, 128}
            dim1, dim2 = target_noise_shape[1], target_noise_shape[2]
            # Heuristic: smaller dimension that looks like channels is the channel dim
            if dim1 in common_channels and dim2 not in common_channels:
                channel_dim_idx = 1  # B,C,F,H,W
                frame_dim_idx = 2
            elif dim2 in common_channels and dim1 not in common_channels:
                channel_dim_idx = 2  # B,F,C,H,W
                frame_dim_idx = 1
            else:
                # Default assumption: B,F,C,H,W format (most common in video models)
                # This must match the default in blending.py _detect_video_format()
                channel_dim_idx = 2
                frame_dim_idx = 1
        else:
            # Explicit frame_dim_idx provided
            channel_dim_idx = 2 if frame_dim_idx == 1 else 1
        channels = target_noise_shape[channel_dim_idx]
    else:
        channels = target_noise_shape[1]
        channel_dim_idx = 1
    
    # Add model information to shader params
    shader_params = shader_params.copy()
    shader_params["target_channels"] = channels
    
    if model is not None:
        _add_model_info_to_params(shader_params, model, model_name, debugger)
    
    if debug_enabled and debug_level >= 2:
        print(f"🎨 Generating {shader_type} shader noise for {'video' if is_video else 'image'}")
        print(f"   Target Shape: {target_noise_shape}, Device: {device}, Seed: {seed}")
        if model_name:
            print(f"   Model: {model_name}")
        print(f"   Target channels: {channels}")
    
    # Generate noise based on tensor type
    if is_video:
        noise = _generate_video_noise(
            target_noise_shape, shader_params, seed, device, 
            frame_count, frame_dim_idx, channels, height, width,
            batch_size, generator_func, latent_samples.dtype, debugger
        )
    else:
        noise = _generate_image_noise(
            target_noise_shape, shader_params, seed, device,
            channels, height, width, batch_size,
            generator_func, latent_samples.dtype, debugger
        )
    
    # Normalize the noise tensor
    noise = normalize_noise(noise)
    
    # Analyze and log the generated noise for debugging
    if debug_enabled and hasattr(debugger, 'analyze_tensor'):
        debugger.analyze_tensor(noise, "shader_noise")
    
    # Final shape check and correction
    if noise.shape != target_noise_shape:
        noise = _correct_noise_shape(noise, target_noise_shape, channel_dim_idx, device, latent_samples.dtype)
    
    return noise


def _add_model_info_to_params(
    shader_params: Dict[str, Any],
    model: Any,
    model_name: Optional[str],
    debugger: Optional[object]
) -> None:
    """Add model information to shader parameters."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    # Get and store model name
    if model_name is None:
        model_name = getattr(model, 'model_name', None)
        if model_name is None and hasattr(model, 'model') and hasattr(model.model, 'model_name'):
            model_name = model.model.model_name
    
    if model_name:
        shader_params["model_name"] = model_name
        if debug_enabled and debug_level >= 2:
            print(f"📝 Added model_name='{model_name}' to shader_params")
    
    # Add model class name
    shader_params["model_class"] = model.__class__.__name__
    
    # Add inner model class name if it exists
    if hasattr(model, 'model'):
        shader_params["inner_model_class"] = model.model.__class__.__name__
        inner_model_name = getattr(model.model, 'model_name', None)
        if inner_model_name:
            shader_params["inner_model_name"] = inner_model_name


def _generate_video_noise(
    target_noise_shape: Tuple[int, ...],
    shader_params: Dict[str, Any],
    seed: int,
    device: str,
    frame_count: int,
    frame_dim_idx: int,
    channels: int,
    height: int,
    width: int,
    batch_size: int,
    generator_func: Optional[Callable],
    dtype: torch.dtype,
    debugger: Optional[object]
) -> torch.Tensor:
    """Generate noise for video latents."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    if debug_enabled and debug_level >= 2:
        print(f"   Video dimensions: Target={target_noise_shape}, Frames={frame_count}")
    
    if generator_func is None:
        # Return random noise if no generator
        return torch.randn(target_noise_shape, device=device, dtype=dtype)
    
    # Generate noise frame by frame
    noise_frames = []
    
    for frame_idx in range(frame_count):
        # Update time parameter for temporal variation
        frame_params = shader_params.copy()
        frame_time_increment = 0.1 if frame_count <= 1 else (1.0 / (frame_count - 1)) * frame_idx
        frame_params["time"] = shader_params.get("time", 0.0) + frame_time_increment
        
        if debug_enabled and debug_level >= 3:
            print(f"   Generating frame {frame_idx} with time={frame_params['time']:.2f}")
        
        # Prepare arguments - use "params" to match BaseNoiseGenerator.generate() signature
        # Wrap dict in ShaderParams for attribute-style access and validate
        generator_args = {
            "params": ShaderParams(frame_params).validate(),
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "device": device,
            "seed": seed + frame_idx if not shader_params.get("useTemporalCoherence", False) else seed,
            "target_channels": channels
        }
        
        # Generate the noise for this frame
        frame_noise = generator_func(**generator_args)
        
        # Validate and correct frame shape
        expected_frame_shape = (batch_size, channels, height, width)
        if frame_noise.shape != expected_frame_shape:
            frame_noise = _correct_frame_shape(frame_noise, expected_frame_shape, device, dtype)
        
        noise_frames.append(frame_noise)
    
    # Stack frames along the correct dimension
    if noise_frames:
        return torch.stack(noise_frames, dim=frame_dim_idx)
    else:
        return torch.zeros(target_noise_shape, device=device, dtype=dtype)


def _generate_image_noise(
    target_noise_shape: Tuple[int, ...],
    shader_params: Dict[str, Any],
    seed: int,
    device: str,
    channels: int,
    height: int,
    width: int,
    batch_size: int,
    generator_func: Optional[Callable],
    dtype: torch.dtype,
    debugger: Optional[object]
) -> torch.Tensor:
    """Generate noise for image latents."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    if debug_enabled and debug_level >= 2:
        print(f"   Image dimensions: Target={target_noise_shape}")
    
    if generator_func is None:
        # Return random noise if no generator
        return torch.randn(target_noise_shape, device=device, dtype=dtype)
    
    # Prepare arguments - use "params" to match BaseNoiseGenerator.generate() signature
    # Wrap dict in ShaderParams for attribute-style access and validate
    generator_args = {
        "params": ShaderParams(shader_params).validate(),
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "target_channels": channels
    }
    
    # Generate the noise
    noise = generator_func(**generator_args)
    
    # Ensure image has correct final shape
    if noise.shape != target_noise_shape:
        noise = _correct_image_shape(noise, target_noise_shape, channels, height, width, device, dtype)
    
    return noise


def _correct_frame_shape(
    frame_noise: torch.Tensor,
    expected_shape: Tuple[int, ...],
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    """Correct frame noise shape to match expected shape."""
    try:
        # FIRST fix spatial dimensions with interpolation
        # This must happen before channel handling to prevent shape mismatches
        if frame_noise.shape[2:] != expected_shape[2:]:
            frame_noise = F.interpolate(
                frame_noise, 
                size=(expected_shape[2], expected_shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
        
        # THEN fix channel count - now spatial dims match
        if frame_noise.shape[1] != expected_shape[1]:
            corrected_frame = torch.zeros(expected_shape, device=device, dtype=dtype)
            min_c = min(frame_noise.shape[1], expected_shape[1])
            corrected_frame[:, :min_c] = frame_noise[:, :min_c]
            frame_noise = corrected_frame
        
        # Final check
        if frame_noise.shape != expected_shape:
            raise ValueError("Shape correction failed")
            
    except Exception:
        # Create a zero tensor as fallback
        frame_noise = torch.zeros(expected_shape, device=device, dtype=dtype)
    
    return frame_noise


def _correct_image_shape(
    noise: torch.Tensor,
    target_shape: Tuple[int, ...],
    channels: int,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    """Correct image noise shape to match target shape."""
    try:
        # FIRST fix spatial dimensions with interpolation
        # This must happen before channel handling to prevent shape mismatches
        if noise.shape[2:] != target_shape[2:]:
            noise = F.interpolate(
                noise, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # THEN fix channel count - now spatial dims match
        if noise.shape[1] != channels:
            corrected_noise = torch.zeros(target_shape, device=device, dtype=dtype)
            min_c = min(noise.shape[1], channels)
            corrected_noise[:, :min_c] = noise[:, :min_c]
            noise = corrected_noise
        
        if noise.shape != target_shape:
            raise ValueError("Shape correction failed")
            
    except Exception:
        noise = torch.zeros(target_shape, device=device, dtype=dtype)
    
    return noise


def _correct_noise_shape(
    noise: torch.Tensor,
    target_shape: Tuple[int, ...],
    channel_dim_idx: int,
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    """Final correction of noise shape to match target."""

    try:
        if len(noise.shape) == 5:
            # 5D video tensor - handle both B,C,F,H,W and B,F,C,H,W formats
            # Determine frame dimension from channel_dim_idx
            frame_dim_idx = 2 if channel_dim_idx == 1 else 1

            # For trilinear interpolation, PyTorch expects (N, C, D, H, W) format
            # If we have B,F,C,H,W format, we need to convert to B,C,F,H,W first
            if channel_dim_idx == 2:  # B,F,C,H,W format
                # Transpose to B,C,F,H,W for interpolation
                noise = noise.transpose(1, 2).contiguous()

            # Now noise is in B,C,F,H,W format, extract target dimensions
            # target_shape is in the original format, so extract appropriately
            if channel_dim_idx == 2:  # Original was B,F,C,H,W
                tgt_frames = target_shape[1]
                tgt_channels = target_shape[2]
            else:  # B,C,F,H,W
                tgt_channels = target_shape[1]
                tgt_frames = target_shape[2]
            tgt_h, tgt_w = target_shape[3], target_shape[4]

            # Use trilinear interpolation for (frames, height, width)
            noise = F.interpolate(
                noise,
                size=(tgt_frames, tgt_h, tgt_w),
                mode='trilinear',
                align_corners=False
            )

            # Handle channel mismatch
            if noise.shape[1] != tgt_channels:
                b, c, f, h, w = noise.shape
                new_noise = torch.zeros((b, tgt_channels, f, h, w), device=device, dtype=dtype)
                min_c = min(c, tgt_channels)
                new_noise[:, :min_c] = noise[:, :min_c]
                noise = new_noise

            # Convert back to original format if needed
            if channel_dim_idx == 2:  # Original was B,F,C,H,W
                noise = noise.transpose(1, 2).contiguous()
        else:
            # 4D image tensor - use bilinear interpolation
            noise = F.interpolate(noise, size=target_shape[2:], mode='bilinear', align_corners=False)

            # Handle channel mismatch
            if noise.shape[1] != target_shape[1]:
                final_noise = torch.zeros(target_shape, device=device, dtype=dtype)
                min_c = min(noise.shape[1], target_shape[1])
                final_noise[:, :min_c, ...] = noise[:, :min_c, ...]
                noise = final_noise

    except Exception:
        # Fallback: for 5D tensors, try frame-by-frame bilinear interpolation
        if len(noise.shape) == 5 and len(target_shape) == 5:
            try:
                noise = _interpolate_video_framewise(noise, target_shape, channel_dim_idx, device, dtype)
            except Exception:
                noise = torch.zeros(target_shape, device=device, dtype=dtype)
        else:
            noise = torch.zeros(target_shape, device=device, dtype=dtype)

    return noise


def _interpolate_video_framewise(
    noise: torch.Tensor,
    target_shape: Tuple[int, ...],
    channel_dim_idx: int,
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    """Fallback frame-by-frame interpolation for video tensors."""
    # Determine frame dimension
    frame_dim = 2 if channel_dim_idx == 1 else 1
    target_h, target_w = target_shape[3], target_shape[4]
    
    frames_list = []
    num_frames = noise.shape[frame_dim]
    
    for i in range(num_frames):
        if frame_dim == 1:
            frame = noise[:, i]  # B,C,H,W
        else:
            frame = noise[:, :, i]  # B,C,H,W
        
        frame_resized = F.interpolate(
            frame,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        frames_list.append(frame_resized)
    
    if frame_dim == 1:
        result = torch.stack(frames_list, dim=1)
    else:
        result = torch.stack(frames_list, dim=2)
    
    # Handle frame count mismatch
    target_frames = target_shape[frame_dim]
    if result.shape[frame_dim] != target_frames:
        # Simple repeat/truncate for frame adjustment
        if result.shape[frame_dim] < target_frames:
            repeats = (target_frames // result.shape[frame_dim]) + 1
            if frame_dim == 1:
                result = result.repeat(1, repeats, 1, 1, 1)[:, :target_frames]
            else:
                result = result.repeat(1, 1, repeats, 1, 1)[:, :, :target_frames]
        else:
            if frame_dim == 1:
                result = result[:, :target_frames]
            else:
                result = result[:, :, :target_frames]
    
    return result
