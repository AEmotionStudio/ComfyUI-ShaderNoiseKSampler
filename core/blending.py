"""
Blending operations for combining base noise with shader noise.
"""
import torch
import torch.nn.functional as F
import contextlib
from typing import Optional, Tuple

from .constants import SUPPORTED_BLEND_MODES


def blend_noises(
    base_noise: torch.Tensor,
    shader_noise: torch.Tensor,
    blend_mode: str,
    strength: float,
    debugger: Optional[object] = None
) -> torch.Tensor:
    """
    Blend base noise with shader noise using the specified blend mode and strength.
    Handles channel dimension mismatches automatically.
    
    Args:
        base_noise: Base noise tensor
        shader_noise: Shader noise tensor
        blend_mode: Blending mode to apply (one of SUPPORTED_BLEND_MODES)
        strength: Strength of the blend [0.0-1.0]
        debugger: Optional debugger instance for logging
        
    Returns:
        Blended noise tensor
    """
    # If strength is 0, return base noise unchanged
    if strength <= 0.0:
        return base_noise
        
    # If strength is 1.0 and blend mode is "normal", return shader noise
    if strength >= 1.0 and blend_mode == "normal":
        # Ensure shader noise has same shape as base noise
        if shader_noise.shape != base_noise.shape:
            shader_noise = _match_tensor_shape(shader_noise, base_noise, debugger)
        return shader_noise
    
    # Ensure compatible dimensions for blending
    if shader_noise.shape != base_noise.shape:
        shader_noise = _match_tensor_shape(shader_noise, base_noise, debugger)
    
    # Apply the blend mode
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    ctx = debugger.time_operation(f"blend_{blend_mode}") if debug_enabled else contextlib.nullcontext()
    
    with ctx:
        result = _apply_blend_mode(base_noise, shader_noise, blend_mode, strength)
    
    # Debug output for blend results
    if debug_enabled:
        _log_blend_stats(base_noise, result, blend_mode, strength, debugger)
    
    return result


def _detect_video_format(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Detect the format of a 5D video tensor and return channel/frame dimension indices.
    
    Args:
        tensor: 5D tensor to analyze
        
    Returns:
        Tuple of (channel_dim, frame_dim) indices
    """
    if len(tensor.shape) != 5:
        raise ValueError(f"Expected 5D tensor, got {len(tensor.shape)}D")
    
    # Common channel counts for latent models
    common_channels = {4, 8, 12, 16, 32, 64, 128}
    
    dim1 = tensor.shape[1]
    dim2 = tensor.shape[2]
    
    # If dim1 looks like channels (common count) and dim2 doesn't
    if dim1 in common_channels and dim2 not in common_channels:
        return 1, 2  # B,C,F,H,W format
    # If dim2 looks like channels and dim1 doesn't
    elif dim2 in common_channels and dim1 not in common_channels:
        return 2, 1  # B,F,C,H,W format
    # If both look like channels, prefer larger as frames (heuristic)
    elif dim1 in common_channels and dim2 in common_channels:
        if dim1 >= dim2:
            return 2, 1  # B,F,C,H,W - dim1 is likely frames
        else:
            return 1, 2  # B,C,F,H,W - dim2 is likely frames
    # Default to B,F,C,H,W format (most common in video models)
    else:
        return 2, 1


def _match_tensor_shape(
    source: torch.Tensor,
    target: torch.Tensor,
    debugger: Optional[object] = None
) -> torch.Tensor:
    """
    Match the shape of source tensor to target tensor.
    Handles both 4D image tensors and 5D video tensors with various formats.
    
    Args:
        source: Source tensor to reshape
        target: Target tensor whose shape to match
        debugger: Optional debugger for logging
        
    Returns:
        Reshaped source tensor
    """
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    if debug_enabled and debug_level >= 1:
        print(f"⚠️ Shape mismatch for blending: base={target.shape}, shader={source.shape}")
    
    # Handle 5D video tensors specially
    is_video = len(source.shape) == 5 and len(target.shape) == 5
    
    if is_video:
        return _match_video_tensor_shape(source, target, debugger)
    
    # Handle 4D tensors (standard image format B,C,H,W)
    result = source.clone()
    
    # FIRST handle spatial dimension mismatches with interpolation
    # This must happen before channel handling to prevent shape mismatches
    if result.shape[2:] != target.shape[2:]:
        try:
            result = F.interpolate(
                result, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        except RuntimeError as e:
            if debug_enabled:
                print(f"⚠️ Error during interpolation: {e}")
            # Fall back to zeros if we can't match shapes
            return torch.zeros_like(target)
            
        if debug_enabled and debug_level >= 1:
            print(f"✅ Matched spatial dimensions: {result.shape}")
    
    # THEN handle channel dimension mismatches (index 1 for 4D)
    # Now spatial dims match, so channel slicing is safe
    if result.shape[1] != target.shape[1]:
        new_result = torch.zeros_like(target)
        min_channels = min(result.shape[1], target.shape[1])
        new_result[:, :min_channels] = result[:, :min_channels]
        result = new_result
        
        if debug_enabled and debug_level >= 1:
            print(f"✅ Matched channel dimensions: {result.shape}")
    
    return result


def _match_video_tensor_shape(
    source: torch.Tensor, 
    target: torch.Tensor,
    debugger: Optional[object] = None
) -> torch.Tensor:
    """
    Match video tensor shapes by handling frame and channel dimensions.
    Supports both B,C,F,H,W and B,F,C,H,W formats.
    
    Args:
        source: Source 5D tensor
        target: Target 5D tensor
        debugger: Optional debugger for logging
        
    Returns:
        Reshaped source tensor matching target shape
    """
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    # Detect formats for both tensors
    src_channel_dim, src_frame_dim = _detect_video_format(source)
    tgt_channel_dim, tgt_frame_dim = _detect_video_format(target)
    
    if debug_enabled and debug_level >= 2:
        print(f"   Source format: channel_dim={src_channel_dim}, frame_dim={src_frame_dim}")
        print(f"   Target format: channel_dim={tgt_channel_dim}, frame_dim={tgt_frame_dim}")
    
    # Extract dimensions
    src_batch = source.shape[0]
    src_channels = source.shape[src_channel_dim]
    src_frames = source.shape[src_frame_dim]
    src_height = source.shape[3]
    src_width = source.shape[4]
    
    tgt_batch = target.shape[0]
    tgt_channels = target.shape[tgt_channel_dim]
    tgt_frames = target.shape[tgt_frame_dim]
    tgt_height = target.shape[3]
    tgt_width = target.shape[4]
    
    # Start with a clone to preserve data
    result = source.clone()
    
    # Step 1: Convert source to target format if different
    if src_channel_dim != tgt_channel_dim:
        # Swap dimensions 1 and 2
        result = result.transpose(1, 2).contiguous()
        if debug_enabled and debug_level >= 2:
            print(f"   Transposed to target format: {result.shape}")
    
    # After potential transpose, dimensions now align with target format
    # Get current channel dim based on target format
    channel_dim = tgt_channel_dim
    frame_dim = tgt_frame_dim
    
    # Step 2: Handle frame count mismatch
    current_frames = result.shape[frame_dim]
    if current_frames != tgt_frames:
        if debug_enabled and debug_level >= 1:
            print(f"   Adjusting frames from {current_frames} to {tgt_frames}")
        
        # Interpolate along frame dimension
        # F.interpolate requires 4D for bilinear, so reshape appropriately
        if frame_dim == 1:  # B,F,C,H,W
            b, f, c, h, w = result.shape
            # Reshape to 4D: (B*C, 1, F, H*W) for bilinear interpolation
            result_4d = result.permute(0, 2, 1, 3, 4).reshape(b * c, 1, f, h * w)
            result_4d = F.interpolate(result_4d, size=(tgt_frames, h * w), mode='bilinear', align_corners=False)
            result = result_4d.reshape(b, c, tgt_frames, h, w).permute(0, 2, 1, 3, 4)
        else:  # B,C,F,H,W (frame_dim == 2)
            b, c, f, h, w = result.shape
            # Reshape to 4D: (B*C, 1, F, H*W) for bilinear interpolation
            result_4d = result.reshape(b * c, 1, f, h * w)
            result_4d = F.interpolate(result_4d, size=(tgt_frames, h * w), mode='bilinear', align_corners=False)
            result = result_4d.reshape(b, c, tgt_frames, h, w)
    
    # Step 3: Handle spatial dimension mismatch FIRST
    # This must happen before channel handling to prevent shape mismatches
    if result.shape[3:] != target.shape[3:]:
        if debug_enabled and debug_level >= 1:
            print(f"   Adjusting spatial dims from {result.shape[3:]} to {target.shape[3:]}")
        
        try:
            # Use trilinear interpolation for 5D tensors
            result = F.interpolate(
                result,
                size=(result.shape[2], tgt_height, tgt_width),
                mode='trilinear',
                align_corners=False
            )
        except RuntimeError:
            # Fallback: interpolate each frame with bilinear
            frames_list = []
            num_frames = result.shape[frame_dim]
            
            for i in range(num_frames):
                if frame_dim == 1:
                    frame = result[:, i]  # B,C,H,W
                else:
                    frame = result[:, :, i]  # B,C,H,W
                
                frame_resized = F.interpolate(
                    frame,
                    size=(tgt_height, tgt_width),
                    mode='bilinear',
                    align_corners=False
                )
                frames_list.append(frame_resized)
            
            if frame_dim == 1:
                result = torch.stack(frames_list, dim=1)
            else:
                result = torch.stack(frames_list, dim=2)
    
    # Step 4: Handle channel mismatch AFTER spatial dims match
    # Now that spatial dims match, channel slicing is safe
    current_channels = result.shape[channel_dim]
    if current_channels != tgt_channels:
        if debug_enabled and debug_level >= 1:
            print(f"   Adjusting channels from {current_channels} to {tgt_channels}")
        
        new_result = torch.zeros_like(target)
        min_channels = min(current_channels, tgt_channels)
        
        if channel_dim == 1:  # B,C,F,H,W
            new_result[:, :min_channels] = result[:, :min_channels]
        else:  # B,F,C,H,W
            new_result[:, :, :min_channels] = result[:, :, :min_channels]
        
        result = new_result
    
    if debug_enabled and debug_level >= 1:
        print(f"✅ Final matched shape: {result.shape}")
    
    return result


def _apply_blend_mode(
    base: torch.Tensor,
    shader: torch.Tensor,
    mode: str,
    strength: float
) -> torch.Tensor:
    """
    Apply the specified blend mode.
    
    Args:
        base: Base noise tensor
        shader: Shader noise tensor
        mode: Blend mode name
        strength: Blend strength [0.0-1.0]
        
    Returns:
        Blended tensor
    """
    if mode == "normal":
        # Linear interpolation
        return base * (1.0 - strength) + shader * strength
    
    elif mode == "add":
        # Add shader to base
        return base + shader * strength
    
    elif mode == "multiply":
        # Multiply base by shader
        return base * (1.0 + (shader - 0.5) * strength * 2)
    
    elif mode == "screen":
        # Screen blend mode
        return 1.0 - (1.0 - base) * (1.0 - shader * strength)
    
    elif mode == "overlay":
        # Overlay blend mode - compute full result then interpolate with base
        # Standard overlay: 2*base*shader if base < 0.5, else 1 - 2*(1-base)*(1-shader)
        overlay_result = torch.where(
            base < 0.5,
            2 * base * shader,
            1 - 2 * (1 - base) * (1 - shader)
        )
        return base * (1.0 - strength) + overlay_result * strength
    
    elif mode == "soft_light":
        # Soft light blend mode
        return ((1.0 - 2.0 * shader) * base**2 + 2.0 * shader * base) * strength + base * (1.0 - strength)
    
    elif mode == "hard_light":
        # Hard light blend mode - compute full result then interpolate with base
        # Standard hard light: 2*base*shader if shader < 0.5, else 1 - 2*(1-base)*(1-shader)
        hard_light_result = torch.where(
            shader < 0.5,
            2 * base * shader,
            1 - 2 * (1 - base) * (1 - shader)
        )
        return base * (1.0 - strength) + hard_light_result * strength
    
    elif mode == "difference":
        # Difference blend mode - interpolate between base and abs difference
        diff_result = torch.abs(base - shader)
        return base * (1.0 - strength) + diff_result * strength
    
    else:
        # Default to normal blend for unknown modes
        return base * (1.0 - strength) + shader * strength


def _log_blend_stats(
    base: torch.Tensor,
    result: torch.Tensor,
    mode: str,
    strength: float,
    debugger: object
) -> None:
    """
    Log statistics about the blend operation.
    
    Args:
        base: Base noise tensor
        result: Result tensor after blending
        mode: Blend mode used
        strength: Blend strength used
        debugger: Debugger instance for logging
    """
    debug_level = getattr(debugger, 'debug_level', 0)
    
    base_stats = {
        "min": float(base.min().item()),
        "max": float(base.max().item()),
        "mean": float(base.mean().item()),
        "std": float(base.std().item())
    }
    
    result_stats = {
        "min": float(result.min().item()),
        "max": float(result.max().item()),
        "mean": float(result.mean().item()),
        "std": float(result.std().item())
    }
    
    if debug_level >= 2:
        print(f"📊 Blend stats ({mode}, strength={strength:.2f}):")
        print(f"   Base: min={base_stats['min']:.4f}, max={base_stats['max']:.4f}, "
              f"mean={base_stats['mean']:.4f}, std={base_stats['std']:.4f}")
        print(f"   Result: min={result_stats['min']:.4f}, max={result_stats['max']:.4f}, "
              f"mean={result_stats['mean']:.4f}, std={result_stats['std']:.4f}")
    
    mean_diff = abs(result_stats["mean"] - base_stats["mean"])
    std_diff = abs(result_stats["std"] - base_stats["std"])
    
    # Gate warning behind debug level to avoid spamming logs in production
    if mean_diff < 0.001 and std_diff < 0.001 and debug_level >= 1:
        print(f"⚠️ Warning: Blend may not be effective - minimal statistical difference detected")
        print(f"   Base: mean={base_stats['mean']:.4f}, std={base_stats['std']:.4f}")
        print(f"   Result: mean={result_stats['mean']:.4f}, std={result_stats['std']:.4f}")
