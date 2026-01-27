"""
Blending operations for combining base noise with shader noise.
"""
import torch
import contextlib
from typing import Optional, Callable

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


def _match_tensor_shape(
    source: torch.Tensor,
    target: torch.Tensor,
    debugger: Optional[object] = None
) -> torch.Tensor:
    """
    Match the shape of source tensor to target tensor.
    
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
    
    result = source
    
    # Handle channel dimension mismatches
    if source.shape[1] != target.shape[1]:
        # Create a new tensor with the same shape as target
        result = torch.zeros_like(target)
        # Copy over the common channels
        min_channels = min(source.shape[1], target.shape[1])
        result[:, :min_channels] = source[:, :min_channels]
        
        if debug_enabled and debug_level >= 1:
            print(f"✅ Matched channel dimensions: {result.shape}")
    
    # Handle spatial dimension mismatches with interpolation
    if result.shape[2:] != target.shape[2:]:
        try:
            result = torch.nn.functional.interpolate(
                result, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        except RuntimeError as e:
            if debug_enabled:
                print(f"⚠️ Error during interpolation: {e}")
            # Handle video tensors specially
            if len(source.shape) == 5 and len(target.shape) == 5:
                result = _match_video_tensor_shape(source, target)
            else:
                # Fall back to zeros if we can't match shapes
                result = torch.zeros_like(target)
            
        if debug_enabled and debug_level >= 1:
            print(f"✅ Matched spatial dimensions: {result.shape}")
    
    return result


def _match_video_tensor_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Match video tensor shapes by handling frame and channel dimensions.
    
    Args:
        source: Source 5D tensor
        target: Target 5D tensor
        
    Returns:
        Reshaped source tensor
    """
    b, f, c, h, w = source.shape
    _, _, target_c, target_h, target_w = target.shape
    
    if c != target_c:
        # Handle channel differences
        new_tensor = torch.zeros_like(target)
        min_c = min(c, target_c)
        # Copy each frame separately
        for i in range(f):
            new_tensor[:, i, :min_c] = source[:, i, :min_c]
        return new_tensor
    
    return torch.zeros_like(target)


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
        # Overlay blend mode - Optimized using torch.where for vectorization
        term1 = 2 * base * (shader * strength)
        term2 = 1 - 2 * (1 - base) * (1 - shader * strength)
        return torch.where(base < 0.5, term1, term2)
    
    elif mode == "soft_light":
        # Soft light blend mode
        return ((1.0 - 2.0 * shader) * base**2 + 2.0 * shader * base) * strength + base * (1.0 - strength)
    
    elif mode == "hard_light":
        # Hard light blend mode - Optimized using torch.where for vectorization
        term1 = 2 * base * shader * strength + base * (1 - strength)
        term2 = 1 - 2 * (1 - base) * (1 - shader) * strength + base * (1 - strength)
        return torch.where(shader < 0.5, term1, term2)
    
    elif mode == "difference":
        # Difference blend mode
        return base + (torch.abs(base - shader) * strength)
    
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
    
    if mean_diff < 0.001 and std_diff < 0.001:
        print(f"⚠️ Warning: Blend may not be effective - minimal statistical difference detected")
        print(f"   Base: mean={base_stats['mean']:.4f}, std={base_stats['std']:.4f}")
        print(f"   Result: mean={result_stats['mean']:.4f}, std={result_stats['std']:.4f}")
