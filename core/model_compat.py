"""
Model compatibility utilities for detecting model types and channel counts.
"""
import torch
from typing import Optional, Any, Tuple

from .constants import (
    DEFAULT_CHANNELS,
    MODEL_CHANNEL_COUNTS,
    VIDEO_MODEL_CHANNELS,
)


def get_model_channel_count(model: Any, debugger: Optional[object] = None) -> int:
    """
    Determine the number of channels in the latent space for a given model.
    This is important for matching the correct noise shape to the model's expectations.
    
    Args:
        model: The model to analyze
        debugger: Optional debugger for logging
        
    Returns:
        Number of channels (4 for SD1.5/2.x, 16 for WAN2.1/Flux/SD3, etc.)
    """
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    # Get model identifiers
    model_name = _get_model_name(model)
    model_type = _get_model_type(model)
    model_class = model.__class__.__name__
    
    if debug_enabled:
        print(f"🔎 Model detection - name: '{model_name}', type: '{model_type}', class: '{model_class}'")
    
    # Check inner model if wrapped
    if hasattr(model, 'model'):
        inner_channels = _check_inner_model(model.model, debugger)
        if inner_channels is not None:
            return inner_channels
    
    # Check latent format
    if hasattr(model, 'latent_format') and hasattr(model.latent_format, 'latent_channels'):
        channels = model.latent_format.latent_channels
        if debug_enabled:
            print(f"✅ Found latent_format.latent_channels: {channels}")
        return channels
    
    # Check diffusion model structure
    channels = _check_diffusion_model(model, debugger)
    if channels is not None:
        return channels
    
    # Check by model name patterns
    channels = _check_model_name_patterns(model_name, model_class, debugger)
    if channels is not None:
        return channels
    
    # Check model path if available
    model_path = getattr(model, 'model_path', "").lower()
    if model_path:
        channels = _check_model_path(model_path, debugger)
        if channels is not None:
            return channels
    
    # Default for most models (SD1.5, SD2.x, SDXL, etc.)
    if debug_enabled:
        print(f"🔄 Using default channel count: {DEFAULT_CHANNELS}")
    return DEFAULT_CHANNELS


def _get_model_name(model: Any) -> str:
    """Extract model name from model object."""
    model_name = getattr(model, 'model_name', "")
    if isinstance(model_name, str):
        return model_name.lower()
    return ""


def _get_model_type(model: Any) -> str:
    """Extract model type from model object."""
    model_type_attr = getattr(model, 'model_type', "")
    if model_type_attr is not None:
        return str(model_type_attr).upper()
    return ""


def _check_inner_model(inner_model: Any, debugger: Optional[object] = None) -> Optional[int]:
    """Check inner model for channel count."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    inner_name = _get_model_name(inner_model)
    inner_type = _get_model_type(inner_model)
    inner_class = inner_model.__class__.__name__
    
    if debug_enabled:
        print(f"🔎 Inner model - name: '{inner_name}', type: '{inner_type}', class: '{inner_class}'")
    
    # Check for specific model classes
    class_channel_map = {
        "HiDream": 16,
        "ACEStep": 8,
        "ACE": 8,
        "WAN21": 16,
        "Mochi": 12,
        "LTXV": 128,
        "CosmosVideo": 16,
        "Cosmos1CV8x8x8": 16,
        "HunyuanVideo": 16,
    }
    
    if inner_class in class_channel_map:
        channels = class_channel_map[inner_class]
        if debug_enabled:
            print(f"✅ Detected {inner_class} in inner model - using {channels} channels")
        return channels
    
    # Check for FLOW type
    if inner_type == "FLOW":
        if debug_enabled:
            print(f"✅ Detected HiDream/Flow in inner model - using 16 channels")
        return 16
    
    # Check for WAN/Warp in name
    if any(ind in inner_name for ind in ["wan", "warp"]):
        if debug_enabled:
            print(f"✅ Detected WAN/Warp in inner model - using 16 channels")
        return 16
    
    # Check for Stable Cascade Prior
    if "stable_cascade_prior" in inner_name or "stablecascade_prior" in inner_name:
        if debug_enabled:
            print(f"✅ Detected Stable Cascade Prior in inner model - using 16 channels")
        return 16
    
    return None


def _check_diffusion_model(model: Any, debugger: Optional[object] = None) -> Optional[int]:
    """Check diffusion model structure for channel count."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    debug_level = getattr(debugger, 'debug_level', 0) if debug_enabled else 0
    
    if not hasattr(model, 'diffusion_model'):
        return None
    
    diffusion_model = model.diffusion_model
    
    # Check input blocks
    if hasattr(diffusion_model, 'input_blocks') and len(diffusion_model.input_blocks) > 0:
        if hasattr(diffusion_model.input_blocks[0], 'in_channels'):
            channels = diffusion_model.input_blocks[0].in_channels
            # In_channels is often double the latent space channels
            if channels in [8, 12, 16]:
                actual_channels = channels // 2
                if debug_enabled:
                    print(f"✅ Detected {actual_channels} channels from model's input_blocks[0].in_channels={channels}")
                return actual_channels
    
    # Check patch embedding
    if hasattr(diffusion_model, 'patch_embedding'):
        patch_shape = diffusion_model.patch_embedding.weight.shape
        if len(patch_shape) >= 2:
            channels = patch_shape[1]
            if debug_enabled:
                print(f"✅ Found patch_embedding with input channels: {channels}")
            return channels
    
    # Debug logging for diffusion model structure
    if debug_enabled and debug_level >= 2:
        if hasattr(diffusion_model, 'in_channels'):
            print(f"🔎 Model's diffusion_model.in_channels: {diffusion_model.in_channels}")
    
    return None


def _check_model_name_patterns(
    model_name: str,
    model_class: str,
    debugger: Optional[object] = None
) -> Optional[int]:
    """Check model name patterns for channel count."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    # Model name patterns
    name_patterns = {
        ("wan", "warp", "pixel", "anime"): 16,
        ("sd3",): 16,
        ("flux",): 16,
        ("mochi",): 12,
        ("ltxv",): 128,
        ("cosmos",): 16,
        ("stable_cascade_prior", "stablecascade_prior"): 16,
    }
    
    for patterns, channels in name_patterns.items():
        if any(p in model_name for p in patterns):
            if debug_enabled:
                print(f"✅ Detected model by name pattern - using {channels} channels")
            return channels
    
    # Special case for non-prior Stable Cascade
    if "stable_cascade" in model_name:
        if debug_enabled:
            print(f"✅ Detected Stable Cascade (non-prior) model by name - using 4 channels")
        return 4
    
    # Check class names
    model_class_lower = model_class.lower()
    
    if model_class_lower == 'flux':
        if debug_enabled:
            print(f"✅ Detected Flux model by class - using 16 channels")
        return 16
    elif model_class_lower == 'hidream':
        if debug_enabled:
            print(f"✅ Detected HiDream model by class - using 16 channels")
        return 16
    
    return None


def _check_model_path(model_path: str, debugger: Optional[object] = None) -> Optional[int]:
    """Check model path for channel count hints."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    wan_indicators = ["wan", "warp", "pixel", "anime"]
    if any(ind in model_path for ind in wan_indicators):
        if debug_enabled:
            print(f"✅ Detected WAN/Warp/Pixel/Anime model from model_path - using 16 channels")
        return 16
    
    return None


def detect_latent_format(
    latent_shape: Tuple[int, ...],
    target_channels: int,
    debugger: Optional[object] = None
) -> Tuple[str, int, int, int, int, int, int]:
    """
    Detect the format of a latent tensor and extract dimensions.
    
    Args:
        latent_shape: Shape of the latent tensor
        target_channels: Expected channel count from model
        debugger: Optional debugger for logging
        
    Returns:
        Tuple of (format_name, batch_size, frames, channels, height, width, 
                  channel_dim_idx, frame_dim_idx)
    """
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    is_video = len(latent_shape) == 5
    batch_size = latent_shape[0]
    height = latent_shape[-2]
    width = latent_shape[-1]
    
    if is_video:
        return _detect_video_format(latent_shape, target_channels, debugger)
    else:
        # 4D: [B, C, H, W]
        channels = latent_shape[1]
        frames = 1
        channel_dim_idx = 1
        frame_dim_idx = -1
        detected_format = "[B, C, H, W]"
        
        # Validate against model channels
        if channels != target_channels:
            if debug_enabled:
                print(f"⚠️ Image shape {latent_shape} channel count ({channels}) differs "
                      f"from model's target channels ({target_channels}). "
                      f"Using model's target channels for noise generation.")
            channels = target_channels
        
        return (detected_format, batch_size, frames, channels, height, width,
                channel_dim_idx, frame_dim_idx)


def _detect_video_format(
    latent_shape: Tuple[int, ...],
    target_channels: int,
    debugger: Optional[object] = None
) -> Tuple[str, int, int, int, int, int, int, int]:
    """Detect video tensor format."""
    debug_enabled = debugger is not None and getattr(debugger, 'enabled', False)
    
    batch_size = latent_shape[0]
    latent_dim1 = latent_shape[1]
    latent_dim2 = latent_shape[2]
    height = latent_shape[-2]
    width = latent_shape[-1]
    
    # Logic to determine Frame vs Channel dimension for 5D tensor
    if latent_dim1 == target_channels and latent_dim2 != target_channels:
        # Format likely [B, C, F, H, W]
        channels = latent_dim1
        frames = latent_dim2
        channel_dim_idx = 1
        frame_dim_idx = 2
        detected_format = "[B, C, F, H, W]"
    elif latent_dim2 == target_channels and latent_dim1 != target_channels:
        # Format likely [B, F, C, H, W]
        channels = latent_dim2
        frames = latent_dim1
        channel_dim_idx = 2
        frame_dim_idx = 1
        detected_format = "[B, F, C, H, W]"
    elif latent_dim1 == target_channels and latent_dim2 == target_channels:
        # Ambiguous - default to B,F,C,H,W
        if debug_enabled:
            print(f"⚠️ Ambiguous video shape {latent_shape} where both dim 1 & 2 match "
                  f"target channels {target_channels}. Assuming [B, F, C, H, W] format.")
        channels = target_channels
        frames = latent_dim1
        channel_dim_idx = 2
        frame_dim_idx = 1
        detected_format = "[B, F, C, H, W] (Ambiguous)"
    else:
        # Neither matches - fallback to B,F,C,H,W
        channels = target_channels
        frames = latent_dim1
        channel_dim_idx = 2
        frame_dim_idx = 1
        detected_format = "[B, F, C, H, W] (Fallback)"
    
    return (detected_format, batch_size, frames, channels, height, width,
            channel_dim_idx, frame_dim_idx)


def build_noise_shape(
    detected_format: str,
    batch_size: int,
    frames: int,
    target_channels: int,
    height: int,
    width: int,
    is_video: bool
) -> Tuple[int, ...]:
    """
    Build the target noise shape based on detected format.
    
    Args:
        detected_format: Detected tensor format string
        batch_size: Batch size
        frames: Number of frames (for video)
        target_channels: Target channel count
        height: Height dimension
        width: Width dimension
        is_video: Whether this is a video tensor
        
    Returns:
        Target noise shape tuple
    """
    if is_video:
        if detected_format.startswith("[B, C, F"):
            # [B, C, F, H, W] format
            return (batch_size, target_channels, frames, height, width)
        else:
            # Default to [B, F, C, H, W] format
            return (batch_size, frames, target_channels, height, width)
    else:
        # [B, C, H, W] format
        return (batch_size, target_channels, height, width)
