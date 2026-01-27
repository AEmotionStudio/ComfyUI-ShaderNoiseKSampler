"""
Base class for shader noise generators.

This module provides the abstract base class that all shader generators
must inherit from, ensuring consistent interface and shared functionality.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

from ..utils.color_utils import apply_color_scheme
from ..utils.shape_masks import apply_shape_mask, apply_mask_to_tensor
from ..utils.noise_utils import create_coordinate_grid
from ..core.params import ShaderParams, get_param_value
from ..core.constants import DEFAULT_CHANNELS


class BaseNoiseGenerator(ABC):
    """
    Abstract base class for all shader noise generators.
    
    Provides common functionality for coordinate grid creation,
    shape mask application, and color scheme handling.
    """
    
    @staticmethod
    @abstractmethod
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
        Generate noise tensor.
        
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
        pass
    
    @staticmethod
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
            dtype: Data type
            range_type: "unit" for [0, 1], "centered" for [-0.5, 0.5], "symmetric" for [-1, 1]
            
        Returns:
            Coordinate tensor [B, H, W, 2] with (x, y) coordinates
        """
        return create_coordinate_grid(batch_size, height, width, device, dtype, range_type)
    
    @staticmethod
    def apply_shape_mask(
        noise: torch.Tensor,
        coords: torch.Tensor,
        params: ShaderParams
    ) -> torch.Tensor:
        """
        Apply shape mask to noise tensor.
        
        Args:
            noise: Input noise tensor [B, H, W, C] or [B, C, H, W]
            coords: Coordinate grid [B, H, W, 2]
            params: Shader parameters containing shape_type and shape_strength
            
        Returns:
            Masked noise tensor
        """
        shape_type = params.shape_type
        shape_strength = params.shape_strength
        time = params.time
        base_seed = params.get("base_seed", 0)
        
        if shape_type in ["none", "0"] or shape_strength <= 0:
            return noise
        
        # Generate shape mask
        mask = apply_shape_mask(coords, shape_type, time, base_seed, shape_strength)
        
        # Apply mask to noise
        return apply_mask_to_tensor(noise, mask, shape_strength)
    
    @staticmethod
    def apply_color_scheme(
        noise: torch.Tensor,
        params: ShaderParams,
        velocity_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply color scheme to noise tensor.
        
        Args:
            noise: Input noise tensor [B, C, H, W]
            params: Shader parameters containing color_scheme and color_intensity
            velocity_field: Optional velocity field for direction-based coloring [B, 2, H, W]
            
        Returns:
            Color-modified noise tensor
        """
        color_scheme = params.color_scheme
        color_intensity = params.color_intensity
        time = params.time
        
        if color_scheme in ["none", "0"] or color_intensity <= 0:
            return noise
        
        return apply_color_scheme(noise, color_scheme, color_intensity, velocity_field, time)
    
    @staticmethod
    def apply_common_postprocessing(
        noise: torch.Tensor,
        params: ShaderParams,
        coords: torch.Tensor,
        velocity_field: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply common postprocessing steps (shape mask, color scheme).
        
        Args:
            noise: Input noise tensor [B, C, H, W]
            params: Shader parameters
            coords: Coordinate grid [B, H, W, 2]
            velocity_field: Optional velocity field for color schemes
            
        Returns:
            Postprocessed noise tensor
        """
        # Apply color scheme first (operates on channels)
        noise = BaseNoiseGenerator.apply_color_scheme(noise, params, velocity_field)
        
        # Apply shape mask using the coords parameter
        noise = BaseNoiseGenerator.apply_shape_mask(noise, coords, params)
        
        return noise
    
    @staticmethod
    def normalize_to_range(
        tensor: torch.Tensor,
        target_min: float = -1.0,
        target_max: float = 1.0
    ) -> torch.Tensor:
        """
        Normalize tensor to target range.
        
        Args:
            tensor: Input tensor
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized tensor
        """
        t_min = tensor.min()
        t_max = tensor.max()
        
        if t_max - t_min < 1e-8:
            # Avoid division by zero for constant tensors
            return torch.full_like(tensor, (target_min + target_max) / 2)
        
        normalized = (tensor - t_min) / (t_max - t_min)
        return normalized * (target_max - target_min) + target_min
    
    @staticmethod
    def expand_channels(
        noise: torch.Tensor,
        target_channels: int,
        params: ShaderParams,
        device: torch.device,
        seed: int = 0
    ) -> torch.Tensor:
        """
        Expand noise tensor to target number of channels.
        
        Args:
            noise: Input noise tensor [B, C, H, W]
            target_channels: Target number of channels
            params: Shader parameters
            device: Target device
            seed: Random seed for channel generation
            
        Returns:
            Expanded noise tensor [B, target_channels, H, W]
        """
        batch, current_channels, height, width = noise.shape
        
        if current_channels >= target_channels:
            return noise[:, :target_channels]
        
        # Create additional channels through variations
        additional_channels = []
        
        for c in range(current_channels, target_channels):
            variation_seed = seed + 500 + (c * 100)
            torch.manual_seed(variation_seed)
            
            # Mix existing channels with slight variations
            if current_channels >= 2:
                mix_ratio = (c * 0.2) % 1.0
                mixed = noise[:, 0:1] * mix_ratio + noise[:, 1:2] * (1.0 - mix_ratio)
            else:
                mixed = noise[:, 0:1].clone()
            
            # Apply unique transformation based on channel number
            if c % 3 == 0:
                mixed = torch.sin(mixed * 3.14159)
            elif c % 3 == 1:
                mixed = torch.abs(mixed) * 2.0 - 1.0
            
            # Normalize
            mixed = (mixed - mixed.mean()) / (mixed.std() + 1e-8)
            additional_channels.append(mixed)
        
        if additional_channels:
            extra = torch.cat(additional_channels, dim=1)
            noise = torch.cat([noise, extra], dim=1)
        
        return noise
    
    @staticmethod
    def get_target_channels(
        params: ShaderParams,
        default: int = DEFAULT_CHANNELS
    ) -> int:
        """
        Get target channel count from parameters.
        
        Args:
            params: Shader parameters
            default: Default channel count
            
        Returns:
            Target number of channels
        """
        return int(params.get("target_channels", default))
