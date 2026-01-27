"""
Color scheme utilities for shader noise generation.

This module provides centralized color interpolation and application functions
used by all shader generators.
"""

import torch
from typing import Tuple, List, Optional, Union

# Centralized color scheme definitions
# Format: scheme_name -> list of (position, (r, g, b)) stops
COLOR_SCHEMES = {
    "viridis": [
        (0.0, (0.267, 0.005, 0.329)),
        (0.33, (0.188, 0.407, 0.553)),
        (0.66, (0.208, 0.718, 0.471)),
        (1.0, (0.992, 0.906, 0.143)),
    ],
    "plasma": [
        (0.0, (0.05, 0.03, 0.53)),
        (0.25, (0.40, 0.00, 0.66)),
        (0.5, (0.70, 0.18, 0.53)),
        (0.75, (0.94, 0.46, 0.25)),
        (1.0, (0.98, 0.80, 0.08)),
    ],
    "inferno": [
        (0.0, (0.001, 0.001, 0.016)),
        (0.25, (0.259, 0.039, 0.408)),
        (0.5, (0.576, 0.149, 0.404)),
        (0.75, (0.867, 0.318, 0.227)),
        (0.85, (0.988, 0.647, 0.039)),
        (1.0, (0.988, 1.000, 0.643)),
    ],
    "magma": [
        (0.0, (0.001, 0.001, 0.016)),
        (0.25, (0.231, 0.059, 0.439)),
        (0.5, (0.549, 0.161, 0.506)),
        (0.75, (0.871, 0.288, 0.408)),
        (0.85, (0.996, 0.624, 0.427)),
        (1.0, (0.988, 0.992, 0.749)),
    ],
    "turbo": [
        (0.0, (0.188, 0.071, 0.235)),
        (0.25, (0.275, 0.408, 0.859)),
        (0.5, (0.149, 0.749, 0.549)),
        (0.65, (0.831, 1.000, 0.314)),
        (0.85, (0.980, 0.718, 0.298)),
        (1.0, (0.729, 0.004, 0.000)),
    ],
    "jet": [
        (0.0, (0.000, 0.000, 0.498)),
        (0.125, (0.000, 0.000, 1.000)),
        (0.375, (0.000, 1.000, 1.000)),
        (0.625, (1.000, 1.000, 0.000)),
        (0.875, (1.000, 0.000, 0.000)),
        (1.0, (0.498, 0.000, 0.000)),
    ],
    "hot": [
        (0.0, (0.0, 0.0, 0.0)),
        (0.375, (1.0, 0.0, 0.0)),
        (0.75, (1.0, 1.0, 0.0)),
        (1.0, (1.0, 1.0, 1.0)),
    ],
    "parula": [
        (0.0, (0.208, 0.165, 0.529)),
        (0.25, (0.059, 0.361, 0.867)),
        (0.5, (0.000, 0.710, 0.651)),
        (0.75, (1.000, 0.765, 0.216)),
        (1.0, (0.988, 0.996, 0.643)),
    ],
    "pink": [
        (0.0, (0.05, 0.05, 0.05)),
        (0.5, (1.0, 0.41, 0.71)),
        (1.0, (1.0, 0.75, 0.80)),
    ],
    "bone": [
        (0.0, (0.0, 0.0, 0.0)),
        (0.375, (0.329, 0.329, 0.455)),
        (0.75, (0.627, 0.757, 0.757)),
        (1.0, (1.0, 1.0, 1.0)),
    ],
    "ocean": [
        (0.0, (0.0, 0.0, 0.0)),
        (0.33, (0.0, 0.0, 0.6)),
        (0.66, (0.0, 0.6, 1.0)),
        (1.0, (0.6, 1.0, 1.0)),
    ],
    "terrain": [
        (0.0, (0.2, 0.2, 0.6)),
        (0.15, (0.0, 0.5, 0.0)),
        (0.33, (0.0, 0.8, 0.4)),
        (0.5, (0.87, 0.87, 0.4)),
        (0.75, (0.6, 0.4, 0.2)),
        (1.0, (1.0, 1.0, 1.0)),
    ],
    "neon": [
        (0.0, (1.0, 0.0, 0.5)),
        (0.33, (0.0, 1.0, 1.0)),
        (0.66, (1.0, 1.0, 0.0)),
        (1.0, (0.5, 0.0, 1.0)),
    ],
    "fire": [
        (0.0, (0.0, 0.0, 0.0)),
        (0.25, (1.0, 0.0, 0.0)),
        (0.6, (1.0, 1.0, 0.0)),
        (1.0, (1.0, 1.0, 1.0)),
    ],
    # Linear interpolation schemes (2 colors)
    "blue_red": [
        (0.0, (0.0, 0.0, 1.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ],
    "cool": [
        (0.0, (0.0, 1.0, 1.0)),
        (1.0, (1.0, 0.0, 1.0)),
    ],
    "autumn": [
        (0.0, (1.0, 0.0, 0.0)),
        (1.0, (1.0, 1.0, 0.0)),
    ],
    "winter": [
        (0.0, (0.0, 0.0, 1.0)),
        (1.0, (0.0, 1.0, 0.5)),
    ],
    "spring": [
        (0.0, (1.0, 0.0, 1.0)),
        (1.0, (1.0, 1.0, 0.0)),
    ],
    "summer": [
        (0.0, (0.0, 0.5, 0.4)),
        (1.0, (1.0, 1.0, 0.4)),
    ],
    "copper": [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (1.0, 0.6, 0.4)),
    ],
}

# List of all available color schemes
AVAILABLE_SCHEMES = ["none"] + list(COLOR_SCHEMES.keys()) + [
    "rainbow", "heatmap", "vorticity", "hsv",
    "rgb", "complementary", "monochrome", "gradient", "fantasy"
]


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between a and b.
    
    Args:
        a: Start value tensor
        b: End value tensor
        t: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated tensor: a + (b - a) * t
    """
    return a + (b - a) * t


def hsv_to_rgb(
    h: torch.Tensor, 
    s: torch.Tensor, 
    v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert HSV color space to RGB.
    
    Args:
        h: Hue tensor [0, 1]
        s: Saturation tensor [0, 1]
        v: Value tensor [0, 1]
        
    Returns:
        Tuple of (r, g, b) tensors each in [0, 1]
    """
    c = v * s
    h_prime = h * 6.0
    x = c * (1.0 - torch.abs(torch.fmod(h_prime, 2.0) - 1.0))
    m = v - c
    
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    
    # Sector 0: h_prime in [0, 1)
    mask0 = (h_prime < 1.0)
    r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], 0.0
    
    # Sector 1: h_prime in [1, 2)
    mask1 = (h_prime >= 1.0) & (h_prime < 2.0)
    r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], 0.0
    
    # Sector 2: h_prime in [2, 3)
    mask2 = (h_prime >= 2.0) & (h_prime < 3.0)
    r[mask2], g[mask2], b[mask2] = 0.0, c[mask2], x[mask2]
    
    # Sector 3: h_prime in [3, 4)
    mask3 = (h_prime >= 3.0) & (h_prime < 4.0)
    r[mask3], g[mask3], b[mask3] = 0.0, x[mask3], c[mask3]
    
    # Sector 4: h_prime in [4, 5)
    mask4 = (h_prime >= 4.0) & (h_prime < 5.0)
    r[mask4], g[mask4], b[mask4] = x[mask4], 0.0, c[mask4]
    
    # Sector 5: h_prime in [5, 6)
    mask5 = (h_prime >= 5.0)
    r[mask5], g[mask5], b[mask5] = c[mask5], 0.0, x[mask5]
    
    return r + m, g + m, b + m


def interpolate_colors(
    stops: List[Tuple[float, Tuple[float, float, float]]],
    t: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interpolate between color stops based on input value.
    
    Uses optimized bucketize for O(N) performance.
    
    Args:
        stops: List of (position, (r, g, b)) color stops
        t: Normalized value tensor [B, 1, H, W] or [B, H, W, 1] in [0, 1]
        device: Target device (inferred from t if not provided)
        
    Returns:
        Tuple of (r, g, b) tensors matching input shape
    """
    if device is None:
        device = t.device
    
    num_stops = len(stops)
    
    # Prepare boundaries and colors
    boundaries = torch.tensor([s[0] for s in stops], device=device, dtype=t.dtype)
    colors = torch.tensor([s[1] for s in stops], device=device, dtype=t.dtype)  # [S, 3]
    
    # Handle different input shapes
    original_shape = t.shape
    is_bhwc = len(original_shape) == 4 and original_shape[-1] == 1
    
    if is_bhwc:
        # Convert from [B, H, W, 1] to [B, 1, H, W]
        t = t.permute(0, 3, 1, 2)
    
    # Find bucket indices
    bucket_indices = torch.bucketize(t, boundaries)
    
    # Clamp indices to valid segment range [1, num_stops-1]
    idx = torch.clamp(bucket_indices, 1, num_stops - 1)
    idx_lower = idx - 1
    idx_upper = idx
    
    # Gather boundary values
    t0 = boundaries[idx_lower]
    t1 = boundaries[idx_upper]
    
    # Gather colors using embedding lookup
    # Need to handle the shape properly - squeeze dim 1 from indices
    idx_squeezed = idx_lower.squeeze(1) if len(idx_lower.shape) == 4 else idx_lower
    c0 = torch.nn.functional.embedding(idx_squeezed, colors)
    c1 = torch.nn.functional.embedding(idx_upper.squeeze(1) if len(idx_upper.shape) == 4 else idx_upper, colors)
    
    # Permute to [B, 3, H, W] format
    if len(c0.shape) == 4:  # [B, H, W, 3]
        c0 = c0.permute(0, 3, 1, 2)
        c1 = c1.permute(0, 3, 1, 2)
    
    # Calculate interpolation factor
    denominator = t1 - t0 + 1e-8
    local_t = torch.clamp((t - t0) / denominator, 0.0, 1.0)
    
    # Interpolate
    final_color = lerp(c0, c1, local_t)
    
    # Split into R, G, B
    r = final_color[:, 0:1]
    g = final_color[:, 1:2]
    b = final_color[:, 2:3]
    
    # Convert back to BHWC if input was BHWC
    if is_bhwc:
        r = r.permute(0, 2, 3, 1)
        g = g.permute(0, 2, 3, 1)
        b = b.permute(0, 2, 3, 1)
    
    return r, g, b


def apply_color_scheme(
    noise_tensor: torch.Tensor,
    scheme_name: str,
    intensity: float = 0.8,
    velocity_field: Optional[torch.Tensor] = None,
    time: float = 0.0
) -> torch.Tensor:
    """
    Apply a color scheme to a noise tensor.
    
    Args:
        noise_tensor: Input tensor [B, C, H, W]
        scheme_name: Name of the color scheme to apply
        intensity: Color intensity [0.0 to 1.0]
        velocity_field: Optional velocity field for direction-based coloring [B, 2, H, W]
        time: Animation time for dynamic schemes
        
    Returns:
        Color-modified tensor [B, C, H, W]
    """
    if scheme_name in ["none", "0"] or intensity <= 0.0:
        return noise_tensor
    
    batch, channels, height, width = noise_tensor.shape
    device = noise_tensor.device
    dtype = noise_tensor.dtype
    
    # Normalize noise to [0, 1] for color mapping
    noise_min = noise_tensor.min()
    noise_max = noise_tensor.max()
    normalized = (noise_tensor - noise_min) / (noise_max - noise_min + 1e-8)
    
    # Use first channel as the color mapping value
    t_color = normalized[:, 0:1]  # [B, 1, H, W]
    
    # Initialize result tensor
    result = torch.zeros_like(noise_tensor)
    
    # Preserve channels beyond the first 3
    if channels > 3:
        result[:, 3:] = noise_tensor[:, 3:]
    
    # Handle velocity-based schemes
    if velocity_field is not None and scheme_name in ["rainbow", "vorticity", "heatmap"]:
        vx = velocity_field[:, 0:1]
        vy = velocity_field[:, 1:2]
        vmag = torch.sqrt(vx**2 + vy**2)
        vmag = vmag / (vmag.max() + 1e-8)
        vangle = torch.atan2(vy, vx)
        vangle = (vangle + 3.14159) / (2 * 3.14159)  # Normalize to [0, 1]
        
        if scheme_name == "rainbow":
            r, g, b = hsv_to_rgb(vangle, torch.ones_like(vangle) * 0.8, vmag)
        elif scheme_name == "vorticity":
            curl_mag = torch.abs(vx - vy)
            curl_norm = curl_mag / (curl_mag.max() + 1e-8)
            positive_mask = (vx > vy).float()
            negative_mask = (vx <= vy).float()
            r = positive_mask * curl_norm
            g = (positive_mask + negative_mask) * (1.0 - curl_norm)
            b = negative_mask * curl_norm
        elif scheme_name == "heatmap":
            r = torch.pow(vmag, 0.5)
            g = torch.pow(vmag, 1.5)
            b = torch.pow(vmag, 3.0)
    
    # Handle predefined color schemes
    elif scheme_name in COLOR_SCHEMES:
        stops = COLOR_SCHEMES[scheme_name]
        r, g, b = interpolate_colors(stops, t_color, device)
    
    # Handle special schemes
    elif scheme_name == "rainbow":
        r, g, b = hsv_to_rgb(t_color, torch.ones_like(t_color) * 0.9, torch.ones_like(t_color) * 0.9)
    
    elif scheme_name == "hsv":
        r, g, b = hsv_to_rgb(t_color, torch.ones_like(t_color) * 0.95, torch.ones_like(t_color) * 0.95)
    
    elif scheme_name == "heatmap":
        r = torch.pow(t_color, 0.5)
        g = torch.pow(t_color, 1.5)
        b = torch.pow(t_color, 3.0)
    
    elif scheme_name == "rgb":
        if channels >= 3:
            result[:, 0] = noise_tensor[:, 0] * 1.5
            result[:, 1] = noise_tensor[:, 1] * 1.3
            result[:, 2] = noise_tensor[:, 2] * 0.8
            return lerp(noise_tensor, result, torch.tensor(intensity, device=device))
    
    elif scheme_name == "complementary":
        if channels >= 3:
            result[:, 0] = noise_tensor[:, 0] * 1.5
            result[:, 1] = -noise_tensor[:, 0] * 0.8
            result[:, 2] = noise_tensor[:, 2] * 1.2
            return lerp(noise_tensor, result, torch.tensor(intensity, device=device))
    
    elif scheme_name == "monochrome":
        if channels > 1:
            base = noise_tensor[:, 0:1].clone()
            scales = torch.tensor([1.0, 0.95, 0.9, 0.85][:channels], device=device).view(1, -1, 1, 1)
            result = base * scales
            return lerp(noise_tensor, result, torch.tensor(intensity, device=device))
    
    elif scheme_name == "gradient":
        if channels >= 3:
            y_norm = torch.linspace(0, 1, height, device=device).view(1, 1, -1, 1).expand(batch, 1, -1, width)
            x_norm = torch.linspace(0, 1, width, device=device).view(1, 1, 1, -1).expand(batch, 1, height, -1)
            result[:, 0:1] = x_norm + noise_tensor[:, 0:1] * 0.4
            result[:, 1:2] = y_norm + noise_tensor[:, 1:2] * 0.4
            result[:, 2:3] = (x_norm + y_norm) / 2 + noise_tensor[:, 2:3] * 0.4
            return lerp(noise_tensor, result, torch.tensor(intensity, device=device))
    
    elif scheme_name == "fantasy":
        if channels >= 3:
            angle = torch.atan2(noise_tensor[:, 1], noise_tensor[:, 0])
            radius = torch.sqrt(noise_tensor[:, 0]**2 + noise_tensor[:, 1]**2)
            result[:, 0] = torch.sin(angle * 2.0 + radius * 3.0) * 0.5 + 0.5
            result[:, 1] = torch.sin(angle * 3.0 - radius * 2.0) * 0.5 + 0.5
            result[:, 2] = torch.sin(radius * 5.0) * 0.5 + 0.5
            result = (result - 0.5) * 2.0
            return lerp(noise_tensor, result, torch.tensor(intensity, device=device))
    
    elif scheme_name == "plasma" and velocity_field is not None:
        # Dynamic plasma with time
        vangle = torch.atan2(velocity_field[:, 1:2], velocity_field[:, 0:1])
        vangle = (vangle + 3.14159) / (2 * 3.14159)
        time_t = torch.tensor(time, device=device, dtype=dtype)
        r = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + time_t)
        g = 0.5 + 0.5 * torch.sin(vangle * 6.28318 + t_color * 3.14159 + time_t * 2.0)
        b = 0.5 + 0.5 * torch.cos(vangle * 3.14159 + t_color * 6.28318 + time_t * 3.0)
    
    else:
        # Unknown scheme - return original
        return noise_tensor
    
    # Assign color channels to result
    if channels >= 1:
        result[:, 0:1] = r
    if channels >= 2:
        result[:, 1:2] = g
    if channels >= 3:
        result[:, 2:3] = b
    
    # Apply intensity blending
    if intensity < 1.0:
        grayscale = (r + g + b) / 3.0
        result[:, 0:1] = lerp(grayscale, r, torch.tensor(intensity, device=device))
        result[:, 1:2] = lerp(grayscale, g, torch.tensor(intensity, device=device))
        result[:, 2:3] = lerp(grayscale, b, torch.tensor(intensity, device=device))
    
    return result


def get_scheme_stops(scheme_name: str) -> Optional[List[Tuple[float, Tuple[float, float, float]]]]:
    """
    Get the color stops for a named scheme.
    
    Args:
        scheme_name: Name of the color scheme
        
    Returns:
        List of color stops or None if scheme not found
    """
    return COLOR_SCHEMES.get(scheme_name)
