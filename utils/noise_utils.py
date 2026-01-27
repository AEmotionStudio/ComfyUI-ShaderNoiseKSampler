"""
Noise generation utilities for shader noise.

This module provides centralized noise generation functions including
simplex noise, FBM, and gradient noise used by all shader generators.
"""

import torch
import math
from typing import Tuple, Optional

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


def random_gradient(p: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Generate pseudo-random gradient vectors based on position.
    
    Args:
        p: Position tensor [..., 2]
        seed: Random seed for reproducibility
        
    Returns:
        Gradient tensor [..., 2]
    """
    # Hash function
    n = p[..., 0] * (127.1 + seed) + p[..., 1] * (311.7 + seed * 0.5)
    n = torch.sin(n) * HASH_CONSTANT
    n = n - torch.floor(n)
    
    # Convert to angle and create gradient
    angle = n * 2.0 * math.pi
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _fade(t: torch.Tensor) -> torch.Tensor:
    """
    Quintic fade function for smooth interpolation.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def _hash2d(p: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    2D hash function for noise generation.
    """
    dot = p[..., 0] * (127.1 + seed) + p[..., 1] * (311.7 + seed * 0.7)
    return torch.frac(torch.sin(dot) * HASH_CONSTANT)


def _hash3d(p: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    3D hash function for noise generation.
    """
    dot = p[..., 0] * (127.1 + seed) + p[..., 1] * (311.7 + seed * 0.7) + p[..., 2] * (74.7 + seed * 0.3)
    return torch.frac(torch.sin(dot) * HASH_CONSTANT)


def simplex_noise_2d(
    coords: torch.Tensor,
    seed: int = 0,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Generate 2D simplex noise.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2] in any range
        seed: Random seed for reproducibility
        scale: Scale factor for noise frequency
        
    Returns:
        Noise tensor [B, H, W] in range [-1, 1]
    """
    # Scale coordinates
    p = coords * scale
    
    # Skewing factors for 2D simplex
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    
    # Skew the input space
    s = (p[..., 0] + p[..., 1]) * F2
    i = torch.floor(p[..., 0] + s)
    j = torch.floor(p[..., 1] + s)
    
    # Unskew
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = p[..., 0] - X0
    y0 = p[..., 1] - Y0
    
    # Determine simplex
    i1 = (x0 > y0).float()
    j1 = 1.0 - i1
    
    # Offsets for corners
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    
    # Gradients
    device = coords.device
    dtype = coords.dtype
    
    def grad2(hash_val: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = (hash_val * 8.0).long() % 8
        u = torch.where(h < 4, x, y)
        v = torch.where(h < 4, y, x)
        return torch.where(h % 2 == 0, u, -u) + torch.where((h // 2) % 2 == 0, v, -v)
    
    # Hash and gradient
    ii = i.long()
    jj = j.long()
    
    # Create permutation-like hash
    torch.manual_seed(seed)
    perm = torch.randperm(256, device=device, dtype=dtype)
    
    def perm_hash(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xi = (x.long() % 256).abs()
        yi = (y.long() % 256).abs()
        return perm[(perm[xi % 256].long() + yi) % 256] / 256.0
    
    g0 = perm_hash(ii, jj)
    g1 = perm_hash(ii + i1.long(), jj + j1.long())
    g2 = perm_hash(ii + 1, jj + 1)
    
    # Contributions from corners
    t0 = 0.5 - x0*x0 - y0*y0
    t1 = 0.5 - x1*x1 - y1*y1
    t2 = 0.5 - x2*x2 - y2*y2
    
    n0 = torch.where(t0 < 0, torch.zeros_like(t0), t0**4 * grad2(g0, x0, y0))
    n1 = torch.where(t1 < 0, torch.zeros_like(t1), t1**4 * grad2(g1, x1, y1))
    n2 = torch.where(t2 < 0, torch.zeros_like(t2), t2**4 * grad2(g2, x2, y2))
    
    # Scale to [-1, 1]
    return 70.0 * (n0 + n1 + n2)


def simplex_noise_3d(
    coords: torch.Tensor,
    time: float = 0.0,
    seed: int = 0,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Generate 3D simplex noise (2D coords + time).
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        time: Time value for animation
        seed: Random seed for reproducibility
        scale: Scale factor for noise frequency
        
    Returns:
        Noise tensor [B, H, W] in range [-1, 1]
    """
    device = coords.device
    dtype = coords.dtype
    batch, height, width, _ = coords.shape
    
    # Create 3D coordinates by adding time dimension
    p = torch.zeros((batch, height, width, 3), device=device, dtype=dtype)
    p[..., 0] = coords[..., 0] * scale
    p[..., 1] = coords[..., 1] * scale
    p[..., 2] = time * scale * 0.5
    
    # 3D simplex constants
    F3 = 1.0 / 3.0
    G3 = 1.0 / 6.0
    
    # Skew
    s = (p[..., 0] + p[..., 1] + p[..., 2]) * F3
    i = torch.floor(p[..., 0] + s)
    j = torch.floor(p[..., 1] + s)
    k = torch.floor(p[..., 2] + s)
    
    # Unskew
    t = (i + j + k) * G3
    x0 = p[..., 0] - (i - t)
    y0 = p[..., 1] - (j - t)
    z0 = p[..., 2] - (k - t)
    
    # Determine simplex traversal order
    e = (x0 >= y0).float()
    f = (y0 >= z0).float()
    g = (z0 >= x0).float()
    
    i1 = e * (1 - g)
    j1 = (1 - e) * f
    k1 = g * (1 - f)
    
    i2 = e + g * (1 - e)
    j2 = (1 - e) + f * e
    k2 = 1 - f * (1 - g)
    
    # Compute corner positions
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3
    
    # Gradient function
    def grad3d(h: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Convert hash to gradient direction
        h_int = (h * 16).long() % 16
        u = torch.where(h_int < 8, x, y)
        v = torch.where(h_int < 4, y, torch.where((h_int == 12) | (h_int == 14), x, z))
        return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)
    
    # Hash function
    torch.manual_seed(seed)
    perm = torch.randperm(256, device=device, dtype=dtype)
    
    def hash_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xi = (x.long() % 256).abs()
        yi = (y.long() % 256).abs()
        zi = (z.long() % 256).abs()
        return perm[(perm[(perm[xi % 256].long() + yi) % 256].long() + zi) % 256] / 256.0
    
    g0 = hash_3d(i, j, k)
    g1 = hash_3d(i + i1, j + j1, k + k1)
    g2 = hash_3d(i + i2, j + j2, k + k2)
    g3 = hash_3d(i + 1, j + 1, k + 1)
    
    # Compute contributions
    t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
    t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
    t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
    t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
    
    n0 = torch.where(t0 < 0, torch.zeros_like(t0), t0**4 * grad3d(g0, x0, y0, z0))
    n1 = torch.where(t1 < 0, torch.zeros_like(t1), t1**4 * grad3d(g1, x1, y1, z1))
    n2 = torch.where(t2 < 0, torch.zeros_like(t2), t2**4 * grad3d(g2, x2, y2, z2))
    n3 = torch.where(t3 < 0, torch.zeros_like(t3), t3**4 * grad3d(g3, x3, y3, z3))
    
    return 32.0 * (n0 + n1 + n2 + n3)


def fbm_noise(
    coords: torch.Tensor,
    octaves: int = 4,
    scale: float = 1.0,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
    time: Optional[float] = None
) -> torch.Tensor:
    """
    Generate Fractional Brownian Motion (FBM) noise.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        octaves: Number of noise octaves to sum
        scale: Base scale for noise
        persistence: Amplitude multiplier per octave
        lacunarity: Frequency multiplier per octave
        seed: Random seed for reproducibility
        time: Optional time value for 3D noise
        
    Returns:
        FBM noise tensor [B, H, W]
    """
    batch, height, width, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    
    result = torch.zeros((batch, height, width), device=device, dtype=dtype)
    amplitude = 1.0
    frequency = scale
    max_value = 0.0
    
    for i in range(octaves):
        if time is not None:
            noise = simplex_noise_3d(coords * frequency, time * (i + 1) * 0.3, seed + i * 100, scale=1.0)
        else:
            noise = simplex_noise_2d(coords * frequency, seed + i * 100, scale=1.0)
        
        result = result + noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalize to [-1, 1]
    return result / max_value


def perlin_noise_2d(
    coords: torch.Tensor,
    seed: int = 0,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Generate 2D Perlin noise.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        seed: Random seed for reproducibility
        scale: Scale factor for noise
        
    Returns:
        Noise tensor [B, H, W] in range approximately [-1, 1]
    """
    device = coords.device
    dtype = coords.dtype
    
    p = coords * scale
    
    # Grid cell corners
    pi = torch.floor(p)
    pf = p - pi
    
    # Fade curves
    u = _fade(pf[..., 0])
    v = _fade(pf[..., 1])
    
    # Gradient vectors at corners
    def gradient_dot(cell_x: torch.Tensor, cell_y: torch.Tensor, 
                     offset_x: float, offset_y: float) -> torch.Tensor:
        # Hash to get gradient
        h = _hash2d(torch.stack([cell_x, cell_y], dim=-1), seed)
        angle = h * 2.0 * math.pi
        gx = torch.cos(angle)
        gy = torch.sin(angle)
        
        # Distance from corner
        dx = pf[..., 0] - offset_x
        dy = pf[..., 1] - offset_y
        
        return gx * dx + gy * dy
    
    # Dot products at four corners
    n00 = gradient_dot(pi[..., 0], pi[..., 1], 0.0, 0.0)
    n10 = gradient_dot(pi[..., 0] + 1, pi[..., 1], 1.0, 0.0)
    n01 = gradient_dot(pi[..., 0], pi[..., 1] + 1, 0.0, 1.0)
    n11 = gradient_dot(pi[..., 0] + 1, pi[..., 1] + 1, 1.0, 1.0)
    
    # Interpolate
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    
    return nx0 + v * (nx1 - nx0)


def worley_noise_2d(
    coords: torch.Tensor,
    seed: int = 0,
    scale: float = 1.0,
    num_points: int = 9
) -> torch.Tensor:
    """
    Generate 2D Worley (cellular) noise.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        seed: Random seed for reproducibility
        scale: Scale factor for cell size
        num_points: Number of feature points per cell (approximate)
        
    Returns:
        Noise tensor [B, H, W] representing distance to nearest point
    """
    device = coords.device
    dtype = coords.dtype
    batch, height, width, _ = coords.shape
    
    p = coords * scale
    pi = torch.floor(p)
    pf = p - pi
    
    # Initialize minimum distance
    min_dist = torch.ones((batch, height, width), device=device, dtype=dtype) * 10.0
    
    # Check surrounding cells
    for di in range(-1, 2):
        for dj in range(-1, 2):
            cell = pi + torch.tensor([di, dj], device=device, dtype=dtype)
            
            # Generate random point in cell
            torch.manual_seed(seed + int(_hash2d(cell, 0).mean().item() * 1000))
            point = cell + torch.rand(2, device=device, dtype=dtype)
            
            # Distance to this point
            diff = p - point
            dist = torch.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
            min_dist = torch.minimum(min_dist, dist)
    
    return min_dist


def value_noise_2d(
    coords: torch.Tensor,
    seed: int = 0,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Generate 2D value noise.
    
    Args:
        coords: Coordinate tensor [B, H, W, 2]
        seed: Random seed for reproducibility
        scale: Scale factor for noise
        
    Returns:
        Noise tensor [B, H, W] in range [0, 1]
    """
    p = coords * scale
    pi = torch.floor(p)
    pf = p - pi
    
    # Smooth interpolation
    u = _fade(pf[..., 0])
    v = _fade(pf[..., 1])
    
    # Values at corners
    n00 = _hash2d(pi, seed)
    n10 = _hash2d(pi + torch.tensor([1.0, 0.0], device=coords.device), seed)
    n01 = _hash2d(pi + torch.tensor([0.0, 1.0], device=coords.device), seed)
    n11 = _hash2d(pi + torch.tensor([1.0, 1.0], device=coords.device), seed)
    
    # Interpolate
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    
    return nx0 + v * (nx1 - nx0)
