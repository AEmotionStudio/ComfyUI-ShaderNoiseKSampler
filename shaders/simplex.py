"""
The simplex primitives, in one place, with a batchable seed.

Every generator carried its own copy of these: the 2D hash appeared three times
and the 3D hash four, all with the body

    h = ix*1619 + iy*31337 + seed*2459   ->   fmod(h*h*h, 1013)

The copies are **not** interchangeable, and unifying them would silently change
three generators, so the variants are kept and named for what they do:

- `simplex_2d(p, seed, rotate=False)` -- `domain_warp` rotates its coordinates by
  `(seed % 628) / 100` radians before skewing; `curl_noise` and `tensor_field` do not.
- `simplex_3d(p, seed, corners=1)` -- `temporal_coherent` sums all four simplex
  corners. The other three copies evaluate corner 0 alone, which is not really
  simplex noise, but it is what eleven golden fixtures and every calibrated preset
  pin. `domain_warp` even hashed the other three corners and discarded them.

**The seed may be a tensor**, one per leading slice, which is the point of this
module. `p` then carries a leading axis of N draws -- `[N, B, H, W, 2]` -- against
a `[N, 1, 1, 1, 1]` seed, and one call renders N channels instead of N calls
rendering one. Two rules make that bit-exact rather than merely close:

1. **The seed tensor must be int64 and the overflow is load-bearing.** `h*h*h`
   overflows int64 for realistic coordinates and the hash depends on how it wraps.
   Any other dtype computes a different function.
2. **The rotation table comes from Python `math.cos`, not `torch.cos`.** The scalar
   path multiplies by `float32(math.cos(float64_angle))`; `torch.cos` on a float32
   tensor is a different code path with no last-ulp guarantee. N is at most 64, so
   the Python loop costs nothing.

Everything else is elementwise or a per-slice reduction, both of which are
bit-identical under batching (verified: mean, std and amax over `dim=(1,2,3,4)`
match per-slice scalar reductions exactly).
"""
import math

import torch
import torch.nn.functional as F

F2 = 0.5 * (math.sqrt(3.0) - 1.0)
G2 = (3.0 - math.sqrt(3.0)) / 6.0
F3 = 1.0 / 3.0
G3 = 1.0 / 6.0

# Gradient table for the four-corner 3D simplex, indexed by hash.
SIMPLEX_GRADIENTS = torch.tensor([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
], dtype=torch.float32)


def _rotate(x, y, seed):
    """
    Turn the coordinates by `(seed % 628) / 100` radians, one angle per slice.

    Deliberately a Python loop producing one slice at a time from the *shared*
    coordinate tensor, rather than a broadcast multiply by a `[N,1,1,1,1]` cosine.
    Two things make the obvious version inexact:

    - `x * python_float` and `x * float32_tensor` are different kernels and
      contract `a*b - c*d` into an fma differently at some shapes -- a 6e-08 drift
      that appeared at 22x38 but not 48x84.
    - slicing an already-expanded `[N,B,H,W,1]` view changes both the rank and the
      contiguity the kernel sees, which moved a different set of shapes.

    Computing each slice from the same rank-4 tensor the scalar path uses makes
    every slice bit-identical to the draw it replaces, by construction. N is at
    most CHANNEL_BASIS and this is six cheap ops per slice against a few hundred
    in the rest of the function.
    """
    if not torch.is_tensor(seed):
        angle = (int(seed) % 628) / 100.0
        cos_r, sin_r = math.cos(angle), math.sin(angle)
        return x * cos_r - y * sin_r, x * sin_r + y * cos_r

    # `x` either still carries the shared coordinates, in which case the rotation is
    # what grows the leading axis, or it already has one slice per seed because an
    # earlier step in the chain materialised it.
    per_slice = x.dim() == seed.dim()
    xs, ys = [], []
    for index, value in enumerate(seed.flatten().tolist()):
        angle = (int(value) % 628) / 100.0
        cos_r, sin_r = math.cos(angle), math.sin(angle)
        xi = x[index:index + 1] if per_slice else x.unsqueeze(0)
        yi = y[index:index + 1] if per_slice else y.unsqueeze(0)
        xs.append(xi * cos_r - yi * sin_r)
        ys.append(xi * sin_r + yi * cos_r)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _fold(seed, modulus):
    """`seed % modulus`, for an int or an int64 tensor of seeds."""
    if torch.is_tensor(seed):
        return seed.to(torch.int64) % modulus
    return int(seed) % modulus


def _grad2(h, gx, gy):
    h_int = h.long() % 8
    u = torch.where(h_int < 4, gx, gy)
    v = torch.where(h_int < 4, gy, gx)
    return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)


def simplex_2d(p, seed, rotate=False):
    """
    2D simplex noise over `p[..., 0:2]`, returning the same shape with a trailing 1.

    `seed` is an int, or an int64 tensor broadcastable against `p`'s leading axes.
    """
    squeeze = p.dim() == 3
    if squeeze:
        p = p.unsqueeze(0)
    if p.shape[-1] < 2:
        p = torch.cat([p, p], dim=-1)

    seed = _fold(seed, 10000)
    x, y = p[..., 0:1], p[..., 1:2]

    if rotate:
        # This is what grows the leading axis when the seed is a tensor: every
        # slice is rotated from the shared coordinates, so each one matches the
        # scalar draw exactly.
        x, y = _rotate(x, y, seed)
    elif torch.is_tensor(seed) and x.dim() == seed.dim() - 1:
        x, y = x.unsqueeze(0), y.unsqueeze(0)

    s = (x + y) * F2
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    t = (i + j) * G2
    x0, y0 = x - (i - t), y - (j - t)

    i1 = (x0 > y0).float()
    j1 = 1.0 - i1
    x1, y1 = x0 - i1 + G2, y0 - j1 + G2
    x2, y2 = x0 - 1.0 + 2.0 * G2, y0 - 1.0 + 2.0 * G2

    seed_term = (seed if torch.is_tensor(seed) else seed) * 2459

    def hash_coord(ix, iy):
        h = ix * 1619 + iy * 31337 + seed_term
        return torch.fmod(h * h * h, 1013)

    i0l, j0l = i.long(), j.long()
    h0 = hash_coord(i0l, j0l)
    h1 = hash_coord(i0l + i1.long(), j0l + j1.long())
    h2 = hash_coord(i0l + 1, j0l + 1)

    # Straight-line rather than a loop over the three corners: at these tensor
    # sizes the draw is dominated by per-op dispatch, and the loop's own overhead
    # measured 7% of the function.
    zero = torch.zeros_like(x0)
    t0 = torch.maximum(0.5 - x0 * x0 - y0 * y0, zero)
    t1 = torch.maximum(0.5 - x1 * x1 - y1 * y1, zero)
    t2 = torch.maximum(0.5 - x2 * x2 - y2 * y2, zero)

    n0 = t0 ** 4 * _grad2(h0, x0, y0)
    n1 = t1 ** 4 * _grad2(h1, x1, y1)
    n2 = t2 ** 4 * _grad2(h2, x2, y2)

    result = 70.0 * (n0 + n1 + n2)
    if squeeze:
        result = result.squeeze(0)
    return result if result.shape[-1] == 1 else result.unsqueeze(-1)


def _grad3(h, gx, gy, gz):
    h_int = h.long() % 12
    u = torch.where(h_int < 8, gx, gy)
    # The copies this replaces wrote `torch.where(h_int < 4, gy,
    # torch.where((h_int == 12) | (h_int == 14), gx, gz))`. After `% 12` the index
    # is in [0, 11], so that inner test can never fire and the branch is always gz.
    v = torch.where(h_int < 4, gy, gz)
    return torch.where(h_int % 2 == 0, u, -u) + torch.where((h_int // 2) % 2 == 0, v, -v)


def simplex_3d(p, seed, corners=1):
    """
    3D simplex noise over `p[..., 0:3]`, with time as the third axis.

    `corners=1` is what domain_warp, curl_noise and tensor_field carry: only the
    first simplex corner contributes, which is not really simplex noise -- it is
    blockier and less isotropic than a proper corner sum -- but it is what the
    golden fixtures and the calibrated presets pin, and it is the only variant
    this function implements. temporal_coherent's four-corner version is a
    different function, not a parameterisation of this one: it picks corners by
    the real simplex ordering and reads gradients from a table, so it lives with
    that generator rather than here.

    `seed` is an int, or an int64 tensor broadcastable against `p`'s leading axes.
    """
    squeeze = p.dim() == 3
    if squeeze:
        p = p.unsqueeze(0)

    x, y = p[..., 0:1], p[..., 1:2]
    z = p[..., 2:3] if p.shape[-1] > 2 else torch.zeros_like(x)

    s = (x + y + z) * F3
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    k = torch.floor(z + s)

    t = (i + j + k) * G3
    x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

    seed_term = seed * 2459

    def hash3(ix, iy, iz):
        h = ix * 1619 + iy * 31337 + iz * 6971 + seed_term
        return torch.fmod(h * h * h, 1013)

    i0, j0, k0 = i.long(), j.long(), k.long()
    falloff = torch.maximum(0.6 - x0 * x0 - y0 * y0 - z0 * z0, torch.zeros_like(x0))
    result = falloff ** 4 * _grad3(hash3(i0, j0, k0), x0, y0, z0)

    result = 32.0 * result
    if squeeze:
        result = result.squeeze(0)
    return result if result.shape[-1] == 1 else result.unsqueeze(-1)


def lattice_hash(ix, iy, seed, iz=None):
    """
    A uniform value in [0, 1) per integer lattice point, from the hash above.

    `ix`, `iy` and `iz` are int64 tensors; `seed` is an int or an int64 tensor
    of seeds that broadcasts against them and grows the leading axis the way
    `simplex_2d` does. The cube alone is linear mod 1013 until it wraps int64,
    which a small seed and small coordinates never do, and then two cells 29 by
    10 apart share a value; one xorshift round after it breaks that up.
    """
    h = ix * 1619 + iy * 31337 + _fold(seed, 10000) * 2459
    if iz is not None:
        h = h + iz * 6971
    h = h * h * h
    h = (h ^ (h >> 21)) * 2654435761
    return torch.remainder(h, 1013).to(torch.float32) / 1013.0


def simplex_3d_full(coords, seed=0):
    """
    Four-corner 3D simplex noise over `coords[..., :3]`.

    temporal_coherent's own function, moved here unchanged so the generators
    added after it can share it; its golden fixture pins the generator across
    the move. `seed` is an int, or an int64 tensor of seeds shaped
    [N, 1, 1, 1, 1] as fill_channels hands them over.
    """
    dim = coords.shape[-1]
    device = coords.device
    if torch.is_tensor(seed):
        # One seed per channel, always shaped [N,1,1,1]. Everything below works
        # on coords[..., k], which is [B,H,W] while the coordinates are still
        # shared and [N,B,H,W] once an earlier step has grown the axis; a rank-4
        # seed broadcasts correctly against both. Deriving the rank from the
        # coordinates instead collapses the batch axis in the shared case.
        seed = seed.reshape(-1, 1, 1, 1)

    gradients = SIMPLEX_GRADIENTS.to(device)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2] if dim > 2 else torch.zeros_like(x)

    s = (x + y + z) * F3
    i = torch.floor(x + s)
    j = torch.floor(y + s)
    k = torch.floor(z + s)

    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)

    # Determine simplex
    x_ge_y = (x0 >= y0).float()
    y_ge_z = (y0 >= z0).float()
    x_ge_z = (x0 >= z0).float()

    i1 = x_ge_y * x_ge_z
    j1 = (1 - x_ge_y) * y_ge_z
    k1 = (1 - x_ge_z) * (1 - y_ge_z)

    i2 = x_ge_y + (1 - x_ge_y) * x_ge_z
    j2 = x_ge_y * (1 - x_ge_z) + (1 - x_ge_y)
    k2 = (1 - x_ge_z) + x_ge_z * (1 - x_ge_y)

    def grad3d(ix, iy, iz, gx, gy, gz):
        h = (ix * 1619 + iy * 31337 + iz * 6971 + seed * 2459)
        h = torch.fmod(h * h * h, 1013)
        grads = F.embedding(h.long() % 12, gradients)
        return grads[..., 0] * gx + grads[..., 1] * gy + grads[..., 2] * gz

    noise = torch.zeros_like(x0)

    t0 = 0.6 - x0*x0 - y0*y0 - z0*z0
    mask0 = (t0 >= 0).float()
    t0 = t0 * t0
    noise = noise + mask0 * t0 * t0 * grad3d(i, j, k, x0, y0, z0)

    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    t1 = 0.6 - x1*x1 - y1*y1 - z1*z1
    mask1 = (t1 >= 0).float()
    t1 = t1 * t1
    noise = noise + mask1 * t1 * t1 * grad3d(i + i1, j + j1, k + k1, x1, y1, z1)

    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    t2 = 0.6 - x2*x2 - y2*y2 - z2*z2
    mask2 = (t2 >= 0).float()
    t2 = t2 * t2
    noise = noise + mask2 * t2 * t2 * grad3d(i + i2, j + j2, k + k2, x2, y2, z2)

    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3
    t3 = 0.6 - x3*x3 - y3*y3 - z3*z3
    mask3 = (t3 >= 0).float()
    t3 = t3 * t3
    noise = noise + mask3 * t3 * t3 * grad3d(i + 1, j + 1, k + 1, x3, y3, z3)

    return (noise * 32.0).unsqueeze(-1)
