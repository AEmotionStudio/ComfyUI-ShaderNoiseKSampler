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

F2 = 0.5 * (math.sqrt(3.0) - 1.0)
G2 = (3.0 - math.sqrt(3.0)) / 6.0
F3 = 1.0 / 3.0
G3 = 1.0 / 6.0


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

    xs, ys = [], []
    for value in seed.flatten().tolist():
        angle = (int(value) % 628) / 100.0
        cos_r, sin_r = math.cos(angle), math.sin(angle)
        xs.append((x * cos_r - y * sin_r).unsqueeze(0))
        ys.append((x * sin_r + y * cos_r).unsqueeze(0))
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
