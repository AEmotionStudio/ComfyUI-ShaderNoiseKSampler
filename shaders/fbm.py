"""
The skeleton the scalar-field generators share.

domain_warp draws a field, standardises it, masks it, hands channel 0 to
fill_channels and draws the rest through the same closures; five golden fixtures
pin that generator to the byte, so its fbm_noise stays where it is. Everything
here is for the generators added after it, which have no fixtures to keep: the
same skeleton, and an FBM whose 3D path uses the four-corner simplex rather than
the corner-0 variant those fixtures froze.

`seed` may be an int or an int64 tensor of seeds, one per channel. The
coordinates then grow a leading axis at the first hash and everything after it
broadcasts, which is how fill_channels draws the whole channel axis in one call;
see shaders/simplex.py.
"""
import torch

from .base import BaseNoiseGenerator
from .simplex import simplex_2d, simplex_3d_full
from ..utils.shape_masks import apply_shape_mask

MAX_OCTAVES = 8


def time_is_an_axis(time_axis, time):
    """
    Whether to evaluate the field in 3D with time as the third axis.

    core.shader_noise sets `time_axis` once per clip from the frame count. A
    direct caller that leaves it unset gets the older test, whether time is
    nonzero, which is what domain_warp's single-image fixtures pin.
    """
    return (time != 0) if time_axis is None else time_axis


def with_time(p, z):
    """`p[..., :2]` with a constant third coordinate."""
    return torch.cat([p[..., :2], torch.full_like(p[..., :1], float(z))], dim=-1)


def simplex_warp(p, seed, strength, frequency):
    """Displace `p` by two low-frequency simplex fields, at most `strength` apart."""
    if strength <= 0.0:
        return p
    q = p * frequency
    dx = simplex_2d(q, seed + 100)
    dy = simplex_2d(q, seed + 200)
    return torch.cat([p[..., 0:1] + dx * strength, p[..., 1:2] + dy * strength], dim=-1)


def fbm(p, octaves, seed, persistence=0.5, lacunarity=2.0, time=None, octave_offset=None):
    """
    Layered simplex noise over `p[..., :2]`.

    With `time` the layers are evaluated in 3D, each a little further along the
    time axis than the last so they do not move in lockstep. `octave_offset` is a
    translation of `[dx, dy]` per layer index, so the layers are drawn from
    different parts of the same field.
    """
    total = None
    amp, freq, norm = 1.0, 1.0, 0.0
    for i in range(min(max(int(octaves), 1), MAX_OCTAVES)):
        q = p[..., :2] * freq
        if octave_offset is not None:
            q = q + octave_offset * i
        if time is None:
            layer = simplex_2d(q, seed + i)
        else:
            layer = simplex_3d_full(with_time(q, time * (0.2 + 0.05 * i)), seed + i)
        total = amp * layer if total is None else total + amp * layer
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm


def standardise(field, seed):
    """
    Zero mean and unit deviation: over the whole draw for one seed, per leading
    slice for a tensor of seeds, so a batched channel matches the draw it replaces.
    """
    if torch.is_tensor(seed):
        dims = tuple(range(1, field.dim()))
        return (field - field.mean(dim=dims, keepdim=True)) / (field.std(dim=dims, keepdim=True) + 1e-8)
    return (field - field.mean()) / (field.std() + 1e-8)


def draw_channels(field, coords, params, seed, target_channels, contrast=1.0, palette=True):
    """
    Fill the channel axis from `field(seed) -> [B, H, W, 1]` (or `[N, B, H, W, 1]`
    for a tensor of seeds), finishing each draw the way every generator does:
    standardised, halved, masked, clamped.

    The shape mask is drawn once and lerped into every channel: it is a spatial
    shape, and apply_shape_mask reseeds the global RNG. Channel 0 stays on the
    scalar path, which is what keeps a one-channel draw and the travel-mode bases
    built from it unchanged; the palette, if any, maps that one field to three.
    """
    shape_type = params.shape_type
    shape_strength = params.shape_strength
    mask = None
    if shape_type not in ["none", "0"] and shape_strength > 0:
        mask = apply_shape_mask(coords, shape_type, params.time, seed, shape_strength)

    def finish(raw, channel_seed):
        out = standardise(raw, channel_seed) * (0.5 * contrast)
        if mask is not None:
            out = torch.lerp(out, out * mask, shape_strength)
        return torch.clamp(out, -1.0, 1.0)

    base = finish(field(seed), seed).permute(0, 3, 1, 2)
    if palette:
        base = BaseNoiseGenerator.palette_channels(base, params)

    def draw(channel_seed):
        return finish(field(channel_seed), channel_seed).permute(0, 3, 1, 2)

    def draw_many(channel_seeds):
        seeds = channel_seeds.reshape(-1, 1, 1, 1, 1)
        # [N, B, H, W, 1] -> [B, N, H, W]
        return finish(field(seeds), seeds).squeeze(-1).permute(1, 0, 2, 3)

    return BaseNoiseGenerator.fill_channels(draw, base, target_channels, seed, render_many=draw_many)
