"""
Shader noise generation for the standard pipeline.

The shape comes from the latent itself. ComfyUI latents are [B, C, H, W] for
images and [B, C, T, H, W] for video, always with channels on dim 1 and time on
dim 2. Legacy instead guessed between [B,C,F,H,W] and [B,F,C,H,W] by comparing
dimensions against the model's channel count, and defaulted to the second when
both matched -- which happens for real videos: a 61-frame Wan or Hunyuan clip
has (61-1)//4+1 = 16 latent frames and 16 channels. Time evolution was then
applied across channels instead of frames, silently.

Two other differences from legacy:
- Generation runs inside torch.random.fork_rng, so the generators' internal
  torch.manual_seed calls cannot disturb the caller's RNG stream.
- Fractional octaves interpolate between the neighbouring integer renders.
  ShaderParams.validate() truncates octaves to int, so the node's 0.1-step
  octaves slider previously did nothing until it crossed a whole number.
"""
from typing import Any, Dict, Tuple

import torch

from .constants import CHANNEL_BASIS
from .params import ShaderParams

# Below this the fractional part is not worth a second render.
_FRACTION_EPSILON = 1e-3


class UnsupportedLatentError(ValueError):
    """The latent has no 2D spatial grid for a shader to draw on."""


def require_spatial_latent(shape: Tuple[int, ...], allow_sequence: bool = False) -> None:
    """
    Refuse latents the shaders cannot draw on, naming what was wrong.

    Every shader paints a height x width grid, so it needs [B, C, H, W] or
    [B, C, T, H, W]. Some models instead carry a plain sequence -- audio
    (Stable Audio, ACE-Step 1.5, MiniMax Music 3), Hunyuan3D's occupancy grid
    and TripoSplat's [B, tokens, channels] -- where there is no grid to paint.

    `allow_sequence` opts those in as a 1 x length strip, which is well defined
    even though it is not what the shaders were written for.
    """
    if len(shape) in (4, 5) or (allow_sequence and len(shape) == 3):
        return
    raise UnsupportedLatentError(
        f"shader noise needs a latent with a 2D spatial grid, either [B, C, H, W] or "
        f"[B, C, T, H, W], but this model's latent is {len(shape)}D {tuple(shape)}. "
        f"Sequence latents (Stable Audio, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D, "
        f"TripoSplat) have no grid to draw on. Set shader_strength to 0.0 to sample "
        f"them with this node as a plain KSampler."
    )


def latent_layout(shape: Tuple[int, ...], allow_sequence: bool = False) -> Dict[str, int]:
    """
    Describe a latent shape the way ComfyUI lays it out.

    Returns batch, channels, frames (1 for images), height and width. A sequence
    latent, when opted in, reads as a single row: height 1, width the length.
    """
    require_spatial_latent(shape, allow_sequence)
    if len(shape) == 5:
        batch, channels, frames, height, width = shape
    elif len(shape) == 4:
        (batch, channels, height, width), frames = shape, 1
    else:
        (batch, channels, width), frames, height = shape, 1, 1
    return {"batch": batch, "channels": channels, "frames": frames, "height": height, "width": width}


def resolve_generator(shader_type: str):
    """
    Look up a generator by name.

    An unknown name raises instead of quietly substituting different noise.
    Legacy fell back to shader_params_reader.generate_noise_tensor, which is no
    help: it raises for every type it is handed, including perlin, cellular and
    waves -- archetypes named in the parameter vocabulary but not shipped here --
    so the fallback only replaced a clear error with a confusing one.

    Consults the registry without importing comfy, so this module stays
    testable on its own.
    """
    from ..shaders.registry import get_shader, list_shaders

    registered = get_shader(shader_type)
    if registered is not None:
        generate = getattr(registered, "generate", None)
        if callable(generate):
            return generate
        if callable(registered):
            return registered

    raise ValueError(
        f"unknown shader type {shader_type!r}; registered: {', '.join(sorted(list_shaders()))}"
    )


def _as_dict(params: Any) -> Dict[str, Any]:
    if hasattr(params, "to_dict"):
        return params.to_dict()
    return dict(params or {})


def _render(generator, params: Dict[str, Any], octaves: int, layout, seed, device) -> torch.Tensor:
    frame_params = dict(params)
    frame_params["octaves"] = octaves
    return generator(
        params=ShaderParams(frame_params).validate(),
        height=layout["height"],
        width=layout["width"],
        batch_size=layout["batch"],
        device=device,
        seed=seed,
        target_channels=layout["channels"],
    )


def _render_octaves(generator, params: Dict[str, Any], layout, seed, device) -> torch.Tensor:
    """Render one frame, interpolating between integer octave counts when asked."""
    requested = float(params.get("octaves", 1.0) or 1.0)
    low = max(1, int(requested))
    fraction = requested - low

    noise = _render(generator, params, low, layout, seed, device)
    if fraction > _FRACTION_EPSILON:
        higher = _render(generator, params, low + 1, layout, seed, device)
        if higher.shape == noise.shape:
            noise = torch.lerp(noise, higher, fraction)
    return noise


# The widest basis a travel mode asks for. The generators fill their own channel
# axis to the same cap (shaders/base.py::fill_channels), so a draw arrives here
# already spanning about this many directions; see _maybe_decorrelate for what
# that leaves this module to do.
#
# 64 was set from measurement on MiniMax H3 at strength 0.75, where noise spanning
# one channel is a solid quilted texture: a basis of 8 was still mostly destroyed,
# 16 coherent and 32 clean.
DECORRELATION_BASIS = CHANNEL_BASIS

# What a random remix of a basis actually reaches, as a share of the directions
# it aims at. Measured at 16, 24 and 128 channels across all four generators: 0.52
# to 0.64. A draw already wider than that cannot be widened by remixing, so it is
# left alone rather than rendered a second time and thrown away.
_MIX_RANK_YIELD = 0.65

# Arbitrary but fixed, so a seed still reproduces.
_MIX_SEED_STRIDE = 7919

def effective_channel_rank(noise: torch.Tensor) -> float:
    """
    Participation ratio of the channel covariance spectrum: how many channels the
    noise really spans. Full for i.i.d. noise, ~1 when every channel is a copy.
    """
    channels = noise.shape[1]
    if channels < 2:
        return float(channels)
    flat = noise.reshape(noise.shape[0], channels, -1)[0].float()
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    spectrum = torch.linalg.svdvals(flat) ** 2
    weights = spectrum / spectrum.sum().clamp_min(1e-12)
    return float(torch.exp(-(weights * weights.clamp_min(1e-12).log()).sum()))


def _decorrelate(latent_shape, params, shader_type, seed, device, dtype,
                 temporal_coherence, generator, basis_size=None) -> torch.Tensor:
    """
    Rebuild the channel axis as mixtures of `basis_size` independent renders.

    Each output channel is a different random mixture of the renders, so the draw
    spans about min(channels, basis) directions -- in practice about 0.6 of that,
    since random mixing is not orthogonal -- and every channel keeps the shader's
    spatial character.

    The generators span their own channels now, so this is how a travel mode
    narrows a draw on purpose: `drift` down to four directions, `jump` to one. It
    still widens a draw that arrives narrow, though no shipped generator does.
    """
    channels = latent_shape[1]
    basis = min(channels, DECORRELATION_BASIS if basis_size is None else max(1, basis_size))
    single = (latent_shape[0], 1) + tuple(latent_shape[2:])

    draws = torch.stack([
        _generate(single, params, shader_type, seed + _MIX_SEED_STRIDE * i, device,
                 dtype=dtype, temporal_coherence=temporal_coherence, generator=generator)
        for i in range(basis)
    ])                                              # [basis, B, 1, ...]

    mixer = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.randn(channels, basis, generator=mixer, dtype=torch.float32)
    weights = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-8)
    weights = weights.to(device=device, dtype=dtype)

    flat = draws.reshape(basis, -1)                 # [basis, everything]
    mixed = (weights @ flat).reshape((channels,) + draws.shape[1:])
    return torch.cat([mixed[c] for c in range(channels)], dim=1)


def generate(
    latent_shape: Tuple[int, ...],
    params: Any,
    shader_type: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    temporal_coherence: bool = False,
    generator=None,
    decorrelate: bool = False,
    allow_sequence: bool = False,
    basis: int = None,
) -> torch.Tensor:
    """
    Generate shader noise matching a latent's shape.

    Video is generated frame by frame and stacked on dim 2, the time axis.
    Without temporal coherence each frame advances the seed, which is what makes
    frames differ; with it, the seed is held and only `time` advances.

    The result is returned unnormalised; core.noise_math.mix_noise standardises
    both sides when blending.

    The draw runs under inference mode, which is worth about a tenth of it and
    costs nothing here: nothing in a render needs autograd. The clone on the way
    out is not optional. An inference tensor raises `Inference tensors cannot be
    saved for backward` the moment it reaches a grad-recording region, and this
    noise is handed to whatever the workflow does next, so it must leave as an
    ordinary tensor. Cloning outside the block is what makes it one -- about 5 ms
    and 14 MB at H3's default latent.
    """
    with torch.inference_mode():
        noise = _generate(latent_shape, params, shader_type, seed, device, dtype,
                          temporal_coherence, generator, decorrelate, allow_sequence, basis)
    return noise.clone() if torch.is_inference(noise) else noise


def _generate(
    latent_shape: Tuple[int, ...],
    params: Any,
    shader_type: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    temporal_coherence: bool = False,
    generator=None,
    decorrelate: bool = False,
    allow_sequence: bool = False,
    basis: int = None,
) -> torch.Tensor:
    """The draw itself. Recursive callers use this directly, so the clone in
    `generate` happens once rather than at every level."""
    latent_shape = tuple(latent_shape)
    if allow_sequence and len(latent_shape) == 3:
        # Paint it as a one-row strip, then fold the row away again. Audio latents
        # and Hunyuan3D's occupancy grid arrive this way.
        batch, channels, length = latent_shape
        strip = _generate((batch, channels, 1, length), params, shader_type, seed, device,
                         dtype=dtype, temporal_coherence=temporal_coherence,
                         generator=generator, decorrelate=decorrelate, basis=basis)
        return strip.reshape(latent_shape)

    layout = latent_layout(latent_shape)
    generator = generator or resolve_generator(shader_type)
    base_params = _as_dict(params)
    base_time = float(base_params.get("time", 0.0) or 0.0)
    # Whether `time` is a real axis for this draw, rather than whether a particular
    # frame's value happens to be zero. A generator that switches between a 2D and a
    # 3D field must make that choice once for the whole clip: deciding it per frame
    # drew frame 0 -- the only frame whose time is exactly 0.0 -- from a different
    # function than every frame after it.
    base_params["time_axis"] = layout["frames"] > 1

    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if torch.device(device).type == "cuda" else []

    with torch.random.fork_rng(devices=devices):
        # A collapse rebuilds every channel from one-channel draws and never looks at
        # the wide draw, so rendering one first is pure waste -- 7.68 s of jump's
        # 7.98 s at H3's default latent, for a tensor that is dropped. The collapse
        # branch in _maybe_decorrelate turns on values all known before rendering, so
        # take it here instead. Verified torch.equal against the old path for all four
        # generators at 4, 16, 24 and 128 channels.
        if decorrelate and latent_shape[1] >= 2 \
                and (DECORRELATION_BASIS if basis is None else max(1, basis)) <= 1:
            return _decorrelate(latent_shape, params, shader_type, seed, device,
                                dtype, temporal_coherence, None, basis_size=1)

        if layout["frames"] == 1 and len(latent_shape) == 4:
            noise = _render_octaves(generator, base_params, layout, seed, device)
            return _maybe_decorrelate(
                _fit(noise, latent_shape, device, dtype), decorrelate, latent_shape,
                params, shader_type, seed, device, dtype, temporal_coherence, basis)

        frames = []
        span = max(layout["frames"] - 1, 1)
        for index in range(layout["frames"]):
            frame_params = dict(base_params)
            frame_params["time"] = base_time + index / span
            frame_seed = seed if temporal_coherence else seed + index
            frames.append(_render_octaves(generator, frame_params, layout, frame_seed, device))

    stacked = torch.stack(frames, dim=2)  # [B, C, T, H, W]
    return _maybe_decorrelate(
        _fit(stacked, latent_shape, device, dtype), decorrelate, latent_shape,
        params, shader_type, seed, device, dtype, temporal_coherence, basis)


def _fit(noise: torch.Tensor, target_shape, device, dtype) -> torch.Tensor:
    """Last-resort shape correction; generators are expected to honour the request."""
    target_shape = tuple(target_shape)
    noise = noise.to(device=device, dtype=dtype)
    if tuple(noise.shape) == target_shape:
        return noise

    corrected = torch.zeros(target_shape, device=device, dtype=dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(noise.shape, target_shape))
    corrected[slices] = noise[slices]
    return corrected


def _maybe_decorrelate(noise, decorrelate, latent_shape, params, shader_type, seed,
                       device, dtype, temporal_coherence, basis=None):
    """
    Remix the channel axis to the width a travel mode asks for.

    Direction decides what is guarded. Narrowing is a request -- `drift` and
    `jump` want fewer directions than the generator drew -- so it is never
    second-guessed. Widening is an improvement that might not be one, so it is
    skipped when the draw is already as wide as a remix could make it, and kept
    only when it actually came out wider.
    """
    if not decorrelate or noise.shape[1] < 2:
        return noise

    size = DECORRELATION_BASIS if basis is None else max(1, basis)
    if size <= 1:
        # A deliberate collapse: every channel carries the same field, so the
        # shader's parameters decide the destination and the seed stops mattering.
        return _decorrelate(tuple(latent_shape), params, shader_type, seed, device,
                            dtype, temporal_coherence, None, basis_size=1)

    target = min(noise.shape[1], size)
    stock_rank = effective_channel_rank(noise)

    # `drift` arrives asking for four directions out of a draw spanning twenty-odd.
    # Before the generators filled their own channels, a basis of four widened the
    # noise and this branch was unreachable; without it the widening guard below
    # sees a wide draw, leaves it alone, and drift silently becomes walk. Only a
    # basis below the widest counts: at 128 channels a draw can exceed 64 directions,
    # and walk asking for 64 is not a request to come down to them.
    if size < DECORRELATION_BASIS and target < stock_rank:
        return _decorrelate(tuple(latent_shape), params, shader_type, seed, device,
                            dtype, temporal_coherence, None, basis_size=size)

    # A remix reaches about _MIX_RANK_YIELD of its target, so a draw already wider
    # is left alone. The guard is on rank rather than correlation because noise
    # that looks correlated can still span most of its channels.
    if stock_rank >= target * _MIX_RANK_YIELD:
        return noise

    # Keep whichever is actually wider, so asking to widen can never narrow.
    remixed = _decorrelate(tuple(latent_shape), params, shader_type, seed, device,
                           dtype, temporal_coherence, None, basis_size=size)
    return remixed if effective_channel_rank(remixed) > stock_rank else noise
