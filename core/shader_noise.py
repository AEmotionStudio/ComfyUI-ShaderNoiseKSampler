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


# Independent draws combined to fill the channel axis, capped so a very wide
# latent cannot demand one render per channel.
#
# Set from measurement, not guessed. On MiniMax H3 at strength 0.75, where stock
# noise is a solid quilted texture, a basis of 8 is still mostly destroyed, 16 is
# coherent and 32 is clean. The achieved rank runs at roughly 85-90% of the basis
# until it hits the channel count, so 64 gives full independence to everything up
# to 64 channels and takes LTXV's 128 from rank 7.6 to 43.
#
# The cost this cap was originally guarding against turned out not to exist: the
# worst case measured is about a second per draw, on a run that takes thirty to
# fifty, and a draw happens once per stage boundary.
DECORRELATION_BASIS = 64

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
    Fill the channel axis with independent draws instead of copies of one.

    The generators build extra channels as pointwise functions of the first one
    or two (`shaders/base.py::_expand_channels`), so the result is effectively
    rank 1-2 however many channels are asked for -- exactly rank 1.00 for
    domain_warp at 4 channels. Samplers expect i.i.d. noise, and that collapse is
    what makes the shader's own pattern surface in the picture so readily.

    Here each output channel is a different mixture of `DECORRELATION_BASIS`
    independent renders, so the rank is min(channels, basis) while the spatial
    character of the shader is preserved -- every basis element is still that
    shader.
    """
    channels = latent_shape[1]
    basis = min(channels, DECORRELATION_BASIS if basis_size is None else max(1, basis_size))
    single = (latent_shape[0], 1) + tuple(latent_shape[2:])

    draws = torch.stack([
        generate(single, params, shader_type, seed + _MIX_SEED_STRIDE * i, device,
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
    """
    latent_shape = tuple(latent_shape)
    if allow_sequence and len(latent_shape) == 3:
        # Paint it as a one-row strip, then fold the row away again. Audio latents
        # and Hunyuan3D's occupancy grid arrive this way.
        batch, channels, length = latent_shape
        strip = generate((batch, channels, 1, length), params, shader_type, seed, device,
                         dtype=dtype, temporal_coherence=temporal_coherence,
                         generator=generator, decorrelate=decorrelate, basis=basis)
        return strip.reshape(latent_shape)

    layout = latent_layout(latent_shape)
    generator = generator or resolve_generator(shader_type)
    base_params = _as_dict(params)
    base_time = float(base_params.get("time", 0.0) or 0.0)

    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if torch.device(device).type == "cuda" else []

    with torch.random.fork_rng(devices=devices):
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
    Remix the channel axis.

    Widening is guarded, because it is an improvement that might not be one.
    Narrowing is not, because a caller asking for a basis of 1 wants the collapse
    and the guards exist precisely to prevent it.
    """
    if not decorrelate or noise.shape[1] < 2:
        return noise

    size = DECORRELATION_BASIS if basis is None else max(1, basis)
    if size <= 1:
        # A deliberate collapse: every channel carries the same field, so the
        # shader's parameters decide the destination and the seed stops mattering.
        return _decorrelate(tuple(latent_shape), params, shader_type, seed, device,
                            dtype, temporal_coherence, None, basis_size=1)

    # Cheap reject first: remixing can only reach about min(channels, basis)
    # independent directions, so a draw already at least that wide is left alone.
    # This is why the guard is on rank and not on correlation -- curl_noise looks
    # correlated but already spans 25 of 128 channels.
    stock_rank = effective_channel_rank(noise)
    if stock_rank >= min(noise.shape[1], size) * 0.9:
        return noise

    # The basis draws are not guaranteed independent either: at four channels
    # curl_noise remixes to a *lower* rank than it started with. Rather than tune
    # a threshold per generator, keep whichever is actually wider, so asking to
    # widen can never make the noise narrower than leaving it alone.
    remixed = _decorrelate(tuple(latent_shape), params, shader_type, seed, device,
                           dtype, temporal_coherence, None, basis_size=size)
    return remixed if effective_channel_rank(remixed) > stock_rank else noise
