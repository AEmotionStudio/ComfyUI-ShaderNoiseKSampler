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


def latent_layout(shape: Tuple[int, ...]) -> Dict[str, int]:
    """
    Describe a latent shape the way ComfyUI lays it out.

    Returns batch, channels, frames (1 for images), height and width.
    """
    if len(shape) == 5:
        batch, channels, frames, height, width = shape
    elif len(shape) == 4:
        (batch, channels, height, width), frames = shape, 1
    else:
        raise ValueError(f"expected a 4D or 5D latent shape, got {tuple(shape)}")
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


def generate(
    latent_shape: Tuple[int, ...],
    params: Any,
    shader_type: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    temporal_coherence: bool = False,
    generator=None,
) -> torch.Tensor:
    """
    Generate shader noise matching a latent's shape.

    Video is generated frame by frame and stacked on dim 2, the time axis.
    Without temporal coherence each frame advances the seed, which is what makes
    frames differ; with it, the seed is held and only `time` advances.

    The result is returned unnormalised; core.noise_math.mix_noise standardises
    both sides when blending.
    """
    layout = latent_layout(tuple(latent_shape))
    generator = generator or resolve_generator(shader_type)
    base_params = _as_dict(params)
    base_time = float(base_params.get("time", 0.0) or 0.0)

    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if torch.device(device).type == "cuda" else []

    with torch.random.fork_rng(devices=devices):
        if layout["frames"] == 1 and len(latent_shape) == 4:
            noise = _render_octaves(generator, base_params, layout, seed, device)
            return _fit(noise, latent_shape, device, dtype)

        frames = []
        span = max(layout["frames"] - 1, 1)
        for index in range(layout["frames"]):
            frame_params = dict(base_params)
            frame_params["time"] = base_time + index / span
            frame_seed = seed if temporal_coherence else seed + index
            frames.append(_render_octaves(generator, frame_params, layout, frame_seed, device))

    stacked = torch.stack(frames, dim=2)  # [B, C, T, H, W]
    return _fit(stacked, latent_shape, device, dtype)


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
