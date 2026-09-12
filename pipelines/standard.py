"""
The standard (corrected) sampling pipeline.

One sigma schedule is built for the whole run and sampled in segments. Shader
noise enters at segment boundaries, so stages are parts of a single trajectory
instead of independent restarts.

At a boundary the latent is split back into a denoised estimate and its noise,
the noise is re-mixed with shader noise, and the next segment resumes from that
pair. The split happens in the model's internal space (`model.process_latent_in`
and `model_sampling.noise_scaling`), which makes it exact: at shader_strength 0
the segmented run reproduces an uninterrupted one, for both EPS and flow models.

Nothing here detects the model. The noise shape is whatever the latent is, so any
channel count works, and multi-stream latents are handled per stream -- MiniMax
H3 arrives as a NestedTensor of a video stream and an audio stream. Only the
first stream is painted; the rest keep the Gaussian noise ComfyUI gave them.

What this fixes relative to legacy, all covered by tests:
- stages no longer restart from maximum noise (flow models discarded the
  previous stage entirely, since noise_scaling is sigma*noise + (1-sigma)*latent)
- `denoise` reaches the schedule instead of being hard-coded to 1.0
- custom sigmas are used, not just counted
- blended noise keeps mean 0 / std 1
- the noise shape comes from the latent, so frames are never confused with channels
- noise_mask, batch_index and the preview callback behave like a stock KSampler
"""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import comfy.sample
import latent_preview

from ..core import noise_math, schedule, shader_noise

logger = logging.getLogger("ShaderNoiseKSampler")

# Seed offset for injection stages, so they cannot collide with sequential ones.
_INJECTION_SEED_BASE = 1000


def _is_nested(samples) -> bool:
    return bool(getattr(samples, "is_nested", False))


def _streams(samples) -> List[torch.Tensor]:
    return list(samples.unbind()) if _is_nested(samples) else [samples]


def _rebuild(reference, streams: Sequence[torch.Tensor]):
    """Repack streams the way `reference` was shaped."""
    if not _is_nested(reference):
        return streams[0]
    from comfy.nested_tensor import NestedTensor
    return NestedTensor(list(streams))


def _latent_space(model):
    """
    The transforms ComfyUI applies around a sample() call, as (into, out of).

    These belong to the model rather than the latent format. MiniMax H3 overrides
    them to carry its audio stream scaled onto the video sigma schedule, and
    `CFGGuider.inner_sample` calls the model's versions -- so inverting through
    the format alone would leave H3's audio off by `audio_scale`. Reading them
    through `get_model_object` also picks up a node's object_patches.
    """
    latent_format = model.get_model_object("latent_format")
    resolved = []
    for on_model, on_format in (("process_latent_in", "process_in"), ("process_latent_out", "process_out")):
        try:
            resolved.append(model.get_model_object(on_model))
        except (AttributeError, KeyError):  # a wrapper that only carries the format
            resolved.append(getattr(latent_format, on_format))
    return tuple(resolved)


def _shader_events(
    total_steps: int,
    sequential_stages: int,
    injection_stages: int,
    shader_strength: float,
    sequential_distribution: str,
    injection_distribution: str,
    seed: int,
    temporal_coherence: bool,
) -> Tuple[List[int], Dict[int, List[Tuple[float, int]]]]:
    """
    Work out where shader noise enters and how strong it is there.

    Returns the segment boundaries and, per boundary step, the list of
    (strength, seed) shader contributions to apply in order.
    """
    starts = schedule.sequential_starts(total_steps, sequential_stages)
    points = schedule.injection_points(total_steps, injection_stages)
    boundaries = schedule.merge_boundaries(total_steps, starts, points)

    sequential = schedule.stage_strengths(shader_strength, max(sequential_stages, 1), sequential_distribution)
    injection = schedule.stage_strengths(shader_strength, injection_stages, injection_distribution)

    events: Dict[int, List[Tuple[float, int]]] = {step: [] for step in boundaries}

    def nearest(step: int) -> int:
        return min(boundaries, key=lambda b: (abs(b - step), b))

    if sequential_stages > 0:
        for index, start in enumerate(starts):
            stage_seed = seed if temporal_coherence else seed + index
            events[nearest(start)].append((sequential[index], stage_seed))
    for index, point in enumerate(points):
        stage_seed = seed if temporal_coherence else seed + _INJECTION_SEED_BASE + index
        events[nearest(point)].append((injection[index], stage_seed))

    return boundaries, events


def _apply_events(
    noise: torch.Tensor,
    events: Sequence[Tuple[float, int]],
    shader_params: Dict[str, Any],
    shader_type: str,
    blend_mode: str,
    noise_transform: str,
    device: torch.device,
    dtype: torch.dtype,
    temporal_coherence: bool,
    normalize_strength: bool = False,
    decorrelate_channels: bool = False,
) -> torch.Tensor:
    """
    Mix each stage's shader noise into `noise`, in order.

    The shape is read off `noise` rather than carried in from the latent the run
    started with, so a boundary residual cannot be painted at a stale shape.
    """
    for strength, stage_seed in events:
        if strength <= 0.0:
            continue
        generated = shader_noise.generate(
            tuple(noise.shape), shader_params, shader_type, stage_seed, device,
            dtype=dtype, temporal_coherence=temporal_coherence,
            decorrelate=decorrelate_channels,
        )
        generated = noise_math.transform_noise(generated, noise_transform)
        noise = noise_math.mix_noise(noise, generated, blend_mode, strength, normalize_strength)
    return noise


def _split_noise(out, x0_internal, sigma, model_sampling, model):
    """
    Split a segment's result into (denoised estimate, its noise), in internal space.

    `noise_scaling(sigma, zeros, L)` is what the next sample() call applies to a
    latent, so inverting through it keeps EPS and flow models on the same path.

    The latent-space transforms come from the model rather than being handed in,
    so this stays the inverse of what the next sample() call will really do. They
    run on the whole latent, not stream by stream, so a model that treats its
    streams differently -- MiniMax H3 scales audio by `audio_scale` -- sees the
    shape it expects.
    """
    process_in, process_out = _latent_space(model)
    streams_x0 = _streams(x0_internal)
    aligned = [t.to(x0.device, x0.dtype) for t, x0 in zip(_streams(out), streams_x0)]
    streams_internal = _streams(process_in(_rebuild(out, aligned)))

    noises = []
    for internal, x0 in zip(streams_internal, streams_x0):
        x_at_sigma = model_sampling.noise_scaling(sigma, torch.zeros_like(internal), internal)
        x0_at_sigma = model_sampling.noise_scaling(sigma, torch.zeros_like(x0), x0)
        noises.append((x_at_sigma - x0_at_sigma) / sigma)

    return process_out(x0_internal), _rebuild(out, noises)


def _segment_callback(preview, offset: int, total_steps: int, captured: Dict[str, Any]):
    """Drive ComfyUI's progress bar/preview and keep the latest denoised estimate."""
    def callback(step, x0, x, total):
        captured["x0"] = x0
        if preview is not None:
            preview(offset + step, x0, x, total_steps)
    return callback


def run(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: Dict[str, Any],
    denoise: float,
    sequential_stages: int,
    injection_stages: int,
    shader_strength: float,
    blend_mode: str,
    noise_transform: str,
    shader_params: Dict[str, Any],
    shader_type: str,
    sequential_distribution: str = "linear_decrease",
    injection_distribution: str = "linear_decrease",
    use_temporal_coherence: bool = False,
    normalize_strength: bool = False,
    decorrelate_channels: bool = False,
    custom_sigmas: Optional[torch.Tensor] = None,
    disable_pbar: bool = False,
) -> Dict[str, Any]:
    """Run the corrected pipeline and return a latent dict."""
    samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    sigmas = schedule.build_sigmas(model, steps, sampler_name, scheduler, denoise, custom_sigmas)
    total_steps = max(len(sigmas) - 1, 1)
    boundaries, events = _shader_events(
        total_steps, sequential_stages, injection_stages, shader_strength,
        sequential_distribution, injection_distribution, seed, use_temporal_coherence,
    )
    segment_list = schedule.segments(boundaries, total_steps)

    primary = _streams(samples)[0]
    device, dtype = primary.device, primary.dtype

    # Refuse a latent the shaders cannot paint on before any sampling happens, rather
    # than at the first boundary that needs one. At shader_strength 0 there is nothing
    # to paint, so those models still sample through here as a plain KSampler.
    if any(strength > 0.0 for stage in events.values() for strength, _ in stage):
        shader_noise.require_spatial_latent(tuple(primary.shape))

    noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
    noise_streams = _streams(noise)
    noise_streams[0] = _apply_events(
        noise_streams[0].to(device), events.get(boundaries[0], []), shader_params,
        shader_type, blend_mode, noise_transform, device, dtype, use_temporal_coherence,
        normalize_strength, decorrelate_channels,
    )
    noise = _rebuild(samples, noise_streams)

    model_sampling = model.get_model_object("model_sampling")
    noise_mask = latent.get("noise_mask", None)
    preview = None if disable_pbar else latent_preview.prepare_callback(model, total_steps)

    current = samples
    for index, (start, end) in enumerate(segment_list):
        captured: Dict[str, Any] = {}
        is_last = index == len(segment_list) - 1
        result = comfy.sample.sample(
            model, noise, end - start, cfg, sampler_name, scheduler, positive, negative, current,
            denoise=1.0,
            sigmas=sigmas[start:end + 1],
            noise_mask=noise_mask,
            callback=_segment_callback(preview, start, total_steps, captured),
            disable_pbar=disable_pbar,
            seed=seed + start,
            force_full_denoise=is_last,
        )
        if is_last:
            return {**latent, "samples": result}

        if "x0" not in captured:  # no steps ran; carry on from where we are
            current, noise = result, noise
            continue

        current, residual = _split_noise(
            result, captured["x0"], sigmas[end], model_sampling, model
        )
        residual_streams = _streams(residual)
        residual_streams[0] = _apply_events(
            residual_streams[0], events.get(end, []), shader_params, shader_type,
            blend_mode, noise_transform, device, dtype, use_temporal_coherence,
            normalize_strength, decorrelate_channels,
        )
        noise = _rebuild(residual, residual_streams)

    return {**latent, "samples": current}
