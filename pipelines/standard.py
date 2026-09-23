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

from ..core import noise_math, presets, schedule, shader_noise

logger = logging.getLogger("ShaderNoiseKSampler")

# Seed offset for injection stages, so they cannot collide with sequential ones.
_INJECTION_SEED_BASE = 1000

# Bounds a shaped stage must stay inside; the node's own widgets allow these.
_MIN_SCALE = 0.1
_MAX_OCTAVES = 8.0


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


# Keeps one stream's pattern from being another's; arbitrary but fixed.
_STREAM_SEED_STRIDE = 104729


# A stream smaller than this carries settings rather than content, so painting it
# would corrupt a parameter instead of varying a picture. TripoSplat's second
# stream is a [B, 1, 5] camera; H3's audio, the smallest real one, is 414 cells.
_METADATA_ELEMENTS = 64


def _paintable(streams, shade_non_spatial: bool):
    """
    Which streams get shader noise.

    Only the first by default: it is the spatial one, and the rest -- MiniMax H3
    and LTXAV's audio -- have no grid to paint. With `shade_non_spatial` they all
    do, except any stream small enough to be metadata. That exception is what
    keeps the option from writing noise into TripoSplat's camera parameters,
    which would move the viewpoint rather than vary the subject.
    """
    if not shade_non_spatial:
        return [0]
    return [i for i, stream in enumerate(streams)
            if i == 0 or stream.numel() // max(stream.shape[0], 1) >= _METADATA_ELEMENTS]


def _shader_events(
    total_steps: int,
    first: int,
    last: int,
    sequential_stages: int,
    injection_stages: int,
    shader_strength: float,
    sequential_distribution: str,
    injection_distribution: str,
    seed: int,
    temporal_coherence: bool,
    stage_progression: str = "uniform",
) -> Tuple[List[int], Dict[int, List[Tuple[float, int, Dict[str, float]]]]]:
    """
    Work out where shader noise enters and how strong it is there.

    Stages are spread across the step window `[first, last]` -- the steps this
    node actually samples -- so a run of the last three steps of a schedule gets
    its stages in those three steps rather than in a range it never reaches.

    Returns the segment boundaries and, per boundary step, the list of
    (strength, seed, shaping) shader contributions to apply in order. `shaping`
    carries the zoom and detail adjustments for that point in the trajectory and
    is empty under the default uniform progression.
    """
    span = last - first
    starts = [first + start for start in schedule.sequential_starts(span, sequential_stages)]
    points = [first + point for point in schedule.injection_points(span, injection_stages)]
    boundaries = schedule.merge_boundaries(last, starts, points, first=first)

    sequential = schedule.stage_strengths(shader_strength, max(sequential_stages, 1), sequential_distribution)
    injection = schedule.stage_strengths(shader_strength, injection_stages, injection_distribution)

    events: Dict[int, List[Tuple[float, int, Dict[str, float]]]] = {step: [] for step in boundaries}

    def shaping(step: int) -> Dict[str, float]:
        # Position in the whole schedule, not stage index and not position in the
        # window: sequential and injection stages interleave, and what matters is
        # how far along the trajectory the noise lands. A node running the last
        # three steps of seven is at the fine end, not starting a fresh sweep.
        return schedule.stage_shaping(stage_progression, step / max(total_steps, 1))

    def nearest(step: int) -> int:
        return min(boundaries, key=lambda b: (abs(b - step), b))

    if sequential_stages > 0:
        for index, start in enumerate(starts):
            stage_seed = seed if temporal_coherence else seed + index
            events[nearest(start)].append((sequential[index], stage_seed, shaping(start)))
    for index, point in enumerate(points):
        stage_seed = seed if temporal_coherence else seed + _INJECTION_SEED_BASE + index
        events[nearest(point)].append((injection[index], stage_seed, shaping(point)))

    return boundaries, events


def _shaped(shader_params: Dict[str, Any], shaping: Dict[str, float]) -> Dict[str, Any]:
    """Apply one stage's zoom and detail adjustments to a copy of the params."""
    if not shaping:
        return shader_params

    shaped = dict(shader_params)
    multiplier = shaping.get("scale_multiplier", 1.0)
    offset = shaping.get("octave_offset", 0.0)

    # The node writes every spelling the generators read, so adjust them all.
    for key in ("scale", "shaderScale"):
        if key in shaped:
            shaped[key] = max(_MIN_SCALE, float(shaped[key]) * multiplier)
    for key in ("octaves", "shaderOctaves"):
        if key in shaped:
            shaped[key] = max(1.0, min(_MAX_OCTAVES, float(shaped[key]) + offset))
    return shaped


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
    travel_mode: str = presets.DEFAULT_TRAVEL_MODE,
    allow_sequence: bool = False,
    stream_seed_offset: int = 0,
) -> torch.Tensor:
    """
    Mix each stage's shader noise into `noise`, in order.

    The shape is read off `noise` rather than carried in from the latent the run
    started with, so a boundary residual cannot be painted at a stale shape.
    """
    for strength, stage_seed, shaping in events:
        if strength <= 0.0:
            continue
        stage_params = _shaped(shader_params, shaping)
        generated = shader_noise.generate(
            tuple(noise.shape), stage_params, shader_type, stage_seed + stream_seed_offset, device,
            dtype=dtype, temporal_coherence=temporal_coherence,
            decorrelate=True, basis=presets.basis_for(travel_mode),
            allow_sequence=allow_sequence,
        )
        generated = noise_math.transform_noise(generated, noise_transform)
        noise = noise_math.mix_noise(noise, generated, blend_mode, strength, normalize_strength)
    return noise


def _paint(
    noise: torch.Tensor,
    events: Sequence[Tuple[float, int, Dict[str, float]]],
    shader_params: Dict[str, Any],
    shader_type: str,
    blend_mode: str,
    noise_transform: str,
    temporal_coherence: bool,
    normalize_strength: bool,
    travel_mode: str,
    shade_non_spatial: bool,
) -> torch.Tensor:
    """
    Mix one boundary's shader stages into every paintable stream of `noise`.

    Each stream is painted where it already lives. An opening boundary's noise comes
    from `prepare_noise` on the CPU, a later one's is recovered from a segment result
    on the sampler's device, and painting only the paintable streams onto a device
    taken from somewhere else would leave the others behind -- a nested AV latent
    then reaches `pack_latents` with its video on one device and its audio on another.
    """
    streams = _streams(noise)
    for index in _paintable(streams, shade_non_spatial):
        stream = streams[index]
        streams[index] = _apply_events(
            stream, events, shader_params, shader_type, blend_mode,
            noise_transform, stream.device, stream.dtype, temporal_coherence, normalize_strength,
            travel_mode, shade_non_spatial, stream_seed_offset=index * _STREAM_SEED_STRIDE,
        )
    return _rebuild(noise, streams)


def starting_noise(
    latent: Dict[str, Any],
    seed: int,
    shader_strength: float,
    shader_params: Dict[str, Any],
    shader_type: str,
    blend_mode: str,
    noise_transform: str,
    use_temporal_coherence: bool = False,
    normalize_strength: bool = True,
    travel_mode: str = presets.DEFAULT_TRAVEL_MODE,
    shade_non_spatial: bool = False,
    stage_progression: str = "uniform",
) -> torch.Tensor:
    """
    The noise a run starts from: ComfyUI's own, with one shader stage painted in.

    Exactly what `run` paints at its opening boundary under the node's default single
    sequential stage, lifted out so a NOISE source can hand the same tensor to any
    custom sampler. There is no schedule here, so the one stage sits at the start of
    the trajectory and `stage_progression` shapes it from there -- which is what `run`
    does with a single stage too, and what keeps a preset meaning the same thing on
    both nodes.
    """
    samples = latent["samples"]

    # No pre-flight refusal for an unpaintable latent, unlike `run`: there is no
    # sampling here to protect, and the generator raises the same error one call down.
    noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
    shaping = schedule.stage_shaping(stage_progression, 0.0)
    return _paint(noise, [(shader_strength, seed, shaping)], shader_params, shader_type,
                  blend_mode, noise_transform, use_temporal_coherence, normalize_strength,
                  travel_mode, shade_non_spatial)


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
    travel_mode: str = presets.DEFAULT_TRAVEL_MODE,
    shade_non_spatial: bool = False,
    stage_progression: str = "uniform",
    custom_sigmas: Optional[torch.Tensor] = None,
    add_noise: bool = True,
    start_at_step: int = 0,
    end_at_step: int = 10000,
    return_with_leftover_noise: bool = False,
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

    # The step window, which is what lets this node be one half of a split run.
    last = min(max(end_at_step, 0), total_steps)
    first = min(max(start_at_step, 0), last)
    if first >= last or len(sigmas) < 2:  # nothing to denoise, or no schedule to do it on
        return {**latent, "samples": samples}
    if last < total_steps and not return_with_leftover_noise:
        # End on a clean latent, the same zeroed last sigma KSampler.sample uses.
        sigmas = torch.cat([sigmas[:last], sigmas.new_zeros(1)])

    boundaries, events = _shader_events(
        total_steps, first, last, sequential_stages, injection_stages, shader_strength,
        sequential_distribution, injection_distribution, seed, use_temporal_coherence,
        stage_progression,
    )
    segment_list = schedule.segments(boundaries, last)

    primary = _streams(samples)[0]

    # Without add_noise the latent arrives carrying its own noise from whatever ran
    # before, so there is none to make and none to paint at the opening boundary: the
    # shader on a zero tensor would add back exactly what was turned off. Later
    # boundaries still paint, since their noise is recovered from the latent.
    opening = events.get(boundaries[0], []) if add_noise else []
    painted = [opening] + [stage for step, stage in events.items() if step != boundaries[0]]

    # Refuse a latent the shaders cannot paint on before any sampling happens, rather
    # than at the first boundary that needs one. At shader_strength 0 there is nothing
    # to paint, so those models still sample through here as a plain KSampler.
    if any(strength > 0.0 for stage in painted for strength, _, _ in stage):
        for stream in (_streams(samples) if shade_non_spatial else [primary]):
            shader_noise.require_spatial_latent(tuple(stream.shape), shade_non_spatial)

    noise = (comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
             if add_noise else comfy.sample.prepare_empty_noise(samples))
    noise = _paint(noise, opening, shader_params, shader_type, blend_mode, noise_transform,
                   use_temporal_coherence, normalize_strength, travel_mode, shade_non_spatial)

    model_sampling = model.get_model_object("model_sampling")
    noise_mask = latent.get("noise_mask", None)
    window_steps = last - first
    preview = None if disable_pbar else latent_preview.prepare_callback(model, window_steps)

    current = samples
    for index, (start, end) in enumerate(segment_list):
        captured: Dict[str, Any] = {}
        is_last = index == len(segment_list) - 1
        result = comfy.sample.sample(
            model, noise, end - start, cfg, sampler_name, scheduler, positive, negative, current,
            denoise=1.0,
            sigmas=sigmas[start:end + 1],
            noise_mask=noise_mask,
            callback=_segment_callback(preview, start - first, window_steps, captured),
            disable_pbar=disable_pbar,
            seed=seed + start,
        )
        if is_last:
            return {**latent, "samples": result}

        if "x0" not in captured:  # no steps ran; carry on from where we are
            current, noise = result, noise
            continue

        current, residual = _split_noise(
            result, captured["x0"], sigmas[end], model_sampling, model
        )
        noise = _paint(residual, events.get(end, []), shader_params, shader_type, blend_mode,
                       noise_transform, use_temporal_coherence, normalize_strength, travel_mode,
                       shade_non_spatial)

    return {**latent, "samples": current}
