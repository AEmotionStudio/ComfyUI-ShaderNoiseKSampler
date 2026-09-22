"""
Sampling schedule for the standard pipeline.

One sigma schedule is built for the whole run and split into segments at the
points where new shader noise enters. Legacy instead ran every stage as its own
complete sampling pass: each stage restarted from maximum noise, so on flow
models (noise_scaling is `sigma * noise + (1 - sigma) * latent`, and sigma is
1.0 at the start) the previous stage's result was multiplied by zero. Two
sequential stages were therefore a half-length generation, not two halves of
one.

The boundaries here are indices into that single schedule, so stages become
segments of one trajectory.
"""
import math
from typing import List, Optional, Sequence, Tuple

import torch

import comfy.samplers

# A segment shorter than this cannot denoise anything meaningful. Legacy's
# injection points produced ranges like (19, 20) for 20 steps with 3 stages,
# making the final image a single step from full noise.
MIN_SEGMENT_STEPS = 2

SUPPORTED_DISTRIBUTIONS = (
    "uniform", "linear_decrease", "linear_increase", "gaussian",
    "first_stronger", "last_stronger",
)


def stage_strengths(base_strength: float, num_stages: int, distribution: str) -> List[float]:
    """Shader strength per stage, following the chosen distribution curve."""
    if num_stages <= 0:
        return []
    if num_stages == 1:
        return [base_strength]

    span = num_stages - 1
    min_factor = 0.25  # keep every stage audible rather than fading one to nothing

    if distribution == "linear_decrease":
        factors = [max(1.0 - i / span, min_factor) for i in range(num_stages)]
    elif distribution == "linear_increase":
        factors = [max(i / span, min_factor) for i in range(num_stages)]
    elif distribution == "gaussian":
        mid, std_dev = span / 2.0, num_stages / 3.0
        factors = [math.exp(-((i - mid) ** 2) / (2 * std_dev ** 2)) for i in range(num_stages)]
    elif distribution == "first_stronger":
        factors = [math.exp(-i) for i in range(num_stages)]
    elif distribution == "last_stronger":
        factors = [math.exp(i - span) for i in range(num_stages)]
    else:  # "uniform" and anything unrecognised
        factors = [1.0] * num_stages

    return [base_strength * f for f in factors]


def sequential_starts(total_steps: int, num_stages: int) -> List[int]:
    """First step of each sequential stage, spread evenly across the schedule."""
    if num_stages <= 1:
        return [0]
    return [int(i * total_steps / num_stages) for i in range(num_stages)]


def injection_points(total_steps: int, num_stages: int) -> List[int]:
    """
    Steps at which injection stages add shader noise.

    These are interior points: injecting at step 0 is what a sequential stage
    already does, and injecting at the last step leaves nothing to sample.
    """
    if num_stages <= 0:
        return []
    return [int(round((i + 1) * total_steps / (num_stages + 1))) for i in range(num_stages)]


def merge_boundaries(
    end_step: int,
    starts: Sequence[int],
    points: Sequence[int],
    min_segment: int = MIN_SEGMENT_STEPS,
    first: int = 0,
) -> List[int]:
    """
    Combine stage starts and injection points into ascending segment boundaries.

    Always begins at `first`, drops duplicates, and discards any boundary that
    would leave a segment shorter than `min_segment` steps.

    `first` and `end_step` bound the node's step window, which is the whole
    schedule unless `start_at_step` or `end_at_step` narrowed it. Both must be
    the window's own edges: measuring the tail against the schedule length
    instead would let a boundary survive one step short of the window's end.
    """
    boundaries: List[int] = []
    for boundary in sorted({first, *starts, *points}):
        if boundary < first or boundary >= end_step:
            continue
        if boundaries and boundary - boundaries[-1] < min_segment:
            continue
        boundaries.append(boundary)

    if not boundaries:
        return [first]
    # The tail is a segment too: drop trailing boundaries that would truncate it.
    while len(boundaries) > 1 and end_step - boundaries[-1] < min_segment:
        boundaries.pop()
    return boundaries


def segments(boundaries: Sequence[int], total_steps: int) -> List[Tuple[int, int]]:
    """Turn boundaries into (start, end) step ranges covering the whole schedule."""
    if not boundaries:
        return [(0, total_steps)]
    ends = list(boundaries[1:]) + [total_steps]
    return [(start, end) for start, end in zip(boundaries, ends) if end > start]


def build_sigmas(
    model,
    steps: int,
    sampler_name: str,
    scheduler: str,
    denoise: float = 1.0,
    custom_sigmas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build the sigma schedule for the whole run.

    Custom sigmas are used as given (they define their own trajectory). Legacy
    wrapped the model to override `model_sampling`, but ComfyUI reads the
    schedule through `model.get_model_object("model_sampling")`, which resolves
    past such a wrapper, so the supplied values never reached the sampler -- only
    their count did.

    Otherwise KSampler builds the schedule, which applies `denoise` (legacy
    hard-coded 1.0 per stage) and the discard-penultimate-sigma samplers.
    """
    if custom_sigmas is not None and torch.is_tensor(custom_sigmas) and custom_sigmas.numel() > 1:
        sigmas = custom_sigmas.detach().clone().float()
        if sigmas[0] < sigmas[-1]:  # accept ascending input, sample descending
            sigmas = sigmas.flip(0)
        return sigmas

    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=getattr(model, "model_options", {}),
    )
    return sampler.sigmas


# How far the zoom and detail controls move across the trajectory, as a
# multiplier on noise_scale and an offset on octaves at the far end. Modest on
# purpose: this shapes the walk, it is not meant to be a second strength knob.
PROGRESSIONS = ("uniform", "coarse_to_fine", "fine_to_coarse")
_SCALE_SPAN = (0.5, 2.0)
_OCTAVE_SPAN = (-1.0, 1.0)


def stage_shaping(progression: str, progress: float) -> dict:
    """
    Per-stage overrides for a boundary `progress` of the way through the schedule.

    The diffusion trajectory is not uniform -- early steps settle composition and
    late steps settle detail -- but every stage has always drawn the same shader
    at the same zoom. `coarse_to_fine` starts zoomed in on large features and ends
    on small ones, which lines the noise up with what each part of the trajectory
    is actually deciding. The README already calls noise_scale the zoom control
    and octaves the detail slider; this just ties them to position.

    Returns multiplicative/additive adjustments, not absolute values, so the
    node's own widget settings stay the centre of the range.
    """
    if progression not in PROGRESSIONS or progression == "uniform":
        return {}

    position = min(max(progress, 0.0), 1.0)
    if progression == "fine_to_coarse":
        position = 1.0 - position

    scale_low, scale_high = _SCALE_SPAN
    octave_low, octave_high = _OCTAVE_SPAN
    return {
        "scale_multiplier": scale_low + (scale_high - scale_low) * position,
        "octave_offset": octave_low + (octave_high - octave_low) * position,
    }
