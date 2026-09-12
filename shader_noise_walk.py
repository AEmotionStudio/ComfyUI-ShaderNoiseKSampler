"""
Walk one shader parameter across a range in a single node.

The point of this pack is exploring the neighbourhood around a seed rather than
jumping between seeds, and a strength ramp shows that clearly: 0.00 to 0.25 in
five steps is a smooth graded walk through one scene. Doing it by hand meant five
separate queue runs with one widget nudged between each. This does the ramp in
one run, with the model resident throughout -- which is what makes it cheap, as
loading a large checkpoint costs far more than the sampling does.

The output is a batched LATENT, so it feeds `Advanced Image Comparer` and
`Video Comparer` directly.
"""
import copy

import torch

import comfy.utils

from .direct_shader_ksampler import DirectShaderNoiseKSampler

# Parameters worth ramping, and whether the value is an integer.
WALKABLE = {
    "shader_strength": False,
    "phase_shift": False,
    "noise_scale": False,
    "warp_strength": False,
    "octaves": False,
    "shape_mask_strength": False,
    "color_intensity": False,
    "seed": True,
}

# A walk is N full sampling runs, so the ceiling is about patience, not memory.
MAX_STEPS = 16


def _ramp(start: float, end: float, count: int):
    """`count` values from start to end inclusive; a single step sits at start."""
    if count < 2:
        return [start]
    span = end - start
    return [start + span * i / (count - 1) for i in range(count)]


def stack_latents(latents):
    """
    Join per-run latents into one batch.

    Multi-stream latents (MiniMax H3, LTXAV) concatenate stream by stream via
    ComfyUI's own helper. `batch_index` is dropped: it selects which noise slot a
    single run uses, and means nothing once several runs are side by side.
    """
    samples = [latent["samples"] for latent in latents]
    if getattr(samples[0], "is_nested", False):
        from comfy.nested_tensor import cat_nested
        joined = cat_nested(samples, dim=0)
    else:
        joined = torch.cat(samples, dim=0)

    carried = {k: v for k, v in latents[0].items() if k not in ("samples", "batch_index")}
    return {**carried, "samples": joined}


class ShaderNoiseWalk(DirectShaderNoiseKSampler):
    """Sample the same seed repeatedly while one shader parameter ramps."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(DirectShaderNoiseKSampler.INPUT_TYPES())
        spec["required"]["walk_parameter"] = (
            list(WALKABLE),
            {"default": "shader_strength",
             "tooltip": "Which parameter ramps across the batch. The others keep their "
                        "widget values. shader_strength walks away from the seed and is "
                        "the one to start with; phase_shift holds the same distance and "
                        "turns the pattern instead, so the subject persists while its "
                        "details rearrange. Walking seed is ordinary seed-hopping, for "
                        "comparison against the rest."},
        )
        spec["required"]["walk_start"] = (
            "FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffff, "step": 0.01,
                      "tooltip": "First value of the ramp. For shader_strength, 0.0 gives "
                                 "a clean reference frame to compare the others against."},
        )
        spec["required"]["walk_end"] = (
            "FLOAT", {"default": 0.25, "min": 0.0, "max": 0xffffffff, "step": 0.01,
                      "tooltip": "Last value of the ramp, inclusive. Past roughly 0.3 the "
                                 "shader's own pattern starts surviving into the picture "
                                 "on video models -- see shader_strength."},
        )
        spec["required"]["walk_steps"] = (
            "INT", {"default": 5, "min": 1, "max": MAX_STEPS, "step": 1,
                    "tooltip": f"How many samples to take along the ramp, endpoints "
                               f"included. This is that many full sampling runs, so the "
                               f"time is roughly walk_steps x a single run -- the model "
                               f"is loaded once and stays resident. Maximum {MAX_STEPS}."},
        )
        return spec

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    OUTPUT_TOOLTIPS = ("One latent per point on the ramp, batched in order, ready for a "
                       "comparer or a single VAE decode.",)
    FUNCTION = "walk"
    CATEGORY = "sampling"
    DESCRIPTION = ("Sample one seed repeatedly while a shader parameter ramps, and return "
                   "the results as a batch. Turns the pack's core idea -- explore around a "
                   "seed rather than hop between seeds -- into one node instead of a run "
                   "per value.")
    DEPRECATED = False

    def walk(self, walk_parameter, walk_start, walk_end, walk_steps, **kwargs):
        if walk_parameter not in WALKABLE:
            raise ValueError(
                f"cannot walk {walk_parameter!r}; choose one of {', '.join(WALKABLE)}"
            )

        values = _ramp(float(walk_start), float(walk_end), int(walk_steps))
        if WALKABLE[walk_parameter]:
            values = [int(round(v)) for v in values]

        # A preset that pins shader_strength would flatten a strength ramp into
        # five identical frames, so the walked parameter is held back from it.
        self._preset_exclude = (walk_parameter,)

        progress = comfy.utils.ProgressBar(len(values))
        latents = []
        for value in values:
            # sample() returns the node's UI envelope; the latent is in "result".
            result = self.walk_sample({**kwargs, walk_parameter: value})
            latents.append(result)
            progress.update(1)

        return (stack_latents(latents),)

    def walk_sample(self, call_kwargs):
        """One point on the ramp, through the ordinary Direct sampling path."""
        output = DirectShaderNoiseKSampler.sample(self, **call_kwargs)
        return output["result"][0] if isinstance(output, dict) else output[0]
