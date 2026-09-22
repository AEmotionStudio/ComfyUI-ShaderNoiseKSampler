"""
Shader noise as a NOISE object, for ComfyUI's custom sampling nodes.

`SamplerCustomAdvanced` takes its starting noise as an object rather than a seed,
which is how a custom guider, sampler and sigma schedule come together. Handing it
this instead of `RandomNoise` puts shader noise in front of guiders the pack has no
node of its own for -- `BasicGuider`, `DualCFGGuider`, whatever another pack
provides -- without a second sampler node to maintain alongside the Direct one.
`AddNoise` takes one too, for shader noise at a chosen sigma with no sampling.

What it is, exactly: the pack's default configuration and nothing more, one shader
stage painted into the noise a run starts from. The sampler's stages re-enter the
shader at segment boundaries partway through the run, which a NOISE object cannot
do -- it is asked for noise once, before any sampling happens. Those still need the
Direct node.
"""
import copy

from .core import presets as preset_table
from .direct_shader_ksampler import DirectShaderNoiseKSampler
from .pipelines import standard as standard_pipeline
from .shader_params_reader import build_shader_params, get_shader_params

# The Direct node's inputs that describe the noise rather than a sampling run.
# Taken from its spec rather than restated, so the tooltips stay one copy.
NOISE_INPUTS = (
    "seed", "shader_strength", "blend_mode", "noise_transform", "use_temporal_coherence",
    "shader_type", "shape_type", "color_scheme", "noise_scale", "octaves", "warp_strength",
    "shape_mask_strength", "phase_shift", "color_intensity",
    "fast_high_channel_noise", "preset", "stage_progression", "shade_non_spatial",
    "travel_mode", "normalize_strength",
)

# stage_progression means something different with one stage, so it does not inherit
# the sampler's wording. Presets set it, so it has to be visible rather than applied
# behind the widgets.
_PROGRESSION_TOOLTIP = (
    "Which end of the trajectory this noise is drawn for. Across a multi-stage run the "
    "sampler ramps zoom and detail with position; there is only one stage here, and it "
    "is the start, so coarse_to_fine draws it zoomed in on large features with fewer "
    "octaves and fine_to_coarse zoomed out on small ones with more. The adjustment spans "
    "0.5x to 2x your noise_scale and plus or minus one octave, so uniform is unchanged. "
    "The roam and video presets set this, which is why they look different from their "
    "strength alone."
)


def _retarget(name, entry):
    """Point a tooltip inherited from the sampler at this node instead."""
    if len(entry) > 1 and isinstance(entry[1], dict) and "tooltip" in entry[1]:
        if name == "stage_progression":
            entry[1]["tooltip"] = _PROGRESSION_TOOLTIP
        else:
            entry[1]["tooltip"] = entry[1]["tooltip"].replace(" Standard sampling only.", "")
    return entry


class ShaderNoise:
    """
    The NOISE interface: a seed, and a way to make the tensor.

    ComfyUI asks for the noise once, handing over the latent it will be sampled
    against, so the shape is the caller's and nothing here holds a tensor.
    """

    def __init__(self, seed, **painting):
        self.seed = seed
        self.painting = painting

    def generate_noise(self, input_latent):
        return standard_pipeline.starting_noise(input_latent, self.seed, **self.painting)


class ShaderNoiseSource:
    """Shader noise for `SamplerCustomAdvanced`, in place of `RandomNoise`."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(DirectShaderNoiseKSampler.INPUT_TYPES())
        return {
            section: {name: _retarget(name, entry)
                      for name, entry in spec[section].items() if name in NOISE_INPUTS}
            for section in ("required", "optional")
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    OUTPUT_TOOLTIPS = ("The noise a run starts from, for SamplerCustomAdvanced's noise "
                       "input or AddNoise's.",)
    FUNCTION = "get_noise"
    CATEGORY = "model/sampling/noise"
    DESCRIPTION = ("Shader noise as a NOISE object, so it can start a run driven by a "
                   "custom guider, sampler and sigma schedule. One shader stage, painted "
                   "into the starting noise; the Direct node's stages need the Direct node.")

    def get_noise(self, seed, shader_strength, blend_mode, noise_transform,
                  use_temporal_coherence, shader_type, shape_type, color_scheme,
                  noise_scale, octaves, warp_strength, shape_mask_strength, phase_shift,
                  color_intensity, fast_high_channel_noise=False, preset="custom",
                  stage_progression="uniform", shade_non_spatial=False, travel_mode="walk",
                  normalize_strength=True):
        chosen = preset_table.apply_preset(preset, dict(
            shader_type=shader_type, shader_strength=shader_strength, blend_mode=blend_mode,
            travel_mode=travel_mode, shape_type=shape_type,
            normalize_strength=normalize_strength, stage_progression=stage_progression,
        ))

        shader_params = build_shader_params(
            get_shader_params(), seed, chosen["shader_type"], chosen["shape_type"],
            color_scheme, noise_scale, octaves, warp_strength, shape_mask_strength,
            phase_shift, color_intensity, use_temporal_coherence, fast_high_channel_noise,
        )

        return (ShaderNoise(
            seed,
            shader_strength=chosen["shader_strength"],
            shader_params=shader_params,
            shader_type=chosen["shader_type"],
            blend_mode=chosen["blend_mode"],
            noise_transform=noise_transform,
            use_temporal_coherence=use_temporal_coherence,
            normalize_strength=chosen["normalize_strength"],
            travel_mode=chosen["travel_mode"],
            shade_non_spatial=shade_non_spatial,
            stage_progression=chosen["stage_progression"],
        ),)
