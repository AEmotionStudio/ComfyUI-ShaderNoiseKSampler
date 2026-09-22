"""
Golden cases for the standard sampling pipeline.

`capture_golden.py` records what the Direct node sends to comfy.sample.sample
for each case. `test_golden.py` replays them and requires identical calls and
outputs: a fast bit-exact net against changing the sampler by accident.

A failure here means the sampler's output changed, not that backwards
compatibility broke. These fixtures were re-captured when the generators
stopped collapsing the channel axis; the pre-2.0 output they used to pin is at
the `pre-collapse-fix` tag.
"""
import os

import torch

from helpers import FakeModel, recorded_sampling, snapshot

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

NODE_DEFAULTS = dict(
    seed=8888, steps=10, cfg=7.0, sampler_name="euler_ancestral", scheduler="beta", denoise=1.0,
    sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
    noise_transform="none", use_temporal_coherence=False, shader_type="domain_warp", shape_type="none",
    color_scheme="none", noise_scale=1.0, octaves=1.0, warp_strength=0.5, shape_mask_strength=1.0,
    phase_shift=0.5, color_intensity=0.8,
    # Pinned rather than left to the node's defaults, so flipping a default shows
    # up as a deliberate edit here instead of silently rewriting what a fixture
    # means.
    sampling_mode="standard", preset="custom", travel_mode="walk", normalize_strength=True,
    stage_progression="uniform", shade_non_spatial=False, sequential_distribution="linear_decrease",
    injection_distribution="linear_decrease", fast_high_channel_noise=False,
    add_noise=True, start_at_step=0, end_at_step=10000, return_with_leftover_noise=False,
)

# Keys that describe the test setup rather than node inputs.
SETUP_KEYS = ("kind", "video", "nested", "latent", "batch", "custom_sigmas")

CASES = {
    "image_default": {},
    "image_multistage_overlay": dict(sequential_stages=2, injection_stages=3, blend_mode="overlay"),
    "image_img2img_denoise": dict(denoise=0.6, latent="random"),
    "image_custom_sigmas": dict(custom_sigmas=True),
    "image_injection_only": dict(sequential_stages=0, injection_stages=2, blend_mode="add"),
    "image_zero_strength": dict(shader_strength=0.0),
    "image_styled": dict(noise_transform="absolute", blend_mode="soft_light", shape_type="radial",
                         color_scheme="viridis", shader_type="tensor_field", octaves=3.5),
    "image_batch": dict(batch=2, shader_type="curl_noise", noise_transform="sin", blend_mode="difference"),
    "video_temporal": dict(kind="flow", video=True, sequential_stages=2, use_temporal_coherence=True,
                           shader_type="tensor_field"),
    "video_curl": dict(kind="flow", video=True, shader_type="curl_noise", blend_mode="screen"),
    "video_nested": dict(kind="flow", video=True, nested=True),
    # temporal_coherent's four-corner 3D simplex moves to shaders/simplex.py so
    # the generators added later can share it; this pins its output across the move.
    "video_temporal_coherent": dict(kind="flow", video=True, shader_type="temporal_coherent",
                                    use_temporal_coherence=True, sequential_stages=2),
    # One image and one video case per generator added after spectral, the video
    # one with temporal coherence, which is where each differs most from a redraw.
    "image_gaussian": dict(shader_type="gaussian"),
    "video_gaussian": dict(kind="flow", video=True, shader_type="gaussian",
                           use_temporal_coherence=True, sequential_stages=2),
    # The spectral generator builds its field from a frequency band rather than
    # per pixel, so it shares none of the others' code below core.shader_noise.
    # One image case and one video case, the video one with temporal coherence
    # because that is where it differs most -- holding the seed and advancing
    # time turns each mode at its own rate instead of redrawing the field.
    "image_spectral": dict(shader_type="spectral", noise_scale=1.5, octaves=2.0),
    "video_spectral": dict(kind="flow", video=True, shader_type="spectral",
                           use_temporal_coherence=True, sequential_stages=2),
    "image_fractal": dict(shader_type="fractal"),
    "video_fractal": dict(kind="flow", video=True, shader_type="fractal",
                       use_temporal_coherence=True, sequential_stages=2),
    "image_perlin": dict(shader_type="perlin"),
    "video_perlin": dict(kind="flow", video=True, shader_type="perlin",
                      use_temporal_coherence=True, sequential_stages=2),
    "image_heterogeneous_fbm": dict(shader_type="heterogeneous_fbm"),
    "video_heterogeneous_fbm": dict(kind="flow", video=True, shader_type="heterogeneous_fbm",
                                 use_temporal_coherence=True, sequential_stages=2),
    "image_interference": dict(shader_type="interference"),
    "video_interference": dict(kind="flow", video=True, shader_type="interference",
                            use_temporal_coherence=True, sequential_stages=2),
    "image_projection_3d": dict(shader_type="projection_3d"),
    "video_projection_3d": dict(kind="flow", video=True, shader_type="projection_3d",
                             use_temporal_coherence=True, sequential_stages=2),
    "image_cellular": dict(shader_type="cellular"),
    "video_cellular": dict(kind="flow", video=True, shader_type="cellular",
                        use_temporal_coherence=True, sequential_stages=2),
    "image_waves": dict(shader_type="waves"),
    "video_waves": dict(kind="flow", video=True, shader_type="waves",
                     use_temporal_coherence=True, sequential_stages=2),
    # The two halves of a split run, the shape a latent upscaler needs. The first
    # stops early and keeps its noise; the second picks the trajectory up without
    # making any, so its shader can only enter at the injection boundary.
    "image_split_first_half": dict(end_at_step=5, return_with_leftover_noise=True),
    "image_split_second_half": dict(start_at_step=5, add_noise=False, injection_stages=1,
                                    latent="random"),
}

CUSTOM_SIGMAS = torch.tensor([14.6, 9.0, 6.0, 4.0, 2.7, 1.8, 1.1, 0.6, 0.3, 0.1, 0.0])


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f"{name}.pt")


def make_latent(setup):
    shape = (setup.get("batch", 1), 16, 5, 8, 8) if setup.get("video") else (setup.get("batch", 1), 4, 16, 16)
    if setup.get("latent") == "random":
        samples = torch.randn(shape, generator=torch.Generator().manual_seed(1234))
    else:
        samples = torch.zeros(shape)
    if setup.get("nested"):
        from comfy.nested_tensor import NestedTensor
        samples = NestedTensor([samples, torch.zeros(1, 8, 12)])
    return {"samples": samples}


def run_case(name):
    """Run one case through the Direct node; return the recorded calls and output."""
    from snk.direct_shader_ksampler import DirectShaderNoiseKSampler

    spec = {**NODE_DEFAULTS, **CASES[name]}
    setup = {key: spec.pop(key) for key in SETUP_KEYS if key in spec}
    if setup.get("custom_sigmas"):
        spec["custom_sigmas"] = CUSTOM_SIGMAS.clone()

    node = DirectShaderNoiseKSampler()
    with recorded_sampling() as calls:
        result = node.sample(model=FakeModel(setup.get("kind", "eps")), positive=[], negative=[],
                             latent_image=make_latent(setup), **spec)
    output = result["result"][0]["samples"] if isinstance(result, dict) else result[0]["samples"]
    return {"calls": calls, "output": snapshot(output)}
