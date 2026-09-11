"""
Golden cases for the legacy sampling pipeline.

`capture_golden.py` records what the pre-refactor node sends to
comfy.sample.sample for each case. `test_legacy_golden.py` replays the cases
in legacy mode and requires identical calls and outputs, so legacy stays
bit-for-bit reproducible for workflows saved before 2.0.
"""
import inspect
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
    """Run one case through the Direct node in legacy mode; return the recorded calls and output."""
    from snk.direct_shader_ksampler import DirectShaderNoiseKSampler

    spec = {**NODE_DEFAULTS, **CASES[name]}
    setup = {key: spec.pop(key) for key in SETUP_KEYS if key in spec}
    if setup.get("custom_sigmas"):
        spec["custom_sigmas"] = CUSTOM_SIGMAS.clone()

    node = DirectShaderNoiseKSampler()
    if "sampling_mode" in inspect.signature(node.sample).parameters:
        spec["sampling_mode"] = "legacy"

    with recorded_sampling() as calls:
        result = node.sample(model=FakeModel(setup.get("kind", "eps")), positive=[], negative=[],
                             latent_image=make_latent(setup), **spec)
    output = result["result"][0]["samples"] if isinstance(result, dict) else result[0]["samples"]
    return {"calls": calls, "output": snapshot(output)}
