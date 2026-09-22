"""
Reconstruct the exact shader field a run injected.

Runs the Direct node itself with a stub sampler and no weights, and captures what
core.shader_noise.generate returned. Only the settings in common.shader_inputs and
the latent's shape decide the field, so this matches what the server generated;
analyze.py checks the CPU render against CUDA before relying on it.
"""
import torch

import common

common.load_pack()

import comfy.sample  # noqa: E402
from helpers import FakeModel  # noqa: E402
from snk.core import shader_noise  # noqa: E402
from snk.direct_shader_ksampler import DirectShaderNoiseKSampler  # noqa: E402

KIND = {"sd15": "eps", "h3": "av", "krea2": "flow"}


def empty_latent(model, width, height, length):
    """The latent the model's workflow starts from, at the size a run used."""
    if model == "sd15":
        return torch.zeros(1, 4, height // 8, width // 8)
    if model == "krea2":
        # Krea 2 samples in Wan 2.1's 16-channel latent space.
        return torch.zeros(1, 16, height // 8, width // 8)
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent
    latent = _empty_av_latent(width=width, height=height, length=length)
    latent = latent[0] if isinstance(latent, tuple) else latent
    return latent["samples"] if isinstance(latent, dict) else latent


def capture(model, latent, seed, strength, travel, phase, scale, shader=common.DEFAULT_SHADER):
    """The shader fields generated for one run, in the order the pipeline made them."""
    fields = []
    depth = 0
    original_generate, original_sample = shader_noise.generate, comfy.sample.sample

    def generate(*args, **kwargs):
        # A travel mode that remixes (drift, jump) calls generate() again for its
        # one-channel basis draws. Only the outermost call is the field the sampler
        # received, so the inner ones are not recorded.
        nonlocal depth
        depth += 1
        try:
            out = original_generate(*args, **kwargs)
        finally:
            depth -= 1
        if depth == 0:
            fields.append(out.detach().clone())
        return out

    def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
        return latent_image

    shader_noise.generate, comfy.sample.sample = generate, sample
    try:
        DirectShaderNoiseKSampler().sample(
            model=FakeModel(KIND[model]), positive=[], negative=[], latent_image={"samples": latent},
            steps=20, cfg=1.0, sampler_name="euler", scheduler="simple",
            **common.shader_inputs(seed, strength, travel, phase, scale, shader))
    finally:
        shader_noise.generate, comfy.sample.sample = original_generate, original_sample
    return fields
