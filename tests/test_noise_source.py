"""
Shader noise as a NOISE object for ComfyUI's custom sampling nodes.

The claim this file exists to pin: the source hands out exactly the noise the
sampler starts from under its default single stage. If that drifts, a workflow
built on SamplerCustomAdvanced stops matching the same settings on the Direct node,
and the two nodes quietly disagree about what a seed means.
"""
import pytest
import torch

import comfy.sample
from helpers import FakeModel, recorded_sampling
from snk.core.shader_noise import UnsupportedLatentError
from snk.direct_shader_ksampler import DirectShaderNoiseKSampler
from snk.shader_noise_source import ShaderNoise, ShaderNoiseSource

# One set of shader settings, spread into both nodes, so a difference in the noise
# cannot come from a difference in what they were asked for.
SHADER = dict(
    seed=8888, shader_strength=0.4, blend_mode="multiply", noise_transform="none",
    use_temporal_coherence=False, shader_type="domain_warp", shape_type="none",
    color_scheme="none", noise_scale=1.0, octaves=1.0, warp_strength=0.5,
    shape_mask_strength=1.0, phase_shift=0.5, color_intensity=0.8,
)


def source_noise(latent, **overrides):
    """The tensor SamplerCustomAdvanced would get from this node."""
    with recorded_sampling():
        noise = ShaderNoiseSource().get_noise(**{**SHADER, **overrides})[0]
    return noise.generate_noise(latent)


def sampler_noise(latent, model=None, **overrides):
    """The tensor the Direct node starts its run from, at one sequential stage."""
    with recorded_sampling() as calls:
        DirectShaderNoiseKSampler().sample(
            model=model or FakeModel("eps"), positive=[], negative=[],
            latent_image=latent, steps=10, cfg=7.0, sampler_name="euler",
            scheduler="normal", denoise=1.0, sequential_stages=1, injection_stages=0,
            **{**SHADER, **overrides},
        )
    return calls[0]["noise"]


def streams(noise):
    if isinstance(noise, list):  # helpers.snapshot unbinds a nested recording
        return noise
    return list(noise.unbind()) if getattr(noise, "is_nested", False) else [noise]


@pytest.mark.parametrize("kind,shape", [
    ("eps", (1, 4, 16, 16)),
    ("flow", (1, 16, 5, 8, 8)),
])
def test_it_hands_out_the_samplers_own_starting_noise(kind, shape):
    latent = {"samples": torch.zeros(shape)}
    assert torch.equal(source_noise(latent), sampler_noise(latent, model=FakeModel(kind)))


def test_an_av_latent_matches_stream_for_stream():
    """MiniMax H3 arrives as video + audio; only the video stream is painted."""
    model = FakeModel("av")
    latent = {"samples": model.empty_latent()}

    ours = streams(source_noise(latent))
    theirs = streams(sampler_noise(latent, model=model))
    stock = comfy.sample.prepare_noise(latent["samples"], 8888, None).unbind()

    assert len(ours) == len(theirs) == 2
    for got, want in zip(ours, theirs):
        assert torch.equal(got, want)
    assert torch.equal(ours[1], stock[1]), "the audio stream must be left alone"
    assert not torch.allclose(ours[0], stock[0]), "the video stream must be painted"


def test_a_preset_reaches_it_the_same_way():
    latent = {"samples": torch.zeros(1, 4, 16, 16)}
    assert torch.equal(source_noise(latent, preset="roam"),
                       sampler_noise(latent, preset="roam"))


def test_zero_strength_is_the_stock_gaussian():
    samples = torch.zeros(1, 4, 16, 16)
    noise = source_noise({"samples": samples}, shader_strength=0.0)
    assert torch.equal(noise, comfy.sample.prepare_noise(samples, 8888, None))


def test_batch_index_is_honoured():
    """It selects which noise slot a latent gets, and the sampler respects it."""
    samples = torch.zeros(2, 4, 16, 16)
    latent = {"samples": samples, "batch_index": [1, 0]}
    assert torch.equal(source_noise(latent), sampler_noise(latent))


def test_the_shape_comes_from_the_latent_it_is_given():
    """
    ComfyUI asks one NOISE object for noise whenever it needs some, so nothing may
    be shaped or cached when the node runs.
    """
    with recorded_sampling():
        noise = ShaderNoiseSource().get_noise(**SHADER)[0]

    small = noise.generate_noise({"samples": torch.zeros(1, 4, 16, 16)})
    large = noise.generate_noise({"samples": torch.zeros(1, 4, 32, 24)})

    assert small.shape == (1, 4, 16, 16)
    assert large.shape == (1, 4, 32, 24)


def test_it_matches_the_noise_interface_comfyui_calls():
    with recorded_sampling():
        noise = ShaderNoiseSource().get_noise(**SHADER)[0]

    assert isinstance(noise, ShaderNoise)
    assert noise.seed == 8888, "SamplerCustomAdvanced passes this through as the sampler seed"
    assert callable(noise.generate_noise)


def test_a_latent_the_shader_cannot_paint_is_refused():
    with pytest.raises(UnsupportedLatentError, match=r"3D \(1, 64, 1024\)"):
        source_noise({"samples": torch.ones(1, 64, 1024)})


def test_it_offers_the_noise_inputs_and_none_of_the_sampling_ones():
    spec = ShaderNoiseSource.INPUT_TYPES()
    names = {*spec["required"], *spec["optional"]}

    assert "shader_strength" in names and "travel_mode" in names
    # stage_progression stays: presets set it, and it shapes the single stage.
    assert "stage_progression" in names
    for sampling_only in ("model", "positive", "negative", "latent_image", "steps", "cfg",
                          "sampler_name", "scheduler", "denoise", "custom_sigmas",
                          "sequential_stages", "injection_stages", "sampling_mode",
                          "start_at_step", "end_at_step", "add_noise",
                          "return_with_leftover_noise"):
        assert sampling_only not in names, sampling_only


def test_the_tooltips_are_the_samplers_own():
    """
    One copy, minus a caveat about a sampling mode this node has none of.
    stage_progression is the exception: with a single stage it describes where in the
    trajectory the noise is drawn, not a ramp across stages, so it says so itself.
    """
    direct = DirectShaderNoiseKSampler.INPUT_TYPES()
    spec = ShaderNoiseSource.INPUT_TYPES()

    for section in ("required", "optional"):
        for name, entry in spec[section].items():
            tooltip = entry[1]["tooltip"]
            assert "Standard sampling only" not in tooltip, name
            if name == "stage_progression":
                assert "only one stage here" in tooltip
                continue
            expected = direct[section][name][1]["tooltip"].replace(" Standard sampling only.", "")
            assert tooltip == expected, name


@pytest.mark.parametrize("progression,zoomed_in", [("coarse_to_fine", True),
                                                   ("fine_to_coarse", False)])
def test_stage_progression_shapes_the_single_stage(progression, zoomed_in):
    """
    The roam and video presets set it, so ignoring it here would have made the same
    preset mean two different things on the two nodes.
    """
    latent = {"samples": torch.zeros(1, 4, 16, 16)}
    uniform = source_noise(latent, stage_progression="uniform")
    shaped = source_noise(latent, stage_progression=progression)

    assert not torch.equal(uniform, shaped)
    assert torch.equal(shaped, sampler_noise(latent, stage_progression=progression))
