"""
The node routes to the right pipeline, and standard mode samples one trajectory.

Measured through the node with a recording sampler:

    standard, 1 stage      1 call  x 20 steps, from sigma 14.61
    standard, 2 stages     2 calls x 10 steps, from 14.61 then 1.48
    standard, denoise 0.6  1 call  x 20 steps, from 2.23
    standard, 3 injections 4 calls x  5 steps, 14.61 / 3.87 / 1.48 / 0.60
    legacy,   2 stages     2 calls, each building its own full schedule

The second sigma in the two-stage standard run is the point: legacy restarted
every stage at 14.61, which on flow models discards the previous stage entirely.
"""
from unittest import mock

import pytest
import torch

import comfy.sample
from helpers import REPO_DIR, FakeModel
from snk.direct_shader_ksampler import DirectShaderNoiseKSampler


@pytest.fixture
def sampler_calls():
    calls = []

    def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, sigmas=None, callback=None,
                    disable_pbar=False, seed=None):
        calls.append({
            "steps": steps,
            "sigmas": None if sigmas is None else sigmas.detach().clone(),
            "denoise": denoise,
        })
        result = latent_image + 0.1 * noise
        if callback is not None:
            callback(max(steps - 1, 0), result * 0.5, result, steps)
        return result

    with mock.patch.object(comfy.sample, "sample", fake_sample):
        yield calls


def run_node(**overrides):
    kwargs = dict(
        model=FakeModel("eps"), seed=8888, steps=20, cfg=7.0, sampler_name="euler",
        scheduler="normal", positive=[], negative=[],
        latent_image={"samples": torch.zeros(1, 4, 16, 16)}, denoise=1.0,
        sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
        noise_transform="none", use_temporal_coherence=False, shader_type="domain_warp",
        shape_type="none", color_scheme="none", noise_scale=1.0, octaves=1.0,
        warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.5, color_intensity=0.8,
    )
    kwargs.update(overrides)
    return DirectShaderNoiseKSampler().sample(**kwargs)


@pytest.mark.parametrize("mode", ["standard", "legacy"])
def test_both_modes_return_a_latent(sampler_calls, mode):
    result = run_node(sampling_mode=mode)
    assert "result" in result and isinstance(result["result"], tuple)
    assert result["result"][0]["samples"].shape == (1, 4, 16, 16)


def test_standard_mode_samples_one_schedule(sampler_calls):
    run_node(sampling_mode="standard", sequential_stages=2)

    assert len(sampler_calls) == 2
    first, second = sampler_calls
    assert first["steps"] == second["steps"] == 10
    assert first["sigmas"] is not None
    # The second segment continues where the first stopped instead of restarting.
    assert float(second["sigmas"][0]) < float(first["sigmas"][0])
    assert torch.equal(first["sigmas"][-1], second["sigmas"][0])


def test_legacy_mode_still_builds_its_own_schedule(sampler_calls):
    """The frozen path passes no sigmas, so KSampler rebuilds a full one per stage."""
    run_node(sampling_mode="legacy", sequential_stages=2)

    assert len(sampler_calls) == 2
    assert all(call["sigmas"] is None for call in sampler_calls)


def test_denoise_only_reaches_the_schedule_in_standard_mode(sampler_calls):
    run_node(sampling_mode="standard", denoise=1.0)
    full_start = float(sampler_calls[0]["sigmas"][0])

    sampler_calls.clear()
    run_node(sampling_mode="standard", denoise=0.6)
    assert float(sampler_calls[0]["sigmas"][0]) < full_start


def test_injection_stages_never_leave_a_one_step_segment(sampler_calls):
    run_node(sampling_mode="standard", injection_stages=3)

    assert sum(call["steps"] for call in sampler_calls) == 20
    assert all(call["steps"] >= 2 for call in sampler_calls)


def test_standard_mode_is_the_default(sampler_calls):
    run_node(sequential_stages=2)
    assert all(call["sigmas"] is not None for call in sampler_calls)


def test_every_advertised_shader_type_can_actually_be_resolved():
    """
    The dropdown must not offer a pattern the generator registry cannot build.

    It did: `temporal_coherent` shipped, was registered, and passed its own
    tests, but was missing from the node's combo, so no workflow could select
    it. Nothing tied the advertised list to the registry until this.
    """
    from snk.core.shader_noise import resolve_generator

    advertised = DirectShaderNoiseKSampler.INPUT_TYPES()["required"]["shader_type"][0]
    assert advertised, "the node must advertise at least one shader type"
    for shader_type in advertised:
        assert callable(resolve_generator(shader_type)), shader_type


def test_the_node_offers_every_registered_generator():
    """The other direction: a shipped generator that no workflow can reach is dead weight."""
    from snk.shaders.registry import get_shader, list_shaders

    advertised = set(DirectShaderNoiseKSampler.INPUT_TYPES()["required"]["shader_type"][0])
    # aliases point at a generator already reachable under its canonical name
    canonical = {name for name in list_shaders()
                 if not any(get_shader(name) is get_shader(other) and other in advertised
                            for other in advertised)}
    assert not canonical - advertised, f"registered but unreachable from the node: {sorted(canonical - advertised)}"


def test_the_preview_has_a_source_for_every_advertised_shader_type():
    """
    The live preview compiles one GLSL program per shader type out of
    web/src/shader_renderer.ts. A type without one leaves the preview on its last
    pattern and logs "Shader source not found", which is where temporal_coherent
    and spectral sat for a while.
    """
    import os
    import re

    with open(os.path.join(REPO_DIR, "web", "src", "shader_renderer.ts")) as source:
        block = source.read().split("const SHADER_SOURCES", 1)[1]
    previews = set(re.findall(r'^    "(\w+)": `', block, re.M))
    advertised = set(DirectShaderNoiseKSampler.INPUT_TYPES()["required"]["shader_type"][0])
    assert previews == advertised, sorted(previews ^ advertised)


def test_sanitising_keeps_every_advertised_shader_type():
    """
    The parameter whitelist rewrote any name it did not know to tensor_field.
    Standard mode never noticed, because the node passes shader_type to the
    pipeline separately; legacy mode reads it back out of the sanitised dict,
    so temporal_coherent sampled tensor_field there without a word.
    """
    from snk.shader_params_reader import ShaderParamsReader

    for shader_type in DirectShaderNoiseKSampler.INPUT_TYPES()["required"]["shader_type"][0]:
        sanitised = ShaderParamsReader.validate_and_sanitize_params({"shader_type": shader_type})
        assert sanitised["shader_type"] == shader_type


def test_legacy_mode_samples_the_shader_type_it_was_given(sampler_calls):
    from snk import shader_noise_ksampler

    with mock.patch.object(shader_noise_ksampler, "get_shader_generator",
                           wraps=shader_noise_ksampler.get_shader_generator) as resolve:
        run_node(sampling_mode="legacy", shader_type="temporal_coherent")

    assert resolve.call_args_list, "legacy mode never asked for a generator"
    assert {call.args[0] for call in resolve.call_args_list} == {"temporal_coherent"}


# --- the step window reaches the pipeline ----------------------------------

WINDOW_INPUTS = ["add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"]


def test_the_window_inputs_come_last():
    """
    ComfyUI maps saved widget values by position, so a new input anywhere but the
    tail re-reads every stored value in every saved workflow. Walk appends its own
    widgets to `required`, which the frontend orders ahead of all of `optional`, so
    the tail of Direct's optional block is the last slot on both nodes.
    """
    from snk.shader_noise_walk import ShaderNoiseWalk

    optional = list(DirectShaderNoiseKSampler.INPUT_TYPES()["optional"])
    assert optional[-4:] == WINDOW_INPUTS
    assert list(ShaderNoiseWalk.INPUT_TYPES()["optional"]) == optional


def test_the_window_defaults_sample_the_whole_schedule(sampler_calls):
    """Every default has to be a no-op, or it changes what saved workflows produce."""
    spec = DirectShaderNoiseKSampler.INPUT_TYPES()["optional"]
    defaults = {name: spec[name][1]["default"] for name in WINDOW_INPUTS}
    assert defaults == {"add_noise": True, "start_at_step": 0, "end_at_step": 10000,
                        "return_with_leftover_noise": False}

    run_node()
    whole = sampler_calls[0]["sigmas"].clone()
    sampler_calls.clear()

    run_node(**defaults)
    assert torch.equal(sampler_calls[0]["sigmas"], whole)


def test_the_node_forwards_the_window_to_the_pipeline():
    from snk import direct_shader_ksampler

    with mock.patch.object(direct_shader_ksampler.standard_pipeline, "run") as run:
        run.return_value = {"samples": torch.zeros(1, 4, 16, 16)}
        run_node(add_noise=False, start_at_step=4, end_at_step=7,
                 return_with_leftover_noise=True)

    assert {key: run.call_args.kwargs[key] for key in WINDOW_INPUTS} == {
        "add_noise": False, "start_at_step": 4, "end_at_step": 7,
        "return_with_leftover_noise": True,
    }


def test_legacy_mode_ignores_the_window(sampler_calls):
    """The frozen path predates it; asking for a window there must not fail the run."""
    run_node(sampling_mode="legacy", start_at_step=5, end_at_step=10,
             return_with_leftover_noise=True)

    assert sampler_calls and all(call["sigmas"] is None for call in sampler_calls)
