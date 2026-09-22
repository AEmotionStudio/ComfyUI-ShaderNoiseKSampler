"""
The walk node ramps one parameter and batches the results.

Exercised through the same recording stand-in the pipeline tests use, so no
diffusion model is needed: the interesting behaviour is which values reach the
sampler and how the results are joined.
"""
from unittest import mock

import pytest
import torch

import comfy.sample
from helpers import FakeModel
from snk.shader_noise_walk import MAX_STEPS, ShaderNoiseWalk, _ramp, stack_latents

SHADER_PARAMS = dict(
    shader_type="domain_warp", shape_type="none", color_scheme="none", noise_scale=1.0,
    octaves=1.0, warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.5,
    color_intensity=0.8,
)


@pytest.fixture
def recorder():
    """Record the shader params each run receives, via the pipeline it calls."""
    calls = []

    def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent_image, denoise=1.0, disable_noise=False, start_step=None,
                    last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None,
                    callback=None, disable_pbar=False, seed=None):
        calls.append({"seed": seed, "noise": noise})
        internal = model.get_model_object("process_latent_in")(latent_image) + noise * 0.1
        if callback is not None:
            callback(max(steps - 1, 0), internal * 0.5, internal, steps)
        return model.get_model_object("process_latent_out")(internal)

    with mock.patch.object(comfy.sample, "sample", fake_sample):
        yield calls


def run_walk(model=None, latent=None, **overrides):
    kwargs = dict(
        model=model or FakeModel("eps"), seed=8888, steps=6, cfg=7.0, sampler_name="euler",
        scheduler="normal", positive=[], negative=[],
        latent_image=latent or {"samples": torch.zeros(1, 4, 16, 16)}, denoise=1.0,
        sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
        noise_transform="none", use_temporal_coherence=False,
        walk_parameter="shader_strength", walk_start=0.0, walk_end=0.4, walk_steps=3,
        **SHADER_PARAMS,
    )
    kwargs.update(overrides)
    return ShaderNoiseWalk().walk(**kwargs)[0]


def test_ramp_includes_both_endpoints():
    assert _ramp(0.0, 1.0, 5) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert _ramp(0.2, 0.2, 3) == pytest.approx([0.2, 0.2, 0.2])
    assert _ramp(0.0, 1.0, 1) == [0.0]


def test_walk_returns_one_latent_per_step(recorder):
    out = run_walk(walk_steps=4)
    assert out["samples"].shape == (4, 4, 16, 16)


def test_each_step_is_its_own_sampling_run(recorder):
    run_walk(walk_steps=3, sequential_stages=2)
    assert len(recorder) == 6, "3 walk steps x 2 segments"


def test_walking_strength_changes_the_noise_but_not_the_seed(recorder):
    run_walk(walk_parameter="shader_strength", walk_start=0.0, walk_end=0.5, walk_steps=3)

    assert {call["seed"] for call in recorder} == {8888}, "the seed must be held"
    first, last = recorder[0]["noise"], recorder[-1]["noise"]
    assert not torch.allclose(first, last), "strength 0.0 and 0.5 must differ"


def test_walking_the_seed_changes_the_seed(recorder):
    run_walk(walk_parameter="seed", walk_start=1, walk_end=3, walk_steps=3)
    assert [call["seed"] for call in recorder] == [1, 2, 3]


def test_an_unwalkable_parameter_is_refused(recorder):
    with pytest.raises(ValueError, match="cannot walk"):
        run_walk(walk_parameter="cfg")
    assert recorder == []


def test_walk_steps_is_capped():
    spec = ShaderNoiseWalk.INPUT_TYPES()["required"]["walk_steps"][1]
    assert spec["max"] == MAX_STEPS


def test_multi_stream_latents_batch_per_stream(recorder):
    """MiniMax H3 and LTXAV arrive nested; each stream has to be joined separately."""
    model = FakeModel("av")
    out = run_walk(model=model, latent={"samples": model.empty_latent()}, walk_steps=3)

    assert out["samples"].is_nested
    assert [tuple(t.shape) for t in out["samples"].unbind()] == [
        (3,) + tuple(s[1:]) for s in FakeModel.AV_SHAPES
    ]


def test_batch_index_is_dropped_but_other_keys_survive():
    latents = [{"samples": torch.zeros(1, 4, 8, 8), "batch_index": [0], "keep": "yes"}
               for _ in range(2)]
    joined = stack_latents(latents)
    assert joined["samples"].shape == (2, 4, 8, 8)
    assert "batch_index" not in joined
    assert joined["keep"] == "yes"


def test_the_walk_node_offers_everything_the_direct_node_does():
    """It subclasses the sampler, so a new sampler input must not go missing here."""
    from snk.direct_shader_ksampler import DirectShaderNoiseKSampler

    direct = DirectShaderNoiseKSampler.INPUT_TYPES()
    walk = ShaderNoiseWalk.INPUT_TYPES()
    for section in ("required", "optional"):
        assert set(direct.get(section, {})) <= set(walk.get(section, {})), section


def test_the_walk_node_passes_the_window_through(recorder):
    """Walk forwards **kwargs into Direct.sample, so a new input has to survive it."""
    batch = run_walk(walk_steps=2, start_at_step=2, end_at_step=5,
                     return_with_leftover_noise=True)

    assert batch["samples"].shape[0] == 2
    assert len(recorder) == 2, "one windowed run per point on the ramp"
