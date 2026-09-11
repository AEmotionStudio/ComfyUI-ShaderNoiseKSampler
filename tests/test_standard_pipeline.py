"""
The standard pipeline samples one trajectory.

Legacy gave each stage its own full schedule starting at maximum noise, hard
-coded denoise to 1.0, and only counted custom sigmas. These tests pin the
corrected behaviour using a recording stand-in for comfy.sample.sample, so no
diffusion model is needed.
"""
from unittest import mock

import pytest
import torch

import comfy.sample
from helpers import FakeModel
from snk.pipelines.standard import _split_noise, run

SHADER_PARAMS = {
    "scale": 1.0, "octaves": 1.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


@pytest.fixture
def recorder():
    """Record every sample() call and feed the callback a denoised estimate."""
    calls = []

    def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, sigmas=None, callback=None,
                    disable_pbar=False, seed=None):
        calls.append({
            "noise": noise.detach().clone(), "latent": latent_image.detach().clone(),
            "steps": steps, "denoise": denoise, "sigmas": sigmas.detach().clone(),
            "noise_mask": noise_mask, "seed": seed, "force_full_denoise": force_full_denoise,
        })
        result = latent_image + 0.1 * noise
        if callback is not None:
            callback(max(steps - 1, 0), result * 0.5, result, steps)
        return result

    with mock.patch.object(comfy.sample, "sample", fake_sample):
        yield calls


def run_pipeline(recorder_unused=None, **overrides):
    kwargs = dict(
        model=FakeModel("eps"), seed=8888, steps=20, cfg=7.0, sampler_name="euler",
        scheduler="normal", positive=[], negative=[],
        latent={"samples": torch.zeros(1, 4, 16, 16)}, denoise=1.0,
        sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
        noise_transform="none", shader_params=dict(SHADER_PARAMS), shader_type="domain_warp",
        disable_pbar=True,
    )
    kwargs.update(overrides)
    return run(**kwargs)


def test_single_stage_runs_one_segment(recorder):
    out = run_pipeline()
    assert len(recorder) == 1
    assert out["samples"].shape == (1, 4, 16, 16)
    assert recorder[0]["force_full_denoise"] is True


def test_stages_slice_one_schedule(recorder):
    run_pipeline(sequential_stages=2)

    assert len(recorder) == 2
    first, second = recorder[0]["sigmas"], recorder[1]["sigmas"]
    # Contiguous halves of a single descending schedule.
    assert torch.equal(first[-1], second[0])
    assert float(first[0]) > float(second[0]) > float(second[-1])
    assert float(second[-1]) == pytest.approx(0.0)
    assert recorder[0]["force_full_denoise"] is False
    assert recorder[1]["force_full_denoise"] is True


def test_every_segment_starts_where_the_previous_ended(recorder):
    run_pipeline(sequential_stages=2, injection_stages=3)

    total = sum(call["steps"] for call in recorder)
    assert total == 20, "segments must cover the full step count exactly once"
    for call in recorder:
        assert call["steps"] >= 2, "no 1-step segment"


def test_denoise_reaches_the_schedule(recorder):
    """Legacy passed 1.0 to every stage, so the denoise input did nothing."""
    run_pipeline(denoise=1.0)
    full_start = float(recorder[0]["sigmas"][0])

    recorder.clear()
    run_pipeline(denoise=0.6)
    partial_start = float(recorder[0]["sigmas"][0])

    assert partial_start < full_start


def test_custom_sigmas_are_sampled_not_just_counted(recorder):
    custom = torch.tensor([12.0, 8.0, 5.0, 3.0, 1.5, 0.6, 0.0])
    run_pipeline(custom_sigmas=custom)

    assert torch.equal(recorder[0]["sigmas"], custom)


def test_zero_strength_uses_exactly_the_stock_ksampler_noise(recorder):
    samples = torch.zeros(1, 4, 16, 16)
    run_pipeline(latent={"samples": samples}, shader_strength=0.0)

    expected = comfy.sample.prepare_noise(samples, 8888, None)
    assert torch.equal(recorder[0]["noise"], expected)


def test_shader_strength_changes_the_noise(recorder):
    run_pipeline(shader_strength=0.0)
    plain = recorder[0]["noise"].clone()

    recorder.clear()
    run_pipeline(shader_strength=0.5)
    assert not torch.allclose(recorder[0]["noise"], plain)


def test_noise_mask_is_forwarded(recorder):
    mask = torch.ones(1, 1, 16, 16)
    run_pipeline(latent={"samples": torch.zeros(1, 4, 16, 16), "noise_mask": mask})

    assert recorder[0]["noise_mask"] is mask


def test_latent_metadata_is_preserved(recorder):
    out = run_pipeline(latent={"samples": torch.zeros(1, 4, 16, 16), "batch_index": [0]})
    assert out["batch_index"] == [0]


@pytest.mark.parametrize("kind,sigma_value", [("eps", 2.7), ("flow", 0.4)])
def test_boundary_split_is_lossless(kind, sigma_value):
    """
    Splitting mid-trajectory must reproduce the interrupted state exactly,
    otherwise multi-stage runs would drift even at shader_strength 0.
    """
    model = FakeModel(kind)
    model_sampling, latent_format = model.sampling, model.latent_format
    sigma = torch.tensor(sigma_value)

    torch.manual_seed(0)
    x0_internal = torch.randn(1, 4, 8, 8)
    eps = torch.randn_like(x0_internal)
    x_internal = model_sampling.noise_scaling(sigma, eps, x0_internal, False)

    # What a finished segment hands back, and what the next one rebuilds from it.
    returned = latent_format.process_out(model_sampling.inverse_noise_scaling(sigma, x_internal))
    next_latent, residual = _split_noise(returned, x0_internal, sigma, model_sampling, latent_format)
    rebuilt = model_sampling.noise_scaling(sigma, residual, latent_format.process_in(next_latent), False)

    assert torch.allclose(rebuilt, x_internal, atol=1e-5)
    assert torch.allclose(residual, eps, atol=1e-4)
