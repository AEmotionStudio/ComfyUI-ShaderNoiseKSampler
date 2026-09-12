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
from helpers import FakeModel, snapshot
from snk.core.shader_noise import UnsupportedLatentError
from snk.pipelines.standard import _rebuild, _split_noise, _streams, run

SHADER_PARAMS = {
    "scale": 1.0, "octaves": 1.0, "warp_strength": 0.5, "phase_shift": 0.5,
    "shape_type": "none", "color_scheme": "none", "time": 0.0, "base_seed": 8888,
}


@pytest.fixture
def recorder():
    """
    Record every sample() call and feed the callback a denoised estimate.

    The estimate goes back in the model's internal space, the way ComfyUI's own
    callback hands it over: process_latent_in has already run. That keeps a model
    which rescales one stream there (MiniMax H3) honest rather than accidentally
    passing.
    """
    calls = []

    def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                    latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, sigmas=None, callback=None,
                    disable_pbar=False, seed=None):
        calls.append({
            "noise": snapshot(noise), "latent": snapshot(latent_image),
            "steps": steps, "denoise": denoise, "sigmas": sigmas.detach().clone(),
            "noise_mask": noise_mask, "seed": seed, "force_full_denoise": force_full_denoise,
        })
        # Mirror comfy.samplers.CFGGuider: the noise and the callback's estimate live in
        # the model's internal space, and only the return value is back in latent space.
        internal = model.get_model_object("process_latent_in")(latent_image) + noise * 0.1
        if callback is not None:
            callback(max(steps - 1, 0), internal * 0.5, internal, steps)
        return model.get_model_object("process_latent_out")(internal)

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


@pytest.mark.parametrize("kind,sigma_value", [("eps", 2.7), ("flow", 0.4), ("av", 0.4)])
def test_boundary_split_is_lossless(kind, sigma_value):
    """
    Splitting mid-trajectory must reproduce the interrupted state exactly,
    otherwise multi-stage runs would drift even at shader_strength 0.

    The "av" case is MiniMax H3, whose model carries the audio stream at
    audio_scale on the way in. Inverting through the latent format alone -- which
    for MiniMaxH3AV is an identity, scale_factor being 1.0 -- would hand the next
    segment an audio residual wrong by that factor.
    """
    model = FakeModel(kind)
    model_sampling = model.sampling
    # Ground truth, straight off the model: exactly what CFGGuider.inner_sample applies
    # around a sample() call. _split_noise has to invert these, whatever it resolves.
    process_in, process_out = model.model.process_latent_in, model.model.process_latent_out
    sigma = torch.tensor(sigma_value)

    torch.manual_seed(0)
    x0_internal = model.empty_latent()
    x0_internal = _rebuild(x0_internal, [torch.randn_like(t) for t in _streams(x0_internal)])
    eps = _rebuild(x0_internal, [torch.randn_like(t) for t in _streams(x0_internal)])

    # noise_scaling is per stream: ComfyUI flattens a nested latent before sampling,
    # so it never sees a NestedTensor and has no __rmul__ to reach one with.
    def per_stream(fn, *values):
        return _rebuild(values[0], [fn(*group) for group in zip(*(_streams(v) for v in values))])

    x_internal = per_stream(lambda e, x: model_sampling.noise_scaling(sigma, e, x, False), eps, x0_internal)

    # What a finished segment hands back, and what the next one rebuilds from it.
    returned = process_out(per_stream(lambda t: model_sampling.inverse_noise_scaling(sigma, t), x_internal))
    next_latent, residual = _split_noise(returned, x0_internal, sigma, model_sampling, model)
    rebuilt = per_stream(
        lambda r, l: model_sampling.noise_scaling(sigma, r, l, False), residual, process_in(next_latent)
    )

    for got, want in zip(_streams(rebuilt), _streams(x_internal)):
        assert torch.allclose(got, want, atol=1e-5)
    for got, want in zip(_streams(residual), _streams(eps)):
        assert torch.allclose(got, want, atol=1e-4)


def test_av_latent_streams_survive_a_multi_stage_run(recorder):
    """
    MiniMax H3 hands the sampler a NestedTensor of a 5D video stream [B,24,T,H,W]
    and a 4D audio stream [B,32,2,T]. Both must reach every segment at their own
    shape and come back out of the pipeline intact.
    """
    model = FakeModel("av")
    expected = [tuple(s) for s in FakeModel.AV_SHAPES]

    out = run_pipeline(model=model, latent={"samples": model.empty_latent()},
                       sequential_stages=2, shader_strength=0.6)

    assert len(recorder) == 2
    for call in recorder:
        assert [tuple(t.shape) for t in call["noise"]] == expected
        assert [tuple(t.shape) for t in call["latent"]] == expected

    assert out["samples"].is_nested
    assert [tuple(t.shape) for t in out["samples"].unbind()] == expected


def test_the_shader_only_paints_the_spatial_stream(recorder):
    """
    H3's audio stream has no spatial grid, so it must keep exactly the Gaussian
    noise a stock KSampler would have given it while the video stream is painted.
    """
    model = FakeModel("av")
    latent = {"samples": model.empty_latent()}
    stock = comfy.sample.prepare_noise(latent["samples"], 8888, None).unbind()

    run_pipeline(model=model, latent=latent, shader_strength=0.6)
    video, audio = recorder[0]["noise"]

    assert torch.equal(audio, stock[1]), "audio stream must be untouched"
    assert not torch.allclose(video, stock[0]), "video stream must be painted"


def test_non_spatial_latent_is_refused_by_name(recorder):
    """Stable Audio, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D and TripoSplat land here."""
    with pytest.raises(UnsupportedLatentError, match=r"3D \(1, 64, 1024\)"):
        run_pipeline(model=FakeModel("flow"), latent={"samples": torch.ones(1, 64, 1024)})

    assert recorder == [], "refused before any sampling started"


def test_non_spatial_latent_still_samples_without_a_shader(recorder):
    """With nothing to paint there is nothing to refuse, so it works as a plain KSampler."""
    run_pipeline(model=FakeModel("flow"), latent={"samples": torch.ones(1, 64, 1024)},
                 shader_strength=0.0)

    assert len(recorder) == 1
    assert tuple(recorder[0]["noise"].shape) == (1, 64, 1024)
