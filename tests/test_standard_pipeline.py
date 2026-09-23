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
import latent_preview
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
    assert float(recorder[0]["sigmas"][-1]) == 0.0


def test_stages_slice_one_schedule(recorder):
    run_pipeline(sequential_stages=2)

    assert len(recorder) == 2
    first, second = recorder[0]["sigmas"], recorder[1]["sigmas"]
    # Contiguous halves of a single descending schedule.
    assert torch.equal(first[-1], second[0])
    assert float(first[0]) > float(second[0]) > float(second[-1])
    assert float(second[-1]) == pytest.approx(0.0)


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


def test_sequence_latents_are_painted_when_opted_in(recorder):
    """
    Stable Audio, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D and TripoSplat carry
    [B, C, L] with no grid. Refused by default; painted as a one-row strip when
    asked for.
    """
    samples = torch.ones(1, 64, 1024)
    stock = comfy.sample.prepare_noise(samples, 8888, None)

    run_pipeline(model=FakeModel("flow"), latent={"samples": samples},
                 shader_strength=0.5, shade_non_spatial=True)

    assert len(recorder) == 1
    assert tuple(recorder[0]["noise"].shape) == (1, 64, 1024)
    assert not torch.allclose(recorder[0]["noise"], stock), "the strip must be painted"


def test_the_audio_stream_is_painted_when_opted_in(recorder):
    """The inverse of test_the_shader_only_paints_the_spatial_stream."""
    model = FakeModel("av")
    latent = {"samples": model.empty_latent()}
    stock = comfy.sample.prepare_noise(latent["samples"], 8888, None).unbind()

    run_pipeline(model=model, latent=latent, shader_strength=0.5, shade_non_spatial=True)
    video, audio = recorder[0]["noise"]

    assert not torch.allclose(audio, stock[1]), "audio must now be painted"
    assert not torch.allclose(video, stock[0]), "video must still be painted"


def test_each_stream_gets_its_own_pattern(recorder):
    """Otherwise both streams would carry the same field wherever shapes allow."""
    model = FakeModel("av")
    run_pipeline(model=model, latent={"samples": model.empty_latent()},
                 shader_strength=0.5, shade_non_spatial=True)
    video, audio = recorder[0]["noise"]
    assert video.shape != audio.shape          # different shapes anyway here,
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_matching_streams_get_their_own_field():
    """
    Two streams of the same shape are the only case where one field could be drawn
    into both. Feeding identical base noise isolates the per-stream seed offset: the
    inputs match, so anything left over in the outputs is the shader's own doing.
    """
    from comfy.nested_tensor import NestedTensor
    from snk.pipelines.standard import _paint

    base = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
    painted = _paint(NestedTensor([base.clone(), base.clone()]), [(0.5, 8888, {})],
                     dict(SHADER_PARAMS), "domain_warp", "multiply", "none",
                     False, True, "walk", True)
    first, second = painted.unbind()

    assert not torch.allclose(first, base), "the streams must be painted at all"
    assert not torch.allclose(first, second), "and not with the same field twice"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device to mix up")
def test_painting_leaves_every_stream_on_its_own_device():
    """
    A boundary residual arrives on the sampler's device while the latent that started
    the run sits on the CPU. Painting only the paintable stream onto a device from
    anywhere else strands the others: MiniMax H3 then reached pack_latents with its
    video on one device and its audio on another, and torch.cat refused.
    """
    from comfy.nested_tensor import NestedTensor
    from snk.pipelines.standard import _paint

    streams = [torch.randn(1, 4, 16, 16, device="cuda"), torch.randn(1, 8, 32, device="cuda")]
    painted = _paint(NestedTensor(streams), [(0.5, 8888, {})], dict(SHADER_PARAMS),
                     "domain_warp", "multiply", "none", False, True, "walk", False)

    assert {stream.device for stream in painted.unbind()} == {streams[0].device}


def test_an_unpaintable_latent_is_refused_before_any_sampling(recorder):
    """
    Why the check runs up front rather than at the boundary that needs it: with
    add_noise off nothing is painted at the opening, so the failure would otherwise
    land after a whole segment had already been sampled.
    """
    with pytest.raises(UnsupportedLatentError):
        run_pipeline(model=FakeModel("flow"), latent={"samples": torch.ones(1, 64, 1024)},
                     shader_strength=0.5, add_noise=False, injection_stages=1)

    assert recorder == [], "refused before any sampling started"


def test_sequence_latents_are_still_refused_by_default(recorder):
    with pytest.raises(UnsupportedLatentError):
        run_pipeline(model=FakeModel("flow"), latent={"samples": torch.ones(1, 64, 1024)},
                     shader_strength=0.5)
    assert recorder == []


def test_stage_progression_varies_the_shader_across_the_run(recorder):
    """Every stage used to draw the same shader at the same zoom."""
    from snk.pipelines.standard import _shader_events

    _, uniform = _shader_events(20, 0, 20, 3, 0, 0.5, "uniform", "uniform", 1, False, "uniform")
    _, shaped = _shader_events(20, 0, 20, 3, 0, 0.5, "uniform", "uniform", 1, False, "coarse_to_fine")

    assert all(not event[2] for stage in uniform.values() for event in stage)
    multipliers = [event[2]["scale_multiplier"] for stage in shaped.values() for event in stage]
    assert len(multipliers) == 3
    assert sorted(multipliers) == multipliers, "must ramp with position in the schedule"
    assert multipliers[0] < 1.0 < multipliers[-1]


def test_a_shaped_run_still_samples_correctly(recorder):
    out = run_pipeline(sequential_stages=3, shader_strength=0.4,
                       stage_progression="coarse_to_fine")
    assert len(recorder) == 3
    assert out["samples"].shape == (1, 4, 16, 16)
    assert torch.isfinite(out["samples"]).all()


def test_shaping_leaves_the_callers_params_alone(recorder):
    """The params dict is shared across stages; shaping must copy, not mutate."""
    params = dict(SHADER_PARAMS, scale=1.0, octaves=2.0)
    run_pipeline(sequential_stages=3, shader_strength=0.4, shader_params=params,
                 stage_progression="coarse_to_fine")
    assert params["scale"] == 1.0 and params["octaves"] == 2.0


def test_metadata_streams_are_never_painted(recorder):
    """
    TripoSplat is (geometry, camera) where the camera is [B, 1, 5]. Painting it
    would move the viewpoint rather than vary the subject, so shade_non_spatial
    has to skip it even though it skips nothing else.
    """
    from comfy.nested_tensor import NestedTensor
    from snk.pipelines.standard import _paintable

    geometry, camera = torch.ones(1, 16, 8, 8), torch.ones(1, 1, 5)
    assert _paintable([geometry, camera], True) == [0]
    assert _paintable([geometry, camera], False) == [0]

    # a real audio stream is far above the threshold and must still be painted
    audio = torch.ones(1, 32, 2, 207)
    assert _paintable([geometry, audio], True) == [0, 1]


# --- the step window -------------------------------------------------------
#
# start_at_step and end_at_step sample part of the schedule, which is what lets
# the node be one half of a split run: early steps here, a latent upscaler in
# between, the rest in a second node.


def full_schedule(recorder, **overrides):
    """The whole schedule as the pipeline builds it, with no window applied."""
    run_pipeline(**overrides)
    sigmas = recorder[0]["sigmas"].clone()
    recorder.clear()
    return sigmas


def test_start_at_step_enters_the_schedule_late(recorder):
    full = full_schedule(recorder)
    run_pipeline(start_at_step=8)

    assert len(recorder) == 1
    assert recorder[0]["steps"] == 12
    assert torch.equal(recorder[0]["sigmas"], full[8:])


def test_end_at_step_stops_early_and_still_finishes_clean(recorder):
    full = full_schedule(recorder)
    run_pipeline(end_at_step=12)

    assert recorder[0]["steps"] == 12
    assert torch.equal(recorder[0]["sigmas"][:-1], full[:12])
    assert float(recorder[0]["sigmas"][-1]) == 0.0, "the window's end must be a clean latent"


def test_leftover_noise_keeps_the_schedules_own_last_sigma(recorder):
    full = full_schedule(recorder)
    run_pipeline(end_at_step=12, return_with_leftover_noise=True)

    assert torch.equal(recorder[0]["sigmas"], full[:13])
    assert float(recorder[0]["sigmas"][-1]) > 0.0, "the latent must still carry its noise"


def test_leftover_noise_does_nothing_at_the_end_of_the_schedule(recorder):
    """KSampler only zeroes a sigma it truncated, so a full run ignores the flag."""
    kept = full_schedule(recorder, return_with_leftover_noise=True)
    assert torch.equal(kept, full_schedule(recorder))


def test_a_stopped_window_keeps_its_interior_boundaries(recorder):
    """
    Only the window's own end is brought to zero. A boundary inside it still hands
    the next segment the sigma the schedule gave it, which is what _split_noise
    divides by.
    """
    run_pipeline(end_at_step=12, sequential_stages=2)
    first, second = recorder[0]["sigmas"], recorder[1]["sigmas"]

    assert torch.equal(first[-1], second[0])
    assert float(first[-1]) > 0.0, "an interior boundary must keep its own sigma"
    assert float(second[-1]) == 0.0


def test_the_two_halves_are_one_trajectory(recorder):
    """
    A split run has to reproduce the unsplit one: the same sigmas in the same
    order, and the same per-segment seed, which drives ancestral and SDE draws.
    """
    run_pipeline(sequential_stages=2)
    whole = [call["sigmas"].clone() for call in recorder]
    seeds = [call["seed"] for call in recorder]
    recorder.clear()

    run_pipeline(end_at_step=10, return_with_leftover_noise=True)
    run_pipeline(start_at_step=10, add_noise=False)

    assert [call["seed"] for call in recorder] == seeds
    for got, want in zip(recorder, whole):
        assert torch.equal(got["sigmas"], want)
    assert torch.equal(recorder[0]["sigmas"][-1], recorder[1]["sigmas"][0])


def test_stages_divide_the_window_not_the_whole_schedule(recorder):
    """
    Two stages over the last ten steps means two stages in those ten steps. Spread
    over the whole schedule instead, the first would sit outside the window and
    never fire.
    """
    run_pipeline(start_at_step=10, sequential_stages=2)
    assert [call["steps"] for call in recorder] == [5, 5]


def test_shaping_stays_measured_against_the_whole_trajectory(recorder):
    """
    A node running the tail of a schedule is at the fine end of coarse_to_fine,
    not starting a fresh coarse-to-fine sweep of its own.
    """
    from snk.pipelines.standard import _shader_events

    args = (2, 0, 0.5, "uniform", "uniform", 1, False, "coarse_to_fine")
    _, early = _shader_events(20, 0, 10, *args)
    _, late = _shader_events(20, 10, 20, *args)

    assert all(e[2]["scale_multiplier"] < 1.0 for stage in early.values() for e in stage)
    assert all(e[2]["scale_multiplier"] > 1.0 for stage in late.values() for e in stage)


def test_an_empty_window_returns_the_latent_untouched(recorder):
    samples = torch.randn(1, 4, 16, 16)
    out = run_pipeline(latent={"samples": samples, "batch_index": [0]},
                       start_at_step=12, end_at_step=8)

    assert recorder == [], "nothing to denoise, so nothing to sample"
    assert torch.equal(out["samples"], samples)
    assert out["batch_index"] == [0]


def test_a_schedule_with_no_steps_returns_the_latent(recorder):
    """denoise 0.0 builds an empty schedule; the sampler would index off the end of it."""
    samples = torch.randn(1, 4, 16, 16)
    out = run_pipeline(latent={"samples": samples}, denoise=0.0)

    assert recorder == []
    assert torch.equal(out["samples"], samples)


# --- add_noise -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["eps", "av"])
def test_add_noise_off_hands_the_sampler_zeros(recorder, kind):
    model = FakeModel(kind)
    latent = {"samples": model.empty_latent() if kind == "av" else torch.randn(1, 4, 16, 16)}
    run_pipeline(model=model, latent=latent, start_at_step=10, add_noise=False)

    noise = recorder[0]["noise"]
    for stream in (noise if isinstance(noise, list) else [noise]):
        assert not stream.any(), "every stream must arrive unnoised"


def test_add_noise_off_skips_the_opening_shader(recorder):
    """Painting the shader onto zeros would add back exactly what was turned off."""
    run_pipeline(latent={"samples": torch.randn(1, 4, 16, 16)}, start_at_step=10,
                 add_noise=False, shader_strength=0.8)

    assert not recorder[0]["noise"].any()


def test_add_noise_off_still_paints_later_boundaries(recorder):
    """Their noise comes out of the latent, so there is something real to paint."""
    run_pipeline(latent={"samples": torch.randn(1, 4, 16, 16)}, add_noise=False,
                 injection_stages=1, shader_strength=0.6)

    assert len(recorder) == 2
    assert not recorder[0]["noise"].any()
    assert recorder[1]["noise"].any(), "the injection boundary must still reach the run"


def test_add_noise_off_does_not_refuse_a_latent_it_will_never_paint(recorder):
    """
    A sequence latent is refused because the shader cannot paint it -- but with
    add_noise off and no interior boundary, nothing was going to be painted.
    """
    run_pipeline(model=FakeModel("flow"), latent={"samples": torch.ones(1, 64, 1024)},
                 add_noise=False, start_at_step=10, shader_strength=0.5)

    assert len(recorder) == 1


# --- progress --------------------------------------------------------------


def test_the_progress_bar_counts_the_windows_own_steps(recorder):
    """
    ProgressBar takes its total from the callback, so reporting absolute positions
    would leave a windowed run starting part-filled and stopping short of the end.
    """
    reported = []

    def fake_prepare_callback(model, steps, x0_output_dict=None):
        return lambda step, x0, x, total: reported.append((step, total))

    with mock.patch.object(latent_preview, "prepare_callback", fake_prepare_callback):
        run_pipeline(end_at_step=12, disable_pbar=False)

    assert reported[-1] == (11, 12), "the last step of the window must fill the bar"


@pytest.mark.parametrize("start,end,leftover", [
    (0, 20, False), (0, 12, False), (0, 12, True),
    (8, 20, False), (8, 12, False), (8, 12, True), (8, 20, True),
])
def test_the_window_slices_exactly_what_ksampler_advanced_slices(recorder, start, end, leftover):
    """
    Against ComfyUI's own KSampler, not a restatement of ours. A split that puts
    this node on one side and a stock KSampler (Advanced) on the other only joins
    up if both index the schedule the same way -- including the zeroed last sigma,
    and including the cases where the flag is supposed to do nothing.
    """
    import comfy.samplers

    model = FakeModel("eps")
    core = []

    def capture(model_, noise, positive, negative, cfg, device, sampler, sigmas, *args, **kwargs):
        core.append(sigmas.clone())
        return kwargs["latent_image"]

    with mock.patch.object(comfy.samplers, "sample", capture):
        sampler = comfy.samplers.KSampler(model, steps=20, device="cpu", sampler="euler",
                                          scheduler="normal", denoise=1.0, model_options={})
        sampler.sample(torch.zeros(1, 4, 16, 16), [], [], cfg=7.0,
                       latent_image=torch.zeros(1, 4, 16, 16),
                       start_step=start, last_step=end, force_full_denoise=not leftover)

    run_pipeline(model=model, start_at_step=start, end_at_step=end,
                 return_with_leftover_noise=leftover)

    assert len(recorder) == 1, "one stage, so one segment to compare"
    assert torch.equal(recorder[0]["sigmas"], core[0])
