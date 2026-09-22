"""
One schedule, split into segments.

Legacy gave every stage its own full schedule, and its injection points could
leave a 1-step tail: 20 steps with 3 injection stages produced ranges
(0,10) (10,19) (19,20), so the final image came out of a single step from full
noise. It also never applied `denoise` to the schedule.
"""
import pytest
import torch

from helpers import FakeModel
from snk.core import schedule
from snk.core.schedule import (
    MIN_SEGMENT_STEPS,
    SUPPORTED_DISTRIBUTIONS,
    build_sigmas,
    injection_points,
    merge_boundaries,
    segments,
    sequential_starts,
    stage_strengths,
)


def boundaries_for(total_steps, sequential, injection):
    return merge_boundaries(
        total_steps, sequential_starts(total_steps, sequential), injection_points(total_steps, injection)
    )


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_stage_strengths_shape(distribution):
    assert stage_strengths(0.3, 0, distribution) == []
    assert stage_strengths(0.3, 1, distribution) == [0.3]

    values = stage_strengths(0.3, 4, distribution)
    assert len(values) == 4
    assert all(0.0 <= v <= 0.3 + 1e-9 for v in values)


def test_single_stage_never_silently_disables_the_shader():
    """linear_increase gave 0.0 for a single stage in the old in-class copy."""
    for distribution in SUPPORTED_DISTRIBUTIONS:
        assert stage_strengths(0.3, 1, distribution) == [0.3]


def test_distribution_direction():
    assert stage_strengths(1.0, 3, "linear_decrease") == pytest.approx([1.0, 0.5, 0.25])
    assert stage_strengths(1.0, 3, "linear_increase") == pytest.approx([0.25, 0.5, 1.0])
    assert stage_strengths(1.0, 3, "uniform") == pytest.approx([1.0, 1.0, 1.0])

    gaussian = stage_strengths(1.0, 3, "gaussian")
    assert gaussian[1] > gaussian[0] and gaussian[1] > gaussian[2]


@pytest.mark.parametrize(
    "total,sequential,injection",
    [(20, 1, 0), (20, 2, 0), (20, 1, 3), (20, 0, 2), (20, 3, 3), (10, 2, 2), (6, 2, 3), (4, 1, 3), (2, 3, 3)],
)
def test_segments_cover_the_schedule_without_short_tails(total, sequential, injection):
    segs = segments(boundaries_for(total, sequential, injection), total)

    assert segs[0][0] == 0, "sampling must start at the first sigma"
    assert segs[-1][1] == total, "sampling must reach the last sigma"
    for (_, end), (next_start, _) in zip(segs, segs[1:]):
        assert end == next_start, "segments must be contiguous"
    if total >= MIN_SEGMENT_STEPS:
        assert all(end - start >= MIN_SEGMENT_STEPS for start, end in segs), segs


def test_injection_points_are_interior():
    """Step 0 is what a sequential stage already does; the last step has nothing left to sample."""
    points = injection_points(20, 3)
    assert points == [5, 10, 15]
    assert all(0 < p < 20 for p in points)


def test_legacy_one_step_tail_is_gone():
    segs = segments(boundaries_for(20, 1, 3), 20)
    assert (19, 20) not in segs
    assert min(end - start for start, end in segs) >= MIN_SEGMENT_STEPS


def test_build_sigmas_applies_denoise():
    model = FakeModel("eps")
    full = build_sigmas(model, 20, "euler_ancestral", "beta", denoise=1.0)
    partial = build_sigmas(model, 20, "euler_ancestral", "beta", denoise=0.6)

    assert len(full) == 21 and len(partial) == 21
    assert full[0] > partial[0], "denoise<1 must start from a lower sigma"
    assert float(full[-1]) == pytest.approx(0.0) and float(partial[-1]) == pytest.approx(0.0)


def test_build_sigmas_is_descending():
    sigmas = build_sigmas(FakeModel("flow"), 12, "euler", "normal", denoise=1.0)
    assert torch.all(sigmas[1:] <= sigmas[:-1])


def test_custom_sigmas_are_used_verbatim():
    """Legacy wrapped the model instead, so only the sigma count ever took effect."""
    model = FakeModel("eps")
    custom = torch.tensor([14.6, 6.0, 2.7, 1.1, 0.3, 0.0])
    result = build_sigmas(model, 20, "euler", "beta", 1.0, custom_sigmas=custom)
    assert torch.equal(result, custom.float())

    ascending = torch.tensor([0.0, 0.3, 1.1, 2.7, 6.0, 14.6])
    flipped = build_sigmas(model, 20, "euler", "beta", 1.0, custom_sigmas=ascending)
    assert torch.equal(flipped, custom.float())


# --- per-stage shaping ----------------------------------------------------------------

def test_uniform_progression_changes_nothing():
    for progress in (0.0, 0.5, 1.0):
        assert schedule.stage_shaping("uniform", progress) == {}
        assert schedule.stage_shaping("not_a_progression", progress) == {}


def test_coarse_to_fine_zooms_in_then_out():
    """Low noise_scale is large, zoomed-in features; high is small, zoomed-out ones."""
    start = schedule.stage_shaping("coarse_to_fine", 0.0)
    end = schedule.stage_shaping("coarse_to_fine", 1.0)

    assert start["scale_multiplier"] < 1.0 < end["scale_multiplier"]
    assert start["octave_offset"] < 0.0 < end["octave_offset"]


def test_fine_to_coarse_is_the_mirror():
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        forward = schedule.stage_shaping("coarse_to_fine", progress)
        backward = schedule.stage_shaping("fine_to_coarse", 1.0 - progress)
        assert forward == pytest.approx(backward)


def test_the_midpoint_is_the_widget_value():
    """The span is centred, so a shaped run still sits on the settings the user chose."""
    middle = schedule.stage_shaping("coarse_to_fine", 0.5)
    assert middle["octave_offset"] == pytest.approx(0.0)
    assert 1.0 < middle["scale_multiplier"] < 1.5   # geometric span, so not exactly 1.0


def window_for(first, last, sequential, injection):
    """Boundaries for a run that samples only steps [first, last] of its schedule."""
    span = last - first
    return merge_boundaries(
        last,
        [first + start for start in sequential_starts(span, sequential)],
        [first + point for point in injection_points(span, injection)],
        first=first,
    )


def test_a_window_keeps_its_boundaries_inside_itself():
    assert window_for(10, 20, 3, 2) == [10, 13, 16]
    assert window_for(4, 7, 3, 3) == [4], "three stages cannot fit in three steps"
    assert window_for(0, 20, 2, 0) == [0, 10], "the whole schedule is just the widest window"


def test_the_tail_rule_measures_against_the_window_not_the_schedule():
    """
    merge_boundaries' first argument is where the window ends. Handing it the
    schedule length instead would keep a boundary one step short of the window's
    end and leave a 1-step tail inside an otherwise long run.
    """
    starts = [10, 13, 16, 19]
    assert merge_boundaries(20, starts, [], first=10) == [10, 13, 16]
    assert merge_boundaries(30, starts, [], first=10) == [10, 13, 16, 19]


@pytest.mark.parametrize("first,last,sequential,injection", [
    (10, 20, 3, 2), (4, 7, 3, 3), (5, 7, 2, 2), (18, 20, 3, 3), (2, 4, 3, 3),
    (6, 7, 5, 5), (0, 1, 1, 0), (0, 20, 2, 3),
])
def test_windowed_segments_cover_the_window_exactly(first, last, sequential, injection):
    segs = segments(window_for(first, last, sequential, injection), last)

    assert segs, "a window must always produce something to sample"
    assert segs[0][0] == first and segs[-1][1] == last
    assert all(end == segs[i + 1][0] for i, (_, end) in enumerate(segs[:-1])), "no gaps"
    assert all(end - start >= 1 for start, end in segs), "no 0-step segment"
    if last - first >= MIN_SEGMENT_STEPS:
        assert all(end - start >= MIN_SEGMENT_STEPS for start, end in segs)
