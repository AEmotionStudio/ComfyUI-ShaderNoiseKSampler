"""
Presets and travel modes: the two controls meant to be usable without tooltips.
"""
import pytest
import torch

from helpers import FakeModel
from snk.core import presets, shader_noise
from snk.core.presets import PRESETS, TRAVEL_MODES, apply_preset, basis_for, is_collapse

CPU = torch.device("cpu")
PARAMS = {"scale": 1.0, "octaves": 2.0, "warp_strength": 0.7, "phase_shift": 0.5,
          "time": 0.0, "shape_type": "none", "color_scheme": "none", "intensity": 0.8}


# --- travel modes ---------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 4, 48, 48), (1, 24, 5, 16, 16), (1, 128, 16, 16)])
def test_the_modes_are_ordered_by_how_wide_they_leave_the_noise(shape):
    ranks = {}
    for mode in ("jump", "drift", "walk"):
        noise = shader_noise.generate(shape, PARAMS, "domain_warp", 8888, CPU,
                                      decorrelate=True, basis=basis_for(mode))
        ranks[mode] = shader_noise.effective_channel_rank(noise)

    assert ranks["jump"] == pytest.approx(1.0, abs=0.01), "jump is a deliberate collapse"
    if shape[1] > basis_for("drift"):
        # Strict, so drift collapsing into walk fails instead of passing as equal.
        assert ranks["jump"] < ranks["drift"] < ranks["walk"], ranks
    else:
        # Four channels are all the directions there are: drift has nothing to narrow.
        assert ranks["jump"] < ranks["drift"] == ranks["walk"], ranks


@pytest.mark.parametrize("shader_type", ["domain_warp", "tensor_field", "curl_noise",
                                         "temporal_coherent", "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"])
def test_jump_collapses_every_generator_not_just_the_narrow_ones(shader_type):
    """
    jump has to mean the same thing whatever shader is chosen, otherwise
    shader_type is not a usable coordinate in jump-space. tensor_field natively
    spans 90 of 128 channels and must still come back at rank 1.
    """
    shape = (1, 24, 5, 16, 16)
    noise = shader_noise.generate(shape, PARAMS, shader_type, 8888, CPU,
                                  decorrelate=True, basis=basis_for("jump"))
    assert shader_noise.effective_channel_rank(noise) == pytest.approx(1.0, abs=0.01)


def test_jump_is_not_blocked_by_the_widening_guards():
    """
    The guards exist to stop a remix narrowing the noise. jump wants exactly that,
    so it must bypass them rather than be silently turned into a no-op.
    """
    shape = (1, 24, 5, 16, 16)
    stock = shader_noise.generate(shape, PARAMS, "tensor_field", 8888, CPU)
    jumped = shader_noise.generate(shape, PARAMS, "tensor_field", 8888, CPU,
                                   decorrelate=True, basis=basis_for("jump"))
    assert shader_noise.effective_channel_rank(stock) > 10
    assert shader_noise.effective_channel_rank(jumped) < 1.5
    assert not torch.equal(stock, jumped)


def test_an_unknown_mode_falls_back_rather_than_raising():
    assert basis_for("teleport") == TRAVEL_MODES[presets.DEFAULT_TRAVEL_MODE]
    assert is_collapse("jump") and not is_collapse("walk")


# --- presets --------------------------------------------------------------------------

def test_custom_changes_nothing():
    values = dict(shader_type="curl_noise", shader_strength=0.9, blend_mode="add",
                  travel_mode="drift", stage_progression="fine_to_coarse",
                  shape_type="rays", normalize_strength=False)
    assert apply_preset("custom", values) == values


def test_an_unknown_preset_is_treated_as_custom():
    """A workflow saved against a later version must still run here."""
    values = dict(shader_strength=0.42)
    assert apply_preset("hyperdrive", values) == values


@pytest.mark.parametrize("name", [n for n in PRESETS if n != "custom"])
def test_every_preset_sets_a_complete_consistent_bundle(name):
    """
    The point of a preset is that the settings agree with each other, so a partial
    one would leave a stale widget value in the middle of a tuned combination.
    """
    overrides = PRESETS[name]
    assert set(overrides) == set(presets.PRESET_KEYS), name
    assert overrides["travel_mode"] in TRAVEL_MODES
    assert 0.0 <= overrides["shader_strength"] <= 1.0
    assert overrides["normalize_strength"] is True, "strengths are quoted on one scale"
    assert name in presets.PRESET_DESCRIPTIONS


def test_a_preset_overrides_the_widgets_it_names():
    got = apply_preset("explore", dict(shader_strength=0.99, blend_mode="add", seed=7))
    assert got["shader_strength"] == PRESETS["explore"]["shader_strength"]
    assert got["blend_mode"] == PRESETS["explore"]["blend_mode"]
    assert got["seed"] == 7, "inputs a preset does not name are left alone"


def test_exclusion_protects_a_walked_parameter():
    """Otherwise a preset that pins strength turns a strength ramp into one frame."""
    got = apply_preset("explore", dict(shader_strength=0.77), exclude=("shader_strength",))
    assert got["shader_strength"] == 0.77


def test_jump_and_stamp_actually_jump():
    assert PRESETS["jump"]["travel_mode"] == "jump"
    assert PRESETS["stamp"]["travel_mode"] == "jump"
    assert PRESETS["stamp"]["shape_type"] != "none", "stamp exists to draw the mask"


def test_the_node_offers_exactly_the_presets_that_exist():
    from snk.direct_shader_ksampler import DirectShaderNoiseKSampler

    spec = DirectShaderNoiseKSampler.INPUT_TYPES()
    offered = spec["optional"]["preset"][0]
    assert set(offered) == set(PRESETS)
    assert offered[0] == "custom", "the do-nothing option should read first"

    modes = spec["optional"]["travel_mode"][0]
    assert set(modes) == set(TRAVEL_MODES)
    assert spec["optional"]["travel_mode"][1]["default"] == presets.DEFAULT_TRAVEL_MODE


def test_the_frontend_is_served_the_table_the_node_applies():
    """The panel writes widgets from this route; a second copy of the table would drift."""
    import asyncio
    import json

    from snk.api_routes import get_presets

    served = json.loads(asyncio.run(get_presets(None)).text)
    assert served["presets"] == PRESETS
    assert served["descriptions"] == presets.PRESET_DESCRIPTIONS
    assert served["keys"] == list(presets.PRESET_KEYS)
