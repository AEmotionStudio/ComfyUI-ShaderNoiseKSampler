"""
Travel modes and presets: the two ways to drive this node without reading
tooltips.

`TRAVEL_MODES` names what the channel rank of the shader noise actually does.
The generators fill every channel with a field of its own, so a draw spans nearly
all of them, and that width is what lets the shader steer instead of overwrite.
Narrowing it deliberately is the opposite and is useful for its own reasons, so
both are exposed rather than one being called correct.

`PRESETS` bundle the settings that have to agree with each other. A strength
value only means something alongside a blend mode, a travel mode and a shader
type; picking one at a time is how people end up at 0.6 with a shape mask and an
image made of hexagons.
"""

# Directions the shader noise is left spanning, per mode.
#
#   walk   the seed anchors and the shader perturbs around it. The generator's
#          own full width, nothing remixed: the wide coherent range.
#   drift  mixed down to four directions. Between the two, and the same as walk
#          on a four-channel latent, where four is all there is.
#   jump   the shader's parameters set the destination and the seed fades out.
#          Narrow coherent range, strong push per unit of strength, and the
#          result is a texture or pattern field in the prompt's material rather
#          than a scene -- the model is being handed noise it was never trained
#          to denoise.
TRAVEL_MODES = {
    "walk": 64,
    "drift": 4,
    "jump": 1,
}
DEFAULT_TRAVEL_MODE = "walk"

# A mode at or below this rank is a deliberate collapse, so the widening guards
# in core.shader_noise do not apply to it.
COLLAPSE_BASIS = 1

# Keys a preset may set. Anything absent is left on the user's own widget value.
PRESET_KEYS = (
    "shader_type", "shader_strength", "blend_mode", "travel_mode",
    "stage_progression", "shape_type", "normalize_strength",
)

# Values come from measured behaviour, not taste, and every preset turns
# normalize_strength on so its strength number means the same thing whatever blend
# mode it names.
#
# Calibrate these against a real prompt at a real working resolution. The first
# cut of `roam` was set to 0.60 from a sweep run with zero conditioning at
# 448x256, where domain_warp held to 0.75. Under an actual prompt at 608x352 it
# was already showing colour bands at 0.55, so the number did not survive the
# change of conditions it was measured under.
PRESETS = {
    "custom": {},
    "nudge": dict(
        shader_type="domain_warp", shader_strength=0.12, blend_mode="multiply",
        travel_mode="walk", stage_progression="uniform", shape_type="none",
        normalize_strength=True,
    ),
    "explore": dict(
        shader_type="domain_warp", shader_strength=0.30, blend_mode="multiply",
        travel_mode="walk", stage_progression="uniform", shape_type="none",
        normalize_strength=True,
    ),
    "roam": dict(
        shader_type="domain_warp", shader_strength=0.45, blend_mode="multiply",
        travel_mode="walk", stage_progression="coarse_to_fine", shape_type="none",
        normalize_strength=True,
    ),
    "video": dict(
        shader_type="temporal_coherent", shader_strength=0.35, blend_mode="multiply",
        travel_mode="walk", stage_progression="coarse_to_fine", shape_type="none",
        normalize_strength=True,
    ),
    "jump": dict(
        shader_type="domain_warp", shader_strength=0.70, blend_mode="multiply",
        travel_mode="jump", stage_progression="uniform", shape_type="none",
        normalize_strength=True,
    ),
    "stamp": dict(
        shader_type="domain_warp", shader_strength=0.90, blend_mode="multiply",
        travel_mode="jump", stage_progression="uniform", shape_type="hexgrid",
        normalize_strength=True,
    ),
}

PRESET_DESCRIPTIONS = {
    "custom": "your own widget values, nothing overridden",
    "nudge": "the smallest change that is still visible",
    "explore": "the recommended starting point",
    "roam": "further from the seed: the shader visibly reshapes the picture, the prompt still reads",
    "video": "tuned for video latents, using the 4D time-aware shader",
    "jump": "destination set by the shader instead of the seed; texture, not a scene",
    "stamp": "jump with a shape mask, so the mask is drawn in the prompt's material",
}


def basis_for(travel_mode: str) -> int:
    """Independent renders to mix, for a travel mode."""
    return TRAVEL_MODES.get(travel_mode, TRAVEL_MODES[DEFAULT_TRAVEL_MODE])


def is_collapse(travel_mode: str) -> bool:
    """True when the mode narrows the noise on purpose rather than widening it."""
    return basis_for(travel_mode) <= COLLAPSE_BASIS


def apply_preset(preset: str, values: dict, exclude=()) -> dict:
    """
    Overlay a preset onto the node's inputs.

    Returns a copy: "custom" hands back what it was given, and an unknown name is
    treated as custom rather than raising, so a workflow saved against a later
    version still runs here.

    `exclude` keeps named inputs on their own values. The walk node needs it: a
    preset that fixes shader_strength would otherwise flatten the very ramp the
    node exists to produce.
    """
    overrides = PRESETS.get(preset) or {}
    if not overrides:
        return dict(values)
    skip = set(exclude)
    return {**values,
            **{k: v for k, v in overrides.items() if k in PRESET_KEYS and k not in skip}}
