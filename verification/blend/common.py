"""
Settings shared by the blend measurements, so the driver and the analysis cannot
drift apart.

Every run holds the shader to one configuration and varies six things: the seed,
the strength, the travel mode, phase_shift and noise_scale for the "streets", and
the shader type. A strength-0 run has no shader, so it is shared by every type.

The prompt and whether the audio stream is painted are fixed for a whole matrix
rather than varied within one, but they still change the output, so run_name carries
them too.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (phase_shift, noise_scale) for every run that is not a street variant.
BASE = (0.5, 1.0)
# The streets: runs that differ from BASE in one shader setting only.
VARIANTS = ((0.0, 1.0), (1.0, 1.0), (0.5, 0.5), (0.5, 2.0))

SEEDS = {"sd15": (8888, 1234, 4242, 777), "h3": (8888, 1234, 4242), "krea2": (8888, 1234, 4242)}
SIZE = {"sd15": dict(width=512, height=512, length=1), "h3": dict(width=608, height=352, length=56),
        "krea2": dict(width=1024, height=1024, length=1)}

# The type every run used before there was a choice; manifests from then carry no
# shader column and are read as this.
DEFAULT_SHADER = "domain_warp"


def load_pack():
    """Import ComfyUI and this pack (as `snk`) the way the test suite does."""
    tests = str(REPO / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import helpers  # noqa: F401


def shader_inputs(seed, strength, travel, phase, scale, shader=DEFAULT_SHADER,
                  shade_non_spatial=False):
    """Direct node inputs shared by every run. Sampler settings are per model."""
    return {
        "seed": seed, "denoise": 1.0, "sequential_stages": 1, "injection_stages": 0,
        "shader_strength": strength, "blend_mode": "multiply", "noise_transform": "none",
        "use_temporal_coherence": False, "shader_type": shader, "shape_type": "none",
        "color_scheme": "none", "noise_scale": scale, "octaves": 2.0, "warp_strength": 0.7,
        "shape_mask_strength": 1.0, "phase_shift": phase, "color_intensity": 0.8,
        "sampling_mode": "standard", "preset": "custom", "travel_mode": travel,
        "normalize_strength": True, "stage_progression": "uniform",
        "shade_non_spatial": shade_non_spatial,
    }


def shader_of(strength, shader):
    """No shader reaches a strength-0 run, so every type shares it."""
    return "none" if strength == 0 else shader


def audio_of(strength, shade_non_spatial):
    """Same reason: with no shader to paint, the control is shared with the picture runs."""
    return bool(shade_non_spatial) and strength != 0


def run_key(row):
    return (row["seed"], row["strength"], row["travel"], row["phase"], row["scale"],
            shader_of(row["strength"], row.get("shader", DEFAULT_SHADER)))


def run_name(seed, strength, travel, phase, scale, shader=DEFAULT_SHADER,
             shade_non_spatial=False, prompt=None):
    """
    What a run is filed under. Anything that changes the output has to reach this
    name, or a rerun skips the run it already holds and one setting stands in for
    another. A setting every earlier run shared is left out, so those runs keep the
    names they were recorded under: the default shader type, the model's own prompt,
    and painting the picture alone.
    """
    # Two decimals unless that would merge distinct strengths, so 0.001 is not named 0.00.
    shown = f"{strength:.2f}" if round(strength, 2) == strength else f"{strength:g}"
    name = f"s{seed}_{travel}_{shown}_p{phase:.1f}_n{scale:.1f}"
    shader = shader_of(strength, shader)
    if shader not in ("none", DEFAULT_SHADER):
        name = f"{name}_{shader}"
    if prompt:
        name = f"{name}_{prompt}"
    if audio_of(strength, shade_non_spatial):
        name = f"{name}_audio"
    return name


def read_manifest(path):
    with open(path) as fh:
        return [dict(json.loads(line), _manifest=str(path)) for line in fh if line.strip()]
