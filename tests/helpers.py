"""
Shared test helpers.

Importing this module loads the node pack as the package ``snk`` inside
ComfyUI's Python environment (no server needed). Run the suite with
ComfyUI's venv, e.g.

    ~/ComfyUI/venv/bin/python -m pytest ~/ComfyUI/custom_nodes/comfyui-shadernoiseksampler/tests

Set COMFYUI_ROOT when the pack is not installed under ComfyUI/custom_nodes.

Set SNK_TEST_CPU=1 to pin the run to the CPU. Importing comfy builds a CUDA
context to read total VRAM, which fails with "CUDA-capable device(s) is/are
busy or unavailable" when a ComfyUI server on the same box is holding the card.
Nothing in the suite needs the GPU, so this makes it runnable while you work.
"""
import contextlib
import importlib.util
import os
import sys
import types
from unittest import mock

import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
COMFY_ROOT = os.environ.get("COMFYUI_ROOT", os.path.dirname(os.path.dirname(REPO_DIR)))

if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

if os.environ.get("SNK_TEST_CPU"):
    # comfy.cli_args parses an empty argv under pytest (comfy.options.args_parsing
    # is False), so --cpu on the command line is ignored. Set the flag directly,
    # before comfy.model_management is first imported and reads it.
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def load_pack():
    """Import the pack (its folder name has hyphens) under the importable name ``snk``."""
    if "snk" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "snk", os.path.join(REPO_DIR, "__init__.py"), submodule_search_locations=[REPO_DIR]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["snk"] = module
        spec.loader.exec_module(module)
    return sys.modules["snk"]


load_pack()

import comfy.latent_formats  # noqa: E402  (needs COMFY_ROOT on sys.path)
import comfy.model_base  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.sample  # noqa: E402


class _EpsSampling(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS):
    pass


class _FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class _AVSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    """How comfy.model_base.model_sampling composes ModelType.FLOW_AV."""


# MiniMax H3's own shifts, so audio_scale is the real 12.0 / 3.0 = 4.0.
_H3_CONFIG = types.SimpleNamespace(sampling_settings={"shift": 12.0, "audio_shift": 3.0})


class _H3LatentSpace(comfy.model_base.MiniMaxH3):
    """
    MiniMaxH3's process_latent_in/out without the diffusion network.

    Only these two methods are wanted: they are the reason a model's latent space
    is not interchangeable with its latent format's.
    """

    def __init__(self, latent_format, model_sampling, latent_shapes):
        torch.nn.Module.__init__(self)
        self.latent_format = latent_format
        self.model_sampling = model_sampling
        self.latent_shapes = latent_shapes


class FakeModel:
    """
    Minimal ModelPatcher stand-in. It carries the real model_sampling and
    latent_format objects that ComfyUI and the pipelines read, so noise math
    is exercised exactly; only the diffusion network itself is missing.

    ``kind="av"`` stands in for MiniMax H3: a MiniMaxH3AV format, a dual-shift
    ModelSamplingAV, and the model's own process_latent_in/out, which carry the
    audio stream at audio_scale.
    """

    #: per-stream shapes of the AV latent the "av" kind expects, video first
    AV_SHAPES = [(1, 24, 7, 12, 12), (1, 32, 2, 40)]

    def __init__(self, kind="eps"):
        if kind == "eps":
            self.sampling, self.latent_format = _EpsSampling(), comfy.latent_formats.SD15()
        elif kind == "flow":
            self.sampling, self.latent_format = _FlowSampling(), comfy.latent_formats.Wan21()
        elif kind == "av":
            self.sampling = _AVSampling(_H3_CONFIG)
            self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.kind = kind
        if kind == "av":
            self.model = _H3LatentSpace(self.latent_format, self.sampling, self.AV_SHAPES)
        else:
            self.model = types.SimpleNamespace(
                latent_format=self.latent_format,
                model_sampling=self.sampling,
                process_latent_in=self.latent_format.process_in,
                process_latent_out=self.latent_format.process_out,
            )
        self.load_device = torch.device("cpu")
        self.model_options = {}

    def get_model_object(self, name):
        return {
            "model_sampling": self.sampling,
            "latent_format": self.latent_format,
            "process_latent_in": self.model.process_latent_in,
            "process_latent_out": self.model.process_latent_out,
        }[name]

    def empty_latent(self):
        """A zeroed latent of the shape this kind samples, nested for "av"."""
        if self.kind == "av":
            from comfy.nested_tensor import NestedTensor
            return NestedTensor(tuple(torch.zeros(s) for s in self.AV_SHAPES))
        return torch.zeros(1, self.latent_format.latent_channels, 16, 16)


def snapshot(value):
    """Copy tensors (and NestedTensor streams) so later in-place ops can't alter a recording."""
    if value is None:
        return None
    if getattr(value, "is_nested", False):
        return [t.detach().clone() for t in value.unbind()]
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


# Defaults of ShaderParamsReader.get_shader_params() when no params file exists.
# Tests pin these so an untracked data/shader_params.json can't change results.
DEFAULT_FILE_PARAMS = {
    "shader_type": "tensor_field",
    "visualization_type": 3,
    "scale": 1.0,
    "phase_shift": 0.0,
    "warp_strength": 0.5,
    "time": 0.0,
    "octaves": 3.0,
    "intensity": 0.8,
    "shapemaskstrength": 1.0,
    "shape_type": "none",
}


@contextlib.contextmanager
def recorded_sampling():
    """
    Replace comfy.sample.sample with a deterministic fake that records every
    call, and pin the params file to its defaults. Yields the list of calls.
    """
    calls = []

    def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                    denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, sigmas=None, callback=None,
                    disable_pbar=False, seed=None):
        calls.append({
            "model": type(model).__name__,
            "noise": snapshot(noise),
            "latent": snapshot(latent_image),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "disable_noise": disable_noise,
            "start_step": start_step,
            "last_step": last_step,
            "force_full_denoise": force_full_denoise,
            "noise_mask": snapshot(noise_mask),
            "sigmas": snapshot(sigmas),
            "seed": seed,
        })
        return latent_image * 0.5 + noise * 0.1

    patches = [mock.patch.object(comfy.sample, "sample", fake_sample)]
    for module_name in ("snk.direct_shader_ksampler", "snk.shader_noise_ksampler",
                        "snk.shader_noise_source"):
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
        if hasattr(module, "get_shader_params"):
            patches.append(mock.patch.object(module, "get_shader_params", lambda: dict(DEFAULT_FILE_PARAMS)))

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield calls


def assert_same(actual, expected, path="result"):
    """Recursive exact comparison of recorded structures (tensors must be bit-identical)."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected a tensor, got {type(actual).__name__}"
        assert actual.shape == expected.shape, f"{path}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        assert actual.dtype == expected.dtype, f"{path}: dtype {actual.dtype} != {expected.dtype}"
        if not torch.equal(actual, expected):
            diff = (actual.double() - expected.double()).abs().max().item()
            raise AssertionError(f"{path}: tensors differ (max abs diff {diff:.3g})")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict"
        assert set(actual) == set(expected), f"{path}: key mismatch {sorted(set(actual) ^ set(expected))}"
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{path}: expected a sequence"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
