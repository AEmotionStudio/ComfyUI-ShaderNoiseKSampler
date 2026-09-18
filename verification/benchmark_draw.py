"""
How long one shader-noise draw takes, and how wide it comes out.

    ~/ComfyUI/venv/bin/python verification/benchmark_draw.py
    ~/ComfyUI/venv/bin/python verification/benchmark_draw.py --save after.json
    ~/ComfyUI/venv/bin/python verification/benchmark_draw.py --baseline after.json

There was no timing harness before this one, and the only per-draw number in the
repo -- HANDOFF's "about a second per draw" -- was measured before the generators
started filling their own channel axis, so it understated the real cost by roughly
an order of magnitude at video shapes. Numbers that nobody can reproduce go stale
without anyone noticing.

**Do not raise the thread count.** This box has 8 logical cores over 4 physical
ones, and forcing 8 threads makes the draw about 2.5x SLOWER -- the renders are
small enough that thread dispatch dominates. torch's default is what ComfyUI runs
with, so that is what this measures. --threads exists to demonstrate the trap, not
to tune around it.

The draw runs on the CPU in a real run: pipelines/standard.py takes its device
from the latent, and comfy's intermediate_device() is the CPU unless ComfyUI was
started with --gpu-only.

Reference run, 4 threads, torch 2.x, Ryzen-class 8-thread CPU, after Tier 0
(commit that added this file), domain_warp at octaves 3.0:

    H3 608x352/56f   walk 1.13s  drift 1.31s  jump 0.05s
    H3 default       walk 5.08s  drift 5.97s  jump 0.22s

The same shapes before Tier 0 were walk 1.38 / 5.70, drift 1.61 / 6.64,
jump 1.44 / 5.99.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_pack():
    """Import ComfyUI and this pack (as `snk`) the way the test suite does."""
    tests = str(REPO / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import helpers  # noqa: F401


# Named for the models they come from, so a regression can be read as "H3 got slower".
SHAPES = {
    "sd15 512x512": (1, 4, 64, 64),
    "wan 16ch/16f": (1, 16, 16, 60, 104),
    "h3 608x352/56f": (1, 24, 17, 22, 38),
    "h3 1344x768/124f": (1, 24, 37, 48, 84),
    "h3 audio 124f": (1, 32, 2, 207),
    "ltxv 128ch": (1, 128, 3, 32, 32),
}

PARAMS = {
    "scale": 1.0, "warp_strength": 0.5, "phase_shift": 0.5, "color_intensity": 0.8,
    "color_scheme": "none", "shape_type": "none", "shape_mask_strength": 1.0, "time": 0.0,
}


def measure(generate, rank_of, shape, shader_type, mode, octaves, basis, repeats):
    import torch

    params = dict(PARAMS, octaves=octaves)
    device = torch.device("cpu")
    kwargs = dict(decorrelate=True, basis=basis, allow_sequence=len(shape) == 3)

    noise = generate(shape, params, shader_type, 8888, device, **kwargs)  # warm
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        noise = generate(shape, params, shader_type, 8888, device, **kwargs)
        best = min(best, time.perf_counter() - start)

    return best, rank_of(noise)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--threads", type=int, help="override torch's default; see the module docstring")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--shapes", nargs="*", default=None, help="substrings of the shape names")
    ap.add_argument("--shaders", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=["walk", "drift", "jump"])
    ap.add_argument("--octaves", nargs="*", type=float, default=[3.0],
                    help="a fractional value renders the stack twice; 3.4 shows what roam pays")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--baseline", type=Path, help="a --save file to diff against")
    ap.add_argument("--profile", action="store_true", help="cProfile one draw instead of timing all")
    args = ap.parse_args()

    load_pack()
    import torch
    from snk.core.shader_noise import effective_channel_rank, generate
    from snk.core import presets
    from snk.shaders.registry import list_shaders

    if args.threads:
        torch.set_num_threads(args.threads)

    shapes = {k: v for k, v in SHAPES.items()
              if not args.shapes or any(s in k for s in args.shapes)}
    shaders = args.shaders or sorted(list_shaders())

    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, "
          f"best of {args.repeats}, cpu")
    if args.threads:
        print("!! --threads overrides ComfyUI's default; see the module docstring")

    if args.profile:
        import cProfile, pstats
        shape = next(iter(shapes.values()))
        pr = cProfile.Profile()
        pr.enable()
        generate(shape, dict(PARAMS, octaves=args.octaves[0]), shaders[0], 8888,
                 torch.device("cpu"), decorrelate=True, basis=presets.basis_for(args.modes[0]))
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(20)
        return

    baseline = json.loads(args.baseline.read_text()) if args.baseline else {}
    results = {}
    header = f"{'shape':18s} {'shader':18s} {'mode':6s} {'oct':>4s} {'sec':>8s} {'rank':>7s}"
    print(header + ("   vs base" if baseline else ""))
    print("-" * (len(header) + (10 if baseline else 0)))
    for name, shape in shapes.items():
        for shader_type in shaders:
            for mode in args.modes:
                for octaves in args.octaves:
                    key = f"{name}|{shader_type}|{mode}|{octaves}"
                    seconds, rank = measure(generate, effective_channel_rank, shape,
                                            shader_type, mode, octaves,
                                            presets.basis_for(mode), args.repeats)
                    results[key] = {"seconds": seconds, "rank": rank, "channels": shape[1]}
                    delta = ""
                    if key in baseline:
                        was = baseline[key]["seconds"]
                        delta = f"   {was / seconds:5.2f}x" if seconds else ""
                    print(f"{name:18s} {shader_type:18s} {mode:6s} {octaves:4.1f} "
                          f"{seconds:8.2f} {rank:7.2f}{delta}")

    if args.save:
        args.save.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
