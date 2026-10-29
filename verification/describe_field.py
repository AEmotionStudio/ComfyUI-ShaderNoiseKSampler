"""
What kind of noise a shader type makes, measured on the noise alone.

    ~/ComfyUI/venv/bin/python verification/describe_field.py
    ~/ComfyUI/venv/bin/python verification/describe_field.py --shaders fractal waves --scale 4 --octaves 1
    ~/ComfyUI/venv/bin/python verification/describe_field.py --shapes "h3 1344" --blend normal

No model and no render: every number is a property of the field, or of the field
mixed into white noise the way the sampler receives it. That makes it the cheap
first look at a new type or a new setting, before spending renders on it.

Columns, each averaged over the seeds:

  modes     how many spatial Fourier modes the field really occupies, as a share
            of all of them (the participation ratio effective_channel_rank uses,
            taken over space instead of channels). White noise is near 1.
  <10 <25   share of the field's power below 10% and 25% of Nyquist.
  rank      effective channel rank, as a share of the latent's channels.
  frames    correlation between neighbouring frames, coherence off -> on. Clips only.
            --rate is temporal_rate: how far a held field evolves across the clip.
  clip      share of values sitting exactly on the +-1 clamp every generator ends in.
  creases   kurtosis of the field's Laplacian over the same for a surrogate with the
            identical spectrum and value distribution but scrambled phases. Near 1
            is a random-phase field; above it are edges, folds and filaments that
            the spectrum does not describe.
  dose      how far the mixed noise's spectrum sits from white, at each strength:
            0.5 * mean(l - 1 - ln l) over the mode variances l, the Gaussian KL
            divergence per latent element, x1000. Zero for white noise, and it
            grows as the field piles its power into few modes.

Under each table is one number for the blend mode rather than the type: how far
the mixed noise's local deviation follows the shader's value, as the largest over
the smallest across the shader's deciles at the middle strength. normal and add
read 1.0, flat. multiply reads about 1.4 whatever the type: it amplitude-modulates
the base noise by the field, which is what a shape mask does and a model draws.

What the dose is good for, and where it stops. Against the per-type movement
recorded on MiniMax H3 (res_multistep, coherence off) it ranks the types at a
Spearman of +0.92 to +0.94, and it explains why holding a pattern across frames
acts like a stronger shader: fewer temporal modes, higher dose. It saturates,
though -- it is capped at -0.5 * ln(1 - share) -- so at Krea 2's 128x128 latent
nine structured types land within 15% of each other and it cannot rank them.

Read it together with "<10", because it counts power the model does not read. waves
at noise_scale 4 and warp 0 is a few plane waves with nothing in the large scales:
"<10" 0.00 against domain_warp's 0.67, and a dose of 34 against 38. Rendered on
Krea 2 at strength 0.5 it moved the picture 0.56 town against domain_warp's 1.34,
about what gaussian does (0.45). The dose says how far the noise is from white;
"<10" says whether that distance is in the band a model takes its layout from, and
only that part moved the picture.

It is a mean per latent element, and that is not what carries across picture sizes.
The same Krea 2 prompt, seeds and settings (domain_warp at 0.5) moved the picture
0.55 town at 512x512 and 1.34 at 1024x1024, where the dose reads 24 and 38. But
doubling noise_scale at 1024 brings the dose back to 24.7, cell for cell the 512
field, and the picture still moved 1.15; and gaussian, dose 0, moved the same at
both sizes (0.49 and 0.45). What does order those runs is the dose summed over the
latent, four times larger for the same mean at twice the side: 1 : 4.1 : 6.3 against
0.55 : 1.15 : 1.34. That is three runs, so it is the reading to test next and not a
rule. What can be relied on: a strength that suits one size is a harder push at a
larger one, for a structured type, and raising noise_scale buys back little of it.

Defaults are the settings every recorded A/B table used -- noise_scale 1, octaves
2, warp 0.7, phase 0.5, walk, multiply with normalize_strength on -- so the numbers
sit beside HANDOFF's.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_draw import SHAPES, load_pack  # noqa: E402

load_pack()

from snk.core import noise_math, presets  # noqa: E402
from snk.core.shader_noise import effective_channel_rank, generate  # noqa: E402
from snk.direct_shader_ksampler import DirectShaderNoiseKSampler  # noqa: E402

CPU = torch.device("cpu")
SEEDS = (8888, 1234, 4242, 777)


def draw(shape, shader_type, seed, args, coherence=False):
    params = {
        "scale": args.scale, "octaves": args.octaves, "warp_strength": args.warp,
        "phase_shift": args.phase, "color_intensity": 0.8, "color_scheme": "none",
        "shape_type": "none", "shape_mask_strength": 1.0, "time": 0.0,
        "use_temporal_coherence": coherence, "time_rate": args.rate,
    }
    return generate(shape, params, shader_type, seed, CPU, temporal_coherence=coherence,
                    decorrelate=True, basis=presets.basis_for(args.travel)).double()


def spectrum(fields):
    """Mean power per Fourier mode over the last two axes, DC first."""
    power = torch.fft.fft2(fields).abs() ** 2 / (fields.shape[-2] * fields.shape[-1])
    return power.reshape(-1, *fields.shape[-2:]).mean(dim=0)


def spatial_stats(fields):
    """Mode share, and the power below 10% and 25% of Nyquist."""
    height, width = fields.shape[-2:]
    power = spectrum(fields)
    power[0, 0] = 0.0
    radius = (torch.fft.fftfreq(height).view(-1, 1) ** 2 + torch.fft.fftfreq(width).view(1, -1) ** 2).sqrt()
    weights = power / power.sum()
    modes = float(torch.exp(-(weights * weights.clamp_min(1e-300).log()).sum())) / (height * width - 1)
    return modes, float(weights[radius < 0.05].sum()), float(weights[radius < 0.125].sum())


def frame_correlation(clip):
    """Mean correlation between a frame and the next, over the whole clip."""
    earlier, later = clip[:, :, :-1].flatten(), clip[:, :, 1:].flatten()
    return float(torch.corrcoef(torch.stack([earlier, later]))[0, 1])


def _laplacian_kurtosis(fields):
    # Interior cells only: the field does not wrap, and its surrogate does.
    lap = (fields[..., 1:-1, :-2] + fields[..., 1:-1, 2:] + fields[..., :-2, 1:-1]
           + fields[..., 2:, 1:-1] - 4.0 * fields[..., 1:-1, 1:-1])
    lap = lap - lap.mean()
    return float((lap ** 4).mean() / (lap ** 2).mean() ** 2)


def creases(fields, seed):
    """Structure the spectrum does not hold: see the module docstring."""
    planes = fields.reshape(-1, *fields.shape[-2:])
    scramble = torch.randn(planes.shape, dtype=planes.dtype, generator=torch.Generator().manual_seed(seed))
    phases = torch.fft.fft2(scramble)
    phases = phases / phases.abs().clamp_min(1e-300)
    surrogate = torch.fft.ifft2(torch.fft.fft2(planes).abs() * phases).real
    # Give the surrogate the field's own values, in the surrogate's order, so a
    # clipped or skewed distribution cannot pass for structure.
    order = surrogate.flatten(1).argsort(dim=1).argsort(dim=1)
    surrogate = planes.flatten(1).sort(dim=1).values.gather(1, order).reshape(planes.shape)
    return _laplacian_kurtosis(planes) / _laplacian_kurtosis(surrogate)


def variance_modulation(mixed, shader):
    """Largest over smallest local deviation of the mix, across the shader's deciles."""
    mixed, shader = mixed.flatten(), noise_math.standardize(shader).flatten()
    edges = torch.quantile(shader[:: max(1, shader.numel() // 200000)], torch.linspace(0, 1, 11, dtype=shader.dtype))
    spread = [mixed[(shader >= low) & (shader <= high)].std() for low, high in zip(edges[:-1], edges[1:])]
    return float(max(spread) / min(spread))


def dose(mixes):
    """KL divergence from white per latent element, x1000, over every non-channel axis."""
    axes = tuple(range(2, mixes[0].ndim))
    power = torch.stack([torch.fft.fftn(m, dim=axes).abs() ** 2 / m[0, 0].numel() for m in mixes])
    samples = power.shape[0] * power.shape[1] * power.shape[2]
    variances = power.mean(dim=(0, 1, 2))
    # The mean of `samples` unit exponentials is not exactly 1, and the log reads the
    # shortfall as dose; this is what white noise itself would score.
    count = torch.tensor(float(samples), dtype=torch.float64)
    bias = 0.5 * (count.log() - torch.special.digamma(count))
    return 1000.0 * float(0.5 * (variances - 1.0 - variances.log()).mean() - bias)


def describe(shape, shader_type, args):
    clip = len(shape) == 5
    seeds = SEEDS[:args.seeds]
    fields = [draw(shape, shader_type, seed, args) for seed in seeds]
    bases = [torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
             for seed in seeds]
    standard = torch.stack([noise_math.standardize(f) for f in fields])

    modes, below_10, below_25 = spatial_stats(standard)
    row = {
        "modes": modes, "below_10": below_10, "below_25": below_25,
        "rank": sum(effective_channel_rank(f) for f in fields) / len(fields) / shape[1],
        "clip": float(torch.stack([(f.abs() >= 1.0).double().mean() for f in fields]).mean()),
        "creases": sum(creases(noise_math.standardize(f), seed) for f, seed in zip(fields, seeds)) / len(fields),
    }
    if clip:
        held = [draw(shape, shader_type, seed, args, coherence=True) for seed in seeds]
        row["frames"] = (sum(map(frame_correlation, fields)) / len(fields),
                         sum(map(frame_correlation, held)) / len(held))

    row["dose"] = []
    for strength in args.strengths:
        mixes = [noise_math.mix_noise(b, f, args.blend, strength, normalize=True) for b, f in zip(bases, fields)]
        row["dose"].append(dose(mixes))
        if strength == args.strengths[len(args.strengths) // 2]:
            row["var_mod"] = sum(variance_modulation(m, f) for m, f in zip(mixes, fields)) / len(fields)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--shapes", nargs="*", default=["h3 608", "krea2"], help="substrings of the shape names")
    ap.add_argument("--shaders", nargs="*", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="noise_scale")
    ap.add_argument("--octaves", type=float, default=2.0)
    ap.add_argument("--warp", type=float, default=0.7, help="warp_strength")
    ap.add_argument("--phase", type=float, default=0.5, help="phase_shift")
    ap.add_argument("--rate", type=float, default=1.0, help="temporal_rate")
    ap.add_argument("--travel", default="walk", choices=sorted(presets.TRAVEL_MODES))
    ap.add_argument("--blend", default="multiply", choices=noise_math.SUPPORTED_MODES)
    ap.add_argument("--strengths", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=len(SEEDS), choices=range(1, len(SEEDS) + 1))
    args = ap.parse_args()

    shapes = {k: v for k, v in SHAPES.items() if any(s in k for s in args.shapes)}
    # The node's list rather than the registry's, which also holds two aliases.
    shaders = args.shaders or DirectShaderNoiseKSampler.INPUT_TYPES()["required"]["shader_type"][0]
    print(f"noise_scale {args.scale}, octaves {args.octaves}, warp {args.warp}, phase {args.phase}, "
          f"rate {args.rate}, {args.travel}, {args.blend}, {args.seeds} seeds")

    doses = "  ".join(f"@{s:g}" for s in args.strengths)
    for name, shape in shapes.items():
        print(f"\n{name} {shape}")
        print(f"{'shader':18s} {'modes':>6s} {'<10':>5s} {'<25':>5s} {'rank':>5s} {'frames':>13s} "
              f"{'clip':>5s} {'creases':>7s}   dose {doses}")
        modulation = []
        for shader_type in shaders:
            row = describe(shape, shader_type, args)
            modulation.append(row["var_mod"])
            frames = "{:+.2f} -> {:+.2f}".format(*row["frames"]) if "frames" in row else "-"
            print(f"{shader_type:18s} {row['modes']:6.3f} {row['below_10']:5.2f} {row['below_25']:5.2f} "
                  f"{row['rank']:5.2f} {frames:>13s} {row['clip']:5.2f} {row['creases']:7.2f}   "
                  + "  ".join(f"{d:5.1f}" for d in row["dose"]))
        print(f"{args.blend}: local deviation follows the field x{sum(modulation) / len(modulation):.2f}")


if __name__ == "__main__":
    main()
