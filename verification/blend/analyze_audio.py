"""
Measure what shader noise did to H3's audio stream, from runs recorded by drive.py.

    python verification/blend/analyze_audio.py MANIFEST [MANIFEST ...] [--json FILE]

Every run is read against the strength-0 control for its own prompt and seed, because
level and brightness vary far more between prompts than between strengths.

H3 denoises both streams in one packed sequence, so a shader painted only on the
picture reaches the sound through joint attention anyway. Runs of both arms -- the
same strengths with and without shade_non_spatial -- therefore belong in one manifest,
and the summary keeps them apart: only the gap between the two arms is what painting
the audio latent itself did.

  rms dB     loudness, over both channels, so it does not move with stereo width
  centroid   spectral centre of mass in Hz: how bright the sound is
  flatness   geometric over arithmetic mean of the power spectrum, per frame. A pure
             tone reads 0; white noise reads about 0.56, not 1, because a single frame's
             periodogram is exponentially distributed. A fall means the model was pushed
             toward tonal content and away from broadband texture
  L/R corr   correlation between the two channels. The shader reads the audio latent's
             stereo pair as the height of a 2 x T grid, and create_coordinate_grid
             samples height 2 at linspace(0, 1, 2) -- the two opposite edges of the
             field -- so the two channels are drawn as far apart as the pattern allows.
             If that matters, this falls as strength rises.
  env Hz     the dominant rate of the amplitude envelope, with the peak's height over
             the band median in brackets. Unmodulated noise stays under about 5 at these
             clip lengths, while a shallow 30% modulation already reads near 50 -- so
             under 5 the rate is whatever noise won and means nothing. This is the
             measure that separates the shader imposing structure over time from it
             only recolouring timbre.

Spectral measures are taken per channel and averaged, so stereo decorrelation cannot
leak into them.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

FRAME, HOP = 2048, 512
# The envelope is read in windows far longer than the hop, so the carrier's own ripple
# is smoothed away instead of aliasing down into the band. ENV_WINDOW is 32 ms and the
# hop leaves 125 Hz of envelope, well above twice the top of the band. The band itself
# is the range a listener hears as pulse or swell rather than as pitch or as one fade.
ENV_WINDOW, ENV_HOP = 1024, 256
ENV_BAND = (0.5, 20.0)


def framed(x, size, hop):
    count = 1 + max(0, (len(x) - size) // hop)
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return x[index] * np.hanning(size)


def spectral(x, rate):
    """Centroid in Hz and spectral flatness, averaged over frames."""
    power = np.abs(np.fft.rfft(framed(x, FRAME, HOP), axis=-1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    centroid = (power * freqs).sum(axis=-1) / power.sum(axis=-1)
    flatness = np.exp(np.log(power).mean(axis=-1)) / power.mean(axis=-1)
    return float(centroid.mean()), float(flatness.mean())


def envelope_rate(x, rate):
    """Dominant modulation rate in Hz, and how far its peak stands above the band."""
    env = np.sqrt((framed(x, ENV_WINDOW, ENV_HOP) ** 2).mean(axis=-1))
    if len(env) < 8:
        return 0.0, 0.0
    level = env.mean()
    if level <= 0:
        return 0.0, 0.0
    # Against its own level, so the rate does not depend on how loud the run came out.
    env = env / level - 1.0
    if env.std() < 1e-6:  # a steady envelope has no rate, and its peak would be noise
        return 0.0, 0.0
    magnitude = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), ENV_HOP / rate)
    band = (freqs >= ENV_BAND[0]) & (freqs <= ENV_BAND[1])
    if not band.any():
        return 0.0, 0.0
    magnitude, freqs = magnitude[band], freqs[band]
    peak = int(magnitude.argmax())
    return float(freqs[peak]), float(magnitude[peak] / (np.median(magnitude) + 1e-12))


def measure(path):
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    rms = float(np.sqrt((audio ** 2).mean()))
    per_channel = [spectral(audio[:, c], rate) for c in range(audio.shape[1])]
    rates = [envelope_rate(audio[:, c], rate) for c in range(audio.shape[1])]
    if audio.shape[1] == 2:
        left, right = audio[:, 0] - audio[:, 0].mean(), audio[:, 1] - audio[:, 1].mean()
        lr = float((left * right).sum() / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
    else:
        lr = 1.0
    return {
        "rms_db": 20 * np.log10(rms + 1e-12),
        "centroid": float(np.mean([c for c, _ in per_channel])),
        "flatness": float(np.mean([f for _, f in per_channel])),
        "lr_corr": lr,
        "env_hz": float(np.mean([h for h, _ in rates])),
        "env_peak": float(np.mean([p for _, p in rates])),
    }


# A control's env Hz is whatever noise won, so a change in it means nothing; the rate
# is only worth reading on its own, next to the peak that says whether it is real.
DELTAS = ("rms_db", "centroid", "flatness", "lr_corr")
ABSOLUTE = ("env_hz", "env_peak")


def line(label, m):
    return (f" {label:22s} {m['rms_db']:7.1f} {m['centroid']:10.0f} {m['flatness']:10.4f} "
            f"{m['lr_corr']:10.2f} {m['env_hz']:8.1f} ({m['env_peak']:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--json", default=None, help="write every measurement here as well")
    args = ap.parse_args()

    rows = {}
    for path in args.manifests:
        for row in common.read_manifest(path):
            rows.setdefault(row["name"], row)
    runs = [r for r in rows.values() if r.get("audio")]
    if not runs:
        raise SystemExit("no run in these manifests saved audio; only H3 does, and only since "
                         "drive.py started writing FLAC")

    measured = {}
    for row in runs:
        if not Path(row["audio"]).exists():
            raise SystemExit(f"{row['name']}: {row['audio']} is gone")
        measured[row["name"]] = measure(row["audio"])

    groups = defaultdict(list)
    for row in runs:
        groups[(row.get("prompt_kind", "forge"), row["seed"])].append(row)

    deltas = defaultdict(list)
    for (prompt, seed), members in sorted(groups.items()):
        members.sort(key=lambda r: (r["strength"], r["name"]))
        controls = [r for r in members if r["strength"] == 0]
        if not controls:
            raise SystemExit(f"{prompt}, seed {seed}: no strength-0 control to read against")
        control = measured[controls[0]["name"]]
        print(f"\n {prompt}, seed {seed}" + " " * 9 + "rms dB   centroid   flatness   L/R corr   env Hz (peak)")
        for row in members:
            m = measured[row["name"]]
            painted = "audio" if row.get("shade_non_spatial") else "picture only"
            print(line(f"{row['strength']:<5g} {painted}", m))
            if row["strength"] != 0:
                summary = {k: m[k] - control[k] for k in DELTAS}
                summary.update({k: m[k] for k in ABSOLUTE})
                deltas[(prompt, row["strength"], bool(row.get("shade_non_spatial")))].append(summary)

    if deltas:
        print("\n averaged over seeds: change from each one's own control, and the rate as it stands")
        print(" prompt     strength  painted   n    d rms dB  d centroid  d flatness  d L/R corr   env Hz (peak)")
        means = {}
        for key, items in sorted(deltas.items()):
            prompt, strength, painted = key
            means[key] = {k: float(np.mean([d[k] for d in items])) for k in DELTAS + ABSOLUTE}
            m = means[key]
            print(f" {prompt:10s} {strength:<9g} {'audio' if painted else 'picture':9s} {len(items):<4d} "
                  f"{m['rms_db']:+8.1f} {m['centroid']:+11.0f} {m['flatness']:+11.4f} "
                  f"{m['lr_corr']:+11.2f} {m['env_hz']:8.1f} ({m['env_peak']:.1f})")

        paired = [k for k in means if k[2] and (k[0], k[1], False) in means]
        if paired:
            print("\n what painting the audio latent did, over painting the picture alone")
            print(" prompt     strength       d rms dB  d centroid  d flatness  d L/R corr    d env Hz")
            for key in sorted(paired):
                audio, picture = means[key], means[(key[0], key[1], False)]
                gap = {k: audio[k] - picture[k] for k in DELTAS + ABSOLUTE}
                print(f" {key[0]:10s} {key[1]:<14g} {gap['rms_db']:+8.1f} {gap['centroid']:+11.0f} "
                      f"{gap['flatness']:+11.4f} {gap['lr_corr']:+11.2f} {gap['env_hz']:+11.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(measured, indent=1) + "\n")
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
