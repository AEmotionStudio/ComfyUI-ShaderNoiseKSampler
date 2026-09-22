"""
Measure how the shader blends into a model's output, from runs recorded by drive.py.

    python verification/blend/analyze.py MANIFEST [MANIFEST ...] [--sheets DIR] [--json FILE]

Distances are read in towns: the mean distance between two seeds' strength-0 results.

  far       distance from the same seed's strength-0 result
  home      that distance over the mean distance to the other seeds' strength-0
            results; below 1, a run is nearer its own seed than the others
  nearest   how many seeds' runs are nearer their own strength-0 result than any other
  street    distance between runs that differ only in phase_shift or noise_scale
  overlap   correlation between a latent change and the shader field that caused it;
            "chance" is the largest such correlation with another seed's field

Latent distances use the saved latent (the video stream, for H3). Image distances use
the decoded image, or a video's middle frame: mean absolute RGB difference at 64x64.
Manifests can be combined; a run recorded twice is checked for determinism and the
first copy is used. Runs of several shader types are reported type by type, all read
against the same strength-0 controls, and a sheet per strength puts the types side by
side.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import field_capture  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from snk.core import shader_noise  # noqa: E402

BASE, VARIANTS = common.BASE, common.VARIANTS


def corr(x, y):
    dims = tuple(range(2, x.ndim))
    x = x - x.mean(dim=dims, keepdim=True)
    y = y - y.mean(dim=dims, keepdim=True)
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))


def pool(x):
    return F.avg_pool2d(x, 4) if x.ndim == 4 else F.avg_pool3d(x, (1, 2, 2))


class Runs:
    def __init__(self, manifests):
        self.rows, self.duplicates = {}, []
        for path in manifests:
            for row in common.read_manifest(path):
                key = common.run_key(row)
                if key in self.rows:
                    self.duplicates.append((self.rows[key], row))
                else:
                    self.rows[key] = row
        models = {r["model"] for r in self.rows.values()}
        if len(models) != 1:
            raise SystemExit(f"manifests must all be one model, got {sorted(models)}")
        self.model = models.pop()
        self.seeds = sorted({key[0] for key in self.rows})
        self.shaders = sorted({key[5] for key in self.rows} - {"none"})
        self._latents, self._images, self._fields = {}, {}, {}

    def has(self, key):
        return key in self.rows

    def latent(self, key, row=None):
        row = row or self.rows[key]
        if row["latent"] not in self._latents:
            self._latents[row["latent"]] = load_file(row["latent"])["latent_tensor"].float()
        return self._latents[row["latent"]]

    def picture(self, key):
        row = self.rows[key]
        if row.get("image"):
            return Image.open(row["image"]).convert("RGB")
        # One folder per manifest: runs in different manifests share names, and a
        # frame cached from one must never stand in for another's.
        frames = Path(row["_manifest"]).parent / f"frames_{Path(row['_manifest']).stem}"
        frames.mkdir(exist_ok=True)
        png = frames / f"{row['name']}.png"
        if not png.exists():
            length = row.get("length", common.SIZE[self.model]["length"])
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{1.15 * length / 56:.3f}",
                            "-i", row["video"], "-frames:v", "1", str(png)], check=True)
        return Image.open(png).convert("RGB")

    def image(self, key):
        if key not in self._images:
            small = self.picture(key).resize((64, 64), Image.BILINEAR)
            self._images[key] = np.asarray(small, dtype=np.float32) / 255
        return self._images[key]

    def field(self, key):
        if key not in self._fields:
            row = self.rows[key]
            size = {k: row.get(k, common.SIZE[self.model][k]) for k in ("width", "height", "length")}
            latent = field_capture.empty_latent(self.model, **size)
            field = field_capture.capture(self.model, latent, *key)[0].float()
            if tuple(field.shape) != tuple(self.latent(key).shape):
                raise SystemExit(f"{row['name']}: field {tuple(field.shape)} does not match latent "
                                 f"{tuple(self.latent(key).shape)}")
            self._fields[key] = field
        return self._fields[key]


def lat_dist(a, b):
    return float((a - b).norm())


def img_dist(a, b):
    return float(np.abs(a - b).mean())


def check_cpu_matches_cuda(model, shaders):
    """The fields are reconstructed on CPU; the server generated them on the GPU."""
    if not torch.cuda.is_available():
        return None
    params = {"scale": 1.0, "octaves": 2.0, "warp_strength": 0.7, "phase_shift": 0.5, "time": 0.0,
              "shape_type": "none", "color_scheme": "none"}
    shape = {"sd15": (1, 4, 32, 32), "krea2": (1, 16, 32, 32)}.get(model, (1, 24, 3, 22, 38))
    diffs = {}
    for shader in shaders:
        try:
            cpu = shader_noise.generate(shape, params, shader, 8888, torch.device("cpu"), decorrelate=True, basis=64)
            gpu = shader_noise.generate(shape, params, shader, 8888, torch.device("cuda"), decorrelate=True, basis=64)
        except RuntimeError as error:
            print(f"CPU vs CUDA field check skipped: {str(error)[:80]}")
            return None
        diffs[shader] = float((cpu - gpu.cpu()).abs().max())
        print(f"CPU vs CUDA field, {shader}: max abs diff {diffs[shader]:.1e}")
    return diffs


def sheet(runs, path, grid, col_labels, row_labels):
    tw, th = (150, 150) if runs.model in ("sd15", "krea2") else (200, 116)
    out = Image.new("RGB", (90 + tw * len(col_labels), 16 + (th + 4) * len(grid)), "white")
    draw = ImageDraw.Draw(out)
    for c, label in enumerate(col_labels):
        draw.text((90 + c * tw + 4, 2), label, fill="black")
    for r, (label, keys) in enumerate(zip(row_labels, grid)):
        draw.text((4, 16 + r * (th + 4) + th // 2), label, fill="black")
        for c, key in enumerate(keys):
            out.paste(runs.picture(key).resize((tw, th)), (90 + c * tw, 16 + r * (th + 4)))
    out.save(path)
    print("sheet", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--sheets", help="directory for contact sheets")
    ap.add_argument("--json", help="write the tables to this file")
    args = ap.parse_args()

    runs = Runs(args.manifests)
    model, seeds, shaders = runs.model, runs.seeds, runs.shaders
    if len(seeds) < 2:
        raise SystemExit("need at least two seeds: distances are read against other seeds")
    report = {"model": model, "seeds": seeds, "shaders": shaders, "drive": [], "road": [], "streets": [],
              "noise_scale": []}

    for first, second in runs.duplicates:
        a, b = runs.latent(None, first), runs.latent(None, second)
        same = torch.equal(a, b)
        print(f"recorded twice, {first['name']}: latents {'identical' if same else f'differ by up to {float((a - b).abs().max()):.2e}'}")
        report.setdefault("duplicates", []).append(dict(name=first["name"], identical=same))
    report["cpu_vs_cuda"] = check_cpu_matches_cuda(model, shaders)

    zero = {k: (k, 0.0, "walk", *BASE, "none") for k in seeds}
    missing = [k for k in seeds if not runs.has(zero[k])]
    if missing:
        raise SystemExit(f"no strength-0 run for seeds {missing}")
    L0 = {k: runs.latent(zero[k]) for k in seeds}
    I0 = {k: runs.image(zero[k]) for k in seeds}
    pairs = list(itertools.combinations(seeds, 2))
    town_l = float(np.mean([lat_dist(L0[a], L0[b]) for a, b in pairs]))
    town_i = float(np.mean([img_dist(I0[a], I0[b]) for a, b in pairs]))
    report.update(town_latent=town_l, town_image=town_i)
    print(f"\n{model}, seeds {seeds}: town = {town_l:.1f} (latent), {town_i:.3f} (image)")

    def key_for(k, s, travel, shader, phase=BASE[0], scale=BASE[1]):
        return zero[k] if s == 0 else (k, s, travel, phase, scale, shader)

    drive = sorted({(key[5], key[2], key[1]) for key in runs.rows if key[1] > 0 and key[3:5] == BASE},
                   key=lambda tk: (tk[0], tk[1] != "walk", tk[2]))
    print("\n shader             travel  strength | far latent/image | home latent/image | nearest latent/image | overlap own | chance")
    for shader, travel, s in drive:
        keys = {k: key_for(k, s, travel, shader) for k in seeds}
        if not all(runs.has(key) for key in keys.values()):
            continue
        acc = dict(lf=[], lh=[], ln=0, i_f=[], ih=[], i_n=0, own=[], chance=[])
        for k in seeds:
            L, I = runs.latent(keys[k]), runs.image(keys[k])
            own_l, oth_l = lat_dist(L, L0[k]), [lat_dist(L, L0[j]) for j in seeds if j != k]
            own_i, oth_i = img_dist(I, I0[k]), [img_dist(I, I0[j]) for j in seeds if j != k]
            acc["lf"].append(own_l / town_l); acc["lh"].append(own_l / np.mean(oth_l)); acc["ln"] += own_l < min(oth_l)
            acc["i_f"].append(own_i / town_i); acc["ih"].append(own_i / np.mean(oth_i)); acc["i_n"] += own_i < min(oth_i)
            delta = L - L0[k]
            acc["own"].append(corr(delta, runs.field(keys[k])))
            acc["chance"] += [abs(corr(delta, runs.field(keys[j]))) for j in seeds if j != k]
        row = dict(shader=shader, travel=travel, strength=s, far_latent=float(np.mean(acc["lf"])),
                   far_image=float(np.mean(acc["i_f"])), home_latent=float(np.mean(acc["lh"])),
                   home_image=float(np.mean(acc["ih"])), nearest_latent=acc["ln"], nearest_image=acc["i_n"],
                   overlap=float(np.mean(acc["own"])), chance=float(np.max(acc["chance"])))
        report["drive"].append(row)
        n = len(seeds)
        print(f" {shader:18s} {travel:6s}  {s:<6g}  |   {row['far_latent']:5.2f} / {row['far_image']:5.2f}  |"
              f"   {row['home_latent']:5.2f} / {row['home_image']:5.2f}  |      {row['nearest_latent']}/{n} / {row['nearest_image']}/{n}"
              f"        |   {row['overlap']:+.3f}    | {row['chance']:.3f}")

    # --- road: how big each step along walk strength is, per type ------------------------
    for shader in shaders:
        walk = sorted({key[1] for key in runs.rows if key[2] == "walk" and key[3:5] == BASE and key[5] == shader
                       and all(runs.has((k, key[1], "walk", *BASE, shader)) for k in seeds)})
        walk = [0.0] + walk
        if len(walk) > 1:
            print(f"\n road, {shader}: step between neighbouring walk strengths | latent / image, in towns")
        for a, b in zip(walk, walk[1:]):
            step_l = float(np.mean([lat_dist(runs.latent(key_for(k, b, "walk", shader)), runs.latent(key_for(k, a, "walk", shader))) / town_l
                                    for k in seeds]))
            step_i = float(np.mean([img_dist(runs.image(key_for(k, b, "walk", shader)), runs.image(key_for(k, a, "walk", shader))) / town_i
                                    for k in seeds]))
            report["road"].append(dict(shader=shader, start=a, end=b, latent=step_l, image=step_i))
            print(f"  {a:g} -> {b:g}: {step_l:.2f} / {step_i:.2f}")

    complete = {}
    for shader in shaders:
        street_strengths = sorted({key[1] for key in runs.rows if key[3:5] != BASE and key[5] == shader})
        complete[shader] = [s for s in street_strengths
                            if all(runs.has((k, s, "walk", p, n, shader)) for k in seeds for p, n in VARIANTS + (BASE,))]
    if any(complete.values()):
        print("\n shader             strength  variant    | street latent/image | overlap own | chance")
    for shader in shaders:
        for s in complete[shader]:
            for phase, scale in VARIANTS:
                lat_d, img_d, own, chance = [], [], [], []
                for k in seeds:
                    kb, kv = (k, s, "walk", *BASE, shader), (k, s, "walk", phase, scale, shader)
                    lat_d.append(lat_dist(runs.latent(kv), runs.latent(kb)) / town_l)
                    img_d.append(img_dist(runs.image(kv), runs.image(kb)) / town_i)
                    change = runs.latent(kv) - runs.latent(kb)
                    own.append(corr(change, runs.field(kv) - runs.field(kb)))
                    chance += [abs(corr(change, runs.field((j, s, "walk", phase, scale, shader))
                                        - runs.field((j, s, "walk", *BASE, shader)))) for j in seeds if j != k]
                label = f"phase {phase:.1f}" if scale == BASE[1] else f"scale {scale:.1f}"
                row = dict(shader=shader, strength=s, variant=label, street_latent=float(np.mean(lat_d)),
                           street_image=float(np.mean(img_d)), overlap=float(np.mean(own)), chance=float(np.max(chance)))
                report["streets"].append(row)
                print(f" {shader:18s} {s:5.2f}   {label:10s} |     {row['street_latent']:5.2f} / {row['street_image']:5.2f}     |"
                      f"   {row['overlap']:+.3f}    | {row['chance']:.3f}")

    if any(complete.values()):
        print("\n shader             strength  noise_scale | imprint own (pooled) | chance | home image | nearest image")
    for shader in shaders:
        for s in complete[shader]:
            for scale in (0.5, 1.0, 2.0):
                own, chance, home, nearest = [], [], [], 0
                for k in seeds:
                    key = (k, s, "walk", BASE[0], scale, shader)
                    delta = pool(runs.latent(key) - L0[k])
                    own.append(corr(delta, pool(runs.field(key))))
                    chance += [abs(corr(delta, pool(runs.field((j, s, "walk", BASE[0], scale, shader))))) for j in seeds if j != k]
                    I = runs.image(key)
                    own_i, oth_i = img_dist(I, I0[k]), [img_dist(I, I0[j]) for j in seeds if j != k]
                    home.append(own_i / np.mean(oth_i)); nearest += own_i < min(oth_i)
                row = dict(shader=shader, strength=s, noise_scale=scale, imprint=float(np.mean(own)),
                           chance=float(np.max(chance)), home_image=float(np.mean(home)), nearest_image=nearest)
                report["noise_scale"].append(row)
                print(f" {shader:18s} {s:5.2f}      {scale:4.1f}    |       {row['imprint']:+.3f}         | {row['chance']:.3f}  |"
                      f"    {row['home_image']:.2f}    |     {nearest}/{len(seeds)}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    if args.sheets:
        out = Path(args.sheets)
        out.mkdir(parents=True, exist_ok=True)
        for shader in shaders:
            cols = [("walk", 0.0)] + [(t, s) for sh, t, s in drive if sh == shader
                                      and all(runs.has(key_for(k, s, t, shader)) for k in seeds)]
            sheet(runs, out / f"{model}_{shader}_drive.png",
                  [[key_for(k, s, t, shader) for t, s in cols] for k in seeds],
                  [f"{t} {s:g}" for t, s in cols], [f"seed {k}" for k in seeds])
            for s in complete[shader]:
                sheet(runs, out / f"{model}_{shader}_streets_{s:.2f}.png",
                      [[(k, s, "walk", *BASE, shader)] + [(k, s, "walk", p, n, shader) for p, n in VARIANTS] for k in seeds],
                      ["base"] + [f"phase {p:.1f}" if n == BASE[1] else f"scale {n:.1f}" for p, n in VARIANTS],
                      [f"seed {k}" for k in seeds])
        if len(shaders) > 1:
            # The types side by side: one sheet per walk strength, a row per type.
            for s in sorted({st for sh, t, st in drive if t == "walk"}):
                rows = [sh for sh in shaders if all(runs.has((k, s, "walk", *BASE, sh)) for k in seeds)]
                if rows:
                    sheet(runs, out / f"{model}_types_{s:.2f}.png",
                          [[zero[k] for k in seeds]] + [[(k, s, "walk", *BASE, sh) for k in seeds] for sh in rows],
                          [f"seed {k}" for k in seeds], ["strength 0"] + rows)


if __name__ == "__main__":
    main()
