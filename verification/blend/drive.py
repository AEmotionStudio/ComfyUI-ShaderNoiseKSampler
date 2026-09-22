"""
Run a blend-measurement matrix through a running ComfyUI server.

    python verification/blend/drive.py {sd15,h3,krea2} --label NAME [--strengths 0.25,0.5]
        [--street-strengths 0.5] [--jump 0.5] [--seeds 8888,1234] [--shader-types a,b] [--dry-run]

Strength 0 is always included for every seed: it is what every distance is read
against. Each run appends a line to <out>/<model>_<label>.jsonl with where its latent
and image or video landed, and a rerun skips what the manifest already holds.

The server has to have the pack loaded, SD 1.5, MiniMax H3 or Krea 2 installed
under the file names below, and nothing else queued: interleaving models makes it
swap them for every prompt.
"""
import argparse
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

import common

PORTRAIT = "a portrait photograph of an elderly fisherman, weathered face, natural window light"
FORGE = ("Close-up of a blacksmith hammering a glowing orange horseshoe on an anvil, "
         "bright sparks bursting with each strike, dim stone workshop lit by forge fire. "
         "Loud rhythmic metallic clangs of hammer on steel, crackling fire, faint bellows hiss.")
RIDER = ("A dramatic digital painting of a Persian woman in her 40s, with warm olive-brown skin, "
         "dark hair, and an amused smirking expression, seated atop a big red elephant.")


def sd15_graph(prefix, seed, strength, travel, phase, scale, shader, size):
    s = dict(common.shader_inputs(seed, strength, travel, phase, scale, shader), steps=20, cfg=7.0,
             sampler_name="euler", scheduler="normal",
             model=["1", 0], positive=["2", 0], negative=["3", 0], latent_image=["4", 0])
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": PORTRAIT}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "blurry, low quality, watermark"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": size["width"], "height": size["height"], "batch_size": 1}},
        "5": {"class_type": "ShaderNoiseKSamplerDirect", "inputs": s},
        "6": {"class_type": "SaveLatent", "inputs": {"samples": ["5", 0], "filename_prefix": prefix}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def h3_graph(prefix, seed, strength, travel, phase, scale, shader, size):
    s = dict(common.shader_inputs(seed, strength, travel, phase, scale, shader), steps=8, cfg=1.0,
             sampler_name="res_multistep", scheduler="simple",
             model=["2", 0], positive=["6", 0], negative=["7", 0], latent_image=["6", 1])
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 6.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": FORGE,
              "width": size["width"], "height": size["height"], "length": size["length"]}},
        "7": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "8": {"class_type": "ShaderNoiseKSamplerDirect", "inputs": s},
        # SaveLatent cannot write a nested latent, so keep the video stream.
        "9": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 0]}},
        "10": {"class_type": "SaveLatent", "inputs": {"samples": ["9", 0], "filename_prefix": prefix}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": 24.0, "audio": ["12", 0]}},
        "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": prefix, "format": "auto", "format.codec": "auto"}},
    }


def krea2_graph(prefix, seed, strength, travel, phase, scale, shader, size):
    # The user's own Krea 2 workflow, with the Direct node in the KSampler's place.
    s = dict(common.shader_inputs(seed, strength, travel, phase, scale, shader), steps=8, cfg=1.0,
             sampler_name="er_sde", scheduler="beta57",
             model=["1", 0], positive=["5", 0], negative=["6", 0], latent_image=["7", 0])
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "lustifyNSFWCheckpoint_v10Krea2_2996235.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "krea2UncensoredLLMCLIP_v10_int8.safetensors", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": RIDER}},
        "5": {"class_type": "ConditioningKrea2Rebalance", "inputs": {"conditioning": ["4", 0], "multiplier": 3.0,
              "per_layer_weights": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": size["width"], "height": size["height"], "batch_size": 1}},
        "8": {"class_type": "ShaderNoiseKSamplerDirect", "inputs": s},
        "9": {"class_type": "SaveLatent", "inputs": {"samples": ["8", 0], "filename_prefix": prefix}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


GRAPHS = {"sd15": (sd15_graph, PORTRAIT), "h3": (h3_graph, FORGE), "krea2": (krea2_graph, RIDER)}


def floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def plan(model, strengths, street_strengths, jump, seeds, shaders):
    """Every strength for every type, and the strength-0 control once per seed."""
    runs = []
    for seed in seeds:
        runs.append((seed, 0.0, "walk", *common.BASE, common.DEFAULT_SHADER))
        for shader in shaders:
            for k in sorted(set(strengths) - {0.0}):
                runs.append((seed, k, "walk", *common.BASE, shader))
            for k in jump:
                runs.append((seed, k, "jump", *common.BASE, shader))
            for k in street_strengths:
                if k not in strengths:
                    runs.append((seed, k, "walk", *common.BASE, shader))
                for phase, scale in common.VARIANTS:
                    runs.append((seed, k, "walk", phase, scale, shader))
    return runs


def submit(server, graph):
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(server + "/prompt", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]


def wait(server, prompt_id, output_dir):
    """Block until the prompt finishes; return its saved files by kind."""
    kinds = {".latent": "latent", ".png": "image", ".mp4": "video"}
    while True:
        time.sleep(1.5)
        history = json.load(urllib.request.urlopen(f"{server}/history/{prompt_id}", timeout=60))
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(entry["status"])[:2000])
        files = {}
        for node_output in entry.get("outputs", {}).values():
            for listing in node_output.values():
                for item in listing if isinstance(listing, list) else []:
                    if isinstance(item, dict) and "filename" in item:
                        kind = kinds.get(os.path.splitext(item["filename"])[1])
                        if kind:
                            files[kind] = os.path.join(output_dir, item.get("subfolder", ""), item["filename"])
        if files:
            return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(GRAPHS))
    ap.add_argument("--label", required=True, help="names the manifest and the output subfolder")
    ap.add_argument("--strengths", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--street-strengths", default="", help="strengths to run the street variants at")
    ap.add_argument("--jump", default="", help="strengths to also run with travel_mode jump")
    ap.add_argument("--seeds", default="", help="default: " + ", ".join(f"{m} {common.SEEDS[m]}" for m in common.SEEDS))
    ap.add_argument("--shader-types", default=common.DEFAULT_SHADER, help="comma-separated shader types to run")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-output", default=os.path.expanduser("~/ComfyUI/output"))
    ap.add_argument("--out", default=None, help="manifest directory (default: <comfy-output>/snk_measure)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or list(common.SEEDS[args.model])
    shaders = [s.strip() for s in args.shader_types.split(",") if s.strip()]
    runs = plan(args.model, floats(args.strengths), floats(args.street_strengths), floats(args.jump), seeds, shaders)
    out = Path(args.out or os.path.join(args.comfy_output, "snk_measure"))
    manifest = out / f"{args.model}_{args.label}.jsonl"
    done = {r["name"] for r in common.read_manifest(manifest)} if manifest.exists() else set()
    todo = [r for r in runs if common.run_name(*r) not in done]
    print(f"{args.model}: {len(runs)} runs planned, {len(todo)} to do -> {manifest}")
    if args.dry_run:
        for r in todo:
            print("  ", common.run_name(*r))
        return

    out.mkdir(parents=True, exist_ok=True)
    graph, prompt = GRAPHS[args.model]
    size = common.SIZE[args.model]
    for run in todo:
        name = common.run_name(*run)
        started = time.time()
        prefix = f"snk_measure/{args.model}/{args.label}/{name}"
        files = wait(args.server, submit(args.server, graph(prefix, *run, size)), args.comfy_output)
        seed, strength, travel, phase, scale, shader = run
        row = dict(name=name, model=args.model, seed=seed, strength=strength, travel=travel, phase=phase,
                   scale=scale, shader=common.shader_of(strength, shader), prompt=prompt, **size,
                   latent=files.get("latent"), image=files.get("image"), video=files.get("video"),
                   seconds=round(time.time() - started, 1))
        with open(manifest, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(name, row["seconds"], flush=True)
    print("DONE", manifest, flush=True)


if __name__ == "__main__":
    main()
