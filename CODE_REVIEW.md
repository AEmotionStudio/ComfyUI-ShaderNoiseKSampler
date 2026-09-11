# Code Review — ComfyUI-ShaderNoiseKSampler

*Reviewed: 2026-09-11 · Version 1.3.5 (web 1.0.4) · Working tree incl. the uncommitted NestedTensor change*
*Checked against the locally installed ComfyUI 0.31.0 source.*

> **Status — addressed in 2.0.0 (branch `quality/v2`).**
> Findings 1–6 and 8–12 are fixed in the new `standard` sampling mode; the old
> pipeline is preserved as `legacy` and verified **pixel-identical** across 12
> rendered configurations, so existing seeds still reproduce. Standard mode at
> `shader_strength=0` is pixel-identical to a stock KSampler.
> Finding 7 (the global params file) is resolved by the Direct node, which takes
> every shader parameter as a node input; the display node is deprecated.
> A crash not in this review was also found and fixed: `tensor_field` raised for
> 46 of 48 parameter combinations, including the node's own defaults.
> Deliberately left as-is: the 163 inert debug/visualiser call sites in finding
> 13 live in the now-frozen legacy module, where rewriting them would risk the
> bit-identical guarantee for no benefit. See CHANGELOG.md for the full list.

---

## TL;DR

The core idea is good: structured, controllable noise as a way to explore the area around one seed. The project around it is also well presented. You get an in-app "Shader Matrix", eight example workflows, a CHANGELOG, a strict TypeScript migration, and careful input sanitization.

**The default configuration works as advertised.** That's 1 sequential stage, 0 injection stages and `denoise = 1.0`, and it amounts to "blend shader noise into the initial noise, then run one normal sampling pass". That's the path most people use, and it's sound.

**Beyond the default, the sampling engine doesn't do what the docs describe:**

- **Multi-stage.** Each stage throws away most or all of the previous stage's result.
- **`denoise`.** It's silently ignored.
- **`custom_sigmas`.** The values are never used.
- **Noise statistics.** Most blend modes and transforms feed the sampler noise with the wrong mean or variance.

These are fixable. Most of the fixes make the code *smaller*, because they mean using ComfyUI's own machinery (`sigmas=`, the latent's own shape, `common_ksampler` conventions) instead of re-deriving it. Because the fixes change what existing seeds produce, ship them behind a "legacy" toggle or a clearly versioned change.

The rest is maintenance debt from a refactor that was never finished:
- a parallel `core/` package the nodes don't actually use
- no-op compatibility shims
- duplicate registries
- inputs that ComfyUI never delivers
- 334 `print` calls
- no Python tests

---

## How this review was done

- Read the Python sampling pipeline end to end: `shader_noise_ksampler.py`, `direct_shader_ksampler.py`, `shader_params_reader.py`, `api_routes.py`, `core/`, `shaders/`, `__init__.py`.
- Read the web save/render flow: `web/src/shader_params_save_button.ts`, `shader_renderer.ts`.
- Checked every claim about ComfyUI's behaviour against the installed source:
  - `comfy/sample.py`, `comfy/samplers.py`, `comfy/model_sampling.py`
  - `execution.py`, `nodes.py`, `server.py`, `comfy/nested_tensor.py`
- `tsc --noEmit`: **clean**. A fresh `tsc` build is **byte-identical** to the committed `web/*.js`.
- `pnpm test` (vitest 4.1.8): **4 files, 85 tests, all passing**.
- Ran a short torch script to measure what each blend mode and transform does to unit-Gaussian noise (results below).
- I did **not** run generations in ComfyUI. Findings about visual effects follow from the code paths and ComfyUI's math, and should be confirmed with a quick A/B test.

---

## What's good

- **Clear product thinking.** The README and the in-app Shader Matrix explain the concept well, and the example workflows cover SD1.5, SDXL, Flux, WAN, Hunyuan, LTXV and AnimateDiff.
- **The TypeScript migration is disciplined.** `strict` mode, a clean typecheck, 85 passing tests, and committed build output that exactly matches source.
- **Security was taken seriously.** `validate_and_sanitize_params` clamps octaves and seeds, rejects NaN/Inf, and whitelists enum strings. The API validates both camelCase and snake_case keys. The params reader has real path-traversal hardening.
- **Accessibility work** on the modal and canvases: `role` and `aria-label` attributes.
- **CHANGELOG discipline**, including a dependency-CVE entry.
- **The uncommitted NestedTensor change is a correct, well-scoped fix.** It fixes the crash in `errors.md` (`'NestedTensor' object has no attribute 'clone'`, from the old `latent_image["samples"].clone()`). Details:
  - `primary_stream` / `clone_latent` / `nested_noise_like` are small and clearly named.
  - Plain Gaussian noise for the companion (audio) streams is the right call.
  - `channels_from_latent` suppresses a warning that would be spurious for these latents.
  - Checking `latent_format` on both the patcher and `model.model` is the correct generic lookup.

  Commit it. See finding 6 for how much of that special-casing would disappear if the noise shape came from the latent.

---

## Findings

Severity reflects impact on generated output, not code style. Line numbers refer to the current working tree.

### 🔴 High — output correctness

#### 1. Stages don't split one diffusion trajectory. Each stage restarts from maximum noise.

`shader_ksampler` ([shader_noise_ksampler.py:454-471](shader_noise_ksampler.py#L454-L471)) calls `comfy.sample.sample(..., steps=stage_steps, denoise=1.0, start_step=0, last_step=steps)`. That builds a **fresh, complete** sigma schedule for every stage, starting at σ_max. The `start_step`/`end_step` ranges ([:1750](shader_noise_ksampler.py#L1750)) are only used to work out *how many* steps each stage gets. They never select a *segment* of a shared schedule.

The previous stage's latent is then passed as `latent_image` and noised from σ_max:

- **Flow models (Flux, SD3, WAN, Hunyuan, LTXV, …)** use `CONST.noise_scaling = σ·noise + (1−σ)·latent` (`comfy/model_sampling.py:94-97`). At σ = 1 the previous latent is multiplied by **zero**. With `sequential_stages = N`, the output is effectively a `steps/N`-step generation from the *last* stage's noise alone.
- **EPS models (SD1.5/SDXL)** compute `noise·√(1+σ²) + latent`, with σ_max ≈ 14.6. The previous stage survives only as roughly a 1/15 bias.

So "apply shader noise over segments of the diffusion process" (README) isn't what currently happens.

**Fix.** Chain stages the way two `KSamplerAdvanced` nodes chain: return with leftover noise, then continue without adding noise. `comfy.sample.sample` already accepts `sigmas=`:

```python
ks = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name,
                             scheduler=scheduler, denoise=denoise, model_options=model.model_options)
full = ks.sigmas                                   # respects denoise (see #2)
x = latent
for i, (a, b) in enumerate(ranges):
    noise = stage0_blended_noise if i == 0 else torch.zeros_like(x)
    x = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, x,
                            sigmas=full[a:b + 1], noise_mask=noise_mask, callback=cb, seed=stage_seed)
```

ComfyUI's `noise_scaling` at a segment start and `inverse_noise_scaling` at a segment end are consistent for both EPS and CONST, so zero-noise continuation is exact.

If later stages should *add* new shader structure instead of just continuing, swap part of the remaining noise at the boundary. Estimate it from the callback's `x0`, mix it with shader noise in a variance-preserving way (see #4), and rebuild `x`. Do this through `model_sampling` so it stays model-agnostic.

#### 2. `denoise` is ignored whenever `sequential_stages ≥ 1`, which is the default.

Both stage loops hard-code `stage_denoise_param_for_sampler = 1.0` ([:1768](shader_noise_ksampler.py#L1768), [:1872](shader_noise_ksampler.py#L1872)). The user's `denoise` only reaches the sampler on the zero-strength fallback, or on the "injection with no sequential stages" pre-pass.

**Impact.** img2img and video-to-video at `denoise < 1` are fully re-generated, not partially denoised. That very likely explains the README Known Issue that i2v/v2v results are "not as pronounced". It fixes itself once #1 builds the schedule with `denoise` (`KSampler.set_steps` handles it).

#### 3. Injection stages can end with a 1-step restart.

`_calculate_step_points` ([:626-636](shader_noise_ksampler.py#L626-L636)) and the range builder ([:2023-2030](shader_noise_ksampler.py#L2023-L2030)) produce, for example, these ranges:

| Settings | Ranges | Last stage |
|---|---|---|
| 20 steps, 3 stages | `(0,10) (10,19) (19,20)` | 1 step |
| 20 steps, 2 stages | `(0,19) (19,20)` | 1 step |

Combined with #1, the **final output is a single sampling step from maximum noise.**

Separately, with `sequential_stages = 0`, injection first runs a complete generation ([:1924-1949](shader_noise_ksampler.py#L1924-L1949)) and then re-samples it. That doubles the cost and, per #1, largely discards the first pass. The segmented design in #1 covers injection too: injection points become segment boundaries on the one schedule.

#### 4. Blended and transformed noise isn't re-normalized before sampling.

- Shader noise is standardized to mean 0, std 1 ([:1044-1054](shader_noise_ksampler.py#L1044-L1054)).
- `_apply_noise_transform` ([:638-660](shader_noise_ksampler.py#L638-L660)) and `_blend_noises` ([:1179-1217](shader_noise_ksampler.py#L1179-L1217)) are then applied.
- The result goes **straight** to the sampler ([:1827](shader_noise_ksampler.py#L1827) → [:1871](shader_noise_ksampler.py#L1871)).

The blend formulas are image-editing formulas written for data in [0, 1] (`shader - 0.5`, `base < 0.5`, `1 - (1-a)(1-b)`), but here they're applied to N(0, 1) noise. Diffusion samplers assume unit-variance, zero-mean initial noise. A mean offset shows up as a global colour or brightness cast; wrong variance shows up as washed-out or burnt, over-contrasted results.

Measured on independent unit-Gaussian inputs (1×4×128×128):

| Blend mode | strength 0.3: mean / std | strength 1.0: mean / std |
|---|---|---|
| normal | 0.00 / **0.76** | 0.00 / 1.00 |
| add | 0.00 / 1.04 | 0.00 / 1.42 |
| multiply *(default)* | 0.00 / 0.92 | 0.00 / **2.00** |
| screen | 0.00 / 1.09 | 0.00 / 1.74 |
| overlay | **+0.39** / 0.94 | **+0.40** / 1.75 |
| soft_light | **+0.30** / 1.44 | **+1.01 / 4.20** |
| hard_light | **+0.33** / 0.81 | **+0.39** / 1.76 |
| difference | **+0.34** / 1.03 | **+1.13** / 1.31 |

| Transform (on N(0,1)) | mean | std | max \|x\| |
|---|---|---|---|
| inverse (`1/(x+1e-8)`, [:646](shader_noise_ksampler.py#L646)) | +1.38 | **196** | **≈ 40 000** |
| absolute | +0.80 | 0.61 | 4.4 |
| square | +1.01 | 1.43 | 19.5 |
| sqrt | +0.82 | 0.35 | 2.1 |
| log | +0.54 | 0.32 | 1.7 |
| sin / cos | 0.00 | 0.71 | 1.0 |

This is a plausible source of the README warning about "flashing images or harsh artifacts". `inverse` in particular produces extreme outliers on any tensor large enough to contain near-zero values, which is every real latent.

**Fix.** Re-standardize after the transform and after the blend, per sample and per channel. Use a variance-preserving mix for `normal`:

```python
def standardize(x, dims):
    return (x - x.mean(dims, keepdim=True)) / x.std(dims, keepdim=True).clamp_min(1e-6)

theta = strength * math.pi / 2
mixed = base * math.cos(theta) + shader * math.sin(theta)   # std stays 1 for independent inputs
```

Clamp or remove `inverse`. The other modes can keep their character: the spatial structure of the shader noise survives standardization, which is the part you actually want.

#### 5. The `custom_sigmas` values are never used.

`CustomSigmaModelWrapper` ([:123-148](shader_noise_ksampler.py#L123-L148)) overrides a `model_sampling` attribute. But `KSampler.calculate_sigmas` reads `self.model.get_model_object("model_sampling")` (`comfy/samplers.py:1425`), and the wrapper's `__getattr__` forwards that call to the original `ModelPatcher`, so the model's own schedule is used. The custom tensor only affects how many steps run (`total_sampling_steps = len(sigmas) − 1`).

**Fix.** Delete the wrapper and pass `sigmas=custom_sigmas`, segmented per stage as in #1.

### 🟠 Medium

#### 6. Video layout guessing gets it wrong when latent frames equal latent channels.

For 5-D latents, `sample()` guesses between `[B,C,F,H,W]` and `[B,F,C,H,W]` by comparing dims to the model's channel count. When both match, it assumes `[B,F,C,H,W]` ([:1564-1572](shader_noise_ksampler.py#L1564-L1572)). ComfyUI video latents are **always** `[B, C, T, H, W]` (e.g. `comfy_extras/nodes_hunyuan.py:61`).

The ambiguous case is common. WAN and Hunyuan use `T = (length−1)//4 + 1`, so a **61-frame** video has 16 latent frames and 16 channels. The node then generates per-"frame" noise along the channel axis and stacks it there. Time evolution runs across channels instead of frames. Shapes still match, so nothing errors.

**Fix.** Derive the noise shape from the latent itself, after the same `comfy.sample.fix_empty_latent_channels(...)` call that `common_ksampler` makes (`nodes.py:1573`). Treat dim 1 as channels and dim 2 as time. Then you can delete:
- the ~170-line name/class heuristic in `get_model_channel_count` ([:662-829](shader_noise_ksampler.py#L662-L829)). Note that its debug-only branch returns *before* the authoritative `latent_format` check, so detection currently depends on the debug level.
- the channel/format guessing in `sample()`
- the noise-resizing fallbacks in `shader_ksampler`
- most of the NestedTensor special-casing, because the primary stream's shape is simply the shape

#### 7. A single global params file drives the display node.

The non-Direct node reads `data/shader_params.json` ([shader_params_reader.py:280-285](shader_params_reader.py#L280-L285)), written by the save button through [api_routes.py](api_routes.py). Consequences:

- **One file for everything.** Two display nodes in one workflow can't have different params, and neither can two browser tabs or two users.
- **Not portable.** Params aren't stored in the workflow, so a shared workflow or PNG metadata doesn't reproduce the result.
- **Not queueable.** You can't queue runs with different params; this is already a README Known Issue.
- **Lives in the install folder.** The file is written into the extension's install directory. It shows as `?? data/` in `git status`, and Manager updates or reinstalls can clobber it.

The frontend already has the values as widgets on the node ([web/src/shader_renderer.ts:1292-1324](web/src/shader_renderer.ts#L1292-L1324)).

**Fix.** Declare them as real (optional) inputs, exactly as the Direct node does, and let the shader display read and write those widgets. The file, the API route, the save button and the "🔄 requires saving" UX all go away. Alternatively, merge the two nodes into *Direct + optional live preview*. All eight example workflows already use the Direct node, which suggests that's the primary path anyway.

#### 8. Diverges from `common_ksampler` in ways users will notice.

Compare `shader_ksampler` with `nodes.py:1571-1593`:

- **No UI progress bar and no live latent preview.** The node's `callback` is `DenoisingStepCallback`, which returns immediately unless the visualizer is on, and that's never (see #9). Chain it with `latent_preview.prepare_callback(model, steps)`.
- **`noise_mask` is ignored**, so inpainting and masked-latent workflows are silently unmasked.
- **`batch_index` is read but unused** ([:436](shader_noise_ksampler.py#L436)).
- **No `fix_empty_latent_channels`.** Covered by #6.

#### 9. Several inputs are unreachable, and bugs hide behind them.

ComfyUI only fills V1 `hidden` inputs whose type is one of the special strings (`PROMPT`, `UNIQUE_ID`, `EXTRA_PNGINFO`, `DYNPROMPT`, …). Tuple-typed entries like `("BOOLEAN", {...})` are never delivered. Nothing in `web/src` supplies them either. So these are **always their defaults** ([:522-532](shader_noise_ksampler.py#L522-L532), [direct_shader_ksampler.py:81-88](direct_shader_ksampler.py#L81-L88)):

`sequential_distribution`, `injection_distribution`, `debug_level`, `save_visualizations`, `denoise_visualization_frequency`, `fast_high_channel_noise`, `target_attribute_changes`.

That means the whole visualizer and debugger path, the distribution curves and the "Parameter Response Mapper" can't be reached. Exposing them (as optional inputs) would surface these latent bugs:

- **`F` is undefined at module scope.** It's only imported locally inside the video branch ([:981](shader_noise_ksampler.py#L981)), but used at [:1037](shader_noise_ksampler.py#L1037) and [:1065](shader_noise_ksampler.py#L1065). Any image-shape correction raises `NameError`, the `except` falls back to `torch.zeros`, and the sampler gets **all-zero shader noise, silently**. Add `import torch.nn.functional as F` at the top.
- **`ParameterResponseMapper` is undefined in the base node** ([:1378](shader_noise_ksampler.py#L1378)). It's defined only in `direct_shader_ksampler.py`, and the `NameError` is swallowed at [:1422](shader_noise_ksampler.py#L1422).
- **`linear_increase` with one stage gives strength 0** in the class's copy ([:583-586](shader_noise_ksampler.py#L583-L586)), which silently disables the shader. The copy in [core/sampler.py:50-58](core/sampler.py#L50-L58) has the fix. The two have drifted (see #12).

### 🟡 Low

#### 10. Web save flow and API route

- [shader_params_save_button.ts:293-305](web/src/shader_params_save_button.ts#L293-L305) deletes **every localStorage key containing "shader"** except `shader_params`. That includes other extensions' keys. The saved `shader_params` value is also never read back (there's no `getItem` anywhere), so the whole localStorage block can go.
- [:318](web/src/shader_params_save_button.ts#L318) depends on `window.storageOptimizer`, a global from some other extension.
- The tooltip ([:378](web/src/shader_params_save_button.ts#L378)) still describes the old manual-download flow ("file must be named shader_params.json").
- [api_routes.py](api_routes.py) passes unknown keys through sanitization, so arbitrary JSON gets persisted and merged into `shader_params`. It also echoes `str(e)` in 500 responses. Whitelist the known keys. (Moot if #7 removes the route.)

#### 11. `IS_CHANGED` is redundant

For `IS_CHANGED`, ComfyUI passes linked inputs as `None` (`execution.py:176-179`). So [the tuple returned at :539-558](shader_noise_ksampler.py#L539-L558) contains only widget values, which are already part of the cache key. It doesn't include the one thing that actually changes outside the graph, the params file. Remove it from both nodes, or have the display node return a hash or mtime of the params file (moot after #7).

#### 12. Dead and duplicate code from the unfinished refactor

- **`core/sampler.py`, `core/blending.py`, `core/transforms.py`, `core/model_compat.py` are never used by the nodes**, which import only `core.debug`, `core.params` and `core.constants`. They're parallel re-implementations that have already drifted (see the `linear_increase` point in #9). Either finish the migration (`nodes/shader_ksampler.py` still says "will be refactored … in a future update") or delete them.
- **Top-level `advanced_comparer.py` is never imported.** `nodes/comparer.py` is the live copy.
- **`add_domain_warp_to_tensor`, `add_tensor_field_to_tensor`, `add_curl_noise_to_tensor` and `integrate_temporal_coherent_noise` are `pass` no-ops**, yet [__init__.py:74-77](__init__.py#L74-L77) calls them as if they did something.
- **Double registration.** Each generator registers itself through `@shader_generator` in `shaders/*.py`, and again in [__init__.py:48-71](__init__.py#L48-L71). That produces four `Overwriting existing shader generator` warnings on every startup (visible in `errors.md`).
- **`get_shader_generator` / `register_shader_generator` exist twice**, in [__init__.py:132-198](__init__.py#L132-L198) and [shader_noise_ksampler.py:154-206](shader_noise_ksampler.py#L154-L206), with slightly different behaviour.
- **These aren't ComfyUI APIs and do nothing:** `CONTEXT_MENUS`, `has_preview`, `REGISTER_MATRIX_BUTTON`, `__js_files__`.
  - ComfyUI loads every `web/**/*.js` recursively (`server.py:364`) in no guaranteed order, so the "ORDER IS CRITICAL" comment is misleading. Your 1.3.3 "load-order independent" fix was the right response to exactly this.
  - The same recursion also serves `web/types/*.js` and `*.js.map` as extensions. Move the type stubs outside `web/`.
- **Smaller items:**
  - `initial_noise` is generated but unused whenever `sequential_stages > 0`.
  - The `custom_path` path-traversal hardening in `get_shader_params` is never called with an argument.
  - `stage_denoise` is computed and logged but never used.

#### 13. Code hygiene

- **Global RNG.** Seeding goes through `torch.manual_seed(...)` everywhere. `torch.manual_seed(torch.seed())` ([:487](shader_noise_ksampler.py#L487), [:2167](shader_noise_ksampler.py#L2167)) doesn't "restore the original random state" as the comment says; it just reseeds randomly. Use local `torch.Generator` objects and leave global state alone.
- **Logging.** There are 334 `print(...)` calls against 7 `logging` calls; `core/logging_config.py` exists but is barely used. Seven lines print unconditionally on every run ([:1645-1651](shader_noise_ksampler.py#L1645-L1651)).
- **No Python tests.** The web side has 85 tests; the Python side has zero. The highest-value targets are pure functions:
  - stage strength and step-range math (would have caught #3 and the `linear_increase` drift)
  - blend and transform output statistics (#4)
  - noise-shape derivation for image, video and nested latents (#6)
  - `validate_and_sanitize_params`

  A `pytest` suite with a stub `comfy` package is enough for all of these.

#### 14. Housekeeping

- **Untracked files.** `errors.md` (a 75 KB ComfyUI log) and `data/` are untracked in the repo root. Add both to `.gitignore`, or move them out.
- **CHANGELOG** has no **1.3.5** entry, though `pyproject.toml` and the README badge say 1.3.5.
- **README:**
  - It advertises "Twelve Shader Noise Archetypes", but both nodes' UIs offer three ([shader_renderer.ts:1296](web/src/shader_renderer.ts#L1296), [direct_shader_ksampler.py:68](direct_shader_ksampler.py#L68)).
  - Typos: `dpmppm_2_ancestral`, "Neighboorhood", the example file `Flux_Shcnell_Basic_SNK_Direct.png`.
  - The License section still says "You may need to create this file".
- **Build tooling.** `package.json`'s `build` (`tsc && cp …`) and `dev` scripts rely on POSIX `cp`/`sleep`. Pointing `outDir` at `web/` removes the copy step and makes it cross-platform.

---

## Suggested roadmap

**1. Quick wins.** These don't change the output of existing seeds.
- Add the top-level `import torch.nn.functional as F`.
- Chain `latent_preview.prepare_callback` so the progress bar and live preview work.
- Pass `noise_mask`.
- Remove the localStorage purge; whitelist API keys.
- Delete the no-op shims, the duplicate registration, the dead `advanced_comparer.py` and the non-API hooks. Move `web/types` out of `web/`.
- Gitignore `errors.md` and `data/`, add the 1.3.5 CHANGELOG entry, fix the README.
- Commit the NestedTensor fix.

**2. Sampling correctness.** Behaviour-changing, so gate it behind a `legacy_sampling` toggle or a major version.
- One sigma schedule honouring `denoise` and `custom_sigmas`, segmented across stages (#1, #2, #3, #5).
- Standardize noise after transform and blend; use a variance-preserving `normal` mix; clamp `inverse` (#4).
- Noise shape from the latent, `[B, C, T, H, W]` (#6).

**3. Architecture.**
- Make shader params real node inputs, and retire the params file, API route and save button (#7). Consider merging the two sampler nodes.
- Either finish moving the node onto `core/` (and delete the in-class copies), or delete `core/`'s unused modules. Keeping both is the worst option.
- Expose the currently hidden options that are worth keeping, as optional inputs, once #9's bugs are fixed.
- Add a `pytest` suite for the pure functions and switch `print` to `logging`.
