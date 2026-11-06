# ComfyUI-ShaderNoiseKSampler

![Version](https://img.shields.io/badge/version-2.3.0-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green)
![License](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

[Quick Start](#-quick-start) • [Seed Universe](#-navigating-the-seed-universe) • [Shader Types](#-shader-types) • [Nodes](#-more-nodes) • [Configuration](#%EF%B8%8F-configuration-reference) • [Models](#-model-compatibility) • [Changelog](CHANGELOG.md)

ComfyUI-ShaderNoiseKSampler is a KSampler that blends shader-generated noise into the noise a run starts from. Standard sampling teleports randomly from one seed to the next. This node instead gives you a structured way to explore the neighbourhood around a seed you chose, using mathematical patterns to move along coherent paths through latent space.

![Every shader type at one seed](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/shader_type_gallery.webp)
*One seed and one prompt on Krea 2, with the shader type the only difference: `shader_strength` 0.5, walk, multiply, octaves 2, 8 steps of er_sde/beta57.*

> [!TIP]
> Fix your `seed` first. The seed sets both the base noise and the shader's own pattern, so with it held, every change you make afterwards explores around that one picture instead of landing somewhere new.

## 🆕 What's New in 2.3

- **`kaleido`, a fourteenth shader type.** It folds its field about the centre of the frame for symmetric compositions, and `octaves` is the order of the fold. On Krea 2 the picture came back symmetric from a strength of about 0.3 at orders 2 and 6.
- **`temporal_rate`**, a socket on the Direct, Walk and Source nodes. It sets how fast a pattern held by `use_temporal_coherence` evolves across a clip. 0 holds one pattern, and 1 is the pace every clip used to run at.
- **Two fixes.** `drift` with temporal coherence on no longer collapses into `jump`, and the `gaussian` control no longer leaves one latent channel untouched. Earlier `gaussian` renders are not reproduced.
- **Size matters.** The same settings push a larger picture harder: on Krea 2, more than twice as far at 1024x1024 as at 512x512. Lower `shader_strength` when you size up.

Earlier releases are in the [Changelog](CHANGELOG.md).

## 🚀 Quick Start

### Installation

**ComfyUI Manager:** search for "**ComfyUI-ShaderNoiseKSampler**" under **Install Custom Nodes**, install, and restart ComfyUI.

**Manual:**

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKsampler.git
```

Restart ComfyUI afterwards. No `pip install` is needed.

### Usage

<img src="https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/ui_direct.webp" alt="Shader Noise KSampler (Direct) with its live shader display" width="300" align="right">

1.  **Add** `Shader Noise KSampler (Direct)` where your KSampler was, and connect `model`, `positive`, `negative` and `latent_image` as usual.
2.  **Sample as usual**: `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` and `denoise` mean what they do on a KSampler. Keep the `seed` fixed while you explore.
3.  **Pick a preset**: `explore` is the recommended start. A preset writes its values into the widgets it controls, so you can see what the run will use. `custom` leaves everything to you.
4.  **Shape the shader**: `shader_type`, `shader_strength` (0.0 turns it off), `noise_scale`, `octaves`, `warp_strength` and `phase_shift`, plus an optional `shape_type` mask and `color_scheme`. Leave `travel_mode` on `walk` to explore around your seed.
5.  **Watch the live display.** Turn on `Show Shader` and the node previews the pattern its inputs will draw, for every shader type. The **📊 Show Shader Matrix** button (Alt+M) opens the in-app documentation.
6.  **Queue it.** Each queued run keeps its own parameters.

`example_workflows/` has ready-made graphs for SDXL, Flux Schnell, AnimateDiff, WAN 2.1, HunyuanVideo, LTXV and MiniMax H3, plus one for each comparer.

> [!TIP]
> Start from the `explore` preset, or a `shader_strength` of 0.1-0.3 with a single sequential stage, and walk outward from there. Each small step can land on a noticeably different neighbour, by an amount that depends on the model and the seed, so compare neighbouring strengths rather than expecting a smooth fade. Try `noise_scale` early: larger features let the shader show more, and smaller ones let the model absorb it into the picture.

> [!NOTE]
> The original `Shader Noise KSampler` is deprecated and hidden from node search. It still loads in existing workflows, where it reads its parameters from `data/shader_params.json` and samples in `legacy` mode. Use the Direct node for anything new.

<br clear="right">

## 🧭 Navigating the Seed Universe

Think of standard image generation, where you try different seeds, as driving from one town to another. Each new seed takes you to a completely different town.

The ShaderNoiseKSampler works differently. It's like picking one town that you find particularly interesting, getting out of the car, and really exploring *that specific town*: its hidden alleyways, its neighbourhoods, the variations it holds.

Holding the seed fixed parks the car, and every change you make afterwards is a change of street, not of town. `shader_strength` is how far from the seed's own image you drive, and the other shader settings choose the streets:

- **🔍 `noise_scale`, the zoom control**: how "zoomed in" or "zoomed out" you are in latent space
- **🔬 `octaves`, the detail slider**: the level of detail and complexity in the pattern
- **🌀 `warp_strength`, the non-linear navigator**: bends the path through latent space
- **🔄 `phase_shift`, the perspective shifter**: reveals different facets of the same core elements

![One seed explored by strength, against seed-hopping](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/neighbourhood_sweep.webp)
*Top: seed 8888 with `domain_warp` from strength 0 to 0.75. Bottom: six seeds at strength 0, the ordinary way to look for variations. Krea 2, same settings throughout.*

Low strengths explore close to the seed's image. Higher strengths blend the shader's pattern and colour progressively into the picture until, at the top, it can take the picture over. How quickly that happens depends on the model and the seed. On an SD 1.5 portrait the picture was re-composed by 0.25 and mostly shader by 0.75. On MiniMax H3 the seed's scene held through 0.5 and blended with the shader at 0.75 to 1.0. `travel_mode: jump` is the deliberate exception: the shader's parameters set the destination and the seed stops mattering.

The idea is to treat latent space as a territory to explore rather than a lottery to play. You're not precisely controlling the output, but you're not at the mercy of randomness either. Over time you build a map of how each parameter moves a picture, and you find variations in the spaces between seeds that random sampling would statistically miss.

![ShaderNoiseKSampler Shader Vehicle](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/I_dont_care_what_you_think_this_shit_rips.webp)

**Regular KSampler Seed Travel vs Shader Noise KSampler Seed Travel**
![Seed travel compared](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_compare.webp)

**Shader Noise Neighborhood Seed Travel**
![Neighbourhood seed travel](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_seed_1.webp)
*(These two recordings predate 2.0.0; the idea is unchanged.)*

## 🎨 Shader Types

![The raw field of every shader type](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/shader_types_atlas.webp)
*The noise each type draws, before it meets the sampler: one seed, `noise_scale` 1, octaves 2, three channels shown as RGB. At this scale `spectral` sits in a fine frequency band, and lowering `noise_scale` brings out its soft, cloud-like fields.*

| Type | Character |
|---|---|
| `domain_warp` | Flowing, intricate distortions. The even-handed default. |
| `tensor_field` | Structured and directional. |
| `curl_noise` | Smooth fluid motion, and the one that shows soonest, at about half the strength of the others. |
| `temporal_coherent` | 4D simplex with time as a real axis, built for smooth animation. |
| `spectral` | Soft fields built from their own frequency band; `noise_scale` sets the band directly. Nearly free to draw. |
| `gaussian` | The control: plain white noise, the same thing the sampler already starts from. It moves you toward another seed's neighbourhood without adding structure. |
| `fractal` | The reference FBM, plain layered simplex. |
| `perlin` | Classic gradient noise, smoother and more lattice-like than simplex. |
| `heterogeneous_fbm` | An FBM whose detail varies across the frame, with rough patches and smooth ones. |
| `interference` | Two FBMs turned into crossing fringes, banded and moiré-like. |
| `projection_3d` | A plane through a 3D field; `phase_shift` is the depth of the slice, so a clip is coherent by construction. |
| `cellular` | Worley cells; `octaves` picks the pattern and `phase_shift` the cell shape. |
| `waves` | Seeded plane waves summed, straight at `warp_strength` 0 and bent above it. |
| `kaleido` | An FBM folded about the centre of the frame for symmetric compositions; `octaves` is the order of the fold. |

The `shader_type` tooltip on the node says what each knob does to each type.

Around the type, four more controls shape the noise before it reaches the sampler:

-   **Blend modes** (`normal`, `add`, `multiply`, `screen`, `overlay`, `soft_light`, `hard_light`, `difference`) decide how the shader noise meets the base noise. With `normalize_strength` on (the default), one `shader_strength` value hands the sampler the same share of shader in every mode, so the modes differ in character rather than amount.
-   **Noise transforms** (`absolute`, `sin`, `sqrt` and others) reshape the raw shader noise before blending: `absolute` creates ridges, `sin` bands.
-   **Shape masks** (`radial`, `checkerboard`, `spiral`, `hexgrid`, `vignette` and ten more) concentrate the noise into hard geometry. That survives denoising far more readily than plain shader noise, so the shape itself starts being drawn into the picture at lower strength.
-   **Colour schemes** transform the noise data itself, not just the preview, so they change what the model is handed:

![domain_warp under several colour schemes](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/color_schemes.webp)

### The Shader Matrix

![The Shader Matrix](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/ui_matrix.webp)

The **📊 Show Shader Matrix** button on the node (or Alt+M) opens the Shader Matrix, a self-contained guide inside ComfyUI. It covers the mathematics of each noise archetype with Python examples, interactive visualisations of the patterns and shape masks, and the philosophy behind the node. It documents twelve of the fourteen types; `temporal_coherent` and `kaleido` are described in the node's tooltips.

## 🧩 More Nodes

### Walking a parameter

<img src="https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/ui_walk.webp" alt="Shader Noise Walk" width="280" align="right">

`Shader Noise Walk` takes every input the Direct node does, plus `walk_parameter`,
`walk_start`, `walk_end` and `walk_steps`. It samples the same seed `walk_steps`
times (up to 16) while the chosen parameter ramps from start to end, and returns
the runs as one batched latent for a single VAE decode or a comparer. The model
loads once, so a walk costs about `walk_steps` times one run.

It can ramp `shader_strength`, `phase_shift`, `noise_scale`, `warp_strength`,
`octaves`, `shape_mask_strength`, `color_intensity`, `temporal_rate` or `seed`. Start with
`shader_strength` from 0.0, which gives a clean reference frame. `phase_shift`
holds the distance and turns the pattern instead. Walking `seed` is ordinary
seed-hopping, for comparison. A preset still applies, except to the parameter
being walked.

<br clear="right">

### Splitting a run

`start_at_step` and `end_at_step` sample part of the schedule instead of all of it,
the way `KSampler (Advanced)` does. One node can take the early steps, something
else can work on the latent, and a second node can finish it.

That is how MiniMax H3 gets its faces fixed: the video stream goes through a latent
upscaler between the halves. An upscaler needs a finished latent to work on, so the
first half ends clean, with `return_with_leftover_noise` off, and the second adds its
own noise again with `add_noise` on. Both halves still get their shader noise.
`example_workflows/MiniMaxH3_Split_Upscale_SNK_Direct.json` is the whole thing wired up.

With nothing in between, do the opposite: `return_with_leftover_noise` on in the
first half and `add_noise` off in the second. The pair then continues one trajectory
exactly, with nothing re-noised at the join. That leaves the shader nothing to paint
at the second half's opening, so raise `injection_stages` or `sequential_stages` to
give it an interior boundary to enter at.

Stages divide the steps a node actually samples, not the whole schedule, so two
sequential stages over a three-step window are two stages in those three steps.

### Custom sampling

`Shader Noise Source` outputs a `NOISE` object, so shader noise can start a run
driven by ComfyUI's custom sampling nodes. Feed it to `SamplerCustomAdvanced` in
place of `RandomNoise`, and the guider, sampler and sigma schedule are yours. That
reaches guidance this pack has no node of its own for: `BasicGuider` with no
negative and no CFG, `DualCFGGuider`, or whatever another pack provides. `AddNoise`
takes one too, for shader noise at a chosen sigma with no sampling at all.
`example_workflows/MiniMaxH3_CustomSampling_SNK_Source.json` is a MiniMax H3 run
built that way, on core nodes and this pack and nothing else.

It carries the same shader inputs as the sampler and hands out exactly the noise
the sampler would start from, so a seed means the same thing on both. What it
cannot do is stages. The sampler re-enters the shader at segment boundaries partway
through a run, but a `NOISE` object is asked for noise once, before any sampling
happens. For `sequential_stages` or `injection_stages`, use the sampler.

### 👁️ Advanced Image Comparer and Video Comparer

<img src="https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v2/ui_comparer.webp" alt="Advanced Image Comparer in Slider mode" width="360" align="right">

Two utilities, both in the `utils` category, for A/B testing your explorations side by side. Each has an `auto_fill` toggle, on by default. With it on, an empty slot is filled with the previous generation from the current session, so you can compare your latest run against the one before it with a single input. Both carry the golden eyeball in the title bar.

**`Advanced Image Comparer`** takes `image_a` and `image_b`, single images or batches compared pair by pair, and has eight modes:

-   **Slider**: hover and move left or right to reveal one image or the other.
-   **Click**: click to toggle between A and B.
-   **Side-by-Side** and **Stacked**: the two next to each other, or one above the other.
-   **Grid**: many pairs at once, for batches such as a Walk.
-   **Carousel** and **Batch**: step through pairs one at a time, or page through them.
-   **Onion Skin**: overlay with an adjustable opacity for B.

<br clear="right">

![Advanced Image Comparer Showcase](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/advanced_image_comparer_showcase.webp)

**`Video Comparer`** takes `video_a`, `video_b` and an `fps` (8 by default), with play, pause and frame-by-frame controls. Its six modes are **Playback**, **Side-by-Side**, **Stacked**, **Slider**, **Onion Skin** and **Sync Compare**, which keeps both clips in step. It loads and caches frames as needed, so long clips don't overload the browser.

![Video Comparer Showcase](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/video_comparer_showcase.webp)

## ⚙️ Configuration Reference

The `Shader Noise KSampler (Direct)` inputs and their defaults. The Walk and Source nodes share the shader inputs. The tooltips on the node say more, with measurements.

| Option                       | Description                                                                                                 | Default           |
|------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------|
| **`seed`**                   | Master seed for reproducibility. This is your "town" in the latent space universe. Use a fixed seed for exploration. | `8888`            |
| **`steps`**                  | Number of sampling iterations.                                                                             | `20`              |
| **`cfg`**                    | Classifier-Free Guidance scale.                                                                            | `7.0`             |
| **`sampler_name`**           | E.g., `euler_ancestral`, `er_sde`.                                                                         | `euler_ancestral` |
| **`scheduler`**              | E.g., `normal`, `beta`, `simple`.                                                                          | `beta`            |
| **`denoise`**                | Denoising strength.                                                                                        | `1.0`             |
| **`add_noise`**              | Make the noise the run starts from. Off, the latent is taken to already carry its own from an earlier sampler. | `true`            |
| **`start_at_step`**          | Enter the schedule here instead of at the first step.                                                      | `0`               |
| **`end_at_step`**            | Stop after this step. Anything at or past `steps` runs to the end.                                         | `10000`           |
| **`return_with_leftover_noise`** | Hand the latent over still noisy when `end_at_step` stopped the run early, instead of finishing it cleanly. | `false`           |
| **`custom_sigmas`**          | Optional sigma schedule that replaces the one `steps` and `scheduler` would build.                          | —                 |
| **`sequential_stages`**      | Number of shader stages applied sequentially.                                                              | `1`               |
| **`injection_stages`**       | Number of shader stages injected at specific steps.                                                        | `0`               |
| **`sequential_distribution`** / **`injection_distribution`** | How `shader_strength` is spread across each kind of stage: `uniform`, `linear_decrease`, `linear_increase`, `gaussian`, `first_stronger` or `last_stronger`. | `linear_decrease` |
| **`stage_progression`**      | `coarse_to_fine` starts on large features with fewer octaves and ends on small ones with more; `fine_to_coarse` reverses it. Spans 0.5x to 2x `noise_scale` and ±1 octave around your values. | `uniform`         |
| **`shader_strength`**        | Global strength of shader noise influence (0.0 to disable).                                                | `0.3`             |
| **`blend_mode`**             | How shader noise combines with base noise (e.g., `multiply`, `add`).                                       | `multiply`        |
| **`normalize_strength`**     | Reads `shader_strength` on `multiply`'s scale, so the same value hands over the same share of shader in every blend mode. | `true`            |
| **`preset`**                 | Sets the shader settings that only mean something together, and writes them into the widgets. `custom` leaves everything alone. | `custom`          |
| **`travel_mode`**            | How the shader moves you: `walk` explores around the seed, `drift` narrows the move, `jump` lets the shader set the destination. | `walk`            |
| **`noise_transform`**        | Math operation on shader noise (e.g., `none`, `absolute`, `sin`).                                          | `none`            |
| **`use_temporal_coherence`** | Hold one seed across every video frame so the pattern evolves only through time.                           | `false`           |
| **`temporal_rate`**          | A socket, no widget. How far a held pattern evolves across a clip under `use_temporal_coherence`: 0 holds one pattern, 1 is its own pace, higher shares less between frames. | `1.0` (unconnected) |
| **`shade_non_spatial`**      | Also paint streams with no picture in them: H3's and LTXAV's audio, and sequence latents, which are otherwise refused. | `false`           |
| **`fast_high_channel_noise`** | A faster, simplified draw for latents with more than 16 channels, such as LTXV.                           | `false`           |
| **`sampling_mode`**          | `standard` samples one schedule split into stage segments, and honours `denoise` and `custom_sigmas`. `legacy` keeps the pre-2.0.0 pipeline and is selected automatically for older workflows (see Known Issues). | `standard`        |
| **`shader_type`**            | The base pattern: one of the fourteen [types](#-shader-types).                                             | `domain_warp`     |
| **`noise_scale`**            | The zoom control: size of the shader's features.                                                           | `1.0`             |
| **`octaves`**                | The detail slider: number of layers combined. Fractional values blend two layer counts.                    | `1`               |
| **`warp_strength`**          | The non-linear navigator: how much the pattern warps.                                                      | `0.5`             |
| **`phase_shift`**            | The perspective shifter: reveals different facets of the same pattern.                                     | `0.5`             |
| **`shape_type`**             | Geometric mask overlay (e.g., `radial`, `checkerboard`).                                                   | `none`            |
| **`shape_mask_strength`**    | Intensity of the shape mask.                                                                               | `1.0`             |
| **`color_scheme`**           | Colour mapping for the noise (e.g., `viridis`, `jet`).                                                     | `none`            |
| **`color_intensity`**        | Strength of the colour scheme.                                                                             | `0.8`             |

### Sampler & Scheduler

The sampler decides how much of the shader survives. Both pairs below were measured on MiniMax H3 across every shader type, three seeds, and strengths 0.25, 0.5 and 0.75:

-   **`er_sde` / `beta57`** never broke: every type held a clean picture at every strength. A stochastic sampler re-noises at every step, though, so it moves the picture about half as far, and the types differ less from each other. Turn on temporal coherence or raise the strength to get more movement out of it.
-   **`res_multistep` / `simple`** gives each type a stronger, more distinct character, but past 0.5, with coherence off, 8 of 13 types washed the frame in rainbow light.

## 🧱 Model Compatibility

In `standard` sampling mode there is no model detection at all. The noise takes
its shape from the latent you hand the node, so **any model whose latent is
`[B, C, H, W]` or `[B, C, T, H, W]` works, at any channel count**, from 3 channels for
the pixel-space models up to 256, without a list to be added to.

| Latent shape | Works | Examples |
| --- | --- | --- |
| `[B, C, H, W]` (image) | Yes, any `C` | SD 1.5, SDXL, SD3, Flux, Flux 2, Chroma, HiDream, Qwen-Image, Z-Image, PixelDiT, Trellis2, HunyuanImage 2.1 |
| `[B, C, T, H, W]` (video) | Yes, any `C` | WAN 2.1 / 2.2, HunyuanVideo and 1.5, LTXV, Mochi, Cosmos, CogVideoX, SeedVR2, Anima |
| `NestedTensor` of streams | Yes: the first stream is painted, and every stream with `shade_non_spatial` | **MiniMax H3** (video + audio), LTXAV |
| `[B, C, L]` (sequence) | Only with `shade_non_spatial`; otherwise refused with a named error | Stable Audio 1 / 3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D v2, TripoSplat |

**MiniMax H3** arrives as a paired video + audio latent. By default the shader
paints the video stream, and the audio stream keeps exactly the Gaussian noise a
stock KSampler would have given it. With `shade_non_spatial` on, the audio is
painted across stereo x time as well. H3 denoises both streams together, so that
reaches the picture too, and the sound changes at far lower strength than the
picture does. Note that H3 itself only supports batch size 1.

**Sequence latents** carry no height and width, so there is nothing for a shader
to draw on. By default the node refuses them with a message naming the shape,
rather than quietly producing something meaningless. `shade_non_spatial` paints
them as a single row instead, which is largely unexplored. Setting
`shader_strength` to `0.0` leaves nothing to paint, and the node then samples
those models as a plain KSampler.

> [!NOTE]
> The above describes `standard` mode. `legacy` mode keeps its original
> channel-count detection and pipeline, along with their quirks; it is frozen, not
> maintained.

## ❓ Troubleshooting

-   **Shader effects not visible**:
    -   Make sure `shader_strength` is above `0.0`.
    -   With `add_noise` off there is no starting noise to paint, so a single stage does nothing. Raise `sequential_stages` or `injection_stages` to give the shader a boundary to enter at.
    -   `gaussian` is plain white noise by design: it moves you toward another seed's neighbourhood without adding any pattern.
    -   A stochastic sampler such as `er_sde` dilutes the shader (see [Sampler & Scheduler](#sampler--scheduler)).

-   **Unexpected results**: small parameter changes can lead to large visual shifts. Even a 0.05 change in `shader_strength` can land on a noticeably different neighbour of your seed's image, or on a near-identical one; which you get depends on the model and the seed. Use the live display to understand the noise before generating.

### Known Issues

-   **Legacy sampling mode**: nodes loaded from workflows saved before 2.0.0 switch to `sampling_mode: legacy`, which keeps the old pipeline, including the `denoise` issue fixed in 2.0.0, which made image-to-image and video-to-video regenerate their input in full. Its seeds no longer reproduce exactly for workflows using `domain_warp`, `temporal_coherent`, or `curl_noise` on latents wider than four channels, because the shader generators both modes share now fill every latent channel. Switch to `standard` for the corrected sampling.

> [!WARNING]
> **Potential for visual instability**: some parameter explorations, particularly at high intensity or with complex interactions, can produce visually disruptive output such as flashing images or harsh artifacts. Iterate with caution.

## 🌱 Grassroots Research & Development

ComfyUI-ShaderNoiseKSampler grew out of personal research, and it is still an active exploration. There is plenty of room for further improvements, new discoveries, and community-driven enhancements as it evolves.

### 🗺️ Roadmap

| Area | Focus |
| :--- | :--- |
| **🌌 Latent Space Cartography** | A "Visual Intent Engine": describe a desired outcome ("enhance fabric texture", "shift the lighting") and have it translated into shader parameters, built on a deeper map of how parameters move a picture. |
| **🧭 Precision Navigation** | More granular, predictable tools for steering latent paths, from direct correlations between noise constructs and the visual features they produce. |
| **🔮 Cross-Modal Exploration** | Structured noise beyond 2D images. Audio and sequence latents are already reachable with `shade_non_spatial` and largely unexplored. |

## 🤝 Contributing

Contributions are welcome! See the [contributing guidelines](CONTRIBUTING.md) to get started.

## 🙏 Acknowledgements

-   The ComfyUI team for creating such a flexible and powerful platform.
-   The developers of libraries and concepts that inspired aspects of this work (e.g., GLSL, various noise algorithms).
-   The ComfyUI community for their continuous innovation and support.
-   Users and contributors who provide feedback and suggestions.

## 🔗 Connect with Æmotion (Developer)

-   YouTube: [AEmotionStudio](https://www.youtube.com/@aemotionstudio/videos)
-   GitHub: [AEmotionStudio](https://github.com/AEmotionStudio)
-   Discord: [Join our community](https://discord.gg/UzC9353mfp)
-   Website: [aemotionstudio.org](https://aemotionstudio.org/)

## ☕ Support

If you find ComfyUI-ShaderNoiseKSampler useful and wish to support its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/aemotionstudio)

Starring the repository, sharing it, and sending feedback or feature requests all help too. For business inquiries or professional support, reach out through the [website](https://aemotionstudio.org/) or the [Discord server](https://discord.gg/UzC9353mfp).

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.
See the `LICENSE` file for details.
