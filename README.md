# ComfyUI-ShaderNoiseKSampler

![Version](https://img.shields.io/badge/version-2.2.1-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green)
![License](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#%EF%B8%8F-configuration-options) • [Changelog](CHANGELOG.md)


ComfyUI-ShaderNoiseKSampler is an advanced custom KSampler node that blends standard noise generation with a multi-stage shader-based system. Unlike traditional sampling that teleports randomly between different seeds, this tool provides a structured vehicle for exploring the neighborhood around a chosen seed, using mathematical patterns to navigate along coherent paths through latent space.

![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_display.webp)

> [!TIP]
> Perfect for artists and researchers looking to explore the latent space with more intention, sculpt image features with mathematical precision, and achieve unique visual styles through deliberate exploration rather than random discovery. (Using a fixed seed is key to this deliberate exploration, allowing you to systematically investigate the variations around a chosen starting point.)

![ShaderNoiseKSampler Showcase PNG](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler.png)

> [!IMPORTANT]
> **Use `Shader Noise KSampler (Direct)`.** Since 2.0.0 it carries both halves: every shader parameter is a node input, and the live shader display is attached to it. Nothing needs saving, each queued run keeps its own parameters, and the display still shows what the controls are doing.
>
> The original `Shader Noise KSampler` is deprecated and hidden from node search. It still loads in existing workflows, where it keeps reading its parameters from `data/shader_params.json` and samples in `legacy` mode.

**Shader Noise KSampler (deprecated)**
![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_save.webp)
**(Only parameters with 🔄 require saving if changed)**

**Shader Noise KSampler (Direct)**
![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_direct.webp)
**(No saving required. This recording predates 2.0.0; the node now carries the live shader display as well)**

## 🚀 Recent Updates

- **Every Shader Type, Split Runs and Custom Sampling - 9/22/26**: Thirteen shader types, up from four: `spectral` plus the eight the Shader Matrix documented but never shipped (`gaussian`, `fractal`, `perlin`, `heterogeneous_fbm`, `interference`, `projection_3d`, `cellular`, `waves`). The live display previews all thirteen. `start_at_step`, `end_at_step`, `add_noise` and `return_with_leftover_noise` let one run be [split](#splitting-a-run) around a latent upscale, and the new `Shader Noise Source` node hands shader noise to [custom sampling](#custom-sampling). Every new input defaults to a no-op, so saved workflows sample what they did before.

- **Shader Blending Upgrade - 9/12/26**: The shader now blends into every channel of the latent instead of stamping one pattern across it, and the first step away from `shader_strength` 0 is only as large as the shader makes it. New `preset`, `travel_mode`, `stage_progression` and `shade_non_spatial` inputs, `normalize_strength` on by default, a [`Shader Noise Walk`](#walking-a-parameter) node that ramps one parameter across a batch, and tooltips that describe what each strength does. Saved workflows render differently at any strength above 0; see the [Changelog](CHANGELOG.md).

- **Comparer Auto-Fill - 6/12/25**: Both the `Advanced Image Comparer` and `Video Comparer` nodes now feature an `auto_fill` toggle. This addition streamlines your workflow by allowing you to compare with a single input. When `auto_fill` is enabled (the default setting), any empty image or video slot will be automatically populated with the output from the previous generation. This makes iterative A/B testing—comparing your latest creation to the one right before it—faster and more intuitive. It does not pull from your output folder but from the cached images or videos of your current session.
![Video Comparer Updated WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/video_comparer_updated.webp)

## 🧭 Navigating the Seed Universe

Think of standard image generation, where you try different seeds, as driving from one town to another. Each new seed takes you to a completely different town.

The ShaderNoiseKSampler works differently. It's like picking one town (one specific seed) that you find particularly interesting, and instead of just driving through to the next one, you decide to stop, get out of the car, and really explore *that specific town*. You can wander down its hidden alleyways, check out its diverse neighborhoods, and discover all the unique details and variations it holds. 

The shader noise is your map and the shader parameters (like Noise Scale, Octaves, Warp Strength, etc.) are your compass for this in-depth local exploration:

- **🔍 Noise Scale: The Zoom Control** - Determines how "zoomed in" or "zoomed out" you are in latent space
- **🔬 Octaves: The Detail Slider** - Controls the level of detail and complexity in your noise pattern
- **🌀 Warp Strength: The Non-Linear Navigator** - Creates non-linear paths through latent space
- **🔄 Phase Shift: The Perspective Shifter** - Reveals different "facets" of the same core elements

In practice, holding the seed fixed parks the car: the seed sets both the base noise and the shader's own pattern, so every change you make afterwards is a change of street, not of town. `shader_strength` is how far from the seed's own image you drive, and the other shader settings choose the streets. Low strengths explore close to that image; higher strengths blend the shader's pattern and colour progressively into the picture until, at the top, it can take the picture over. How quickly that happens depends on the model and the seed. On an SD 1.5 portrait the picture was re-composed by 0.25 and mostly shader by 0.75, while on MiniMax H3 the seed's scene held through 0.5 and blended with the shader at 0.75 to 1.0. `travel_mode: jump` is the deliberate exception: the shader's parameters set the destination and the seed stops mattering.

The core innovation is treating latent space as a territory to be explored rather than a lottery to be played - turning the act of AI image generation into a journey of deliberate artistic discovery guided by the elegant language of mathematical patterns.

![ShaderNoiseKSampler Shader Vehicle](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/I_dont_care_what_you_think_this_shit_rips.webp)

**Regular KSampler Seed Travel vs Regular Shader Noise KSampler Seed Travel**
![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_compare.webp)
*(Same settings are used in both samplers)*

**Shader Noise Neighborhood Seed Travel**
![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_seed_1.webp)
*(Same settings are used in both samplers)*

**Shader Noise KSampler Palettes**
![ShaderNoiseKSampler Showcase WEBP](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_palettes.webp)
*(Each shader noise palette offers a unique lens into latent space)*

## 🔮 The Innovation: Controllable Noise

While traditional samplers rely on pure randomness, **ShaderNoiseKSampler** introduces a paradigm shift: mathematically controllable noise patterns derived from shader technology (shader noise). This isn't just navigation - it's the invention of a new vehicle.

By replacing the standard random noise distribution with structured shader-generated patterns (shader noise), we transform the diffusion process from a random walk into a deliberate journey. The noise itself becomes an artistic medium you can sculpt through mathematical parameters:

- Scale, octaves, and warp create precise noise topographies
- Transformations and blending modes give you noise "vocabulary"
- Shape masks and color schemes provide spatial and frequency control

This controlled noise approach bridges the gap between the deterministic world of procedural generation and the probabilistic nature of diffusion models, offering creative control without sacrificing the generative AI's creative potential.

## 📖 The Shader Matrix: In-Depth Documentation

![ShaderNoiseKSampler Shader Matrix](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/shader_noise_ksampler_matrix.webp)

A core feature of this project is the **"📊 Show Shader Matrix"** button available on the `ShaderNoiseKSampler` (and related) nodes. Clicking this button opens an extensive, self-contained modal dialog—The Shader Matrix—which provides:

-   Detailed explanations of all shader noise types, mathematical foundations, and parameters.
-   Interactive visualizations of noise patterns and shape masks.
-   Python code examples for noise generation.
-   A comprehensive guide to the philosophy and usage of the node.

This README provides an overview, but the Shader Matrix is your ultimate guide for deep dives!

## ✨ Features

-   **🚀 Advanced KSampler Replacement**: Integrates directly into your workflow as a KSampler.
-   **🔬 Multi-Stage Shader Application**:
    -   **Sequential Stages**: Apply shader noise over segments of the diffusion process.
    -   **Injection Stages**: Apply shader noise at specific, discrete steps.
-   **🎨 Thirteen Shader Noise Types**: the twelve archetypes the Shader Matrix documents -- `domain_warp`, `tensor_field`, `curl_noise`, `spectral`, `gaussian`, `fractal`, `perlin`, `heterogeneous_fbm`, `interference`, `projection_3d`, `cellular` and `waves` -- plus `temporal_coherent`, built for video. Each is a distinct lens on the latent neighbourhood, and each supports shape masks, colour schemes and transforms.
-   **🎭 Sophisticated Blending & Transformations**:
    -   **Blend Modes**: Combine shader noise with base noise using Normal, Multiply, Add, Overlay, Screen, Soft Light, Hard Light or Difference.
    -   **Noise Transformations**: Apply mathematical operations (Absolute, Sin, Square Root, etc.) to shader noise before blending.
-   **💠 Shape Masks**: Spatially modulate noise with geometric overlays (Radial, Linear, Checkerboard, Vignette, Spiral, Hexgrid, etc.) with adjustable strength.
-   **🌈 Color Schemes Integration**: Apply color transformations (Inferno, Magma, Viridis, Jet, Turbo, etc.) to the noise *before* it influences the diffusion model, subtly guiding structure and aesthetics. Adjustable intensity.
-   **⏳ Temporal Coherence**:
    -   Generate frame-consistent evolving noise for animations.
    -   Ensure consistent base noise for predictable exploration when tweaking parameters for still images.
-   **🎛️ Granular Control**:
    -   Global `shader_strength`, spread across stages by `sequential_distribution` and `injection_distribution`.
    -   Adjust `noise_scale`, `octaves`, `warp_strength`, `phase_shift`, and more; `stage_progression` varies the zoom and detail from stage to stage.
-   **🎚️ Presets**: `nudge`, `explore`, `roam`, `video`, `jump` and `stamp` set the shader settings that only mean something together, and write them into the node's widgets so you can see what the run will use.
-   **🧭 Travel Modes**: `walk` explores around the seed, `drift` narrows how the shader moves you, and `jump` lets the shader's parameters choose the destination.
-   **⚖️ Consistent Strength**: With `normalize_strength` (on by default), one `shader_strength` value hands the sampler the same share of shader in every blend mode.
-   **👁️ Live Shader Display**: The Direct node previews the pattern its inputs will draw, for every shader type.
-   **🚶 Shader Noise Walk**: Ramp one parameter across a batch in a single run, with the model loaded once, and feed the result to the comparers.
-   **✂️ Split Runs and Custom Sampling**: Sample part of a schedule like `KSampler (Advanced)`, or take shader noise into `SamplerCustomAdvanced` through `Shader Noise Source`.
-   **💾 Parameter Management**: The deprecated `Shader Noise KSampler` saves its parameters to a file; the Direct node needs no saving.
-   **📊 "Show Shader Matrix" Button**: Access comprehensive, interactive documentation and visualizations directly within ComfyUI (Alt+M shortcut).
-   **🤝 Compatibility**: Shape-driven rather than a model list — any image or video latent at any channel count, including multi-stream video+audio latents like MiniMax H3, and sequence latents with `shade_non_spatial` (see Model Compatibility section).
-   **🧠 Latent Space Cartography**: Create a map of the territory surrounding your seed, developing an intuitive understanding of how to navigate to specific effects.
-   **🔄 Persistent Identities in Variation**: Observe how similar elements persist across parameter adjustments, revealing how the model encodes concepts and their relationships.
-   **💎 Discovery of "Hidden Gems"**: Find interesting variations that exist in the spaces "between" seeds that random sampling might statistically miss.

## 🖼️ Advanced Image Comparer

ComfyUI-ShaderNoiseKSampler includes an `AdvancedImageComparer` node, a versatile utility for visually comparing two images or batches of images directly within your workflow. This node is invaluable for evaluating the subtle (or significant) differences produced by varying parameters, seeds, or even different models.

![Advanced Image Comparer Showcase](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/advanced_image_comparer_showcase.webp) *

### Features:

-   **Eight Comparison Modes**: Choose the best way to visualize differences:
    -   **Slider**: Overlay images and use a slider to reveal one or the other.
    -   **Click**: Toggle between the two images with a click.
    -   **Side-by-Side**: Display images next to each other.
    -   **Stacked**: Display images one above the other.
    -   **Grid**: View multiple image pairs in a grid layout, ideal for batch comparisons.
    -   **Carousel**: Cycle through image pairs one by one.
    -   **Batch**: Display multiple pairs in a paginated list.
    -   **Onion Skin**: Overlay images with adjustable opacity for the top image.
-   **Batch Processing**: Efficiently compare multiple sets of images (Image A1 vs Image B1, Image A2 vs Image B2, etc.).
-   **Interactive Controls**: Easily navigate through image pairs in Carousel and Batch modes.
-   **Customizable Layout**: Adjust the node size and select your preferred layout mode via a dropdown menu.
-   **Distinctive UI**: Features a unique golden eyeball design in the node's title bar for easy identification.

### Usage:

1.  **Add Node**: Add the `Advanced Image Comparer` node (found in the `utils` category) to your ComfyUI graph.
2.  **Connect Inputs**:
    -   `image_a`: Connect the first image or batch of images.
    -   `image_b`: Connect the second image or batch of images.
3.  **Select Mode**: Use the "Layout Mode" dropdown on the node to choose your preferred comparison view.
4.  **Interact**:
    -   **Slider Mode**: Hover your mouse over the image and move it left or right.
    -   **Click Mode**: Click on the image to toggle between A and B.
    -   **Grid Mode**: View multiple pairs at once in a grid layout.
    -   **Carousel/Batch Modes**: Use the provided UI controls (buttons, pair selector) to navigate.
    -   **Onion Skin Mode**: Adjust the "Opacity B" slider to control the transparency of the second image.

This tool is designed to enhance your A/B testing and iterative refinement process, making it easier to observe the impact of your creative choices.

## 🎬 Video Comparer

The `VideoComparer` node provides a powerful way to visually compare two videos directly within your ComfyUI workflow. This node is perfect for comparing different generation settings, model outputs, or any video-related experiments you're conducting.

![Video Comparer Showcase](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/video_comparer_showcase.webp) *

### Features:

-   **Six Viewing Modes**: Choose the comparison method that works best for your needs:
    -   **Playback**: Standard video playback with the ability to switch between videos A and B.
    -   **Side-by-Side**: Display both videos next to each other for direct comparison.
    -   **Stacked**: View videos stacked vertically, one above the other.
    -   **Slider**: Overlay videos with a draggable slider to reveal portions of each.
    -   **Onion Skin**: Overlay videos with adjustable opacity for the top video.
    -   **Sync Compare**: Synchronized comparison mode that keeps both videos in perfect time alignment.
-   **Interactive Playback Controls**: Play, pause, and navigate through frames with an intuitive control interface.
-   **Frame-by-Frame Navigation**: Precisely compare specific frames with frame counter display.
-   **Adjustable FPS**: Set the playback speed to suit your analysis needs.
-   **Memory-Efficient Design**: Smart loading and caching system to handle large videos without overloading your browser.
-   **Distinctive UI**: Features the same golden eyeball design as the Advanced Image Comparer for a consistent experience.

### Usage:

1.  **Add Node**: Add the `Video Comparer` node (found in the `utils` category) to your ComfyUI graph.
2.  **Connect Inputs**:
    -   `video_a`: Connect the first video.
    -   `video_b`: Connect the second video for comparison.
    -   `fps`: Adjust the playback speed (default is 8 fps).
3.  **Interact**:
    -   Use the dropdown menu to select your preferred viewing mode.
    -   Navigate using the playback controls at the bottom of the node.
    -   In Slider mode, move your mouse left/right to reveal different portions of each video.
    -   In Onion Skin mode, adjust the opacity using the controls provided.

This tool is especially valuable for comparing subtle differences in video generation outcomes, helping you fine-tune your workflows for optimal results.

## 📥 Installation

### Option 1: Using ComfyUI Manager

1.  Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you don't have it already.
2.  Open ComfyUI, go to the **Manager** tab.
3.  Click on **Install Custom Nodes**.
4.  Search for "**ComfyUI-ShaderNoiseKSampler**" and click **Install**.
5.  Restart ComfyUI.

### Option 2: Manual Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKsampler.git
```

Restart ComfyUI after installation. No additional `pip install` steps are required for the core functionality.

## 🚀 Usage

1.  **Add Node**: Add the `Shader Noise KSampler (Direct)` node to your ComfyUI graph.
2.  **Connect Inputs**:
    -   `model`: Your primary AI model.
    -   `positive`, `negative`: Your conditioning prompts.
    -   `latent_image`: The input latent (e.g., from an Empty Latent Image node).
3.  **Basic Sampling Parameters**:
    -   Set `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, and `denoise` as you would for a standard KSampler.
    -   It is recommended to use a fixed `seed` number when you want to explore the neighborhood around that specific seed. This allows the shader noise parameters to navigate the latent space coherently from a consistent starting point.
4.  **Configure Shader Noise**: This is where the exploration begins!
    -   **Preset**: Pick `explore` to start, or leave it on `custom` to set everything yourself. A preset writes its values into the widgets it controls.
    -   **Travel Mode**: Leave `travel_mode` on `walk` to explore around your seed; `jump` hands the destination to the shader.
    -   **Stages**: Define `sequential_stages` and `injection_stages`, and how strength is spread across them with `sequential_distribution` and `injection_distribution`.
    -   **Global Controls**: Set `shader_strength` (0.0 to disable shaders), `blend_mode`, and `noise_transform`.
    -   **Shader Controls**: Every stage draws from the same settings:
        -   `shader_type` (e.g., `domain_warp`, `perlin`, `cellular`)
        -   `noise_scale` (zoom control), `octaves` (detail level), `warp_strength` (non-linear navigation), `phase_shift` (perspective shift)
        -   `shape_type` and `shape_mask_strength`
        -   `color_scheme` and `color_intensity`
    -   **Stage Progression**: `coarse_to_fine` or `fine_to_coarse` varies `noise_scale` and `octaves` across the run instead of drawing the same shader at every stage.
    -   **Temporal Coherence**: Enable `use_temporal_coherence` for animations or consistent exploration.
5.  **Explore**: Watch the live shader display as you change the inputs, and use the "📊 Show Shader Matrix" button (or Alt+M) to better understand the noise patterns you're creating.
6.  **Generate**: Queue your prompt and witness the shader-guided generation!

`example_workflows/` has ready-made graphs for SDXL, Flux Schnell, AnimateDiff, WAN 2.1, HunyuanVideo, LTXV and MiniMax H3, plus one for each comparer.

### Walking a parameter

`Shader Noise Walk` takes every input the Direct node does, plus `walk_parameter`,
`walk_start`, `walk_end` and `walk_steps`. It samples the same seed `walk_steps`
times (up to 16) while the chosen parameter ramps from start to end, and returns
the runs as one batched latent for a single VAE decode or a comparer. The model
loads once, so a walk costs about `walk_steps` times one run.

It can ramp `shader_strength`, `phase_shift`, `noise_scale`, `warp_strength`,
`octaves`, `shape_mask_strength`, `color_intensity` or `seed`. Start with
`shader_strength` from 0.0, which gives a clean reference frame; `phase_shift`
holds the distance and turns the pattern instead; walking `seed` is ordinary
seed-hopping, for comparison. A preset still applies, except to the parameter
being walked.

### Splitting a run

`start_at_step` and `end_at_step` sample part of the schedule instead of all of it,
the way `KSampler (Advanced)` does, so one node can take the early steps, something
else can work on the latent, and a second node can finish it.

That is how MiniMax H3 gets its faces fixed: the video stream goes through a latent
upscaler between the halves. An upscaler needs a finished latent to work on, so the
first half ends clean -- `return_with_leftover_noise` off -- and the second adds its
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
driven by ComfyUI's custom sampling nodes: feed it to `SamplerCustomAdvanced` in
place of `RandomNoise` and the guider, sampler and sigma schedule are yours. That
reaches guidance this pack has no node of its own for -- `BasicGuider` with no
negative and no CFG, `DualCFGGuider`, whatever another pack provides. `AddNoise`
takes one too, for shader noise at a chosen sigma with no sampling at all.
`example_workflows/MiniMaxH3_CustomSampling_SNK_Source.json` is a MiniMax H3 run
built that way, on core nodes and this pack and nothing else.

It carries the same shader inputs as the sampler and hands out exactly the noise
the sampler would start from, so a seed means the same thing on both. What it
cannot do is stages: the sampler re-enters the shader at segment boundaries partway
through a run, and a `NOISE` object is asked for noise once, before any sampling
happens. For `sequential_stages` or `injection_stages`, use the sampler.

> [!TIP]
> Fix your `seed` first, then start from the `explore` preset or a `shader_strength` around 0.1-0.3 with a single `sequential_stage`, and walk outward from there. Each small step can still land on a noticeably different neighbour, by an amount that depends on the model and the seed, so compare neighbouring strengths rather than expecting a smooth fade. Try `noise_scale` early: larger features let the shader show more, smaller ones let the model absorb it into the picture.

## 🧠 Latent Space Navigation

Unlike random seed exploration, the ShaderNoiseKSampler provides a methodical way to navigate the latent space:

- **Creative Control vs. Serendipity**: Find a balance between intentional direction and unexpected discovery. You're not precisely controlling the output, but you're not completely at the mercy of randomness either - you're steering through possibility space.

- **Frequency Domain Exploration**: The mathematical nature of this tool allows you to explore how different frequency patterns map to semantic features in the generated images, revealing fundamental patterns in how visual information is encoded in the model.

- **Methodical Discovery**: As you experiment with different parameter combinations, you'll develop an intuitive understanding of how they affect the output, allowing for more deliberate creative choices.

- **Fine-Grained Control**: You can introduce a wide spectrum of changes, from barely noticeable subtleties to more significant transformations. This gives you the flexibility to choose your desired degree of alteration, all while preserving the core meaning, motion, and recognizable objects (semantic elements) of the image.

## ⚙️ Configuration Options

`Shader Noise KSampler (Direct)` offers extensive control. Key parameters are listed below with their navigational significance. For an exhaustive list and explanations, please refer to the **"📊 Show Shader Matrix"** documentation within ComfyUI.

| Option                       | Description & Navigational Significance                                                                     | Default (Example) |
|------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------|
| **`seed`**                   | Master seed for reproducibility. This is your "town" in the latent space universe. Use fixed seed for exploration.                          | `8888`            |
| **`steps`**                  | Number of sampling iterations.                                                                             | `20`              |
| **`cfg`**                    | Classifier-Free Guidance scale.                                                                            | `7.0`             |
| **`sampler_name`**           | E.g., `euler_ancestral`, `dpm_2_ancestral`.                                                                          | `euler_ancestral` |
| **`scheduler`**              | E.g., `normal`, `beta`, `simple`.                                                                   | `beta`            |
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
| **`use_temporal_coherence`** | For consistent noise in animations or exploration.                                                         | `false`           |
| **`shade_non_spatial`**      | Also paint streams with no picture in them: H3's and LTXAV's audio, and sequence latents, which are otherwise refused. | `false`           |
| **`fast_high_channel_noise`** | A faster, simplified draw for latents with more than 16 channels, such as LTXV.                           | `false`           |
| **`sampling_mode`**          | `standard` samples one schedule split into stage segments, and honours `denoise` and `custom_sigmas`. `legacy` keeps the pre-2.0.0 pipeline and is selected automatically for older workflows, but no longer reproduces every earlier seed exactly (see Known Issues). | `standard`        |
| **`shader_type`**            | The base pattern: one of the thirteen types, `domain_warp` by default. The node's tooltip describes each one's character and what its knobs do. | `domain_warp`     |
| **`noise_scale`**            | The "Zoom Control" - determines how "zoomed in" or "zoomed out" you are in latent space.                  | `1.0`             |
| **`octaves`**                | The "Detail Slider" - controls the level of detail and complexity in your noise pattern.                  | `1`               |
| **`warp_strength`**          | The "Non-Linear Navigator" - creates non-linear paths through latent space.                                | `0.5`             |
| **`phase_shift`**            | The "Perspective Shifter" - reveals different "facets" of the same core elements.                          | `0.5`             |
| **`shape_type`**             | Geometric mask overlay (e.g., `radial`, `checkerboard`).                                                   | `none`            |
| **`shape_mask_strength`**    | Intensity of the shape mask.                                                                               | `1.0`             |
| **`color_scheme`**           | Color mapping for noise (e.g., `viridis`, `jet`).                                                          | `none`            |
| **`color_intensity`**        | Strength of the color scheme influence.                                                                    | `0.8`             |

### Sampler & Scheduler Compatibility

Generally, `ShaderNoiseKSampler` aims for broad compatibility. The following are often good starting points (refer to the Shader Matrix for more details):

-   **Recommended Samplers**: `euler_ancestral`, `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `lcm`
-   **Recommended Schedulers**: `beta` (often preferred), `normal`, `simple`, `kl_optimal`

### 🧱 Model Compatibility

In `standard` sampling mode there is no model detection at all. The noise takes
its shape from the latent you hand the node, so **any model whose latent is
`[B, C, H, W]` or `[B, C, T, H, W]` works, at any channel count** — 3 channels for
the pixel-space models up to 256, without a list to be added to.

| Latent shape | Works | Examples |
| --- | --- | --- |
| `[B, C, H, W]` (image) | Yes, any `C` | SD 1.5, SDXL, SD3, Flux, Flux 2, Chroma, HiDream, Qwen-Image, Z-Image, PixelDiT, Trellis2, HunyuanImage 2.1 |
| `[B, C, T, H, W]` (video) | Yes, any `C` | WAN 2.1 / 2.2, HunyuanVideo and 1.5, LTXV, Mochi, Cosmos, CogVideoX, SeedVR2, Anima |
| `NestedTensor` of streams | Yes — the first stream is painted, every stream with `shade_non_spatial` | **MiniMax H3** (video + audio), LTXAV |
| `[B, C, L]` (sequence) | Only with `shade_non_spatial` — otherwise refused with a named error | Stable Audio 1 / 3, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D v2, TripoSplat |

**MiniMax H3** arrives as a paired video + audio latent. By default the shader
paints the video stream and the audio stream keeps exactly the Gaussian noise a
stock KSampler would have given it. With `shade_non_spatial` on, the audio is
painted across stereo x time as well; H3 denoises both streams together, so that
reaches the picture too, and the sound changes at far lower strength than the
picture does. Note that H3 itself only supports batch size 1.

**Sequence latents** carry no height and width, so there is nothing for a shader
to draw on. By default the node refuses them with a message naming the shape
rather than quietly producing something meaningless. `shade_non_spatial` paints
them as a single row instead; that is largely unexplored. Setting
`shader_strength` to `0.0` leaves nothing to paint, and the node then samples
those models as a plain KSampler.

> [!NOTE]
> The above describes `standard` mode. `legacy` mode keeps its original
> channel-count detection and pipeline, along with their quirks; it is frozen, not
> maintained. It shares the shader generators, though, so it no longer reproduces
> every pre-2.0 seed exactly (see Known Issues).

## 🔬 Shader Noise Deep Dive (Brief Overview)

The true depth of `ShaderNoiseKSampler` lies in its components. The "Shader Matrix" covers these extensively.

-   **Shader Noise Types**: Start with `Domain Warp` for intricate, flowing distortions, `Tensor Field` for structured and directional patterns, and `Curl Noise` for smooth, fluid dynamics. Each offers a **unique visual lens** 🔭 for navigating latent space. The other ten -- `spectral`, `gaussian`, `fractal`, `perlin`, `heterogeneous_fbm`, `interference`, `projection_3d`, `cellular`, `waves` and `temporal_coherent` -- are all included; the Shader Matrix documents every one of them, and the `shader_type` tooltip says what each knob does to each.

**Domain Warp**

![Domain Warp Noise Preview](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/domain_warp_preview.webp)

**Tensor Field**

![Tensor Field Noise Preview](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/tensor_field_preview.webp)

**Curl Noise**

![Curl Noise Preview](https://github.com/AEmotionStudio/ComfyUI-ShaderNoiseKSampler/releases/download/assets-v1/curl_noise_preview.webp)

-   **Blend Modes**: Determine how the crafted shader noise interacts with the underlying base noise. `Multiply` can create depth, `Add` can introduce highlights, and `Overlay` can enhance contrast.
-   **Noise Transformations**: Apply mathematical functions like `absolute` (creates ridges), `sin` (creates bands), or `sqrt` (compresses highlights) to the raw shader noise before blending, dramatically altering its characteristics.
-   **Shape Masks**: Impose geometric forms onto your noise. A `radial` mask can create focus, a `checkerboard` can introduce blocky structures. Strength is key.
-   **Color Schemes**: More than just a visual flair for the noise preview, these schemes (`viridis`, `inferno`, `jet`, etc.) transform the noise data itself. This "colored" noise can then guide the diffusion model in unique ways, influencing texture, features, and mood by altering how the model "perceives" the noise structure.

## ❓ Troubleshooting

-   **Shader Effects Not Visible**:
    -   Ensure `shader_strength` is greater than `0.0`.
    -   With `add_noise` off there is no starting noise to paint, so a single stage does nothing. Raise `sequential_stages` or `injection_stages` to give the shader a boundary to enter at.
    -   `gaussian` is plain white noise by design: it moves you toward another seed's neighbourhood without adding any pattern.

-   **Unexpected Results**: Small parameter changes can sometimes lead to large visual shifts. Even a 0.05 change in `shader_strength` can land on a noticeably different neighbour of your seed's image, or on a near-identical one; which you get depends on the model and the seed. Use the shader visualizer to understand the noise before generating. Experimentation is encouraged.

-   **Consult the Shader Matrix**: The in-app documentation is your best friend for detailed troubleshooting and understanding.

### Known Issues

-   **Fixed in 2.0.0 — Image-to-image and video-to-video**: `denoise` was ignored whenever a sequential stage ran (the default), so the input was fully regenerated instead of partially denoised. In `standard` sampling mode `denoise` now reaches the schedule. This was the cause of the weaker i2v/v2v results reported for earlier versions.

-   **Fixed in 2.0.0 — Parameter queuing**: the deprecated `ShaderNoiseKSampler` read its parameters from a file at runtime, so queued runs could not carry different settings. `Shader Noise KSampler (Direct)` takes every shader parameter as a node input, so queued runs each keep their own values.

-   **Legacy sampling mode**: nodes loaded from workflows saved before 2.0.0 switch to `sampling_mode: legacy`, which keeps the old pipeline, including the issues above. Its seeds no longer reproduce exactly for workflows using `domain_warp`, `temporal_coherent`, or `curl_noise` on latents wider than four channels, because the shader generators both modes share now fill every latent channel. Switch to `standard` for the corrected sampling.

  > [!WARNING]
    **Potential for Visual Instability**: Certain parameter explorations, particularly with high intensity or complex interactions, may result in visually disruptive outputs such as flashing images or harsh artifacts. Users are advised to iterate with caution.

## 🌱 Grassroots Research & Development Nature

ComfyUI-ShaderNoiseKSampler is born from dedicated personal research and investigation. While it introduces exciting ways to navigate latent space, please consider it an active exploration. This means there's a vast potential for further improvements, new discoveries, and community-driven enhancements as the project evolves. Your understanding of its current research-driven phase is appreciated!

### 🗺️ Roadmap: The Next Evolution

The journey into guided latent space exploration is just beginning. Here's a glimpse of what's on the horizon:

| Area                                            | Focus                                                                                                                                                                                                                                                           | Technologies Involved (Examples)          |
| :---------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- |
| **🌌 Advanced Latent Space Cartography**        | Evolve current research into an intuitive "Visual Intent Engine". This system will allow users to express desired visual outcomes semantically (e.g., "enhance fabric texture," "shift lighting ambiance"). The engine will then intelligently translate these intents into optimal shader parameter configurations, fostering a more direct and expressive artistic workflow by deeply mapping the interplay between parameters and visual impact. | Semantic AI, Parameter Response Modeling, Machine Learning |
| **🧭 Precision Navigation Tools**               | Create more granular and predictable tools for manipulating latent pathways. Research direct correlations between mathematical noise constructs and emergent visual features for greater artistic intent.                                                    | Mathematical Modelling, Latent Space Analysis |
| **🔮 Cross-Modal Exploration**                  | Investigate applying structured noise principles to modalities beyond 2D images, such as 3D and audio, opening new avenues for creative exploration.                                                                                                | Signal Processing, Generative Models for Audio/3D |

Embark on a journey down the unseen path, exploring uncharted territories within generative models. This work is dedicated to forging new, genuine approaches to advance our understanding of latent space and unlock the vast possibilities held within.

## 🤝 Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information on how to get started.

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

If you find ComfyUI-ShaderNoiseKSampler useful and wish to support its development, consider:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/aemotionstudio)

Your support helps dedicate more time to maintaining and improving this project, developing new features, and creating better documentation and tutorials.

### 💖 Additional Ways to Support

-   ⭐ Star the repository on GitHub.
-   📢 Share it with others in the AI art community.
-   🛠️ Contribute to its development (see Contributing section).
-   💡 Provide feedback and feature requests.

For business inquiries or professional support, please contact me through my [website](https://aemotionstudio.org/) or join my [Discord server](https://discord.gg/UzC9353mfp).

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.
See the `LICENSE` file for details.
