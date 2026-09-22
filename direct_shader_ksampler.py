import comfy.sample
from .shader_params_reader import build_shader_params, get_shader_params
from .shader_noise_ksampler import ShaderNoiseKSampler, get_visualizer, set_debug_level
from .core import presets as preset_table
from .pipelines import standard as standard_pipeline


class DirectShaderNoiseKSampler(ShaderNoiseKSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The AI model used for image generation"}),
                "seed": ("INT", {"default": 8888, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for generation. Same seed with same parameters will generate the same image."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of sampling steps. Higher values can produce better results but take longer"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Higher values follow the prompt more closely"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral", "tooltip": "Algorithm used for the sampling process"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta", "tooltip": "Scheduler used to determine noise level at each step"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning/prompts that guide what to include in the image"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning/prompts that guide what to exclude from the image"}),
                "latent_image": ("LATENT", {"tooltip": "Input latent image to be processed"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength. Lower values preserve more of the original image"}),
                "sequential_stages": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1, "tooltip": "Number of sequential shader stages to apply before injection stages"}),
                "injection_stages": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Number of injection shader stages to apply after sequential stages"}),
                "shader_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How far from the seed the shader takes you. 0.0 disables it. Keep the seed fixed and it is your starting point: low values explore close to the image that seed makes, higher values go further, and the shader's own pattern and colour blend progressively into the picture until, at the top, they can take it over. How fast that happens belongs to the model, the seed and noise_scale rather than to this number. Measured with domain_warp on one prompt each: SD 1.5 at 512x512 re-composes by 0.25, blends visibly at 0.5 and is mostly shader by 0.75; MiniMax H3 stays a close neighbour of the seed's scene through 0.5, lets the shader's colour in at 0.75 and blends it strongly into the scene at 1.0. curl_noise, shape masks and temporal coherence reach each stage at roughly half the value."}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference"], {"default": "multiply", "tooltip": "How shader noise is mixed into the base noise. Each mode hands the sampler a different share of the shader at the same strength -- normal the most, difference the least -- and shapes its structure differently. With normalize_strength on, the default, strength is rescaled so every mode delivers the same share as multiply, and the modes then differ in character rather than in amount. All of them keep mean 0 / std 1, so the sampler still gets noise it can denoise."}),
                "noise_transform": (["none", "reverse", "inverse", "absolute", "square", "sqrt", "log", "sin", "cos"], {"default": "none", "tooltip": "Apply mathematical transformations to the noise for creative effects"}),
                "use_temporal_coherence": ("BOOLEAN", {"default": False, "tooltip": "Hold one seed across every video frame so the shader pattern evolves only through time, instead of redrawing per frame. Frames share one pattern, so it reinforces rather than averaging out and shows at lower strength than a redrawn pattern: on MiniMax H3 it took over the picture at 0.5 and was hard to see at 0.2. No effect on single images."}),

                # New direct shader parameters
                "shader_type": (["domain_warp", "tensor_field", "curl_noise", "temporal_coherent", "spectral", "gaussian", "fractal", "perlin", "heterogeneous_fbm", "interference", "projection_3d", "cellular", "waves"], {"default": "domain_warp", "tooltip": "Which noise pattern to blend in; each has its own character when it shows. domain_warp: flowing, intricate distortions, the even-handed default. tensor_field: structured and directional. curl_noise: smooth fluid motion, and the one that shows soonest, at about half the strength of the others. temporal_coherent: 4D simplex with time as a real axis, built for smooth animation; on MiniMax H3 it shows sooner than domain_warp, as a dot-grid pattern past about 0.5. spectral: soft cloud-like fields built from their own frequency band rather than pixel by pixel -- no filaments or swirls, and colour schemes do nothing to it, but noise_scale sets the band directly and it costs almost nothing to draw (about 90ms where the others take seconds on a long video). Its character is untested against real prompts, so treat its strengths as unknown rather than calibrated. gaussian: the control -- plain white noise, the same thing the sampler already starts from, so any strength moves you toward another seed's neighbourhood without adding structure; noise_scale, octaves, warp_strength, phase_shift and colour schemes do nothing to it, shape masks still apply, and with temporal coherence a clip drifts smoothly from one white field to a second. fractal: the reference FBM, plain layered simplex; warp_strength sets the spacing between the layers and phase_shift slides each layer to a different part of the field, so neither does anything at octaves 1. perlin: classic gradient noise, smoother and more lattice-like than simplex; warp_strength swirls the detail layers while the base layer keeps its shape, and phase_shift is contrast. heterogeneous_fbm: an FBM whose detail varies across the frame, rough patches and smooth ones; warp_strength is how different they are and phase_shift how much of the frame is rough, and both need octaves above 1. interference: two FBMs turned into crossing cos and sin fringes, banded and moire-like; warp_strength sets the fringe density and phase_shift retunes the second field. projection_3d: a plane through a 3D field; phase_shift is the depth of the slice and time slides it, so a clip is coherent by construction, and warp_strength bends the plane. cellular: Worley cells; octaves picks the pattern (1 nearest distance, 2 second-nearest, 3 edges, 4 product, fractional values blend two), phase_shift the cell shape (0 diamond, 0.5 round, 2 square) and warp_strength bends the lattice. waves: octaves seeded plane waves summed, straight at warp_strength 0 and bent above it; phase_shift rearranges the same waves into a different interference pattern."}),
                "shape_type": (["none", "radial", "linear", "spiral", "checkerboard", "spots", "hexgrid", "stripes", "gradient", "vignette", "cross", "stars", "triangles", "concentric", "rays", "zigzag"], {"default": "none", "tooltip": "Mask the shader noise into a shape before it reaches the sampler (not post-processing). A mask concentrates the noise into hard geometry, which survives denoising far more readily than plain shader noise, so the shape itself starts being drawn into the picture at lower strength: on MiniMax H3 it was at 0.6, which is what the stamp preset uses. Around 0.2 the mask shapes the noise without being drawn."}),
                "color_scheme": (["none", "blue_red", "viridis", "plasma", "inferno", "magma", "turbo", "jet", "rainbow", "cool", "hot", "parula", "hsv", "autumn", "winter", "spring", "summer", "copper", "pink", "bone", "ocean", "terrain", "neon", "fire"], {"default": "none", "tooltip": "Choose a color palette to apply to the shader noise visualization [not post processing - is applied to the shader noise pattern before rendering]"}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.001, "tooltip": "Size of the shader's features: lower is larger and zoomed in, higher is smaller and zoomed out. It changes how the shader shows as much as strength does. Large features carry the pattern into the result and pull away from the seed; small ones are mostly absorbed into the picture while still steering it. On SD 1.5 at strength 0.5, 0.5 turned every seed abstract and 2.0 gave clean portraits again; on MiniMax H3 at the same strength both stayed photographic, with 0.5 re-composing the scene most. Small shifts can lead to large variations."}),
                "octaves": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Number of shader noise layers to combine - higher values add more detail and complexity. Fractional values blend between two layer counts (standard sampling only)"}),
                "warp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.001, "tooltip": "Control how much the shader noise pattern warps and distorts - higher values create more swirling or complex transformations [small adjustments are good for subtle variations]"}),
                "shape_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Adjust the intensity of the shape mask\'s effect on the shader noise pattern - higher values make the shape more prominent [small adjustments are good for subtle variations - not effective without shape mask]"}),
                "phase_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Shift the phase of the shader noise pattern to create different variations or animate patterns over time [small adjustments are good for subtle variations]"}),
                "color_intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Adjust the intensity of the color scheme application - lower values are more desaturated, higher values are more vibrant [small adjustments are good for subtle variations - not effective without color scheme]"}),
            },
            # Appended after the required widgets on purpose: saved workflows map
            # widget values by position, so new widgets must come last.
            "optional": {
                "custom_sigmas": ("SIGMAS", {"tooltip": "Optional custom sigma schedule to override the model's default schedule"}),
                "sampling_mode": (["standard", "legacy"], {"default": "standard", "tooltip": "standard: stages are segments of one sampling run, denoise and custom sigmas are honoured, and blended noise keeps the distribution the model expects. legacy: the pre-2.0 behaviour, kept so older workflows reproduce their seeds."}),
                "sequential_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across sequential stages"}),
                "injection_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across injection stages"}),
                "fast_high_channel_noise": ("BOOLEAN", {"default": False, "tooltip": "Use a faster, simplified noise generation method for models with many channels (>16), like LTXV"}),
                "preset": (["custom", "nudge", "explore", "roam", "video", "jump", "stamp"], {"default": "custom", "tooltip": "Pick one and go. A preset sets shader_type, shader_strength, blend_mode, travel_mode, stage_progression and shape_type together, and turns normalize_strength on so its strength number means the same thing in any blend mode -- those settings only mean anything in combination. nudge: the smallest visible change. explore: the recommended start. roam: further from the seed, where the shader visibly reshapes the picture and the prompt still reads. video: the 4D time-aware shader, for video latents. jump: destination set by the shader, texture rather than a scene. stamp: jump with a shape mask, so the mask itself is drawn in your prompt's material. custom leaves every widget alone. The Walk node keeps whichever parameter it is ramping."}),
                "stage_progression": (["uniform", "coarse_to_fine", "fine_to_coarse"], {"default": "uniform", "tooltip": "Vary the shader across the run instead of drawing the same one at every stage. The trajectory is not uniform -- early steps settle composition, late steps settle detail -- but every stage has always used the same zoom. coarse_to_fine starts zoomed in on large features with fewer octaves and ends zoomed out on small ones with more, so the noise matches what each part of the run is deciding; fine_to_coarse reverses it. The adjustment spans 0.5x to 2x your noise_scale and plus or minus one octave, centred on your widget values, so uniform is unchanged. The ramp needs more than one stage, but a single stage is not left alone: it sits at the start of the trajectory, so coarse_to_fine draws it zoomed in and fine_to_coarse zoomed out. Standard sampling only."}),
                "shade_non_spatial": ("BOOLEAN", {"default": False, "tooltip": "Also paint the streams that have no picture in them. Off, the shader touches only the spatial stream and everything else keeps the Gaussian noise ComfyUI gave it -- on MiniMax H3 and LTXAV that means the audio is left alone, and sequence latents (Stable Audio, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D, TripoSplat) are refused outright. On, an audio stream is painted across stereo x time, and a sequence latent is painted as a single row. Video and audio are denoised together on H3, so this reaches the picture too. Largely unexplored: audio has no busy scene for structure to blend into, so the shader changes the sound at far lower strength than it changes the picture -- on H3 the sound measurably changed by 0.05. Streams too small to be content are skipped, so TripoSplat's camera parameters are left alone. Standard sampling only."}),
                "travel_mode": (["walk", "drift", "jump"], {"default": "walk", "tooltip": "How the shader moves you. walk: the seed anchors the picture and the shader steers around it -- the wide, coherent range this node is for, and the right answer unless you want otherwise. drift: halfway, a stronger push over a narrower range. jump: the shader's parameters set the destination and the seed stops mattering; the result is a texture or pattern field in your prompt's material rather than a scene, because the model is being handed noise it was never trained to denoise. The difference is how many independent directions the noise spans across the latent's channels: walk keeps the generator's own, one field per channel; drift mixes them down to four; jump folds them into one. On a four-channel latent such as SD 1.5, four is all there is, so drift and walk come out the same. Standard sampling only."}),
                "normalize_strength": ("BOOLEAN", {"default": True, "tooltip": "Make shader_strength mean the same thing in every blend mode. Untouched, the modes differ by up to twenty-three times at the same setting: at 0.5 normal hands the sampler 0.71 of the shader and difference only 0.03. With this on, strength is read on multiply's scale, so the default mode is unchanged and the others are rescaled to match -- soft_light needs about 1.6x its old number, add and hard_light about half. difference cannot reach the top of the scale at all and saturates. On by default. On multiply it changes nothing, so the only reason to turn it off is to reproduce a workflow saved before it was on, and only if that workflow used another blend mode. Standard sampling only."}),
                "add_noise": ("BOOLEAN", {"default": True, "tooltip": "Make the noise this run starts from. Off, the latent is taken to already carry its own -- handed over by an earlier sampler that stopped with return_with_leftover_noise on -- and this node only carries on denoising it. It belongs with start_at_step: left at step 0 there is no trajectory to continue, and a flow model such as MiniMax H3 is handed an empty latent. It also leaves the shader nothing to paint at the opening, so with the default single stage the shader does nothing at all; raise injection_stages or sequential_stages and it enters at those boundaries instead, where the noise is recovered from the latent rather than made here. Standard sampling only."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Enter the schedule at this step instead of the first. With end_at_step it samples a window of the run rather than all of it, which is how a generation gets split: this node takes the early steps, something works on the latent in between -- a latent upscaler is the usual reason -- and a second node finishes from where this one stopped. The schedule is still the full steps long and denoise still shapes it; this only chooses where to enter. Stages spread across the window, so they divide the steps actually sampled. Standard sampling only."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "Stop after this step instead of running the schedule out; anything at or past steps runs to the end. With return_with_leftover_noise off the latent is still brought to a clean finish here, which is what an upscaler or any node that works on a picture needs. Standard sampling only."}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False, "tooltip": "Hand the latent over still noisy when end_at_step stops the run early, instead of finishing it cleanly. Turn it on for the first half of a split whose second half has add_noise off: the two then continue one trajectory exactly, with nothing re-noised in between. Leave it off when something between the halves needs a finished latent to work on. No effect when the run reaches the end of its schedule. Standard sampling only."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    # The base class is deprecated; this node is not. Without this the flag
    # would be inherited and ComfyUI would hide this node from node search too.
    DEPRECATED = False

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               denoise=1.0, sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
               noise_transform="none", use_temporal_coherence=False,
               shader_type="domain_warp", shape_type="none", color_scheme="none", noise_scale=1.0, octaves=1.0,
               warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.5, color_intensity=0.8,
               sampling_mode="standard", sequential_distribution="linear_decrease",
               injection_distribution="linear_decrease", fast_high_channel_noise=False,
               normalize_strength=True, travel_mode="walk", shade_non_spatial=False,
               stage_progression="uniform", preset="custom", custom_sigmas=None,
               add_noise=True, start_at_step=0, end_at_step=10000, return_with_leftover_noise=False,
               # Accepted for the legacy path and for older callers; not exposed as inputs.
               debug_level="0-Off", denoise_visualization_frequency="25% intervals", target_attribute_changes=""):
        """Run the shader noise sampler with direct parameter inputs."""
        # A preset speaks for several widgets at once, so it has to land before
        # anything reads them. `_preset_exclude` lets a subclass keep an input it
        # is driving itself -- the Walk node uses it for the parameter it ramps.
        chosen = preset_table.apply_preset(preset, dict(
            shader_type=shader_type, shader_strength=shader_strength, blend_mode=blend_mode,
            travel_mode=travel_mode, stage_progression=stage_progression, shape_type=shape_type,
            normalize_strength=normalize_strength,
        ), exclude=getattr(self, "_preset_exclude", ()))
        shader_type = chosen["shader_type"]
        shader_strength = chosen["shader_strength"]
        blend_mode = chosen["blend_mode"]
        travel_mode = chosen["travel_mode"]
        stage_progression = chosen["stage_progression"]
        shape_type = chosen["shape_type"]
        normalize_strength = chosen["normalize_strength"]

        debugger = set_debug_level(int(debug_level.split("-")[0]))
        get_visualizer()

        shader_params = build_shader_params(
            get_shader_params(), seed, shader_type, shape_type, color_scheme, noise_scale,
            octaves, warp_strength, shape_mask_strength, phase_shift, color_intensity,
            use_temporal_coherence, fast_high_channel_noise,
        )

        if debugger.enabled:
            print(f"🔧 Direct shader parameters: type={shader_type} shape={shape_type} colour={color_scheme} "
                  f"scale={noise_scale} octaves={octaves} warp={warp_strength} phase={phase_shift}")

        if sampling_mode == "legacy":
            return super().sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                sequential_stages=sequential_stages,
                injection_stages=injection_stages,
                shader_strength=shader_strength,
                blend_mode=blend_mode,
                noise_transform=noise_transform,
                sequential_distribution=sequential_distribution,
                injection_distribution=injection_distribution,
                use_temporal_coherence=use_temporal_coherence,
                debug_level=debug_level,
                fast_high_channel_noise=fast_high_channel_noise,
                denoise_visualization_frequency=denoise_visualization_frequency,
                custom_sigmas=custom_sigmas,
                target_attribute_changes=target_attribute_changes,
                shader_params_override=shader_params,
            )

        result = standard_pipeline.run(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=denoise,
            sequential_stages=sequential_stages,
            injection_stages=injection_stages,
            shader_strength=shader_strength,
            blend_mode=blend_mode,
            noise_transform=noise_transform,
            shader_params=shader_params,
            shader_type=shader_type,
            sequential_distribution=sequential_distribution,
            injection_distribution=injection_distribution,
            use_temporal_coherence=use_temporal_coherence,
            normalize_strength=normalize_strength,
            travel_mode=travel_mode,
            shade_non_spatial=shade_non_spatial,
            stage_progression=stage_progression,
            custom_sigmas=custom_sigmas,
            add_noise=add_noise,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            return_with_leftover_noise=return_with_leftover_noise,
        )

        shader_info = {
            "shader_type": shader_type,
            "shader_strength": shader_strength,
            "sequential_stages": sequential_stages,
            "injection_stages": injection_stages,
            "blend_mode": blend_mode,
            "noise_transform": noise_transform,
            "sampling_mode": sampling_mode,
            "normalize_strength": normalize_strength,
            "travel_mode": travel_mode,
            "preset": preset,
            "shade_non_spatial": shade_non_spatial,
            "stage_progression": stage_progression,
        }
        return {"ui": {"images": [], "shader_info": shader_info}, "result": (result,)}
