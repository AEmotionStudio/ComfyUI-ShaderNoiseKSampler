import comfy.sample
from .shader_params_reader import get_shader_params, ShaderParamsReader
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
                "shader_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much shader noise replaces the base noise. 0.0 disables it. Raising it walks further from the seed, but past a point the shader's own structure survives denoising and shows up in the image as colour casts and blobs. Where that point sits depends on the model and the scene. Measured with domain_warp under real prompts: MiniMax H3 holds to about 0.75, and SD 1.5 at 512x512 to about 0.35 on a portrait and 0.45 on a busy landscape. Large dark or flat areas give way first. Start low and climb. temporal_coherent gives way sooner, past about 0.5 on H3, and curl_noise, shape masks and temporal coherence all need roughly half the value you would use otherwise."}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference"], {"default": "multiply", "tooltip": "How shader noise is mixed into the base noise. Gentlest first: difference and soft_light tolerate the highest strength, then multiply and normal, then overlay, screen and hard_light; add is the most aggressive and needs the lowest strength. All of them keep mean 0 / std 1, so the sampler still gets the distribution it expects."}),
                "noise_transform": (["none", "reverse", "inverse", "absolute", "square", "sqrt", "log", "sin", "cos"], {"default": "none", "tooltip": "Apply mathematical transformations to the noise for creative effects"}),
                "use_temporal_coherence": ("BOOLEAN", {"default": False, "tooltip": "Hold one seed across every video frame so the shader pattern evolves only through time, instead of redrawing per frame. Ties frames together, but because the pattern no longer varies between them it reinforces rather than averages out: halve your shader_strength when you turn this on. On MiniMax H3 at 0.5 it swamps the picture, while 0.2 is clean. No effect on single images."}),

                # New direct shader parameters
                "shader_type": (["domain_warp", "tensor_field", "curl_noise", "temporal_coherent"], {"default": "domain_warp", "tooltip": "Which noise pattern to walk with. domain_warp: flowing, intricate distortions, the most even-handed default and the most tolerant of strength -- under a real prompt on MiniMax H3 it holds to about 0.75. tensor_field: structured and directional. curl_noise: smooth fluid motion, but the most aggressive -- use roughly half the strength. temporal_coherent: 4D simplex with time as a real axis, built for smooth animation, but it gives way sooner than domain_warp: on H3 it turns into a dot-grid pattern past about 0.5. The live preview only draws the first three; picking temporal_coherent leaves the preview on its last pattern, which does not affect sampling."}),
                "shape_type": (["none", "radial", "linear", "spiral", "checkerboard", "spots", "hexgrid", "stripes", "gradient", "vignette", "cross", "stars", "triangles", "concentric", "rays", "zigzag"], {"default": "none", "tooltip": "Mask the shader noise into a shape before it reaches the sampler (not post-processing). A mask concentrates the noise into hard geometry, so it survives denoising far more readily than plain shader noise -- on MiniMax H3 at strength 0.6 the mask itself is drawn into the picture. Keep strength at or below about 0.2 when a shape is active."}),
                "color_scheme": (["none", "blue_red", "viridis", "plasma", "inferno", "magma", "turbo", "jet", "rainbow", "cool", "hot", "parula", "hsv", "autumn", "winter", "spring", "summer", "copper", "pink", "bone", "ocean", "terrain", "neon", "fire"], {"default": "none", "tooltip": "Choose a color palette to apply to the shader noise visualization [not post processing - is applied to the shader noise pattern before rendering]"}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.001, "tooltip": "Adjust the scale of the shader noise pattern - lower values create larger, zoomed-in features; higher values create smaller, zoomed-out features [small value shifts can lead to larger variations]"}),
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
                "preset": (["custom", "nudge", "explore", "roam", "video", "jump", "stamp"], {"default": "custom", "tooltip": "Pick one and go. A preset sets shader_type, shader_strength, blend_mode, travel_mode, stage_progression and shape_type together, and turns normalize_strength on so its strength number means the same thing in any blend mode -- those settings only mean anything in combination, and choosing them one at a time is how you end up at 0.6 with a shape mask and a picture made of hexagons. nudge: the smallest visible change. explore: the recommended start. roam: as far as the picture reliably holds. video: the 4D time-aware shader, for video latents. jump: destination set by the shader, texture rather than a scene. stamp: jump with a shape mask, so the mask itself is drawn in your prompt's material. custom leaves every widget alone. The Walk node keeps whichever parameter it is ramping."}),
                "stage_progression": (["uniform", "coarse_to_fine", "fine_to_coarse"], {"default": "uniform", "tooltip": "Vary the shader across the run instead of drawing the same one at every stage. The trajectory is not uniform -- early steps settle composition, late steps settle detail -- but every stage has always used the same zoom. coarse_to_fine starts zoomed in on large features with fewer octaves and ends zoomed out on small ones with more, so the noise matches what each part of the run is deciding; fine_to_coarse reverses it. The adjustment spans 0.5x to 2x your noise_scale and plus or minus one octave, centred on your widget values, so uniform is unchanged. Needs more than one stage to do anything. Standard sampling only."}),
                "shade_non_spatial": ("BOOLEAN", {"default": False, "tooltip": "Also paint the streams that have no picture in them. Off, the shader touches only the spatial stream and everything else keeps the Gaussian noise ComfyUI gave it -- on MiniMax H3 and LTXAV that means the audio is left alone, and sequence latents (Stable Audio, ACE-Step 1.5, MiniMax Music 3, Hunyuan3D, TripoSplat) are refused outright. On, an audio stream is painted across stereo x time, and a sequence latent is painted as a single row. Video and audio are denoised together on H3, so this reaches the picture too. Unexplored and easy to overdo: audio has no busy scene to hide structure in, so start near 0.05. Streams too small to be content are skipped, so TripoSplat's camera parameters are left alone. Standard sampling only."}),
                "travel_mode": (["walk", "drift", "jump"], {"default": "walk", "tooltip": "How the shader moves you. walk: the seed anchors the picture and the shader steers around it -- the wide, coherent range this node is for, and the right answer unless you want otherwise. drift: halfway, a stronger push over a narrower range. jump: the shader's parameters set the destination and the seed stops mattering; the result is a texture or pattern field in your prompt's material rather than a scene, because the model is being handed noise it was never trained to denoise. The difference is how many independent directions the noise spans across the latent's channels: walk keeps the generator's own, one field per channel; drift mixes them down to four; jump folds them into one. On a four-channel latent such as SD 1.5, four is all there is, so drift and walk come out the same. Standard sampling only."}),
                "normalize_strength": ("BOOLEAN", {"default": True, "tooltip": "Make shader_strength mean the same thing in every blend mode. Untouched, the modes differ by up to twenty-three times at the same setting: at 0.5 normal hands the sampler 0.71 of the shader and difference only 0.03. With this on, strength is read on multiply's scale, so the default mode is unchanged and the others are rescaled to match -- soft_light needs about 1.6x its old number, add and hard_light about half. difference cannot reach the top of the scale at all and saturates. On by default. On multiply it changes nothing, so the only reason to turn it off is to reproduce a workflow saved before it was on, and only if that workflow used another blend mode. Standard sampling only."}),
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

        # Start from the saved params file, then override with this node's inputs.
        shader_params = get_shader_params()

        # Every generator reads a different spelling of these, so set all variants.
        shader_params["shader_type"] = shader_type
        shader_params["shaderType"] = shader_type

        shader_params["shape_type"] = shape_type
        shader_params["shaderShapeType"] = shape_type

        shader_params["colorScheme"] = color_scheme
        shader_params["color_scheme"] = color_scheme

        shader_params["scale"] = noise_scale
        shader_params["shaderScale"] = noise_scale

        shader_params["octaves"] = float(octaves)
        shader_params["shaderOctaves"] = float(octaves)

        shader_params["warp_strength"] = warp_strength
        shader_params["shaderWarpStrength"] = warp_strength

        shader_params["shapemaskstrength"] = shape_mask_strength
        shader_params["shaderShapeStrength"] = shape_mask_strength
        shader_params["shapeMaskStrength"] = shape_mask_strength
        shader_params["shape_mask_strength"] = shape_mask_strength
        shader_params["shape_strength"] = shape_mask_strength

        shader_params["phase_shift"] = phase_shift
        shader_params["shaderPhaseShift"] = phase_shift

        shader_params["intensity"] = color_intensity
        shader_params["shaderColorIntensity"] = color_intensity

        shader_params["time"] = shader_params.get("time", 0.0)
        shader_params["base_seed"] = seed
        shader_params["useTemporalCoherence"] = use_temporal_coherence
        shader_params["temporal_coherence"] = use_temporal_coherence
        shader_params["fast_high_channel_noise"] = fast_high_channel_noise
        shader_params["visualization_type"] = shader_params.get("visualization_type", 3)

        # Clamp octaves, seeds and enum values before they reach noise generation.
        shader_params = ShaderParamsReader.validate_and_sanitize_params(shader_params)
        # Sanitising truncates octaves to an integer; the standard pipeline
        # interpolates between integer renders, so keep the requested value.
        shader_params["octaves"] = float(octaves)

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
