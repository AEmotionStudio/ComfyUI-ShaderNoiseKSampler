import comfy.sample
from .shader_params_reader import get_shader_params, ShaderParamsReader
from .shader_noise_ksampler import ShaderNoiseKSampler, get_visualizer, set_debug_level
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
                "shader_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the shader noise influence for both stage types. Set to 0.0 to disable and use only base noise."}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference"], {"default": "multiply", "tooltip": "Method used to blend the shader noise with the base noise"}),
                "noise_transform": (["none", "reverse", "inverse", "absolute", "square", "sqrt", "log", "sin", "cos"], {"default": "none", "tooltip": "Apply mathematical transformations to the noise for creative effects"}),
                "use_temporal_coherence": ("BOOLEAN", {"default": False, "tooltip": "Ensures consistent noise patterns. For sequences like video frames, it helps maintain frame-to-frame consistency (e.g., using the same base seed and 4D noise). For single image generations, it ensures the base noise is derived consistently from the main seed."}),

                # New direct shader parameters
                "shader_type": (["domain_warp", "tensor_field", "curl_noise"], {"default": "domain_warp", "tooltip": "Select the type of shader noise pattern to use in the visualization [different types have different characteristic outputs]"}),
                "shape_type": (["none", "radial", "linear", "spiral", "checkerboard", "spots", "hexgrid", "stripes", "gradient", "vignette", "cross", "stars", "triangles", "concentric", "rays", "zigzag"], {"default": "none", "tooltip": "Apply a shape mask to the shader noise pattern to create more complex structures [not post processing - is applied to shader noise pattern before rendering]"}),
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
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
               denoise=1.0, sequential_stages=1, injection_stages=0, shader_strength=0.3, blend_mode="multiply",
               noise_transform="none", use_temporal_coherence=False,
               shader_type="domain_warp", shape_type="none", color_scheme="none", noise_scale=1.0, octaves=1.0,
               warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.5, color_intensity=0.8,
               sampling_mode="standard", sequential_distribution="linear_decrease",
               injection_distribution="linear_decrease", fast_high_channel_noise=False, custom_sigmas=None,
               # Accepted for the legacy path and for older callers; not exposed as inputs.
               debug_level="0-Off", denoise_visualization_frequency="25% intervals", target_attribute_changes=""):
        """Run the shader noise sampler with direct parameter inputs."""
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
        }
        return {"ui": {"images": [], "shader_info": shader_info}, "result": (result,)}
