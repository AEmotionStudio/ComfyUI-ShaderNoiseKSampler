import comfy.sample
from .shader_params_reader import get_shader_params
from .shader_noise_ksampler import ShaderNoiseKSampler, get_visualizer, set_debug_level

# Define a simple parameter response mapper if needed
class ParameterResponseMapper:
    """Simplified version of ParameterResponseMapper for direct shader parameters"""
    def __init__(self, model_type="SD1.5"):
        self.model_type = model_type
    
    def get_recommended_adjustments(self, target_attrs, current_params, max_params=2):
        """Return recommended parameter adjustments based on target attributes"""
        # This is a simple implementation that could be expanded
        adjustments = {}
        
        if not target_attrs or not isinstance(target_attrs, dict):
            return adjustments
            
        # Simple mapping from attributes to shader parameters
        for attr, value in target_attrs.items():
            # Only adjust if the attribute is strongly present (value > 0.5)
            if value < 0.5:
                continue
                
            # Map common attributes to shader parameters
            if attr == "detailed" and "octaves" in current_params:
                adjustments["octaves"] = min(current_params["octaves"] + 1, 8)
            elif attr == "smooth" and "octaves" in current_params:
                adjustments["octaves"] = max(current_params["octaves"] - 1, 1)
            elif attr == "twisted" and "warp_strength" in current_params:
                adjustments["warp_strength"] = min(current_params["warp_strength"] + 0.2, 5.0)
            elif attr == "flowing" and "phase_shift" in current_params:
                adjustments["phase_shift"] = (current_params["phase_shift"] + 0.5) % 6.28
            elif attr == "large_scale" and "scale" in current_params:
                adjustments["scale"] = min(current_params["scale"] + 0.5, 10.0)
            elif attr == "small_scale" and "scale" in current_params:
                adjustments["scale"] = max(current_params["scale"] - 0.5, 0.1)
                
            # Limit number of adjustments
            if len(adjustments) >= max_params:
                break
                
        return adjustments

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
                "octaves": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Number of shader noise layers to combine - higher values add more detail and complexity [small value shifts can lead to larger variations]"}),
                "warp_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.001, "tooltip": "Control how much the shader noise pattern warps and distorts - higher values create more swirling or complex transformations [small adjustments are good for subtle variations]"}),
                "shape_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Adjust the intensity of the shape mask\'s effect on the shader noise pattern - higher values make the shape more prominent [small adjustments are good for subtle variations - not effective without shape mask]"}),
                "phase_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.0001, "tooltip": "Shift the phase of the shader noise pattern to create different variations or animate patterns over time [small adjustments are good for subtle variations]"}),
                "color_intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Adjust the intensity of the color scheme application - lower values are more desaturated, higher values are more vibrant [small adjustments are good for subtle variations - not effective without color scheme]"}),
            },
            "optional": {
                "custom_sigmas": ("SIGMAS", {"tooltip": "Optional custom sigma schedule to override the model's default schedule"}),
            },
            "hidden": {
                "debug_level": (["0-Off", "1-Basic", "2-Detailed", "3-Verbose"], {"default": "0-Off", "tooltip": "Enable debugging at specified level to understand what's happening during shader generation and sampling."}),
                "fast_high_channel_noise": ("BOOLEAN", {"default": False, "tooltip": "Use a faster, simplified noise generation method for models with many channels (>16), like LTXV."}),
                "sequential_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across sequential stages"}),
                "injection_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across injection stages"}),
                "denoise_visualization_frequency": (["Every step", "25% intervals", "10% intervals", "4 steps", "2 steps"], {"default": "Every step", "tooltip": "How often to save images during the denoising process. Higher frequency means more images but slower generation."}),
                "target_attribute_changes": ("STRING", {"forceInput": True, "tooltip": "Connect output from ParameterResponseMapperNode here"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    @classmethod
    def REGISTER_MATRIX_BUTTON(s):
        return True

    @classmethod
    def IS_CHANGED(s, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                  denoise=1.0, sequential_stages=2, injection_stages=3, shader_strength=0.3, blend_mode="multiply", 
                  noise_transform="none", sequential_distribution="linear_decrease", injection_distribution="linear_decrease",
                  use_temporal_coherence=False, debug_level="0-Off", fast_high_channel_noise=False, 
                  denoise_visualization_frequency="25% intervals", custom_sigmas=None, target_attribute_changes="",
                  shader_type="domain_warp", shape_type="none", color_scheme="none", noise_scale=1.0, octaves=1,
                  warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.5, color_intensity=0.8):
        # Ensure all direct parameters also trigger re-execution when changed
        return (seed, steps, cfg, sampler_name, scheduler, denoise, sequential_stages,
                injection_stages, shader_strength, blend_mode, noise_transform,
                sequential_distribution, injection_distribution, use_temporal_coherence,
                debug_level, denoise_visualization_frequency, custom_sigmas, 
                target_attribute_changes, fast_high_channel_noise,
                # Include direct shader parameters
                shader_type, shape_type, color_scheme, noise_scale, octaves,
                warp_strength, shape_mask_strength, phase_shift, color_intensity)

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
              denoise=1.0, sequential_stages=2, injection_stages=3, shader_strength=0.3, blend_mode="multiply", 
              noise_transform="none", sequential_distribution="linear_decrease", injection_distribution="linear_decrease",
              use_temporal_coherence=False, debug_level="0-Off", fast_high_channel_noise=False, 
              denoise_visualization_frequency="25% intervals", custom_sigmas=None, target_attribute_changes="",
              shader_type="tensor_field", shape_type="none", color_scheme="none", noise_scale=1.0, octaves=3.0,
              warp_strength=0.5, shape_mask_strength=1.0, phase_shift=0.0, color_intensity=0.8):
        """
        Run the multi-stage shader noise k-sampler with direct parameter inputs
        """
        # Parse debug level from the selected option
        debug_level_value = int(debug_level.split("-")[0])
        
        # Set the debug level in the shader debugger
        debugger = set_debug_level(debug_level_value)
        
        # Get the visualizer
        visualizer = get_visualizer()

        # Get device early from latent_image
        device = latent_image["samples"].device
        if debugger.enabled:
            print(f"ℹ️ Using device: {device}")

        # Get the shader parameters, but we'll modify them with our direct inputs
        shader_params = get_shader_params()
        
        # Override shader parameters with direct inputs - ensure all naming variants are set
        # Shader Type (set all variants)
        shader_params["shader_type"] = shader_type
        shader_params["shaderType"] = shader_type
        
        # Shape Type (set all variants)
        shader_params["shape_type"] = shape_type
        shader_params["shaderShapeType"] = shape_type
        
        # Color Scheme
        shader_params["colorScheme"] = color_scheme
        shader_params["color_scheme"] = color_scheme
        
        # Noise Scale (set all variants)
        shader_params["scale"] = noise_scale
        shader_params["shaderScale"] = noise_scale
        
        # Octaves (set all variants)
        shader_params["octaves"] = float(octaves)  # Ensure float for compatibility
        shader_params["shaderOctaves"] = float(octaves)
        
        # Warp Strength (set all variants)
        shader_params["warp_strength"] = warp_strength
        shader_params["shaderWarpStrength"] = warp_strength
        
        # Shape Mask Strength (set all variants)
        shader_params["shapemaskstrength"] = shape_mask_strength
        shader_params["shaderShapeStrength"] = shape_mask_strength
        shader_params["shapeMaskStrength"] = shape_mask_strength  # Added capital M version
        shader_params["shape_mask_strength"] = shape_mask_strength  # Added underscore version
        shader_params["shape_strength"] = shape_mask_strength  # Added alternative name checked in shader_to_tensor.py
        
        # Phase Shift (set all variants)
        shader_params["phase_shift"] = phase_shift
        shader_params["shaderPhaseShift"] = phase_shift
        
        # Color Intensity (set all variants)
        shader_params["intensity"] = color_intensity
        shader_params["shaderColorIntensity"] = color_intensity
        
        # Other parameters
        shader_params["time"] = shader_params.get("time", 0.0)  # Keep existing time or set to 0
        shader_params["base_seed"] = seed  # Set base seed for temporal coherence
        shader_params["useTemporalCoherence"] = use_temporal_coherence
        shader_params["temporal_coherence"] = use_temporal_coherence  # Add alternative name
        shader_params["fast_high_channel_noise"] = fast_high_channel_noise
        
        # Visualization type (default to 3/ellipses as in the default parameters)
        shader_params["visualization_type"] = shader_params.get("visualization_type", 3)

        # --- Handle Parameter Response Mapper Integration ---
        if target_attribute_changes and target_attribute_changes.strip():
            if debugger.enabled:
                print("🧠 Applying Parameter Response Mapper adjustments...")
            try:
                import json
                target_attrs = json.loads(target_attribute_changes)
                
                if isinstance(target_attrs, dict) and target_attrs:
                    # Determine model type for mapper (simple inference)
                    model_name_lower = getattr(model, 'model_name', "").lower()
                    if "sdxl" in model_name_lower:
                        mapper_model_type = "SDXL"
                    else:
                        mapper_model_type = "SD1.5" # Default
                    
                    # Use our local ParameterResponseMapper
                    mapper = ParameterResponseMapper(model_type=mapper_model_type)
                    
                    # Extract current shader parameters relevant to the mapper
                    current_mapper_params = {
                        "scale": noise_scale,
                        "octaves": float(octaves),
                        "warp_strength": warp_strength,
                        "phase_shift": phase_shift
                    }

                    if debugger.enabled and debugger.debug_level >= 2:
                        print(f"   Mapper using model_type: {mapper_model_type}")
                        print(f"   Target Attributes: {target_attrs}")
                        print(f"   Current Params for Mapper: {current_mapper_params}")

                    # Get recommendations
                    recommendations = mapper.get_recommended_adjustments(
                        target_attrs,
                        current_mapper_params,
                        max_params=2
                    )
                    
                    if recommendations:
                        if debugger.enabled:
                            print(f"✅ Mapper recommended adjustments: {recommendations}")
                        # Apply recommendations to the shader_params - set ALL variants
                        for param, value in recommendations.items():
                            # Update all possible parameter naming variants
                            if param == "scale":
                                shader_params["scale"] = value
                                shader_params["shaderScale"] = value
                            elif param == "octaves":
                                shader_params["octaves"] = float(value)
                                shader_params["shaderOctaves"] = float(value)
                            elif param == "warp_strength":
                                shader_params["warp_strength"] = value
                                shader_params["shaderWarpStrength"] = value
                            elif param == "phase_shift":
                                shader_params["phase_shift"] = value
                                shader_params["shaderPhaseShift"] = value
                            else:
                                # For any other parameters
                                shader_params[param] = value
            except Exception as e:
                print(f"❌ Error processing target_attribute_changes: {e}. Skipping mapper adjustments.")

        if debugger.enabled:
            # Debug output for direct parameters
            print(f"🔧 Direct Shader Parameters:")
            print(f"   Shader Type: {shader_type}")
            print(f"   Shape Type: {shape_type}")
            print(f"   Color Scheme: {color_scheme}")
            print(f"   Noise Scale: {noise_scale}")
            print(f"   Octaves: {octaves}")
            print(f"   Warp Strength: {warp_strength}")
            print(f"   Shape Mask Strength: {shape_mask_strength}")
            print(f"   Phase Shift: {phase_shift}")
            print(f"   Color Intensity: {color_intensity}")
        
        # Call parent class sample method with the modified shader params
        # Use the parent class implementation from ShaderNoiseKSampler
        result = super().sample(
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
            shader_params_override=shader_params,  # Pass our modified shader params to parent
        )
        
        return result 