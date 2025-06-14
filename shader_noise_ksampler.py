import torch
import math
import comfy.sample
import contextlib
from nodes import common_ksampler
from .shader_params_reader import get_shader_params, generate_noise_tensor
# Add imports for custom sigma handling
import comfy.samplers
import comfy.model_sampling
import comfy.model_base # For ModelType enum
import comfy.latent_formats # For default latent format

# Stub visualizer class as replacement for removed visualizer
class StubVisualizer:
    """A stub visualizer that implements the same interface but does nothing. Temporary replacement for removed visualizer."""
    def __init__(self):
        self.enabled = False
        
    def enable(self, seed=None, shader_type=None, additional_metadata=None):
        """Stub for enable method."""
        pass
        
    def disable(self):
        """Stub for disable method."""
        pass
        
    def save_latent_visualization(self, tensor, label, stage_info=None, is_sample=False):
        """Stub for save_latent_visualization method."""
        pass
        
    def save_denoising_step(self, tensor, stage_info, current_step, total_steps):
        """Stub for save_denoising_step method."""
        pass
        
    def capture_shader_process(self, phase, stage_idx, stage_type, stage_data, base_noise, shader_noise, blended_noise, result):
        """Stub for capture_shader_process method."""
        pass
        
    def capture_final_result(self, tensor, metadata=None):
        """Stub for capture_final_result method."""
        pass
        
    def get_ui_image_paths(self):
        """Stub for get_ui_image_paths method."""
        return {
            "base_noise": None,
            "shader_noise": None,
            "blended_noise": None,
            "stage_results": None,
            "final_result": None,
            "grids": []
        }

# Also need a stub for the debugger
class StubDebugger:
    """A stub debugger that implements the same interface but does nothing."""
    def __init__(self):
        self.enabled = False
        self.debug_level = 0
        
    def reset(self):
        """Stub for reset method."""
        pass
        
    def time_operation(self, name):
        """Stub for time_operation method."""
        return contextlib.nullcontext()
        
    def analyze_tensor(self, tensor, name):
        """Stub for analyze_tensor method."""
        pass
        
    def log_parameters(self, params):
        """Stub for log_parameters method."""
        pass
        
    def log_stage_start(self, stage_type, stage_idx, params):
        """Stub for log_stage_start method."""
        pass
        
    def log_stage_end(self, stage_type, stage_idx):
        """Stub for log_stage_end method."""
        pass
        
    def log_blend_operation(self, base, shader, result, mode, strength):
        """Stub for log_blend_operation method."""
        pass

# Functions to get the stub instances
def get_visualizer():
    """Return the stub visualizer instance."""
    return StubVisualizer()

def get_debugger():
    """Return the stub debugger instance."""
    return StubDebugger()

def set_debug_level(level):
    """Return the stub debugger with the specified level."""
    debugger = get_debugger()
    debugger.debug_level = level
    debugger.enabled = level > 0
    return debugger

class CustomSigmaProvider:
    """
    A helper class to provide sigma values from a custom tensor,
    mimicking the interface of comfy.model_sampling classes.
    """
    def __init__(self, sigmas_tensor):
        # Ensure sigmas are sorted descending
        if sigmas_tensor.numel() > 1: # Avoid error for single-element tensors
             if torch.all(sigmas_tensor[1:] <= sigmas_tensor[:-1]): # Allow non-strict inequality
                 self.sigmas = sigmas_tensor.float()
             elif torch.all(sigmas_tensor[1:] >= sigmas_tensor[:-1]): # Allow non-strict inequality
                 self.sigmas = torch.flip(sigmas_tensor, (0,)).float()
             else:
                 # If not sorted or has duplicates, sort descending
                 self.sigmas = torch.sort(sigmas_tensor, descending=True)[0].float()
        else:
            self.sigmas = sigmas_tensor.float() # Handle single value tensor

        self.log_sigmas = self.sigmas.log()
        self.num_timesteps = len(self.sigmas)

    @property
    def sigma_min(self):
        return self.sigmas[-1]

    @property
    def sigma_max(self):
        return self.sigmas[0]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        log_sigmas_dev = self.log_sigmas.to(log_sigma.device)
        dists = log_sigma - log_sigmas_dev[:, None]
        indices = torch.sum(dists < 0, dim=0) # Find last index where log_sigma >= log_sigmas_dev[index]
        # Handle edge case where sigma is larger than sigma_max
        indices[log_sigma > log_sigmas_dev[0]] = 0
        indices = torch.clamp(indices, 0, self.num_timesteps - 1)
        return indices.view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        sigmas_dev = self.sigmas.to(timestep.device)
        log_sigmas_dev = self.log_sigmas.to(timestep.device)

        t = torch.clamp(timestep.float(), min=0, max=(self.num_timesteps - 1))
        # Use integer indices directly if timestep is integer
        if torch.all(t == t.long()):
            return sigmas_dev[t.long()].to(timestep.device)
        else:
             # Interpolate for float timesteps
             low_idx = t.floor().long()
             high_idx = t.ceil().long()
             # Clamp indices to be within bounds [0, num_timesteps - 1]
             low_idx = torch.clamp(low_idx, 0, self.num_timesteps - 1)
             high_idx = torch.clamp(high_idx, 0, self.num_timesteps - 1)
             w = t.frac()
             # Perform interpolation using sigmas on the correct device
             log_sigma_val = (1 - w) * log_sigmas_dev[low_idx] + w * log_sigmas_dev[high_idx]
             return log_sigma_val.exp().to(timestep.device)

class CustomSigmaModelWrapper:
    """Wraps a model to override its model_sampling attribute."""
    # Add device argument to __init__
    def __init__(self, model, sigmas_tensor, device):
        self.original_model = model
        # Use the explicitly passed device
        self.model_sampling = CustomSigmaProvider(sigmas_tensor.to(device))
        # Copy essential attributes
        self.model_type = getattr(model, 'model_type', comfy.model_base.ModelType.EPS)
        self.latent_format = getattr(model, 'latent_format', comfy.latent_formats.SD15())
        # Store the explicitly passed device
        self.device = device

    def __getattr__(self, name):
        if name in ['original_model', 'model_sampling', 'model_type', 'latent_format', 'device']:
            return object.__getattribute__(self, name)
        try:
            return getattr(self.original_model, name)
        except AttributeError:
             raise AttributeError(f"'{type(self).__name__}' object (wrapping '{type(self.original_model).__name__}') has no attribute '{name}'")

    def apply_model(self, *args, **kwargs):
         if hasattr(self.original_model, 'apply_model') and callable(self.original_model.apply_model):
             return self.original_model.apply_model(*args, **kwargs)
         else:
             raise NotImplementedError(f"The wrapped model '{type(self.original_model).__name__}' does not have an 'apply_model' method.")

# Define a registry for shader generators
SHADER_GENERATORS = {}
# Define a registry for shader generators
SHADER_GENERATORS = {}

# Function to get the appropriate generator
def get_shader_generator(shader_type):
    """Get the appropriate shader generator function based on shader type"""
    generator = SHADER_GENERATORS.get(shader_type)
    if generator:
        return generator
    else:
        return generate_noise_tensor

# Function to register shader generators
def register_shader_generator(shader_type, generator_function):
    """Register a shader generator function for a specific shader type"""
    SHADER_GENERATORS[shader_type] = generator_function
    return generator_function

# Create a custom sampling callback class
class DenoisingStepCallback:
    def __init__(self, visualizer=None, stage_info=None, frequency="25% intervals"):
        self.visualizer = visualizer
        self.stage_info = stage_info or {}
        self.frequency = frequency
        self.enable_logging = visualizer is not None and visualizer.enabled
        self.last_saved_step = -1
        # For animation compatibility
        self.total_steps_value = None
        self.step_counter = 0
    
    def __call__(self, step, x0, x, total_steps):
        if not self.enable_logging:
            return False
        
        try:
            # We'll use a step counter instead of the potentially problematic step input
            current_step = self.step_counter
            
            # On first call, try to extract total_steps safely
            if self.total_steps_value is None:
                # Try to safely get a scalar value
                if isinstance(total_steps, int):
                    self.total_steps_value = total_steps
                elif hasattr(total_steps, 'item') and hasattr(total_steps, 'numel') and total_steps.numel() == 1:
                    self.total_steps_value = total_steps.item()
                else:
                    # Default value if we can't determine
                    self.total_steps_value = 20
                    print(f"⚠️ Using default value of {self.total_steps_value} steps for visualization")
            
            # If we're at the beginning of a new sampling iteration, reset our counter
            if isinstance(step, int) and step == 0:
                self.step_counter = 0
            
            # Determine if we should save this step based on the selected frequency
            should_save = False
            
            # Always save first and last step
            if current_step == 0 or current_step >= self.total_steps_value - 1:
                should_save = True
            # Apply frequency filter
            elif self.frequency == "Every step":
                should_save = True
            elif self.frequency == "25% intervals":
                step_percent = (current_step / self.total_steps_value) * 100
                interval_size = (1 / self.total_steps_value) * 100
                should_save = (step_percent % 25 < interval_size)
            elif self.frequency == "10% intervals":
                step_percent = (current_step / self.total_steps_value) * 100
                interval_size = (1 / self.total_steps_value) * 100
                should_save = (step_percent % 10 < interval_size)
            elif self.frequency == "4 steps":
                should_save = (current_step % max(1, self.total_steps_value // 4) == 0)
            elif self.frequency == "2 steps":
                should_save = (current_step % max(1, self.total_steps_value // 2) == 0)
                
            # Check to avoid duplicate steps
            if should_save and current_step != self.last_saved_step:
                try:
                    # Safely get a copy of the tensor
                    if hasattr(x, 'clone'):
                        x_copy = x.clone()
                    else:
                        x_copy = x
                    
                    # Capture intermediate denoising state
                    self.visualizer.save_denoising_step(
                        x_copy, 
                        self.stage_info, 
                        current_step, 
                        self.total_steps_value
                    )
                    self.last_saved_step = current_step
                except Exception as e:
                    print(f"⚠️ Error saving denoising step: {e}")
            
            # Increment our step counter
            self.step_counter += 1
            
        except Exception as e:
            print(f"⚠️ Error in denoising callback: {e}")
        
        # Always return False to continue processing
        return False

# Modify the shader_ksampler function to use our callback
def shader_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, noise_tensor, denoise=1.0, stage_info=None, denoise_visualization_frequency="25% intervals"):
    """
    Custom modification of common_ksampler that allows us to inject our shader noise
    """
    # Get the debugger
    debugger = get_debugger()
    
    # Get the visualizer
    visualizer = get_visualizer()
    
    # Create stage info if not provided
    if stage_info is None:
        stage_info = {
            "stage_type": "standard",
            "stage_idx": 0
        }
    
    # Get the device from the latent image
    device = latent_image["samples"].device
    
    # Set seed for consistent sampling
    torch.manual_seed(seed)
    
    # Extract latent samples
    latent_samples = latent_image["samples"] 
    is_video = len(latent_samples.shape) == 5
    channel_dim = 2 if is_video else 1
    spatial_dims_start = 3 if is_video else 2
    num_spatial_dims = len(latent_samples.shape) - spatial_dims_start
    
    # Log initial tensors if debugging enabled
    if debugger.enabled:
        with debugger.time_operation("shader_ksampler_setup"):
            debugger.analyze_tensor(latent_samples, "initial_latent")
            debugger.analyze_tensor(noise_tensor, "initial_noise")
    
    # Save initial latent and noise if visualizer is enabled
    if visualizer.enabled:
        visualizer.save_latent_visualization(
            latent_samples.clone(), 
            "Initial Latent", 
            stage_info=stage_info,
            is_sample=True
        )
        visualizer.save_latent_visualization(
            noise_tensor.clone(), 
            "Initial Noise", 
            stage_info=stage_info
        )
    
    # Make sure noise is on the right device
    noise = noise_tensor.to(device)
    
    # Check if shapes match
    if noise.shape != latent_samples.shape:
        if debugger.enabled and debugger.debug_level >= 1:
            print(f"⚠️ Shape mismatch detected: Noise {noise.shape} vs Latent {latent_samples.shape}")
        
        # Handle channel mismatch 
        if noise.shape[channel_dim] != latent_samples.shape[channel_dim]:
            # Create a new noise tensor with the correct number of channels
            new_noise_shape = list(latent_samples.shape)
            new_noise = torch.randn(new_noise_shape, device=device, dtype=noise.dtype)
            
            # Copy over the values from the original noise tensor for channels that exist
            min_channels = min(noise.shape[channel_dim], latent_samples.shape[channel_dim])
            
            # Use slicing based on dimension type
            if is_video:
                new_noise[:, :, :min_channels, ...] = noise[:, :, :min_channels, ...]
            else:
                new_noise[:, :min_channels, ...] = noise[:, :min_channels, ...]
                
            noise = new_noise
            
            if debugger.enabled and debugger.debug_level >= 1:
                print(f"✅ Resized noise channels to match latent dimensions: {noise.shape}")
                if debugger.debug_level >= 2:
                    debugger.analyze_tensor(noise, "resized_noise_channels")
            # Save resized noise if visualizer is enabled
            if visualizer.enabled:
                visualizer.save_latent_visualization(
                    noise.clone(), 
                    "Resized Noise (Channels)", 
                    stage_info=stage_info
                )
        
        # Handle spatial dimension mismatch (only if channels already match or were just fixed)
        else:
            # Check if any spatial dimension mismatches
            spatial_mismatch = False
            for i in range(num_spatial_dims):
                dim_index = spatial_dims_start + i
                if noise.shape[dim_index] != latent_samples.shape[dim_index]:
                    spatial_mismatch = True
                    break
            
            if spatial_mismatch:
                target_spatial_size = latent_samples.shape[spatial_dims_start:]
                if debugger.enabled and debugger.debug_level >= 1:
                      print(f"Attempting to resize spatial dimensions {noise.shape[spatial_dims_start:]} -> {target_spatial_size}")
                      
                # Interpolate spatial dimensions
                try:
                    # Determine interpolation mode based on dimensions
                    if num_spatial_dims == 1:
                        mode = 'linear'
                    elif num_spatial_dims == 2:
                        mode = 'bilinear'
                    elif num_spatial_dims == 3:
                        mode = 'trilinear'
                    else:
                         # Fallback for unexpected dimensions
                         mode = 'nearest' 
                         print(f"⚠️ Unexpected number of spatial dimensions ({num_spatial_dims}). Using 'nearest' interpolation.")
                    
                    noise = torch.nn.functional.interpolate(
                        noise,
                        size=target_spatial_size, 
                        mode=mode,
                        align_corners=False if mode in ['linear', 'bilinear', 'trilinear'] else None
                    )
                    if debugger.enabled and debugger.debug_level >= 1:
                        print(f"✅ Resized noise spatial dimensions to match latent: {noise.shape}")
                except Exception as e:
                    print(f"❌ Error during spatial interpolation: {e}. Skipping resize.")
                    # Optionally fallback: noise = torch.zeros_like(latent_samples) ?

                # Save resized noise if visualizer is enabled
                if visualizer.enabled:
                    visualizer.save_latent_visualization(
                        noise.clone(), 
                        "Resized Noise (Spatial)", 
                        stage_info=stage_info
                    )
    
    # Get batch index if available
    batch_inds = latent_image.get("batch_index", None)
    
    # Reset random seed before sampling to ensure deterministic behavior
    # This is critical for temporal coherence
    torch.manual_seed(seed)

    # Log before sampling
    if debugger.enabled and debugger.debug_level >= 2:
        print(f"🚀 Starting sampling with {sampler_name}, {steps} steps, CFG {cfg:.2f}, denoise {denoise:.2f}")
    
    # Create our denoising step callback with frequency parameter
    callback = DenoisingStepCallback(visualizer, stage_info, denoise_visualization_frequency)
    
    # Use comfy.sample.sample directly to apply our noise
    with debugger.time_operation("sampling_process") if debugger.enabled else contextlib.nullcontext():
        output_samples = comfy.sample.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_samples,
            denoise=denoise,
            disable_noise=True,  # We've already generated the noise
            start_step=0,
            last_step=steps,
            force_full_denoise=True,
            seed=seed,  # Pass seed value explicitly
            callback=callback.__call__  # Add our callback to capture intermediate steps
        )
    
    # Log after sampling
    if debugger.enabled:
        debugger.analyze_tensor(output_samples, "output_samples")
    
    # Save output samples if visualizer is enabled
    if visualizer.enabled:
        visualizer.save_latent_visualization(
            output_samples.clone(), 
            "Sampled Output", 
            stage_info=stage_info,
            is_sample=True
        )
    
    # Restore original random state
    torch.manual_seed(torch.seed())
    
    
    # Format output while preserving any metadata that might be useful for temporal coherence
    out = latent_image.copy()
    out["samples"] = output_samples
    
    return (out, )

class ShaderNoiseKSampler:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The AI model used for image generation"}),
                "seed": ("INT", {"default": 8888, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for generation. Same seed with same parameters will generate the same image. Use fixed seed for shader noise exploration."}),
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
                "shader_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the shader noise influence for both stage types - set to 0.0 to disable and use only the base noise"}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay", "soft_light", "hard_light", "difference"], {"default": "multiply", "tooltip": "Method used to blend the shader noise with the base noise"}),
                "noise_transform": (["none", "reverse", "inverse", "absolute", "square", "sqrt", "log", "sin", "cos"], {"default": "none", "tooltip": "Apply mathematical transformations to the noise for creative effects"}),
                "use_temporal_coherence": ("BOOLEAN", {"default": False, "tooltip": "Ensures consistent noise patterns. For sequences like video frames, it helps maintain frame-to-frame consistency (e.g., using the same base seed and 4D noise). For single image generations, it ensures the base noise is derived consistently from the main seed."}),
                               
            },
            "optional": { # Add the optional input here
                 "custom_sigmas": ("SIGMAS", {"tooltip": "Optional custom sigma schedule to override the model's default schedule"}),
            },
            "hidden": {
                "show_default_preview": ("BOOLEAN", {"default": False}),
                "show_custom_preview": ("BOOLEAN", {"default": False, "tooltip": "Whether to show the custom positioned preview outside the node"}),
                "fast_high_channel_noise": ("BOOLEAN", {"default": False, "tooltip": "Use a faster, simplified noise generation method for models with many channels (>16), like LTXV."}),
                "sequential_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across sequential stages"}),
                "injection_distribution": (["uniform", "linear_decrease", "linear_increase", "gaussian", "first_stronger", "last_stronger"], {"default": "linear_decrease", "tooltip": "How shader strength is distributed across injection stages"}),
                "debug_level": (["0-Off", "1-Basic", "2-Detailed", "3-Verbose"], {"default": "0-Off", "tooltip": "Enable debugging at specified level to understand what's happening during shader generation and sampling"}),
                "save_visualizations": ("BOOLEAN", {"default": False, "tooltip": "Save visualizations of the process stages to disk regardless of debug level"}),
                "denoise_visualization_frequency": (["Every step", "25% intervals", "10% intervals", "4 steps", "2 steps"], {"default": "Every step", "tooltip": "How often to save images during the denoising process. Higher frequency means more images but slower generation."}),
                "target_attribute_changes": ("STRING", {"forceInput": True, "tooltip": "Connect output from ParameterResponseMapperNode here"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    @classmethod
    def IS_CHANGED(s, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                   denoise, sequential_stages, injection_stages, shader_strength, blend_mode, noise_transform="none",
                   sequential_distribution="linear_decrease", injection_distribution="linear_decrease", use_temporal_coherence=False, debug_level="0-Off",
                   save_visualizations=False, denoise_visualization_frequency="Every step", custom_sigmas=None, # Add custom_sigmas here
                   target_attribute_changes="", # Add target_attribute_changes here
                   fast_high_channel_noise=False, # Add fast_high_channel_noise here
                   show_custom_preview=False, show_default_preview=False, shader_params_override=None):
        # This will force a re-execution of the node when params change
        # Include custom_sigmas in the check. Use a simple hash or just the object itself.
        # Using the object itself works for comparison; hashing tensor content is more robust if needed
        # but usually comparing the tensor object ID or a simple representation is sufficient for IS_CHANGED.
        # We return a tuple of things that, if changed, should trigger a re-run.
        # Note: Comparing the tensor directly might not always work if a new identical tensor is passed.
        # A more robust way might involve hashing, but let's start simple.
        return (seed, steps, cfg, sampler_name, scheduler, denoise, sequential_stages,
                injection_stages, shader_strength, blend_mode, noise_transform,
                sequential_distribution, injection_distribution, use_temporal_coherence,
                debug_level, save_visualizations, denoise_visualization_frequency,
                custom_sigmas, target_attribute_changes, fast_high_channel_noise) # Add custom_sigmas, target_attribute_changes and fast_high_channel_noise to the tuple

    @classmethod
    def CONTEXT_MENUS(s):
        return {
            "Show Custom Preview": lambda self, **kwargs: {"show_custom_preview": False},
            "Hide Custom Preview": lambda self, **kwargs: {"show_custom_preview": False}
        }

    def has_preview(self, show_default_preview):
        # Use the parameter to decide whether to allow the default preview
        return show_default_preview

    def _calculate_stage_strengths(self, base_strength, num_stages, distribution):
        """Calculate strength for each stage based on distribution type"""
        strengths = []
        
        if distribution == "uniform":
            strengths = [base_strength] * num_stages
        elif distribution == "linear_decrease":
            min_factor = 0.25  # Ensure a minimum shader strength in each stage
            for i in range(num_stages):
                factor = 1.0 - (i / (num_stages - 1 if num_stages > 1 else 1))
                factor = max(factor, min_factor)
                strengths.append(base_strength * factor)
        elif distribution == "linear_increase":
            for i in range(num_stages):
                factor = i / (num_stages - 1 if num_stages > 1 else 1)
                strengths.append(base_strength * factor)
        elif distribution == "gaussian":
            # Create a bell curve with peak in the middle
            mid_point = (num_stages - 1) / 2
            for i in range(num_stages):
                # Standard deviation as 1/3 of the range
                std_dev = num_stages / 3
                factor = math.exp(-((i - mid_point) ** 2) / (2 * std_dev ** 2))
                strengths.append(base_strength * factor)
        elif distribution == "first_stronger":
            # First stage strongest, rapid falloff
            for i in range(num_stages):
                factor = math.exp(-i)  # Exponential decay
                strengths.append(base_strength * factor)
        elif distribution == "last_stronger":
            # Last stage strongest, exponential increase
            for i in range(num_stages):
                factor = math.exp(i - num_stages + 1)  # Exponential increase
                strengths.append(base_strength * factor)
        else:
            # Default to uniform if unknown distribution
            strengths = [base_strength] * num_stages
            
        return strengths
    
    def _calculate_step_ranges(self, total_steps, num_stages):
        """Calculate step ranges for sequential stages"""
        if num_stages <= 0:
            return []
            
        step_ranges = []
        step_size = total_steps / num_stages
        
        for i in range(num_stages):
            start_step = int(i * step_size)
            end_step = int((i + 1) * step_size) if i < num_stages - 1 else total_steps
            step_ranges.append((start_step, end_step))
            
        return step_ranges

    def _calculate_step_points(self, total_steps, num_stages):
        """Calculate at which steps to apply different shader noises for injection stages"""
        if num_stages <= 1:
            return [0]
            
        step_points = []
        for i in range(num_stages):
            step_point = int(i * total_steps / (num_stages - 1)) if num_stages > 1 else 0
            step_points.append(min(step_point, total_steps - 1))  # Ensure we don't exceed total steps
            
        return step_points

    def _apply_noise_transform(self, noise, transform):
        """Apply mathematical transformations to noise"""
        if transform == "none":
            return noise
        elif transform == "reverse":
            return -noise
        elif transform == "inverse":
            # Add small epsilon to avoid division by zero
            return 1.0 / (noise + 1e-8)
        elif transform == "absolute":
            return torch.abs(noise)
        elif transform == "square":
            return noise ** 2
        elif transform == "sqrt":
            return torch.sqrt(torch.abs(noise))
        elif transform == "log":
            return torch.log(torch.abs(noise) + 1.0)
        elif transform == "sin":
            return torch.sin(noise * math.pi)
        elif transform == "cos":
            return torch.cos(noise * math.pi)
        else:
            return noise  # Default to no transform

    def get_model_channel_count(self, model):
        """
        Determine the number of channels in the latent space for a given model.
        This is important for matching the correct noise shape to the model's expectations.
        
        Args:
            model: The model to analyze
            
        Returns:
            int: Number of channels (4 for SD1.5/2.x, 16 for WAN2.1, 32 for SD3)
        """
        # Get debugger
        debugger = get_debugger()
        
        # Check for different model types
        model_name = getattr(model, 'model_name', "").lower() if isinstance(getattr(model, 'model_name', ""), str) else ""
        # Safely convert model_type to string before using upper()
        model_type_attr = getattr(model, 'model_type', "")
        model_type = str(model_type_attr).upper() if model_type_attr is not None else ""
        model_class = model.__class__.__name__
        
        if debugger.enabled:
            print(f"🔎 Model detection - name: '{model_name}', type: '{model_type}', class: '{model_class}'")
            # Check if model is a ModelPatcher or similar wrapper
            if hasattr(model, 'model'):
                inner_model = model.model
                inner_name = getattr(inner_model, 'model_name', "").lower() if isinstance(getattr(inner_model, 'model_name', ""), str) else ""
                # Safely convert inner_model_type to string before using upper()
                inner_type_attr = getattr(inner_model, 'model_type', "")
                inner_type = str(inner_type_attr).upper() if inner_type_attr is not None else ""
                inner_class = inner_model.__class__.__name__
                print(f"🔎 Inner model - name: '{inner_name}', type: '{inner_type}', class: '{inner_class}'")

                # Add check for HiDream / FLOW type specifically for 16 channels
                if inner_class == "HiDream" or inner_type == "FLOW":
                     print(f"✅ Detected HiDream/Flow in inner model - using 16 channels")
                     return 16
                # Check for ACEStep/ACE for 8 channels
                elif inner_class == "ACEStep" or inner_class == "ACE":
                    print(f"✅ Detected {inner_class} in inner model - using 8 channels")
                    return 8
                # Check for corrected WAN21/Warp (now 16 channels)
                elif "wan" in inner_name or "warp" in inner_name or inner_class == "WAN21":
                    print(f"✅ Detected WAN/Warp in inner model - using 16 channels") # Corrected from 6
                    return 16
                # Check for specific high-channel video/3D models
                elif inner_class == "Mochi":
                    print(f"✅ Detected Mochi in inner model - using 12 channels")
                    return 12
                elif inner_class == "LTXV":
                    print(f"✅ Detected LTXV in inner model - using 128 channels")
                    return 128
                elif inner_class == "CosmosVideo": # Add specific check for CosmosVideo
                    print(f"✅ Detected CosmosVideo in inner model - using 16 channels")
                    return 16
                elif inner_class == "Cosmos1CV8x8x8":
                    print(f"✅ Detected Cosmos1CV8x8x8 in inner model - using 16 channels")
                    return 16
                # Explicit check for HunyuanVideo variants which use more channels
                elif inner_class == "HunyuanVideo":
                    print(f"✅ Detected HunyuanVideo class in inner model - using 16 channels")
                    return 16
                # Check for Stable Cascade Prior
                elif "stable_cascade_prior" in inner_name or "stablecascade_prior" in inner_name:
                    print(f"✅ Detected Stable Cascade Prior in inner model - using 16 channels")
                    return 16
            
            # Print model's unet structure if available for debugging
            if hasattr(model, 'diffusion_model') and debugger.debug_level >= 2:
                if hasattr(model.diffusion_model, 'in_channels'):
                    print(f"🔎 Model's diffusion_model.in_channels: {model.diffusion_model.in_channels}")
                if hasattr(model.diffusion_model, 'input_blocks') and hasattr(model.diffusion_model.input_blocks[0], 'in_channels'):
                    print(f"🔎 Model's input_blocks[0].in_channels: {model.diffusion_model.input_blocks[0].in_channels}")
        
        # Check latent shape by looking at both the model and any wrapped models
        if hasattr(model, 'latent_format') and hasattr(model.latent_format, 'latent_channels'):
            channels = model.latent_format.latent_channels
            if debugger.enabled:
                print(f"✅ Found latent_format.latent_channels: {channels}")
            return channels
        
        # Examine diffusion model's channel structure
        if hasattr(model, 'diffusion_model'):
            # Look at the first input block's in_channels which often reveals the model's channel count
            if hasattr(model.diffusion_model, 'input_blocks') and len(model.diffusion_model.input_blocks) > 0:
                if hasattr(model.diffusion_model.input_blocks[0], 'in_channels'):
                    channels = model.diffusion_model.input_blocks[0].in_channels
                    # In_channels is often double the latent space channels
                    if channels in [8, 12, 16]:
                        actual_channels = channels // 2
                        if debugger.enabled:
                            print(f"✅ Detected {actual_channels} channels from model's input_blocks[0].in_channels={channels}")
                        return actual_channels
            
            # Check for patch_embedding
            if hasattr(model.diffusion_model, 'patch_embedding'):
                patch_shape = model.diffusion_model.patch_embedding.weight.shape
                if len(patch_shape) >= 2:
                    channels = patch_shape[1]
                    if debugger.enabled:
                        print(f"✅ Found patch_embedding with input channels: {channels}")
                    return channels
        
        # Specific model family detection (check outer model too)
        wan_indicators = ["wan", "warp", "pixel", "anime"] # Removed "flow" as it's handled above for inner HiDream/16ch
        if any(ind in model_name for ind in wan_indicators) or model_class == "WAN21":
            if debugger.enabled:
                print(f"✅ Detected WAN/Warp/Pixel/Anime model by name/type - using 16 channels") # Corrected from 6
            return 16
        elif "sd3" in model_name:
            if debugger.enabled:
                print(f"✅ Detected SD3 model by name - using 16 channels") # Corrected from 8
            return 16
        elif "flux" in model_name:
             if debugger.enabled:
                print(f"✅ Detected Flux model by name - using 16 channels")
             return 16
        elif "mochi" in model_name:
             if debugger.enabled:
                print(f"✅ Detected Mochi model by name - using 12 channels")
             return 12
        elif "ltxv" in model_name:
             if debugger.enabled:
                print(f"✅ Detected LTXV model by name - using 128 channels")
             return 128
        elif "cosmos" in model_name:
             if debugger.enabled:
                print(f"✅ Detected Cosmos model by name - using 16 channels")
             return 16
        elif "stable_cascade_prior" in model_name or "stablecascade_prior" in model_name:
             if debugger.enabled:
                print(f"✅ Detected Stable Cascade Prior model by name - using 16 channels")
             return 16
        elif "stable_cascade" in model_name: # Assuming non-prior Stable Cascade is 4 channels
            if debugger.enabled:
                print(f"✅ Detected Stable Cascade (non-prior) model by name - using 4 channels")
            return 4
            
        # Explicit checks for model classes (Flux, HiDream)
        model_class_name = model.__class__.__name__.lower()
        inner_model_class_name = getattr(model.model, '__class__', None).__name__.lower() if hasattr(model, 'model') else None

        if 'flux' == model_class_name or 'flux' == inner_model_class_name:
            if debugger.enabled:
                print(f"✅ Detected Flux model by class ({model_class_name}/{inner_model_class_name}) - using 16 channels")
            return 16
        elif 'hidream' == model_class_name or 'hidream' == inner_model_class_name:
             # This might be redundant if name check catches it, but good as fallback
             if debugger.enabled:
                print(f"✅ Detected HiDream model by class ({model_class_name}/{inner_model_class_name}) - using 16 channels")
             return 16

        # Look at model path if available (CheckpointLoaderSimple might set this)
        model_path = getattr(model, 'model_path', "").lower()
        # Update path check for WAN indicators (now 16 channels)
        if model_path and any(ind in model_path for ind in wan_indicators):
            if debugger.enabled:
                print(f"✅ Detected WAN/Warp/Pixel/Anime model from model_path - using 16 channels") # Corrected from 6
            return 16
        
        # Default for most models (SD1.5, SD2.x, SDXL, Stable Cascade B, SD_X4 etc)
        if debugger.enabled:
            print(f"🔄 Using default channel count: 4")
        return 4

    def _generate_shader_noise(self, latent_samples, target_noise_shape, shader_params, shader_type, seed, device="cuda", model=None, model_name=None, frame_count=1, frame_dim_idx=-1):
        """
        Generate noise using the specified shader
        
        Args:
            latent_samples: The input latent samples (for context/shape)
            target_noise_shape: Expected shape of the final noise tensor
            shader_params: Parameters for shader generation
            shader_type: Type of shader generator to use
            seed: Random seed
            device: Device to create tensor on
            model: The model object (for channel detection)
            model_name: Optional model name for customized generation
            frame_count: Number of frames to generate (deduced from shape)
            frame_dim_idx: Index of the frame dimension (-1 if not video)
            
        Returns:
            torch.Tensor: Generated noise tensor with shape matching target_noise_shape
        """
        # Get debugger
        debugger = get_debugger()
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Extract shape information from the target_noise_shape
        is_video = len(target_noise_shape) == 5
        batch_size = target_noise_shape[0]
        height = target_noise_shape[-2]
        width = target_noise_shape[-1]
        
        # Get channels from target_noise_shape (use correct index)
        if is_video:
            # Determine channel index based on target shape format
            # If dim 1 matches model channels, likely [B, C, F, H, W]
            # If dim 2 matches model channels, likely [B, F, C, H, W]
            # Use the passed frame_dim_idx to deduce channel_dim_idx
            channel_dim_idx = 3 - frame_dim_idx # If frame=1, channel=2; if frame=2, channel=1
            channels = target_noise_shape[channel_dim_idx]
        else:
            # 4D: [B, C, H, W]
            channels = target_noise_shape[1]
            channel_dim_idx = 1

        # Add model information to shader params to aid in detection
        shader_params = shader_params.copy()
        
        # If a model is provided, extract all possible identifying information
        if model is not None:
            # Get and store model name
            if model_name is None:
                model_name = getattr(model, 'model_name', None)
                if model_name is None and hasattr(model, 'model') and hasattr(model.model, 'model_name'):
                    model_name = model.model.model_name
            
            # Always store model_name in shader_params if available
            if model_name:
                shader_params["model_name"] = model_name
                if debugger.enabled and debugger.debug_level >= 2:
                    print(f"📝 Added model_name='{model_name}' to shader_params")
            
            # Add model class name which can be useful for detection
            shader_params["model_class"] = model.__class__.__name__
            
            # Add inner model class name if it exists
            if hasattr(model, 'model'):
                shader_params["inner_model_class"] = model.model.__class__.__name__
                inner_model_name = getattr(model.model, 'model_name', None)
                if inner_model_name:
                    shader_params["inner_model_name"] = inner_model_name
            
            # Get target channels from model (should match 'channels' variable now)
            model_target_channels = self.get_model_channel_count(model)
            if model_target_channels != channels:
                print(f"⚠️ Discrepancy: Model expects {model_target_channels} channels, but target noise shape indicates {channels}.")
                # We will proceed with the 'channels' derived from target_noise_shape
        
        if debugger.enabled and debugger.debug_level >= 2:
            print(f"🎨 Generating {shader_type} shader noise for {'video' if is_video else 'image'}")
            print(f"   Target Shape: {target_noise_shape}, Device: {device}, Seed: {seed}")
            if model_name:
                print(f"   Model: {model_name}")
            print(f"   Target channels: {channels}")
        
        # Add channel count to shader parameters
        shader_params["target_channels"] = channels
        
        # Log the final shader params for debugging
        if debugger.enabled and debugger.debug_level >= 2:
            print(f"📝 Final shader_params for noise generation: {shader_params}")

        # For video latents
        if is_video:
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"   Video dimensions: Target={target_noise_shape}, Frames={frame_count}")
            
            # Get the shader generator function
            generator_func = get_shader_generator(shader_type)
            
            # Generate noise frame by frame
            noise_frames = []
            # Use the correctly determined frame_count
            for frame_idx in range(frame_count):
                # Update time parameter for temporal variation
                frame_params = shader_params.copy()
                # Ensure time increments correctly even if frame_count < 10
                frame_time_increment = 0.1 if frame_count <= 1 else (1.0 / (frame_count - 1)) * frame_idx
                frame_params["time"] = shader_params.get("time", 0.0) + frame_time_increment
                
                if debugger.enabled and debugger.debug_level >= 3:
                    print(f"   Generating frame {frame_idx} with time={frame_params['time']:.2f}")
                
                # Prepare arguments based on what the function accepts
                # Note: The generator usually expects [B, C, H, W] per frame
                generator_args = {
                    "shader_params": frame_params,
                    "height": height, 
                    "width": width, 
                    "batch_size": batch_size,
                    "device": device,
                    "seed": seed + frame_idx if not shader_params.get("useTemporalCoherence", False) else seed,
                    "target_channels": channels  # Always include target_channels
                }
                
                # Generate the noise for this frame using named parameters
                # Expected output shape: [batch_size, channels, height, width]
                frame_noise = generator_func(**generator_args)
                
                # Validate frame_noise shape
                expected_frame_shape = (batch_size, channels, height, width)
                if frame_noise.shape != expected_frame_shape:
                    # print(f"⚠️ Shader generated frame shape {frame_noise.shape} instead of expected {expected_frame_shape}")
                    
                    # Attempt to correct the frame shape
                    try:
                        # If channel count is wrong, fix it
                        if frame_noise.shape[1] != channels:
                            # print(f"   Correcting frame channel count from {frame_noise.shape[1]} to {channels}")
                            corrected_frame = torch.zeros(expected_frame_shape, device=device, dtype=frame_noise.dtype)
                            min_c = min(frame_noise.shape[1], channels)
                            corrected_frame[:, :min_c] = frame_noise[:, :min_c]
                            frame_noise = corrected_frame
                        
                        # If spatial dimensions are wrong, resize
                        # This requires importing torch.nn.functional as F at the top of the file
                        if frame_noise.shape[2:] != expected_frame_shape[2:]:
                            # print(f"   Resizing frame spatial dimensions from {frame_noise.shape[2:]} to {expected_frame_shape[2:]}")
                            import torch.nn.functional as F # Ensure F is available
                            frame_noise = F.interpolate(frame_noise, size=(height, width), mode='bilinear', align_corners=False)
                            
                        # Final check
                        if frame_noise.shape != expected_frame_shape:
                            raise ValueError("Shape correction failed")
                            
                    except Exception as e:
                        # print(f"❌ Error correcting frame shape: {e}. Skipping frame.")
                        # Create a zero tensor as fallback to avoid crashing
                        frame_noise = torch.zeros(expected_frame_shape, device=device, dtype=latent_samples.dtype)
                    
                noise_frames.append(frame_noise)
            
            # Stack frames along the correct dimension
            if noise_frames:
                noise = torch.stack(noise_frames, dim=frame_dim_idx)
            else:
                # Handle case where no frames were generated
                print("❌ No frames generated for video noise. Creating zeros.")
                noise = torch.zeros(target_noise_shape, device=device, dtype=latent_samples.dtype)
            
        # For image latents
        else:
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"   Image dimensions: Target={target_noise_shape}")
            
            # Get the shader generator function
            generator_func = get_shader_generator(shader_type)
            
            # Prepare arguments based on what the function accepts
            generator_args = {
                "shader_params": shader_params,
                "height": height, 
                "width": width, 
                "batch_size": batch_size,
                "device": device,
                "seed": seed,
                "target_channels": channels  # Always include target_channels
            }
            
            # Generate the noise with named parameters
            noise = generator_func(**generator_args)
            
            # Ensure image has correct final shape
            if noise.shape != target_noise_shape:
                print(f"⚠️ Image shader generated shape {noise.shape} instead of expected {target_noise_shape}")
                # Attempt correction (similar to frame correction)
                try:
                    if noise.shape[1] != channels:
                         corrected_noise = torch.zeros(target_noise_shape, device=device, dtype=noise.dtype)
                         min_c = min(noise.shape[1], channels)
                         corrected_noise[:, :min_c] = noise[:, :min_c]
                         noise = corrected_noise
                    if noise.shape[2:] != target_noise_shape[2:]:
                         noise = F.interpolate(noise, size=(height, width), mode='bilinear', align_corners=False)
                    if noise.shape != target_noise_shape:
                         raise ValueError("Shape correction failed")
                except Exception as e:
                    print(f"❌ Error correcting image shape: {e}. Creating zeros.")
                    noise = torch.zeros(target_noise_shape, device=device, dtype=latent_samples.dtype)
        
        # Normalize the noise tensor to have standard deviation of 1.0 and mean of 0.0
        # This ensures consistent blending behavior regardless of the underlying distribution
        if noise.numel() > 0:  # Only attempt normalization if tensor has elements
            # Calculate current mean and standard deviation
            current_mean = noise.mean()
            current_std = noise.std()
            
            # Only normalize if standard deviation is not very close to zero
            if current_std > 1e-6:
                # Normalize: (x - mean) / std
                noise = (noise - current_mean) / current_std
        
        # Analyze and log the generated noise for debugging
        if debugger.enabled:
            debugger.analyze_tensor(noise, "shader_noise")
        
        # Final shape check
        if noise.shape != target_noise_shape:
             print(f"❌ FINAL SHAPE MISMATCH: Noise shape {noise.shape} != Target shape {target_noise_shape}")
             # Attempt final resize as last resort
             try:
                 noise = F.interpolate(noise, size=target_noise_shape[2:], mode='bilinear', align_corners=False)
                 # If channels are still wrong, adjust
                 if noise.shape[channel_dim_idx] != target_noise_shape[channel_dim_idx]:
                      final_noise = torch.zeros(target_noise_shape, device=device, dtype=noise.dtype)
                      min_c = min(noise.shape[channel_dim_idx], target_noise_shape[channel_dim_idx])
                      # Slicing needs to be careful based on dim order
                      if channel_dim_idx == 1: # [B, C, F, H, W]
                          final_noise[:,:min_c,...] = noise[:,:min_c,...]
                      elif channel_dim_idx == 2: # [B, F, C, H, W]
                          final_noise[:,:,:min_c,...] = noise[:,:,:min_c,...]
                      else: # 4D
                          final_noise[:,:min_c,...] = noise[:,:min_c,...]
                      noise = final_noise
             except Exception as e:
                  print(f"❌ Final resize attempt failed: {e}")
                  noise = torch.zeros(target_noise_shape, device=device, dtype=latent_samples.dtype)
        
        return noise

    def _blend_noises(self, base_noise, shader_noise, blend_mode, strength):
        """
        Blend base noise with shader noise using the specified blend mode and strength.
        Handles channel dimension mismatches automatically.
        
        Args:
            base_noise: Base noise tensor
            shader_noise: Shader noise tensor
            blend_mode: Blending mode to apply
            strength: Strength of the blend [0.0-1.0]
            
        Returns:
            torch.Tensor: Blended noise tensor
        """
        # Get the debugger
        debugger = get_debugger()
        
        # If strength is 0, return base noise unchanged
        if strength <= 0.0:
            return base_noise
            
        # If strength is 1.0 and blend mode is "normal", return shader noise
        if strength >= 1.0 and blend_mode == "normal":
            # Ensure shader noise has same shape as base noise
            if shader_noise.shape != base_noise.shape:
                # Handle channel dimension mismatches
                if shader_noise.shape[1] != base_noise.shape[1]:
                    # Create a new tensor with the right number of channels
                    resized_shader = torch.zeros_like(base_noise)
                    # Copy over the common channels
                    min_channels = min(shader_noise.shape[1], base_noise.shape[1])
                    resized_shader[:, :min_channels] = shader_noise[:, :min_channels]
                    return resized_shader
                else:
                    # Handle any other dimension mismatches with interpolation
                    return torch.nn.functional.interpolate(
                        shader_noise, 
                        size=base_noise.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
            return shader_noise
        
        # Ensure compatible dimensions for blending
        if shader_noise.shape != base_noise.shape:
            if debugger.enabled and debugger.debug_level >= 1:
                print(f"⚠️ Shape mismatch for blending: base={base_noise.shape}, shader={shader_noise.shape}")
            
            # Handle channel dimension mismatches
            if shader_noise.shape[1] != base_noise.shape[1]:
                # Create a new tensor with the same shape as base_noise
                resized_shader = torch.zeros_like(base_noise)
                # Copy over the common channels
                min_channels = min(shader_noise.shape[1], base_noise.shape[1])
                resized_shader[:, :min_channels] = shader_noise[:, :min_channels]
                shader_noise = resized_shader
                
                if debugger.enabled and debugger.debug_level >= 1:
                    print(f"✅ Matched channel dimensions: {shader_noise.shape}")
            
            # Handle spatial dimension mismatches with interpolation
            if shader_noise.shape[2:] != base_noise.shape[2:]:
                # Use a try/except block to handle any interpolation issues with video tensors
                try:
                    shader_noise = torch.nn.functional.interpolate(
                        shader_noise, 
                        size=base_noise.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                except RuntimeError as e:
                    if debugger.enabled:
                        print(f"⚠️ Error during interpolation: {e}")
                    # If interpolation fails, reshape tensor to match if possible
                    if len(shader_noise.shape) == 5 and len(base_noise.shape) == 5:
                        # For video tensors, try to reshape directly
                        b, f, c, h, w = shader_noise.shape
                        _, _, target_c, target_h, target_w = base_noise.shape
                        if c != target_c:
                            # Handle channel differences
                            new_shader = torch.zeros_like(base_noise)
                            min_c = min(c, target_c)
                            # Copy each frame separately
                            for i in range(f):
                                new_shader[:, i, :min_c] = shader_noise[:, i, :min_c]
                            shader_noise = new_shader
                    else:
                        # Fall back to zeros if we can't match shapes
                        shader_noise = torch.zeros_like(base_noise)
                
                if debugger.enabled and debugger.debug_level >= 1:
                    print(f"✅ Matched spatial dimensions: {shader_noise.shape}")
        
        # Apply the blend mode
        with debugger.time_operation(f"blend_{blend_mode}") if debugger.enabled else contextlib.nullcontext():
            if blend_mode == "normal":
                # Linear interpolation
                result = base_noise * (1.0 - strength) + shader_noise * strength
            
            elif blend_mode == "add":
                # Add shader to base
                result = base_noise + shader_noise * strength
            
            elif blend_mode == "multiply":
                # Multiply base by shader
                result = base_noise * (1.0 + (shader_noise - 0.5) * strength * 2)
            
            elif blend_mode == "screen":
                # Screen blend mode
                result = 1.0 - (1.0 - base_noise) * (1.0 - shader_noise * strength)
            
            elif blend_mode == "overlay":
                # Overlay blend mode
                mask = base_noise < 0.5
                result = torch.zeros_like(base_noise)
                result[mask] = 2 * base_noise[mask] * (shader_noise[mask] * strength)
                result[~mask] = 1 - 2 * (1 - base_noise[~mask]) * (1 - shader_noise[~mask] * strength)
            
            elif blend_mode == "soft_light":
                # Soft light blend mode
                result = ((1.0 - 2.0 * shader_noise) * base_noise**2 + 2.0 * shader_noise * base_noise) * strength + base_noise * (1.0 - strength)
            
            elif blend_mode == "hard_light":
                # Hard light blend mode
                mask = shader_noise < 0.5
                result = torch.zeros_like(base_noise)
                result[mask] = 2 * base_noise[mask] * shader_noise[mask] * strength + base_noise[mask] * (1 - strength)
                result[~mask] = 1 - 2 * (1 - base_noise[~mask]) * (1 - shader_noise[~mask]) * strength + base_noise[~mask] * (1 - strength)
            
            elif blend_mode == "difference":
                # Difference blend mode
                result = base_noise + (torch.abs(base_noise - shader_noise) * strength)
            
            else:
                # Default to normal blend for unknown modes
                result = base_noise * (1.0 - strength) + shader_noise * strength
        
        # Debug output for blend results
        if debugger.enabled:
            base_stats = {
                "min": float(base_noise.min().item()),
                "max": float(base_noise.max().item()),
                "mean": float(base_noise.mean().item()),
                "std": float(base_noise.std().item())
            }
            
            result_stats = {
                "min": float(result.min().item()),
                "max": float(result.max().item()),
                "mean": float(result.mean().item()),
                "std": float(result.std().item())
            }
            
            if debugger.debug_level >= 2:
                print(f"📊 Blend stats ({blend_mode}, strength={strength:.2f}):")
                print(f"   Base: min={base_stats['min']:.4f}, max={base_stats['max']:.4f}, mean={base_stats['mean']:.4f}, std={base_stats['std']:.4f}")
                print(f"   Result: min={result_stats['min']:.4f}, max={result_stats['max']:.4f}, mean={result_stats['mean']:.4f}, std={result_stats['std']:.4f}")
            
            mean_diff = abs(result_stats["mean"] - base_stats["mean"])
            std_diff = abs(result_stats["std"] - base_stats["std"])
            
            if mean_diff < 0.001 and std_diff < 0.001:
                print(f"⚠️ Warning: Blend may not be effective - minimal statistical difference detected")
                print(f"   Base: mean={base_stats['mean']:.4f}, std={base_stats['std']:.4f}")
                print(f"   Result: mean={result_stats['mean']:.4f}, std={result_stats['std']:.4f}")
        
        return result

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
              denoise=1.0, sequential_stages=2, injection_stages=3, shader_strength=0.3, blend_mode="multiply", 
              noise_transform="none", sequential_distribution="linear_decrease", injection_distribution="linear_decrease",
              use_temporal_coherence=False, debug_level="0-Off", save_visualizations=False, denoise_visualization_frequency="25% intervals",
              custom_sigmas=None, # Add custom_sigmas here
              target_attribute_changes="", # Add target_attribute_changes here
              fast_high_channel_noise=False, # Add fast_high_channel_noise here
              show_custom_preview=False, show_default_preview=False, shader_params_override=None):
        """
        Run the multi-stage shader noise k-sampler
        """
        # Parse debug level from the selected option
        debug_level_value = int(debug_level.split("-")[0])
        
        # Set the debug level in the shader debugger
        debugger = set_debug_level(debug_level_value)
        
        # Get the visualizer
        visualizer = get_visualizer()

        # --- Get device early from latent_image ---
        device = latent_image["samples"].device
        if debugger.enabled:
            print(f"ℹ️ Using device: {device}")

        # Enable the visualizer with the current parameters if save_visualizations is True
        if save_visualizations:
            additional_metadata = {
                "model": getattr(model, 'model_name', "unknown"),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "sequential_stages": sequential_stages,
                "injection_stages": injection_stages,
                "shader_strength": shader_strength,
                "blend_mode": blend_mode,
                "noise_transform": noise_transform,
                "sequential_distribution": sequential_distribution,
                "injection_distribution": injection_distribution,
                "use_temporal_coherence": use_temporal_coherence,
                "fast_high_channel_noise": fast_high_channel_noise, # Log the new param
            }
            
            # Extract shader type from parameters - use override if provided
            if shader_params_override is not None:
                shader_params = shader_params_override
            else:
                shader_params = get_shader_params()
            shader_type = shader_params.get("shader_type", "tensor_field")
            
            # Enable the visualizer with save_to_output_dir option
            visualizer.enable(
                seed=seed, 
                shader_type=shader_type, 
                additional_metadata=additional_metadata
            )
        
        if debugger.enabled and debug_level_value >= 1:
            print(f"🔍 Starting shader debugging at level {debug_level}")
            
            # Log parameters for diagnostics
            debug_params = {
                "shader_type": self.shader_type if hasattr(self, 'shader_type') else "unknown",
                "seed": seed,
                "sequential_stages": sequential_stages,
                "injection_stages": injection_stages,
                "shader_strength": shader_strength,
                "blend_mode": blend_mode,
                "noise_transform": noise_transform,
                "sequential_distribution": sequential_distribution,
                "injection_distribution": injection_distribution,
                "use_temporal_coherence": use_temporal_coherence,
                "debug_level": debug_level
            }
            debugger.log_parameters(debug_params)
        
        # Get model information if available for debugging
        # Use original model name for metadata/logging consistency
        model_name = getattr(model, 'model_name', getattr(getattr(model, 'model', None), 'model_name', None))

        if debugger.enabled and debugger.debug_level >= 1:
            print(f"ℹ️ Using model: {model_name if model_name else 'Unknown model'}")
            
        # Start a new debugging session if enabled
        if debugger.enabled:
            debugger.reset()
            print(f"🔍 Starting shader debugging at level {debug_level}")
        
        # Get the current shader parameters - use override if provided
        if shader_params_override is not None:
            shader_params = shader_params_override
            if debugger.enabled:
                print("🔄 Using provided shader parameters override")
        else:
            shader_params = get_shader_params()
        
        # Extract values for generation
        shader_type = shader_params.get("shader_type", "tensor_field")
        
        # --- Parameter Response Mapper Integration --- START ---
        if target_attribute_changes and target_attribute_changes.strip():
            if debugger.enabled:
                print("🧠 Applying Parameter Response Mapper adjustments...")
            try:
                import json # Need json for parsing
                target_attrs = json.loads(target_attribute_changes)
                
                if isinstance(target_attrs, dict) and target_attrs:
                    # Determine model type for mapper (simple inference)
                    model_name_lower = model_name.lower() if model_name else ""
                    if "sdxl" in model_name_lower:
                        mapper_model_type = "SDXL"
                    else:
                        mapper_model_type = "SD1.5" # Default
                    
                    mapper = ParameterResponseMapper(model_type=mapper_model_type)
                    
                    # Extract current shader parameters relevant to the mapper
                    current_mapper_params = {
                        p: shader_params.get(p, shader_params.get(f"shader{p.capitalize()}", 0.0)) 
                        for p in ["scale", "octaves", "warp_strength", "phase_shift"]
                    }
                    # Ensure correct types (e.g., octaves is float for mapper)
                    current_mapper_params["octaves"] = float(current_mapper_params.get("octaves", 3.0))

                    if debugger.enabled and debugger.debug_level >= 2:
                        print(f"   Mapper using model_type: {mapper_model_type}")
                        print(f"   Target Attributes: {target_attrs}")
                        print(f"   Current Params for Mapper: {current_mapper_params}")

                    # Get recommendations
                    recommendations = mapper.get_recommended_adjustments(
                        target_attrs,
                        current_mapper_params,
                        max_params=2 # Adjust max parameters as needed
                    )
                    
                    if recommendations:
                        if debugger.enabled:
                            print(f"✅ Mapper recommended adjustments: {recommendations}")
                        # Apply recommendations to the main shader_params
                        for param, value in recommendations.items():
                            # Update both plain and prefixed keys if they exist
                            shader_params[param] = value
                            prefixed_key = f"shader{param.capitalize()}"
                            if prefixed_key in shader_params:
                                shader_params[prefixed_key] = value
                        # Re-log parameters if modified
                        if debugger.enabled:
                             debugger.log_parameters({"shader_params_after_mapper": shader_params})
                    elif debugger.enabled:
                        print("ℹ️ Mapper provided no recommendations.")
                else:
                    if debugger.enabled:
                         print("⚠️ Target attribute changes were provided but not a valid dictionary or empty.")

            except Exception as e:
                print(f"❌ Error processing target_attribute_changes: {e}. Skipping mapper adjustments.")
        # --- Parameter Response Mapper Integration --- END ---

        # Log all parameters if debugging enabled
        if debugger.enabled:
            # Collect all parameters for logging
            all_params = {
                "shader_type": shader_type,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "sequential_stages": sequential_stages,
                "injection_stages": injection_stages,
                "shader_strength": shader_strength,
                "blend_mode": blend_mode,
                "noise_transform": noise_transform,
                "sequential_distribution": sequential_distribution,
                "injection_distribution": injection_distribution,
                "use_temporal_coherence": use_temporal_coherence,
                "fast_high_channel_noise": fast_high_channel_noise, # Log the new param
                "shader_params": shader_params
            }
            debugger.log_parameters(all_params)
        
        # Determine which model object to use (original or wrapper)
        model_to_use = model
        using_custom_sigmas = False
        total_sampling_steps = steps # Default to input steps

        if custom_sigmas is not None:
            if debugger.enabled:
                 print("ℹ️ Attempting to use custom sigma schedule.")
            # Basic validation: Check if it's a tensor and has values
            if isinstance(custom_sigmas, torch.Tensor) and custom_sigmas.numel() > 1:
                try:
                    # Pass the determined device to the wrapper
                    model_to_use = CustomSigmaModelWrapper(model, custom_sigmas, device)
                    using_custom_sigmas = True
                    actual_sigma_count = len(model_to_use.model_sampling.sigmas)
                    
                    if actual_sigma_count > 1:
                        total_sampling_steps = actual_sigma_count - 1 # Derive steps from sigmas
                        if debugger.enabled:
                            print(f"✅ Successfully wrapped model with {actual_sigma_count} custom sigmas.")
                            print(f"ℹ️ Overriding sampling steps. Using {total_sampling_steps} steps based on custom sigmas (Input steps: {steps}).")
                    else:
                        # Handle case of single sigma value - technically invalid for sampling
                        print(f"⚠️ Warning: Custom sigma tensor has only {actual_sigma_count} value(s). Cannot derive steps. Using input steps {steps}.")
                        using_custom_sigmas = False # Revert flag if sigmas are invalid for steps
                        model_to_use = model # Revert model
                        total_sampling_steps = steps # Fallback to input steps

                    # Remove previous sigma length vs steps comparison logic as we now override steps
                    # if actual_sigma_count < steps: ...
                    # elif actual_sigma_count != expected_sigma_count: ...

                except Exception as e:
                     print(f"❌ Error wrapping model with custom sigmas: {e}. Falling back to default model sigmas and input steps.")
                     model_to_use = model # Fallback
                     using_custom_sigmas = False
                     total_sampling_steps = steps # Fallback
            else:
                 if debugger.enabled:
                    print(f"⚠️ Custom sigmas input is not a valid tensor or has <= 1 value. Falling back to default model sigmas and input steps.")
                 model_to_use = model # Fallback
                 using_custom_sigmas = False
                 total_sampling_steps = steps # Fallback
        elif debugger.enabled:
             print(f"ℹ️ Using model's default sigma schedule and {steps} input steps.")


        # If shader_strength is 0 or both stage types are 0, use the standard ksampler instead
        if shader_strength == 0.0 or (sequential_stages <= 0 and injection_stages <= 0):
            if debugger.enabled:
                print(f"ℹ️ Using standard ksampler (no shader stages enabled or zero strength) with {total_sampling_steps} steps.")
            # IMPORTANT: Pass model_to_use and the potentially overridden total_sampling_steps
            samples = common_ksampler(model_to_use, seed, total_sampling_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]

            # Save final output if visualizer is enabled
            if visualizer.enabled:
                visualizer.capture_final_result(samples["samples"], {"method": "standard_ksampler", "stages": 0, "total_steps": total_sampling_steps})
                visualizer.disable()
            
            return {"ui": {"images": [], "show_custom_preview": [show_custom_preview]}, "result": (samples,)}
        
        # Determine if we're working with a video or an image
        latent_shape = latent_image["samples"].shape
        device = latent_image["samples"].device
        is_video = len(latent_shape) == 5
        batch_size = latent_shape[0]
        
        if debugger.enabled:
            print(f"ℹ️ Total steps for sampling process: {total_sampling_steps} (derived from {'custom sigmas' if using_custom_sigmas else 'input'})")

        # Determine the target channel count early on using the model
        # Use model_to_use here to correctly get channels if model was wrapped (though usually wrapper forwards this)
        target_channels = self.get_model_channel_count(model_to_use)
        
        # Initialize dimension indices and counts
        frames = 1
        channels = target_channels # Start assuming model's count
        height = latent_shape[-2]
        width = latent_shape[-1]
        channel_dim_idx = -1
        frame_dim_idx = -1
        detected_format = "Unknown"

        if is_video:
            latent_dim1 = latent_shape[1]
            latent_dim2 = latent_shape[2]
            
            # Logic to determine Frame vs Channel dimension for 5D tensor
            # Priority 1: If one dimension matches latent shape and the other matches model channels
            if latent_dim1 == target_channels and latent_dim2 != target_channels:
                # Format likely [B, C, F, H, W]
                channels = latent_dim1
                frames = latent_dim2
                channel_dim_idx = 1
                frame_dim_idx = 2
                detected_format = "[B, C, F, H, W]"
            elif latent_dim2 == target_channels and latent_dim1 != target_channels:
                # Format likely [B, F, C, H, W]
                channels = latent_dim2
                frames = latent_dim1
                channel_dim_idx = 2
                frame_dim_idx = 1
                detected_format = "[B, F, C, H, W]"
            # Priority 2: If both match target_channels (ambiguous, default to B,F,C,H,W)
            elif latent_dim1 == target_channels and latent_dim2 == target_channels:
                # Could be B,C,F or B,F,C - Assume B,F,C common format
                print(f"⚠️ Ambiguous video shape {latent_shape} where both dim 1 & 2 match target channels {target_channels}. Assuming [B, F, C, H, W] format.")
                channels = target_channels
                frames = latent_dim1 # Use dim 1 as frames
                channel_dim_idx = 2
                frame_dim_idx = 1
                detected_format = "[B, F, C, H, W] (Ambiguous)"
            # Priority 3: If neither matches target_channels (Fallback, assume B,F,C,H,W)
            else:
                # print(f"⚠️ Video shape {latent_shape} doesn't match target channels {target_channels} in dim 1 or 2. Assuming [B, F, C, H, W] format and using model's target channels.")
                channels = target_channels # Use model's target channels
                frames = latent_dim1 # Assume dim 1 is frames
                channel_dim_idx = 2
                frame_dim_idx = 1
                detected_format = "[B, F, C, H, W] (Fallback)"

        else: # 4D Tensor
            # Assume [B, C, H, W] format
            channels = latent_shape[1]
            frames = 1
            channel_dim_idx = 1
            frame_dim_idx = -1 # No frame dimension
            detected_format = "[B, C, H, W]"
            # Validate against model channels
            if channels != target_channels:
                print(f"⚠️ Image shape {latent_shape} channel count ({channels}) differs from model's target channels ({target_channels}). Using model's target channels for noise generation.")
                channels = target_channels # Prioritize model's expectation for noise

        if debugger.enabled:
            print(f"ℹ️ Detected Format: {detected_format}")
            print(f"ℹ️ Latent Shape: {latent_shape}")
            print(f"ℹ️ Target Channels (Model): {target_channels}")
            print(f"ℹ️ Deduced Dimensions: Frames={frames}, Channels={channels}, Height={height}, Width={width}")
            print(f"ℹ️ Deduced Indices: Frame Dim={frame_dim_idx}, Channel Dim={channel_dim_idx}")

        # Construct the target noise shape using detected dimensions and model's target channels
        # Order matters based on detected format!
        if is_video:
            if detected_format.startswith("[B, C, F"):
                # [B, C, F, H, W] format
                noise_shape = (batch_size, target_channels, frames, height, width)
            else:
                # Defaulting to [B, F, C, H, W] format
                noise_shape = (batch_size, frames, target_channels, height, width)
        else:
            # [B, C, H, W] format
            noise_shape = (batch_size, target_channels, height, width)

        if debugger.enabled:
            print(f"ℹ️ Target Noise Shape: {noise_shape} (based on detected format and model channels)")

        # Check if input latent channels match target noise channels before proceeding
        latent_channel_dim_idx = channel_dim_idx if is_video else 1
        if latent_shape[latent_channel_dim_idx] != noise_shape[latent_channel_dim_idx]:
             # This check is crucial as resizing happens later in shader_ksampler
             # print(f"⚠️ Input latent channel count ({latent_shape[latent_channel_dim_idx]}) differs from target noise channel count ({noise_shape[latent_channel_dim_idx]}). Noise resizing will occur in ksampler.")
             pass
        elif debugger.enabled:
             print(f"✅ Input latent channel count matches target noise channels ({target_channels}).")

        if debugger.enabled:
            print(f"ℹ️ Processing {'video' if is_video else 'image'} with shape {latent_shape}")
            print(f"ℹ️ Model detected requires {target_channels} channels.")
        
        # Handle temporal coherence
        # Add time parameter if not present
        shader_params["time"] = shader_params.get("time", 0.0)
        # Make sure we have a base_seed for temporal coherence
        shader_params["base_seed"] = seed
        # Add the temporal coherence setting to shader params
        shader_params["useTemporalCoherence"] = use_temporal_coherence
        shader_params["temporal_coherence"] = use_temporal_coherence # Ensure this alternative key is also set
        # Add the fast high channel noise setting to shader params
        shader_params["fast_high_channel_noise"] = fast_high_channel_noise
        
        if debugger.enabled and use_temporal_coherence:
            print(f"🔄 Using temporal coherence with base seed {seed}")
        
        # Add print statements for the requested parameters
        print(f"ShaderNoiseKSampler: Using shader_type: {shader_type}")
        print(f"ShaderNoiseKSampler: Shape Mask: type={shader_params.get('shaderShapeType', shader_params.get('shape_type', 'none'))}, strength={shader_params.get('shaderShapeStrength', shader_params.get('shapemaskstrength', 1.0))}")
        print(f"ShaderNoiseKSampler: Color Scheme: scheme={shader_params.get('colorScheme', 'none')}, intensity={shader_params.get('shaderColorIntensity', shader_params.get('intensity', 0.8))}")
        print(f"ShaderNoiseKSampler: Scale={shader_params.get('shaderScale', shader_params.get('scale', 1.0))}")
        print(f"ShaderNoiseKSampler: Octaves={shader_params.get('shaderOctaves', shader_params.get('octaves', 3.0))}")
        print(f"ShaderNoiseKSampler: Warp Strength={shader_params.get('shaderWarpStrength', shader_params.get('warp_strength', 0.5))}")
        print(f"ShaderNoiseKSampler: Phase Shift={shader_params.get('shaderPhaseShift', shader_params.get('phase_shift', 0.5))}")
        
        # Set initial random seed for consistency
        torch.manual_seed(seed)
        
        # Extract the latent samples
        latent_samples = latent_image["samples"].clone()
        
        # Generate initial noise if using full denoise
        if denoise > 0.0:
            if debugger.enabled:
                print(f"🎲 Generating initial noise with seed {seed}")
                
            # Generate noise with the CORRECT noise_shape derived above
            initial_noise = torch.randn(noise_shape, device=device)
            
            if debugger.enabled and debugger.debug_level >= 2:
                debugger.analyze_tensor(initial_noise, "initial_noise")
            
            # Save initial noise if visualizer is enabled
            if visualizer.enabled:
                visualizer.save_latent_visualization(
                    initial_noise.clone(),
                    "Initial Random Noise",
                    stage_info={"stage_type": "startup"}
                )
        else:
            # No noise needed for zero denoise
            initial_noise = None
            if debugger.enabled:
                print("ℹ️ No initial noise needed (denoise=0)")
        
        # Current latent state
        current_latent = {"samples": latent_samples}
        
        # Save initial latent if visualizer is enabled
        if visualizer.enabled:
            visualizer.save_latent_visualization(
                latent_samples.clone(),
                "Input Latent",
                stage_info={"stage_type": "startup"},
                is_sample=True
            )
        
        # PHASE 1: Sequential Shader Stages
        # Only proceed if we have sequential stages to run
        if sequential_stages > 0:
            if debugger.enabled:
                print(f"🔄 Starting Sequential Phase with {sequential_stages} stages over {total_sampling_steps} total steps")
                
            # Calculate the strength for each sequential stage
            sequential_strengths = self._calculate_stage_strengths(shader_strength, sequential_stages, sequential_distribution)
            
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"💪 Sequential strengths: {[f'{s:.3f}' for s in sequential_strengths]}")
            
            # Calculate the step ranges for each sequential stage based on total_sampling_steps
            sequential_ranges = self._calculate_step_ranges(total_sampling_steps, sequential_stages)
            
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"🔢 Sequential step ranges: {sequential_ranges}")
            
            # Generate base noise for blending for sequential stages
            torch.manual_seed(seed)
            
            if debugger.enabled:
                print(f"🎲 Generating base sequential noise with seed {seed}")
                
            # Generate noise with the CORRECT noise_shape derived above
            base_noise_sequential = torch.randn(noise_shape, device=device)
            
            if debugger.enabled and debugger.debug_level >= 2:
                debugger.analyze_tensor(base_noise_sequential, "base_noise_sequential")
            
            # Save base sequential noise if visualizer is enabled
            if visualizer.enabled:
                visualizer.save_latent_visualization(
                    base_noise_sequential.clone(),
                    "Base Sequential Noise",
                    stage_info={"stage_type": "sequential_base"}
                )
            
            # Process each sequential stage
            for stage_idx, (start_step, end_step) in enumerate(sequential_ranges):
                # Start logging this stage
                if debugger.enabled:
                    debugger.log_stage_start("sequential", stage_idx, {
                        "start_step": start_step,
                        "end_step": end_step,
                        "strength": sequential_strengths[stage_idx]
                    })
                
                # Calculate stage-specific parameters
                stage_steps = end_step - start_step # Steps for this specific stage call
                stage_strength = sequential_strengths[stage_idx]
                
                # Update shader parameters for this stage
                stage_shader_params = shader_params.copy()
                stage_shader_params["stage"] = stage_idx
                # Ensure temporal coherence setting is explicitly added to stage params
                stage_shader_params["useTemporalCoherence"] = use_temporal_coherence
                
                # Calculate denoising strength for this stage (using total_sampling_steps)
                # Note: This might not be strictly necessary if shader_ksampler ignores denoise with force_full_denoise=True
                # but let's keep it consistent for potential future changes or logging.
                progress_done = start_step / total_sampling_steps if total_sampling_steps > 0 else 0
                # Use the original denoise value as the upper limit for the first stage
                max_denoise_for_stage = denoise if start_step == 0 else 1.0
                stage_denoise = max(0.0, max_denoise_for_stage - progress_done) * (total_sampling_steps / stage_steps if stage_steps > 0 else 1.0) # Rescale denoise? Let's simplify.
                # Simpler approach: Assume each stage denoises fully relative to its start point.
                # The denoise=1.0 passed to shader_ksampler combined with force_full_denoise=True handles this.
                stage_denoise_param_for_sampler = 1.0

                if debugger.enabled and debugger.debug_level >= 1:
                    print(f"📊 Sequential stage {stage_idx}: steps {start_step}-{end_step} ({stage_steps} steps this stage), strength {stage_strength:.3f}")
                
                # Determine seed for this stage based on consistency mode
                stage_seed = seed if use_temporal_coherence else seed + stage_idx
                
                # Generate shader noise for this stage
                if debugger.enabled:
                    print(f"🔳 Generating shader noise for sequential stage {stage_idx}")
                    
                with debugger.time_operation(f"gen_seq_shader_{stage_idx}") if debugger.enabled else contextlib.nullcontext():
                    stage_shader_noise = self._generate_shader_noise(
                        latent_samples=latent_samples, # Pass latent samples to derive shape if needed
                        target_noise_shape=noise_shape, # Pass the target noise shape
                        shader_params=stage_shader_params, 
                        shader_type=shader_type, 
                        seed=stage_seed,
                        device=device,
                        model=model_to_use,
                        model_name=model_name,
                        frame_count=frames, # Pass correct frame count
                        frame_dim_idx=frame_dim_idx # Pass frame dimension index
                    )
                
                if debugger.enabled and debugger.debug_level >= 2:
                    debugger.analyze_tensor(stage_shader_noise, f"seq_shader_noise_{stage_idx}")
                
                # Save shader noise if visualizer is enabled
                if visualizer.enabled:
                    visualizer.save_latent_visualization(
                        stage_shader_noise.clone(),
                        f"Sequential {stage_idx} Shader Noise",
                        stage_info={"stage_type": "sequential", "stage_idx": stage_idx}
                    )
                
                # Apply noise transform
                if debugger.enabled and noise_transform != "none":
                    print(f"🔄 Applying {noise_transform} transform to noise")
                
                stage_shader_noise = self._apply_noise_transform(stage_shader_noise, noise_transform)
                
                if debugger.enabled and debugger.debug_level >= 3 and noise_transform != "none":
                    debugger.analyze_tensor(stage_shader_noise, f"seq_transformed_noise_{stage_idx}")
                
                # Save transformed noise if visualizer is enabled and transform was applied
                if visualizer.enabled and noise_transform != "none":
                    visualizer.save_latent_visualization(
                        stage_shader_noise.clone(),
                        f"Sequential {stage_idx} Transformed Noise",
                        stage_info={"stage_type": "sequential", "stage_idx": stage_idx, "transform": noise_transform}
                    )
                
                # Blend with base noise
                if debugger.enabled:
                    print(f"🔀 Blending noise with {blend_mode} mode at strength {stage_strength:.3f}")
                    
                final_noise = self._blend_noises(base_noise_sequential, stage_shader_noise, blend_mode, stage_strength)
                
                if debugger.enabled and debugger.debug_level >= 2:
                    debugger.log_blend_operation(base_noise_sequential, stage_shader_noise, final_noise, blend_mode, stage_strength)
                
                # Save blended noise if visualizer is enabled
                if visualizer.enabled:
                    visualizer.save_latent_visualization(
                        final_noise.clone(),
                        f"Sequential {stage_idx} Blended Noise",
                        stage_info={"stage_type": "sequential", "stage_idx": stage_idx, "blend_mode": blend_mode, "strength": stage_strength}
                    )
                
                # Create a copy of current latent for sampling
                stage_latent = current_latent.copy()
                
                # Run the sampling for this stage
                stage_seed = seed if use_temporal_coherence else seed + stage_idx
                
                if debugger.enabled:
                    print(f"🚀 Running sequential sampling stage {stage_idx} with seed {stage_seed}")
                    if using_custom_sigmas:
                        print("   (Using custom sigma schedule)")

                # Create stage info for this sequential stage
                sequential_stage_info = {
                    "stage_type": "sequential",
                    "stage_idx": stage_idx,
                    "start_step": start_step,
                    "end_step": end_step,
                    "strength": stage_strength,
                    "denoise": stage_denoise
                }

                stage_result = shader_ksampler(
                    model=model_to_use, # Pass the potentially wrapped model
                    seed=stage_seed,
                    steps=stage_steps, # Pass the steps for THIS stage
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=stage_latent,
                    noise_tensor=final_noise,
                    denoise=stage_denoise_param_for_sampler, # Pass 1.0 as it does full denoise per stage
                    stage_info=sequential_stage_info,
                    denoise_visualization_frequency=denoise_visualization_frequency
                )

                # --- DEBUG PRINT ---
                # print(f"[DEBUG] shader_ksampler returned (type: {type(stage_result)}): {stage_result}")
                # --- END DEBUG PRINT ---

                # Update current latent with stage result - MODIFIED TO BE SAFER
                if isinstance(stage_result, tuple) and len(stage_result) > 0 and isinstance(stage_result[0], dict):
                    current_latent = stage_result[0]
                elif isinstance(stage_result, dict) and 'samples' in stage_result:
                    print("[DEBUG] Handling shader_ksampler result as dict in sequential loop.")
                    current_latent = stage_result # Assign dict directly
                else:
                    print(f"[ERROR] Unexpected stage_result type or format in sequential loop: {type(stage_result)}. Content: {stage_result}")
                    raise TypeError(f"shader_ksampler returned unexpected type/format: {type(stage_result)}")
                
                # Capture shader process if visualizer is enabled
                if visualizer.enabled:
                    stage_data = {
                        "start_step": start_step,
                        "end_step": end_step,
                        "strength": stage_strength,
                        "denoise": stage_denoise,
                        "seed": stage_seed
                    }
                    
                    visualizer.capture_shader_process(
                        phase="sequential",
                        stage_idx=stage_idx,
                        stage_type="shader_sampling",
                        stage_data=stage_data,
                        base_noise=base_noise_sequential.clone(),
                        shader_noise=stage_shader_noise.clone(),
                        blended_noise=final_noise.clone(),
                        result=current_latent["samples"].clone()
                    )
                
                if debugger.enabled:
                    debugger.log_stage_end("sequential", stage_idx)
                    if debugger.debug_level >= 2:
                        debugger.analyze_tensor(current_latent["samples"], f"seq_stage{stage_idx}_result")
        
        # PHASE 2: Injection Shader Stages S{αi}∘K{βi}
        # Only proceed if we have injection stages to run
        if injection_stages > 0:
            if debugger.enabled:
                print(f"💉 Starting Injection Phase with {injection_stages} stages over {total_sampling_steps} total steps")
                
            # If we skipped sequential stages, current_latent might not be set properly
            if sequential_stages <= 0 and denoise > 0.0:
                # Run a standard ksampler for initial denoising using total_sampling_steps
                if debugger.enabled:
                    print(f"🔄 Performing initial denoising before injection stages ({total_sampling_steps} steps)")
                    
                initial_result = shader_ksampler(
                    model=model_to_use, # Pass the potentially wrapped model
                    seed=seed,
                    steps=total_sampling_steps, # Use total_sampling_steps
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image={"samples": latent_samples},
                    noise_tensor=initial_noise,
                    denoise=denoise, # Use the overall denoise parameter here
                    stage_info={
                        "stage_type": "initial_denoising",
                        "stage_idx": 0,
                        "start_step": 0,
                        "end_step": total_sampling_steps,
                        "denoise": denoise
                    },
                    denoise_visualization_frequency=denoise_visualization_frequency
                ) # Call first, then check type

                # Check result type - MODIFIED TO BE SAFER
                if isinstance(initial_result, tuple) and len(initial_result) > 0 and isinstance(initial_result[0], dict):
                    current_latent = initial_result[0]
                elif isinstance(initial_result, dict) and 'samples' in initial_result:
                    print("[DEBUG] Handling initial shader_ksampler result as dict.")
                    current_latent = initial_result # Assign dict directly
                else:
                    print(f"[ERROR] Unexpected initial_result type or format: {type(initial_result)}. Content: {initial_result}")
                    raise TypeError(f"Initial shader_ksampler returned unexpected type/format: {type(initial_result)}")

                if debugger.enabled and debugger.debug_level >= 2:
                    debugger.analyze_tensor(current_latent["samples"], "after_initial_denoising")
                
                # Save initial denoising result if visualizer is enabled
                if visualizer.enabled:
                    visualizer.save_latent_visualization(
                        current_latent["samples"].clone(),
                        "Initial Denoising Result",
                        stage_info={"stage_type": "initial_denoising"},
                        is_sample=True
                    )
            
            # Calculate the strength for each injection stage
            injection_strengths = self._calculate_stage_strengths(shader_strength, injection_stages, injection_distribution)
            
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"💪 Injection strengths: {[f'{s:.3f}' for s in injection_strengths]}")
            
            # Calculate at which steps to apply shader noise for injection based on total_sampling_steps
            step_points = self._calculate_step_points(total_sampling_steps, injection_stages)
            
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"🔢 Injection step points: {step_points}")
            
            # Generate a base noise pattern for blending in injection stages
            # Use the same seed if temporal coherence is enabled
            injection_base_seed = seed if use_temporal_coherence else seed + sequential_stages
            
            if debugger.enabled:
                print(f"🎲 Generating base injection noise with seed {injection_base_seed}")
                
            torch.manual_seed(injection_base_seed)
            # Generate noise with the CORRECT noise_shape derived above
            base_noise_injection = torch.randn(noise_shape, device=device)
            
            if debugger.enabled and debugger.debug_level >= 2:
                debugger.analyze_tensor(base_noise_injection, "base_noise_injection")
            
            # Generate base shader noise for injection stages
            base_shader_noise_injection = self._generate_shader_noise(
                latent_samples=latent_samples, # Pass latent samples
                target_noise_shape=noise_shape, # Pass the target noise shape
                shader_params=shader_params, 
                shader_type=shader_type, 
                seed=seed,
                device=device,
                model=model_to_use,
                model_name=model_name,
                frame_count=frames, # Pass correct frame count
                frame_dim_idx=frame_dim_idx # Pass frame dimension index
            )
            
            # Save base shader noise for injection if visualizer is enabled
            if visualizer.enabled:
                visualizer.save_latent_visualization(
                    base_shader_noise_injection.clone(),
                    "Base Injection Shader Noise",
                    stage_info={"stage_type": "injection_base_shader"}
                )
            
            # Create step ranges for injection stages based on total_sampling_steps
            injection_ranges = []
            for i in range(len(step_points)):
                start_step = step_points[i]
                # For last stage, go to the end (total_sampling_steps)
                end_step = step_points[i + 1] if i < len(step_points) - 1 else total_sampling_steps
                # Only add if there are steps to process
                if end_step > start_step:
                    injection_ranges.append((start_step, end_step))
            
            if debugger.enabled and debugger.debug_level >= 2:
                print(f"🔢 Injection ranges: {injection_ranges}")
            
            # Process each injection stage
            for stage_idx, (start_step, end_step) in enumerate(injection_ranges):
                # Start logging this stage
                if debugger.enabled:
                    debugger.log_stage_start("injection", stage_idx, {
                        "start_step": start_step,
                        "end_step": end_step,
                        "strength": injection_strengths[stage_idx]
                    })
                
                # Calculate stage-specific parameters
                stage_steps = end_step - start_step # Steps for this specific stage call
                stage_strength = injection_strengths[stage_idx]
                
                # Calculate remaining denoise for this stage (see sequential stage comment)
                progress_done = start_step / total_sampling_steps if total_sampling_steps > 0 else 0
                max_denoise_for_stage = denoise if start_step == 0 else 1.0 # Likely always 1.0 here
                stage_denoise_param_for_sampler = 1.0 # Pass 1.0 as it does full denoise per stage

                if debugger.enabled and debugger.debug_level >= 1:
                    print(f"📊 Injection stage {stage_idx}: steps {start_step}-{end_step} ({stage_steps} steps this stage), strength {stage_strength:.3f}")
                
                # Use base shader noise but apply stage-specific transformations
                shader_noise = base_shader_noise_injection.clone()
                
                if debugger.enabled and noise_transform != "none":
                    print(f"🔄 Applying {noise_transform} transform to injection noise")
                    
                shader_noise = self._apply_noise_transform(shader_noise, noise_transform)
                
                if debugger.enabled and debugger.debug_level >= 3 and noise_transform != "none":
                    debugger.analyze_tensor(shader_noise, f"inj_transformed_noise_{stage_idx}")
                
                # Save transformed noise if visualizer is enabled and transform was applied
                if visualizer.enabled and noise_transform != "none":
                    visualizer.save_latent_visualization(
                        shader_noise.clone(),
                        f"Injection {stage_idx} Transformed Noise",
                        stage_info={"stage_type": "injection", "stage_idx": stage_idx, "transform": noise_transform}
                    )
                
                # Blend with base noise
                if debugger.enabled:
                    print(f"🔀 Blending injection noise with {blend_mode} mode at strength {stage_strength:.3f}")
                    
                final_noise = self._blend_noises(base_noise_injection, shader_noise, blend_mode, stage_strength)
                
                if debugger.enabled and debugger.debug_level >= 2:
                    debugger.log_blend_operation(base_noise_injection, shader_noise, final_noise, blend_mode, stage_strength)
                
                # Save blended noise if visualizer is enabled
                if visualizer.enabled:
                    visualizer.save_latent_visualization(
                        final_noise.clone(),
                        f"Injection {stage_idx} Blended Noise",
                        stage_info={"stage_type": "injection", "stage_idx": stage_idx, "blend_mode": blend_mode, "strength": stage_strength}
                    )
                
                # Create a copy of current latent for sampling
                stage_latent = current_latent.copy()
                
                # Run the sampling for this injection stage
                stage_seed = seed if use_temporal_coherence else seed + sequential_stages + stage_idx
                
                if debugger.enabled:
                    print(f"🚀 Running injection sampling stage {stage_idx} with seed {stage_seed}")
                    if using_custom_sigmas:
                         print("   (Using custom sigma schedule)")

                # Create stage info for this injection stage
                injection_stage_info = {
                    "stage_type": "injection",
                    "stage_idx": stage_idx,
                    "start_step": start_step,
                    "end_step": end_step,
                    "strength": stage_strength,
                    "denoise": stage_denoise_param_for_sampler
                }

                stage_result = shader_ksampler(
                    model=model_to_use, # Pass the potentially wrapped model
                    seed=stage_seed,
                    steps=stage_steps, # Pass the steps for THIS stage
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=stage_latent,
                    noise_tensor=final_noise,
                    denoise=stage_denoise_param_for_sampler, # Pass 1.0
                    stage_info=injection_stage_info,
                    denoise_visualization_frequency=denoise_visualization_frequency
                )

                # Update current latent with stage result - MODIFIED TO BE SAFER
                if isinstance(stage_result, tuple) and len(stage_result) > 0 and isinstance(stage_result[0], dict):
                    current_latent = stage_result[0]
                elif isinstance(stage_result, dict) and 'samples' in stage_result:
                    print("[DEBUG] Handling shader_ksampler result as dict in injection loop.")
                    current_latent = stage_result # Assign dict directly
                else:
                    print(f"[ERROR] Unexpected stage_result type or format in injection loop: {type(stage_result)}. Content: {stage_result}")
                    raise TypeError(f"shader_ksampler returned unexpected type/format: {type(stage_result)}")
                
                # Capture shader process if visualizer is enabled
                if visualizer.enabled:
                    stage_data = {
                        "start_step": start_step,
                        "end_step": end_step,
                        "strength": stage_strength,
                        "denoise": stage_denoise_param_for_sampler,
                        "seed": stage_seed
                    }
                    
                    visualizer.capture_shader_process(
                        phase="injection",
                        stage_idx=stage_idx,
                        stage_type="shader_sampling",
                        stage_data=stage_data,
                        base_noise=base_noise_injection.clone(),
                        shader_noise=shader_noise.clone(),
                        blended_noise=final_noise.clone(),
                        result=current_latent["samples"].clone()
                    )
                
                if debugger.enabled:
                    debugger.log_stage_end("injection", stage_idx)
                    if debugger.debug_level >= 2:
                        debugger.analyze_tensor(current_latent["samples"], f"inj_stage{stage_idx}_result")
        
        # Reset random seed state when done
        torch.manual_seed(torch.seed())
        
        # Format output
        out = latent_image.copy()
        out["samples"] = current_latent["samples"]
        
        # Include shader parameters in the UI information
        shader_info = {
            "shader_type": shader_type,
            "shader_strength": shader_strength,
            "sequential_stages": sequential_stages,
            "injection_stages": injection_stages,
            "sequential_distribution": sequential_distribution,
            "injection_distribution": injection_distribution,
            "noise_transform": noise_transform,
            "temporal_coherence": use_temporal_coherence
        }
        shader_info["using_custom_sigmas"] = using_custom_sigmas # Add info about sigma usage

        # Capture final result if visualizer is enabled - this doesn't affect the output
        if visualizer.enabled:
            final_metadata = {
                "shader_type": shader_type,
                "sequential_stages": sequential_stages,
                "injection_stages": injection_stages,
                "blend_mode": blend_mode,
                "noise_transform": noise_transform
            }
            # Use a copy of the tensor to ensure we don't modify the output
            visualizer.capture_final_result(current_latent["samples"].clone(), final_metadata)
            # Get visualization paths for UI display
            viz_paths = visualizer.get_ui_image_paths()
            visualizer.disable()
        else:
            viz_paths = {
                "base_noise": None,
                "shader_noise": None,
                "blended_noise": None,
                "stage_results": None,
                "final_result": None,
                "grids": []
            }
        
        # --- Explicitly delete the model reference before returning ---
        # This helps ensure the wrapper (if created) is released promptly.
        if 'model_to_use' in locals():
            del model_to_use
        # --- End explicit deletion ---

        # Return both the generated samples and UI information - visualization does not affect this
        return {"ui": {
            "images": [], 
            "show_custom_preview": [show_custom_preview], 
            "shader_info": shader_info,
            "viz_paths": viz_paths  # Add visualization paths to UI data
        }, "result": (out,)}
