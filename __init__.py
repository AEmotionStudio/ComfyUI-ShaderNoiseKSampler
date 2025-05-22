# First import components from shader noise ksampler
from .shader_noise_ksampler import (
    ShaderNoiseKSampler, 
    register_shader_generator, 
    SHADER_GENERATORS
)

# Import DirectShaderNoiseKSampler
from .direct_shader_ksampler import DirectShaderNoiseKSampler

# Next import tensor class
from .shader_to_tensor import ShaderToTensor

# Import and integrate domain warp
from .shaders.domain_warp import add_domain_warp_to_tensor, generate_domain_warp_tensor

# Apply domain warp integration
print("=== INTEGRATING DOMAIN WARP TO SHADER SYSTEM ===")
add_domain_warp_to_tensor(ShaderToTensor)
register_shader_generator("domain_warp", generate_domain_warp_tensor)
print(f"Domain warp integration complete - has domain_warp: {'domain_warp' in dir(ShaderToTensor)}")

# Import and integrate tensor field
from .shaders.tensor_field import add_tensor_field_to_tensor, generate_tensor_field_tensor

# Apply tensor field integration
print("=== INTEGRATING TENSOR FIELD TO SHADER SYSTEM ===")
add_tensor_field_to_tensor(ShaderToTensor)
register_shader_generator("tensor_field", generate_tensor_field_tensor)
print(f"Tensor field integration complete - has tensor_field: {'tensor_field' in dir(ShaderToTensor)}")

# Import and integrate Curl Noise
from .shaders.curl_noise import add_curl_noise_to_tensor, generate_curl_noise_tensor

# Apply Curl Noise integration
print("=== INTEGRATING CURL NOISE TO SHADER SYSTEM ===")
add_curl_noise_to_tensor(ShaderToTensor)
register_shader_generator("curl", generate_curl_noise_tensor)
register_shader_generator("curl_noise", generate_curl_noise_tensor)
print(f"Curl noise integration complete - has curl_noise: {'curl_noise' in dir(ShaderToTensor)}")

# Import and integrate temporal coherent noise
from .shaders.temporal_coherent_noise import integrate_temporal_coherent_noise, generate_temporal_coherent_noise_tensor

# Apply temporal coherent noise integration
print("=== INTEGRATING TEMPORAL COHERENT NOISE TO SHADER SYSTEM ===")
integrate_temporal_coherent_noise()
register_shader_generator("temporal_coherent", generate_temporal_coherent_noise_tensor)
register_shader_generator("temporal_coherent_noise", generate_temporal_coherent_noise_tensor)
print(f"Temporal coherent noise integration complete - has temporal_coherent_noise: {'temporal_coherent_noise' in dir(ShaderToTensor)}")

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ShaderNoiseKSampler": ShaderNoiseKSampler,
    "ShaderNoiseKSamplerDirect": DirectShaderNoiseKSampler,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShaderNoiseKSampler": "Shader Noise KSampler",
    "ShaderNoiseKSamplerDirect": "Shader Noise KSampler (Direct)",
}

# Add web directory for UI components
WEB_DIRECTORY = "./web"

# List of JS files to be loaded
__js_files__ = ["gradient_title.js", "shader_renderer.js", "shader_params_save_button.js", "shader_info_button.js", "noise_visualizer.js"]


# List of exported elements
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__js_files__", "__css_files__", "SHADER_GENERATORS"]