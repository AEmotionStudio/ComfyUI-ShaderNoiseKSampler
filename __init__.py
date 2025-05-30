# First import components from shader noise ksampler
from .shader_noise_ksampler import (
    ShaderNoiseKSampler, 
    register_shader_generator, 
    SHADER_GENERATORS
)

# Import DirectShaderNoiseKSampler
from .direct_shader_ksampler import DirectShaderNoiseKSampler

# Import AdvancedImageComparer
from .advanced_comparer import AdvancedImageComparer

# Import VideoComparer
from .video_comparer import VideoComparer

# Next import tensor class
from .shader_to_tensor import ShaderToTensor

# Import and integrate domain warp
from .shaders.domain_warp import add_domain_warp_to_tensor, generate_domain_warp_tensor

# Apply domain warp integration
add_domain_warp_to_tensor(ShaderToTensor)
register_shader_generator("domain_warp", generate_domain_warp_tensor)

# Import and integrate tensor field
from .shaders.tensor_field import add_tensor_field_to_tensor, generate_tensor_field_tensor

# Apply tensor field integration
add_tensor_field_to_tensor(ShaderToTensor)
register_shader_generator("tensor_field", generate_tensor_field_tensor)

# Import and integrate Curl Noise
from .shaders.curl_noise import add_curl_noise_to_tensor, generate_curl_noise_tensor

# Apply Curl Noise integration
add_curl_noise_to_tensor(ShaderToTensor)
register_shader_generator("curl", generate_curl_noise_tensor)
register_shader_generator("curl_noise", generate_curl_noise_tensor)

# Import and integrate temporal coherent noise
from .shaders.temporal_coherent_noise import integrate_temporal_coherent_noise, generate_temporal_coherent_noise_tensor

# Apply temporal coherent noise integration
integrate_temporal_coherent_noise()
register_shader_generator("temporal_coherent", generate_temporal_coherent_noise_tensor)
register_shader_generator("temporal_coherent_noise", generate_temporal_coherent_noise_tensor)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ShaderNoiseKSampler": ShaderNoiseKSampler,
    "ShaderNoiseKSamplerDirect": DirectShaderNoiseKSampler,
    "AdvancedImageComparer": AdvancedImageComparer,
    "Video Comparer": VideoComparer,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShaderNoiseKSampler": "Shader Noise KSampler",
    "ShaderNoiseKSamplerDirect": "Shader Noise KSampler (Direct)",
    "AdvancedImageComparer": "Advanced Image Comparer",
    "Video Comparer": "Video Comparer",
}

# Add web directory for UI components
WEB_DIRECTORY = "./web"

# List of JS files to be loaded - ORDER IS CRITICAL
__js_files__ = [
    "gradient_title.js", 
    "shader_renderer.js",
    "matrix_button.js", 
    "shader_params_save_button.js", 
    "noise_visualizer.js",
    "advanced_comparer.js",           
    "video_comparer.js"               
]

# List of exported elements
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__js_files__", "SHADER_GENERATORS"]