"""
ComfyUI-ShaderNoiseKSampler

A custom KSampler node that uses shader-based noise patterns
for creative image generation.
"""

# Import node classes from nodes package
from .nodes import (
    ShaderNoiseKSampler,
    DirectShaderNoiseKSampler,
    AdvancedImageComparer,
    VideoComparer,
)
from .shader_to_tensor import ShaderToTensor

# Import shader registry
from .shaders.registry import (
    ShaderRegistry,
    register_shader,
    get_shader,
    list_shaders,
)

# Import shader generators
from .shaders.domain_warp import (
    DomainWarpGenerator,
    add_domain_warp_to_tensor,
    generate_domain_warp_tensor,
)
from .shaders.tensor_field import (
    TensorFieldGenerator,
    add_tensor_field_to_tensor,
    generate_tensor_field_tensor,
)
from .shaders.curl_noise import (
    CurlNoiseGenerator,
    add_curl_noise_to_tensor,
    generate_curl_noise_tensor,
)
from .shaders.temporal_coherent_noise import (
    TemporalCoherentNoiseGenerator,
    integrate_temporal_coherent_noise,
    generate_temporal_coherent_noise_tensor,
)

# Register all shader generators with the centralized registry
register_shader("domain_warp", DomainWarpGenerator, {
    "description": "Domain warping noise using FBM",
    "supports_temporal": True,
})
register_shader("tensor_field", TensorFieldGenerator, {
    "description": "Tensor field based noise patterns",
    "supports_temporal": True,
})
register_shader("curl", CurlNoiseGenerator, {
    "description": "Curl/fluid noise patterns",
    "supports_temporal": True,
})
register_shader("curl_noise", CurlNoiseGenerator, {
    "description": "Curl/fluid noise patterns (alias)",
    "supports_temporal": True,
})
register_shader("temporal_coherent", TemporalCoherentNoiseGenerator, {
    "description": "Temporally coherent noise for animations",
    "supports_temporal": True,
})
register_shader("temporal_coherent_noise", TemporalCoherentNoiseGenerator, {
    "description": "Temporally coherent noise (alias)",
    "supports_temporal": True,
})

# Apply shader integrations to ShaderToTensor for backward compatibility
add_domain_warp_to_tensor(ShaderToTensor)
add_tensor_field_to_tensor(ShaderToTensor)
add_curl_noise_to_tensor(ShaderToTensor)
integrate_temporal_coherent_noise()

# Legacy SHADER_GENERATORS dict for backward compatibility
# Maps shader type names to generator functions
SHADER_GENERATORS = {
    "domain_warp": generate_domain_warp_tensor,
    "tensor_field": generate_tensor_field_tensor,
    "curl": generate_curl_noise_tensor,
    "curl_noise": generate_curl_noise_tensor,
    "temporal_coherent": generate_temporal_coherent_noise_tensor,
    "temporal_coherent_noise": generate_temporal_coherent_noise_tensor,
}


def get_shader_generator(shader_type: str):
    """
    Get the appropriate shader generator function based on shader type.
    
    This function provides backward compatibility with the old API
    while using the new registry system internally.
    
    Args:
        shader_type: Name of the shader type
        
    Returns:
        Generator function (static method) for the shader type, or None if not found
    """
    # First try the legacy dict for backward compatibility
    if shader_type in SHADER_GENERATORS:
        return SHADER_GENERATORS[shader_type]
    
    # Fall back to registry - return the static generate method
    generator_class = get_shader(shader_type)
    if generator_class is not None:
        # Return the static generate method directly (consistent with shader_noise_ksampler.py)
        return generator_class.generate
    
    # Return None if not found
    return None


def register_shader_generator(shader_type: str, generator_function):
    """
    Register a shader generator function.
    
    This function provides backward compatibility with the old API.
    Registers to both the legacy SHADER_GENERATORS dict and the new registry.
    
    Args:
        shader_type: Name of the shader type
        generator_function: Generator function or class to register
    """
    # Add to legacy dict for backward compatibility
    SHADER_GENERATORS[shader_type] = generator_function
    # Also register to the new registry so shader_noise_ksampler.py can find it
    register_shader(shader_type, generator_function)


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
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__js_files__",
    "SHADER_GENERATORS",
    "get_shader_generator",
    "register_shader_generator",
    "ShaderRegistry",
    "register_shader",
    "get_shader",
    "list_shaders",
]
