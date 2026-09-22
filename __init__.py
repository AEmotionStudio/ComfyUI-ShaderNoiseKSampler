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
from .shader_noise_walk import ShaderNoiseWalk
from .shader_noise_source import ShaderNoiseSource
from .shader_to_tensor import ShaderToTensor

# Import shader registry
from .shaders.registry import (
    ShaderRegistry,
    register_shader,
    get_shader,
    list_shaders,
)

# Importing the generator modules runs their @shader_generator decorators,
# which is what registers every shader type. Only the aliases below are
# registered here.
from .shaders.domain_warp import (
    DomainWarpGenerator,
    generate_domain_warp_tensor,
)
from .shaders.tensor_field import (
    TensorFieldGenerator,
    generate_tensor_field_tensor,
)
from .shaders.curl_noise import (
    CurlNoiseGenerator,
    generate_curl_noise_tensor,
)
from .shaders.temporal_coherent_noise import (
    TemporalCoherentNoiseGenerator,
    generate_temporal_coherent_noise_tensor,
)
from .shaders.spectral import (
    SpectralNoiseGenerator,
    generate_spectral_tensor,
)
from .shaders.gaussian import (
    GaussianNoiseGenerator,
    generate_gaussian_tensor,
)
from .shaders.waves import (
    WavesNoiseGenerator,
    generate_waves_tensor,
)
from .shaders.cellular import (
    CellularNoiseGenerator,
    generate_cellular_tensor,
)
from .shaders.projection_3d import (
    Projection3DNoiseGenerator,
    generate_projection_3d_tensor,
)
from .shaders.interference import (
    InterferenceNoiseGenerator,
    generate_interference_tensor,
)
from .shaders.heterogeneous_fbm import (
    HeterogeneousFBMGenerator,
    generate_heterogeneous_fbm_tensor,
)
from .shaders.perlin import (
    PerlinNoiseGenerator,
    generate_perlin_tensor,
)
from .shaders.fractal import (
    FractalNoiseGenerator,
    generate_fractal_tensor,
)

register_shader("curl", CurlNoiseGenerator, {
    "description": "Curl/fluid noise patterns (alias)",
    "supports_temporal": True,
})
register_shader("temporal_coherent_noise", TemporalCoherentNoiseGenerator, {
    "description": "Temporally coherent noise (alias)",
    "supports_temporal": True,
})

# Register API routes for server-side parameter saving
try:
    from server import PromptServer
    from .api_routes import setup_routes
    setup_routes(PromptServer.instance)
except ImportError:
    # PromptServer not available (e.g., running tests without ComfyUI)
    pass
except Exception as e:
    print(f"[ShaderNoiseKSampler] Warning: Could not register API routes: {e}")

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


def _wrap_legacy_generator(legacy_func):
    """
    Wrap a legacy generator function to accept the new 'params' keyword argument.

    Legacy functions expect 'shader_params' as a dict, but the new convention uses
    'params' which may be a ShaderParams instance. This wrapper translates between
    the two conventions and converts ShaderParams to dict.

    Args:
        legacy_func: Legacy generator function expecting shader_params as dict

    Returns:
        Wrapped function accepting params (ShaderParams or dict)
    """
    def wrapper(**kwargs):
        # If 'params' is provided but not 'shader_params', translate it
        if 'params' in kwargs and 'shader_params' not in kwargs:
            params = kwargs.pop('params')
            # Convert ShaderParams to dict if needed for legacy function
            if hasattr(params, 'to_dict'):
                shader_params = params.to_dict()
            elif hasattr(params, '__iter__'):
                shader_params = dict(params)
            else:
                shader_params = params
            kwargs['shader_params'] = shader_params
        return legacy_func(**kwargs)
    return wrapper


def get_shader_generator(shader_type: str):
    """
    Get the appropriate shader generator function based on shader type.

    This function provides backward compatibility with the old API
    while using the new registry system internally. The returned function
    accepts both 'params' (new convention) and 'shader_params' (legacy convention).

    Args:
        shader_type: Name of the shader type

    Returns:
        Generator function for the shader type. Falls back to generate_noise_tensor
        if not found (consistent with shader_noise_ksampler.py behavior).
    """
    # Import here to avoid circular imports
    from .shader_params_reader import generate_noise_tensor

    # First try the legacy dict for backward compatibility
    # Wrap legacy functions to accept 'params' keyword argument
    if shader_type in SHADER_GENERATORS:
        return _wrap_legacy_generator(SHADER_GENERATORS[shader_type])

    # Fall back to registry - return the static generate method
    generator_class = get_shader(shader_type)
    if generator_class is not None:
        # Return the static generate method directly (consistent with shader_noise_ksampler.py)
        return generator_class.generate

    # Fallback: wrap generate_noise_tensor to translate params -> shader_params
    # This matches the behavior in shader_noise_ksampler.py
    def fallback_wrapper(params, height, width, batch_size, device, seed, target_channels, **kwargs):
        # Convert ShaderParams to dict if needed for legacy function
        if hasattr(params, 'to_dict'):
            shader_params = params.to_dict()
        elif hasattr(params, '__iter__'):
            shader_params = dict(params)
        else:
            shader_params = {}
        return generate_noise_tensor(
            shader_params=shader_params,
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            seed=seed,
            target_channels=target_channels,
            **kwargs
        )
    return fallback_wrapper


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
    "ShaderNoiseWalk": ShaderNoiseWalk,
    "ShaderNoiseSource": ShaderNoiseSource,
    "AdvancedImageComparer": AdvancedImageComparer,
    "Video Comparer": VideoComparer,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShaderNoiseKSampler": "Shader Noise KSampler",
    "ShaderNoiseKSamplerDirect": "Shader Noise KSampler (Direct)",
    "ShaderNoiseWalk": "Shader Noise Walk",
    "ShaderNoiseSource": "Shader Noise Source",
    "AdvancedImageComparer": "Advanced Image Comparer",
    "Video Comparer": "Video Comparer",
}

# Add web directory for UI components
WEB_DIRECTORY = "./web"

# List of exported elements
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "SHADER_GENERATORS",
    "get_shader_generator",
    "register_shader_generator",
    "ShaderRegistry",
    "register_shader",
    "get_shader",
    "list_shaders",
]
