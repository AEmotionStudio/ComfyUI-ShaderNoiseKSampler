"""
API routes for ShaderNoiseKSampler extension.

Provides server-side endpoints for saving shader parameters from the frontend.
"""

from aiohttp import web
import json
import logging
import os
from .shader_params_reader import ShaderParamsReader

logger = logging.getLogger("ShaderNoiseKSampler")

# Get extension directory
EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))

# Keys the save button sends. Anything else is dropped rather than persisted:
# the file is read back into shader params on every run, so unknown keys would
# flow into noise generation untouched by validate_and_sanitize_params.
ALLOWED_KEYS = frozenset({
    "shaderType",
    "shaderScale",
    "shaderOctaves",
    "shaderWarpStrength",
    "shaderShapeType",
    "shaderShapeStrength",
    "shaderPhaseShift",
    "colorScheme",
    "shaderColorIntensity",
    "visualization_type",
})


async def save_shader_params(request):
    """
    API endpoint to save shader parameters to JSON file.

    Receives JSON data from the frontend and writes it to data/shader_params.json.
    This enables automatic parameter persistence without manual file downloads.
    """
    try:
        data = await request.json()
        if not isinstance(data, dict):
            return web.json_response(
                {"status": "error", "message": "Expected a JSON object"}, status=400
            )

        params_file = os.path.join(EXTENSION_DIR, "data", "shader_params.json")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(params_file), exist_ok=True)

        # Keep only known keys, then validate and sanitize what is left
        known = {key: value for key, value in data.items() if key in ALLOWED_KEYS}
        sanitized_data = ShaderParamsReader.validate_and_sanitize_params(known)

        with open(params_file, 'w') as f:
            json.dump(sanitized_data, f, indent=2)

        logger.info("Saved shader params to %s", params_file)
        return web.json_response({"status": "success"})
    except json.JSONDecodeError:
        return web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)
    except Exception:
        # Log the detail, but do not hand internals back to the caller
        logger.exception("Error saving shader params")
        return web.json_response({"status": "error", "message": "Could not save parameters"}, status=500)


async def get_presets(request):
    """
    Serve the preset table, so the frontend can write a preset's values into the
    widgets without keeping a copy of its own. Preset values get recalibrated
    against real runs, and a second copy would drift the first time one did.
    """
    from .core import presets
    return web.json_response({
        "presets": presets.PRESETS,
        "descriptions": presets.PRESET_DESCRIPTIONS,
        "keys": list(presets.PRESET_KEYS),
    })

def setup_routes(server):
    """
    Register API routes with ComfyUI's PromptServer.

    Args:
        server: The PromptServer instance from ComfyUI
    """
    if hasattr(server, 'app') and hasattr(server.app, 'router'):
        server.app.router.add_post("/shader_noise_ksampler/save_params", save_shader_params)
        logger.info("Registered API route: POST /shader_noise_ksampler/save_params")
        server.app.router.add_get("/shader_noise_ksampler/presets", get_presets)
        logger.info("Registered API route: GET /shader_noise_ksampler/presets")
