"""
API routes for ShaderNoiseKSampler extension.

Provides server-side endpoints for saving shader parameters from the frontend.
"""

from aiohttp import web
import json
import os

# Get extension directory
EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))


async def save_shader_params(request):
    """
    API endpoint to save shader parameters to JSON file.
    
    Receives JSON data from the frontend and writes it to data/shader_params.json.
    This enables automatic parameter persistence without manual file downloads.
    """
    try:
        data = await request.json()
        params_file = os.path.join(EXTENSION_DIR, "data", "shader_params.json")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        
        with open(params_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[ShaderNoiseKSampler] Saved shader params to {params_file}")
        return web.json_response({"status": "success"})
    except Exception as e:
        print(f"[ShaderNoiseKSampler] Error saving shader params: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


def setup_routes(server):
    """
    Register API routes with ComfyUI's PromptServer.
    
    Args:
        server: The PromptServer instance from ComfyUI
    """
    if hasattr(server, 'app') and hasattr(server.app, 'router'):
        server.app.router.add_post("/shader_noise_ksampler/save_params", save_shader_params)
        print("[ShaderNoiseKSampler] Registered API route: POST /shader_noise_ksampler/save_params")

