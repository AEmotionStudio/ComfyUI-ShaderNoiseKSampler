"""
Debugging and visualization utilities for shader noise generation.

This module provides stub implementations for debugging and visualization
that can be replaced with full implementations when needed.
"""

import contextlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class StubVisualizer:
    """
    A stub visualizer that implements the same interface but does nothing.
    
    This can be replaced with a full implementation for debugging purposes.
    """
    
    def __init__(self):
        self.enabled = False
        
    def enable(self, seed: Optional[int] = None, shader_type: Optional[str] = None, 
               additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Enable visualization."""
        pass
        
    def disable(self) -> None:
        """Disable visualization."""
        pass
        
    def save_latent_visualization(self, tensor, label: str, 
                                   stage_info: Optional[Dict[str, Any]] = None, 
                                   is_sample: bool = False) -> None:
        """Save a latent visualization."""
        pass
        
    def save_denoising_step(self, tensor, stage_info: Dict[str, Any], 
                            current_step: int, total_steps: int) -> None:
        """Save a denoising step visualization."""
        pass
        
    def capture_shader_process(self, phase: str, stage_idx: int, stage_type: str, 
                               stage_data: Dict[str, Any], base_noise, shader_noise, 
                               blended_noise, result) -> None:
        """Capture shader process state."""
        pass
        
    def capture_final_result(self, tensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Capture final result."""
        pass
        
    def get_ui_image_paths(self) -> Dict[str, Any]:
        """Get UI image paths."""
        return {
            "base_noise": None,
            "shader_noise": None,
            "blended_noise": None,
            "stage_results": None,
            "final_result": None,
            "grids": []
        }


class StubDebugger:
    """
    A stub debugger that implements the same interface but does nothing.
    
    This can be replaced with a full implementation for debugging purposes.
    """
    
    def __init__(self):
        self.enabled = False
        self.debug_level = 0
        
    def reset(self) -> None:
        """Reset debugger state."""
        pass
        
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return contextlib.nullcontext()
        
    def analyze_tensor(self, tensor, name: str) -> None:
        """Analyze a tensor and log statistics."""
        pass
        
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass
        
    def log_stage_start(self, stage_type: str, stage_idx: int, 
                        params: Dict[str, Any]) -> None:
        """Log stage start."""
        pass
        
    def log_stage_end(self, stage_type: str, stage_idx: int) -> None:
        """Log stage end."""
        pass
        
    def log_blend_operation(self, base, shader, result, mode: str, 
                            strength: float) -> None:
        """Log blend operation."""
        pass


# Global instances
_visualizer: Optional[StubVisualizer] = None
_debugger: Optional[StubDebugger] = None


def get_visualizer() -> StubVisualizer:
    """Get the global visualizer instance."""
    global _visualizer
    if _visualizer is None:
        _visualizer = StubVisualizer()
    return _visualizer


def get_debugger() -> StubDebugger:
    """Get the global debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = StubDebugger()
    return _debugger


def set_debug_level(level: int) -> StubDebugger:
    """Set the debug level and return the debugger."""
    debugger = get_debugger()
    debugger.debug_level = level
    debugger.enabled = level > 0
    return debugger
