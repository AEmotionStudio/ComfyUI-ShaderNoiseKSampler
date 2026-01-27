"""
Parameter normalization and validation for shader noise.

This module provides a single source of truth for parameter handling,
normalizing the various naming conventions used across the codebase.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from .constants import (
    DEFAULT_SCALE,
    DEFAULT_OCTAVES,
    DEFAULT_WARP_STRENGTH,
    DEFAULT_PHASE_SHIFT,
    DEFAULT_SHAPE_STRENGTH,
    DEFAULT_COLOR_INTENSITY,
    DEFAULT_TIME,
    MAX_OCTAVES,
    MIN_SCALE,
    MAX_SCALE,
)

# Parameter alias mapping
# Maps canonical parameter names to all known aliases
PARAM_ALIASES = {
    # Scale parameter
    "scale": [
        "shaderScale",
        "scale",
        "noise_scale",
    ],
    # Octaves parameter
    "octaves": [
        "shaderOctaves",
        "octaves",
    ],
    # Warp strength parameter
    "warp_strength": [
        "shaderWarpStrength",
        "warp_strength",
        "warp",
    ],
    # Phase shift parameter
    "phase_shift": [
        "shaderPhaseShift",
        "phase_shift",
        "phase",
    ],
    # Shape type parameter
    "shape_type": [
        "shaderShapeType",
        "shape_type",
        "shapetype",
    ],
    # Shape mask strength parameter
    "shape_strength": [
        "shaderShapeStrength",
        "shapemaskstrength",
        "shapeMaskStrength",
        "shape_mask_strength",
        "shape_strength",
    ],
    # Color scheme parameter
    "color_scheme": [
        "colorScheme",
        "color_scheme",
    ],
    # Color intensity parameter
    "color_intensity": [
        "shaderColorIntensity",
        "intensity",
        "color_intensity",
    ],
    # Shader type parameter
    "shader_type": [
        "shaderType",
        "shader_type",
    ],
    # Time parameter
    "time": [
        "time",
        "animation_time",
    ],
    # Temporal coherence parameter
    "use_temporal_coherence": [
        "useTemporalCoherence",
        "temporal_coherence",
        "use_temporal_coherence",
    ],
    # Base seed parameter
    "base_seed": [
        "base_seed",
        "baseSeed",
    ],
    # Visualization type parameter
    "visualization_type": [
        "visualization_type",
        "visualizationType",
    ],
}

# Reverse mapping: alias -> canonical name
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in PARAM_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def normalize_param_name(name: str) -> str:
    """
    Normalize a parameter name to its canonical form.
    
    Args:
        name: Parameter name (any alias)
        
    Returns:
        Canonical parameter name
    """
    return _ALIAS_TO_CANONICAL.get(name.lower(), name)


def get_param_value(
    params: Dict[str, Any],
    canonical_name: str,
    default: Any = None
) -> Any:
    """
    Get a parameter value from a dict, checking all known aliases.
    
    Args:
        params: Parameter dictionary
        canonical_name: Canonical parameter name
        default: Default value if not found
        
    Returns:
        Parameter value or default
    """
    # Check canonical name first
    if canonical_name in params:
        return params[canonical_name]
    
    # Check all aliases
    aliases = PARAM_ALIASES.get(canonical_name, [])
    for alias in aliases:
        if alias in params:
            return params[alias]
    
    return default


class ShaderParams:
    """
    Normalized parameter container with validation.
    
    This class provides a consistent interface for accessing shader parameters
    regardless of which naming convention was used in the input.
    """
    
    def __init__(self, raw_params: Optional[Dict[str, Any]] = None):
        """
        Initialize shader parameters from a raw dictionary.
        
        Args:
            raw_params: Raw parameter dictionary (may use any naming convention)
        """
        self._params: Dict[str, Any] = {}
        if raw_params:
            self._normalize(raw_params)
        self._apply_defaults()
    
    def _normalize(self, raw: Dict[str, Any]) -> None:
        """
        Normalize raw parameters to canonical names.
        
        Args:
            raw: Raw parameter dictionary
        """
        for key, value in raw.items():
            canonical = normalize_param_name(key)
            # Don't overwrite if we already have a value for this canonical name
            if canonical not in self._params:
                self._params[canonical] = value
    
    def _apply_defaults(self) -> None:
        """Apply default values for missing parameters."""
        defaults = {
            "scale": DEFAULT_SCALE,
            "octaves": DEFAULT_OCTAVES,
            "warp_strength": DEFAULT_WARP_STRENGTH,
            "phase_shift": DEFAULT_PHASE_SHIFT,
            "shape_strength": DEFAULT_SHAPE_STRENGTH,
            "color_intensity": DEFAULT_COLOR_INTENSITY,
            "time": DEFAULT_TIME,
            "shader_type": "tensor_field",
            "shape_type": "none",
            "color_scheme": "none",
            "use_temporal_coherence": False,
            "visualization_type": 3,
        }
        
        for key, value in defaults.items():
            if key not in self._params:
                self._params[key] = value
    
    def validate(self) -> "ShaderParams":
        """
        Validate and sanitize parameter values.
        
        Returns:
            Self for chaining
        """
        # Validate octaves
        if "octaves" in self._params:
            try:
                val = float(self._params["octaves"])
                self._params["octaves"] = int(max(1.0, min(val, MAX_OCTAVES)))
            except (ValueError, TypeError):
                self._params["octaves"] = int(DEFAULT_OCTAVES)
        
        # Validate scale
        if "scale" in self._params:
            try:
                val = float(self._params["scale"])
                self._params["scale"] = max(MIN_SCALE, min(val, MAX_SCALE))
            except (ValueError, TypeError):
                self._params["scale"] = DEFAULT_SCALE
        
        # Validate float parameters
        float_params = [
            ("warp_strength", DEFAULT_WARP_STRENGTH),
            ("phase_shift", DEFAULT_PHASE_SHIFT),
            ("shape_strength", DEFAULT_SHAPE_STRENGTH),
            ("color_intensity", DEFAULT_COLOR_INTENSITY),
            ("time", DEFAULT_TIME),
        ]
        
        for param_name, default_val in float_params:
            if param_name in self._params:
                try:
                    self._params[param_name] = float(self._params[param_name])
                except (ValueError, TypeError):
                    self._params[param_name] = default_val
        
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            key: Parameter name (canonical or alias)
            default: Default value if not found
            
        Returns:
            Parameter value
        """
        canonical = normalize_param_name(key)
        return self._params.get(canonical, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a parameter value.
        
        Args:
            key: Parameter name (will be normalized)
            value: Value to set
        """
        canonical = normalize_param_name(key)
        self._params[canonical] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary with canonical names.
        
        Returns:
            Dictionary of parameters
        """
        return self._params.copy()
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary including all alias names.
        
        This is useful for compatibility with existing code that expects
        specific parameter names.
        
        Returns:
            Dictionary with all parameter aliases populated
        """
        result = self._params.copy()
        
        for canonical, aliases in PARAM_ALIASES.items():
            if canonical in self._params:
                value = self._params[canonical]
                for alias in aliases:
                    result[alias] = value
        
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        canonical = normalize_param_name(key)
        return canonical in self._params
    
    # Property accessors for common parameters
    @property
    def scale(self) -> float:
        return self.get("scale", DEFAULT_SCALE)
    
    @scale.setter
    def scale(self, value: float) -> None:
        self.set("scale", value)
    
    @property
    def octaves(self) -> int:
        return int(self.get("octaves", DEFAULT_OCTAVES))
    
    @octaves.setter
    def octaves(self, value: int) -> None:
        self.set("octaves", value)
    
    @property
    def warp_strength(self) -> float:
        return self.get("warp_strength", DEFAULT_WARP_STRENGTH)
    
    @warp_strength.setter
    def warp_strength(self, value: float) -> None:
        self.set("warp_strength", value)
    
    @property
    def phase_shift(self) -> float:
        return self.get("phase_shift", DEFAULT_PHASE_SHIFT)
    
    @phase_shift.setter
    def phase_shift(self, value: float) -> None:
        self.set("phase_shift", value)
    
    @property
    def shape_type(self) -> str:
        return self.get("shape_type", "none")
    
    @shape_type.setter
    def shape_type(self, value: str) -> None:
        self.set("shape_type", value)
    
    @property
    def shape_strength(self) -> float:
        return self.get("shape_strength", DEFAULT_SHAPE_STRENGTH)
    
    @shape_strength.setter
    def shape_strength(self, value: float) -> None:
        self.set("shape_strength", value)
    
    @property
    def color_scheme(self) -> str:
        return self.get("color_scheme", "none")
    
    @color_scheme.setter
    def color_scheme(self, value: str) -> None:
        self.set("color_scheme", value)
    
    @property
    def color_intensity(self) -> float:
        return self.get("color_intensity", DEFAULT_COLOR_INTENSITY)
    
    @color_intensity.setter
    def color_intensity(self, value: float) -> None:
        self.set("color_intensity", value)
    
    @property
    def shader_type(self) -> str:
        return self.get("shader_type", "tensor_field")
    
    @shader_type.setter
    def shader_type(self, value: str) -> None:
        self.set("shader_type", value)
    
    @property
    def time(self) -> float:
        return self.get("time", DEFAULT_TIME)
    
    @time.setter
    def time(self, value: float) -> None:
        self.set("time", value)
    
    @property
    def use_temporal_coherence(self) -> bool:
        return bool(self.get("use_temporal_coherence", False))
    
    @use_temporal_coherence.setter
    def use_temporal_coherence(self, value: bool) -> None:
        self.set("use_temporal_coherence", value)


def load_shader_params(
    custom_path: Optional[str] = None,
    extension_dir: Optional[str] = None
) -> ShaderParams:
    """
    Load shader parameters from a JSON file.
    
    Args:
        custom_path: Optional path to a custom JSON file
        extension_dir: Extension directory for security checks
        
    Returns:
        ShaderParams instance with loaded values
    """
    if extension_dir is None:
        extension_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine file path
    if custom_path:
        # Security check for path traversal
        try:
            resolved_path = os.path.realpath(custom_path)
            extension_real_path = os.path.realpath(extension_dir)
            data_dir_real_path = os.path.realpath(os.path.join(extension_real_path, "data"))
            
            if not resolved_path.lower().endswith('.json'):
                params_file = os.path.join(extension_dir, "data", "shader_params.json")
            else:
                resolved_norm = os.path.normcase(resolved_path)
                extension_norm = os.path.normcase(extension_real_path)
                data_dir_norm = os.path.normcase(data_dir_real_path)
                
                is_inside_extension = os.path.commonpath([resolved_norm, extension_norm]) == extension_norm
                is_in_data = os.path.commonpath([resolved_norm, data_dir_norm]) == data_dir_norm
                
                if is_inside_extension and is_in_data:
                    params_file = resolved_path
                else:
                    params_file = os.path.join(extension_dir, "data", "shader_params.json")
        except (ValueError, OSError):
            params_file = os.path.join(extension_dir, "data", "shader_params.json")
    else:
        # Try root directory first
        params_file = os.path.join(extension_dir, "shader_params.json")
        if not os.path.exists(params_file):
            params_file = os.path.join(extension_dir, "data", "shader_params.json")
    
    # Load from file
    try:
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                raw_params = json.load(f)
                # Validate that JSON is a dict before passing to ShaderParams
                if not isinstance(raw_params, dict):
                    raise ValueError("JSON file must contain a dictionary")
                return ShaderParams(raw_params).validate()
    except (json.JSONDecodeError, IOError, ValueError, AttributeError, TypeError):
        # Handle: invalid JSON, IO errors, non-dict JSON, attribute access on non-dict
        pass
    
    # Return defaults if loading failed
    return ShaderParams().validate()
