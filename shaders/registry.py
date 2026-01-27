"""
Shader generator registry for plugin-based registration.

This module provides a centralized registry for shader generators,
replacing the monkey-patching approach used in __init__.py.
"""

from typing import Dict, Type, Optional, List, Any
import logging

from .base import BaseNoiseGenerator

logger = logging.getLogger(__name__)


class ShaderRegistry:
    """
    Centralized registry for shader noise generators.
    
    Provides a clean plugin-based approach to registering and
    accessing shader generators.
    """
    
    _generators: Dict[str, Type[BaseNoiseGenerator]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        generator_class: Type[BaseNoiseGenerator],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a shader generator.
        
        Args:
            name: Name of the shader type (e.g., "domain_warp")
            generator_class: Class implementing BaseNoiseGenerator
            metadata: Optional metadata about the generator
        """
        if name in cls._generators:
            logger.warning(f"Overwriting existing shader generator: {name}")
        
        cls._generators[name] = generator_class
        cls._metadata[name] = metadata or {}
        logger.debug(f"Registered shader generator: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseNoiseGenerator]]:
        """
        Get a registered shader generator by name.
        
        Args:
            name: Name of the shader type
            
        Returns:
            Generator class or None if not found
        """
        return cls._generators.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered generator.
        
        Args:
            name: Name of the shader type
            
        Returns:
            Metadata dictionary (empty if not found)
        """
        return cls._metadata.get(name, {})
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available shader generator names.
        
        Returns:
            List of registered shader names
        """
        return list(cls._generators.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a shader generator is registered.
        
        Args:
            name: Name of the shader type
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._generators
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a shader generator.
        
        Args:
            name: Name of the shader type
            
        Returns:
            True if was registered, False if not found
        """
        if name in cls._generators:
            del cls._generators[name]
            cls._metadata.pop(name, None)
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered generators.
        """
        cls._generators.clear()
        cls._metadata.clear()


# Global registry instance for convenience functions
_default_registry = ShaderRegistry

# Public alias for the registry (for external imports)
shader_registry = ShaderRegistry


def register_shader(
    name: str,
    generator_class: Type[BaseNoiseGenerator],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a shader generator to the global registry.
    
    Args:
        name: Name of the shader type
        generator_class: Class implementing BaseNoiseGenerator
        metadata: Optional metadata about the generator
    """
    _default_registry.register(name, generator_class, metadata)


def get_shader(name: str) -> Optional[Type[BaseNoiseGenerator]]:
    """
    Get a shader generator from the global registry.
    
    Args:
        name: Name of the shader type
        
    Returns:
        Generator class or None if not found
    """
    return _default_registry.get(name)


def list_shaders() -> List[str]:
    """
    List all available shader generators.
    
    Returns:
        List of registered shader names
    """
    return _default_registry.list_available()


# Decorator for registering shader generators
def shader_generator(name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to register a shader generator class.
    
    Usage:
        @shader_generator("my_shader")
        class MyShaderGenerator(BaseNoiseGenerator):
            ...
    
    Args:
        name: Name to register the shader under
        metadata: Optional metadata
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseNoiseGenerator]) -> Type[BaseNoiseGenerator]:
        register_shader(name, cls, metadata)
        return cls
    return decorator
