"""
Logging configuration for ComfyUI-ShaderNoiseKSampler.

This module provides centralized logging setup for the package.
"""

import logging
import sys
from typing import Optional

# Package-wide logger
PACKAGE_NAME = "comfyui_shader_noise"

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "[%(name)s] %(levelname)s: %(message)s"


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None
) -> logging.Logger:
    """
    Set up logging for the package.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Log format string (default: SIMPLE_FORMAT)
        handler: Custom handler (default: StreamHandler to stdout)
        
    Returns:
        Root logger for the package
    """
    if format_string is None:
        format_string = SIMPLE_FORMAT
    
    # Get the package logger
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
    
    # Set format
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    
    # Add handler
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (will be prefixed with package name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"{PACKAGE_NAME}.{name}")


# Convenience functions for common log levels
def set_debug_logging() -> logging.Logger:
    """Enable debug-level logging."""
    return setup_logging(level=logging.DEBUG)


def set_info_logging() -> logging.Logger:
    """Enable info-level logging."""
    return setup_logging(level=logging.INFO)


def set_warning_logging() -> logging.Logger:
    """Enable warning-level logging (quieter)."""
    return setup_logging(level=logging.WARNING)


def set_quiet_logging() -> logging.Logger:
    """Enable error-only logging (quietest)."""
    return setup_logging(level=logging.ERROR)
