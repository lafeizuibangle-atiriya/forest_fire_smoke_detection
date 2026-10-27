"""Project-wide exception hierarchy."""

from __future__ import annotations


class FFSDError(Exception):
    """Base exception for this project."""


class ConfigError(FFSDError):
    """Raised when a config file is missing or invalid."""


class DataError(FFSDError):
    """Raised when dataset layout or labels are invalid."""


class ModelError(FFSDError):
    """Raised when a model cannot be loaded or exported."""
