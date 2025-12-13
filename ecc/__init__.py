"""Engagement Clip Creator package."""

from importlib import metadata

try:
    __version__ = metadata.version("ecc")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
