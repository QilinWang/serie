"""Package and protocol version stamps."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PROTOCOL_VERSION = "serie.protocol.v1"

try:
    __version__ = version("serie")
except PackageNotFoundError:
    __version__ = "0.1.1+local"
