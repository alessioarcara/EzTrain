"""Framework-neutral wrappers for rich log payloads.

Metrics and callbacks produce these instead of tracker-specific objects
(``wandb.Image``/``wandb.Video``); each ``Logger`` implementation converts
them to whatever its backend understands, or drops them.
"""

from __future__ import annotations

from typing import Any


class Image:
    """A static image: a matplotlib Figure or an HxWxC array."""

    def __init__(self, data: Any) -> None:
        self.data = data


class Video:
    """A video as an array of frames (e.g. TxCxHxW), plus playback fps."""

    def __init__(self, frames: Any, fps: int = 1) -> None:
        self.frames = frames
        self.fps = fps
