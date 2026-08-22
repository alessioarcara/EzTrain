"""Checkpointer protocol.

The schedule (when to save/restore) lives in
:class:`~eztrainer.callbacks.CheckpointCallback`; the mechanics (what a
checkpoint physically is) live in framework-specific implementations
(torch/orbax extras). A trainer advertises what to persist through its
``checkpointables`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from eztrainer.trainer import Trainer


@runtime_checkable
class Checkpointer(Protocol):
    def setup(self, trainer: Trainer) -> None:
        """Prepare storage for ``trainer.run.run_id`` and, if
        ``trainer.run.restore_dir`` is set, restore according to
        ``trainer.run.run_type`` (CONTINUE: full state, set
        ``trainer.start_iteration``; FORK: weights only)."""
        ...

    def save(
        self, trainer: Trainer, iteration: int, metrics: Mapping[str, Any]
    ) -> None: ...

    def close(self) -> None: ...
