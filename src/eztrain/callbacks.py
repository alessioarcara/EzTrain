"""Callbacks: observe and steer a trainer through named hooks.

``Callback`` is a nominal base class (so config systems like EzConfy can
type a polymorphic ``list[Callback]``), but dispatch is dynamic by hook
name — see ``Trainer.call_hook`` — so callbacks may also implement custom
hooks emitted by project-specific trainers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from loguru import logger as log

from eztrain.checkpoint import Checkpointer

if TYPE_CHECKING:
    from eztrain.trainer import Trainer


class Callback:
    def on_train_start(self, trainer: Trainer) -> None:
        pass

    def on_iteration_end(self, trainer: Trainer, iteration: int) -> None:
        pass

    def on_eval_end(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


class MonitorCallback(Callback):
    """Tracks the best value of ``trainer.history[monitor]``.

    Base for anything that reacts to "the metric improved" (early stopping,
    best-model checkpointing). Subclasses call :meth:`improved`.
    """

    def __init__(self, *, monitor: str, mode: Literal["min", "max"] = "min") -> None:
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None

    def improved(self, trainer: Trainer) -> bool:
        value = trainer.history.get(self.monitor)
        if value is None:
            log.warning("'{}' not found in trainer history; skipping.", self.monitor)
            return False
        try:
            value = float(value)
        except (TypeError, ValueError):
            log.warning(
                "'{}' value {!r} is not a number; skipping.", self.monitor, value
            )
            return False

        if self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        ):
            self.best = value
            return True
        return False


class EarlyStopping(MonitorCallback):
    """Sets ``trainer.should_stop`` after ``patience`` evaluations without
    improvement of ``monitor``."""

    def __init__(
        self,
        *,
        monitor: str,
        mode: Literal["min", "max"] = "min",
        patience: int = 10,
    ) -> None:
        super().__init__(monitor=monitor, mode=mode)
        self.patience = patience
        self.counter = 0

    def on_eval_end(self, trainer: Trainer) -> None:
        if self.improved(trainer):
            self.counter = 0
            return
        self.counter += 1
        log.info(
            "No improvement in '{}' for {}/{} evaluations.",
            self.monitor,
            self.counter,
            self.patience,
        )
        if self.counter >= self.patience:
            log.info("Early stopping triggered.")
            trainer.should_stop = True


class CheckpointCallback(Callback):
    """Checkpoint *schedule*: restore on train start, save every
    ``save_freq`` iterations and once more at the end if needed.

    The *mechanics* (file formats, best/latest policies, what CONTINUE vs
    FORK restores) belong to the injected
    :class:`~eztrain.checkpoint.Checkpointer`.
    """

    def __init__(self, *, checkpointer: Checkpointer, save_freq: int = 1) -> None:
        self.checkpointer = checkpointer
        self.save_freq = save_freq
        self._last_saved: int | None = None

    def on_train_start(self, trainer: Trainer) -> None:
        self.checkpointer.setup(trainer)

    def on_iteration_end(self, trainer: Trainer, iteration: int) -> None:
        if iteration % self.save_freq == 0:
            self.checkpointer.save(trainer, iteration, trainer.history)
            self._last_saved = iteration

    def on_train_end(self, trainer: Trainer) -> None:
        if trainer.iteration > 0 and self._last_saved != trainer.iteration:
            self.checkpointer.save(trainer, trainer.iteration, trainer.history)
            self._last_saved = trainer.iteration
        self.checkpointer.close()
