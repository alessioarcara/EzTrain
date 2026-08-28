from collections.abc import Mapping
from typing import Any

from eztrain import (
    CheckpointCallback,
    EarlyStopping,
    MonitorCallback,
    Trainer,
)


class ScriptedTrainer(Trainer):
    """Logs a scripted series of values for a 'val/loss' metric."""

    def __init__(self, values: list[float], **kwargs: Any) -> None:
        super().__init__(max_iterations=len(values), eval_freq=1, **kwargs)
        self.values = values

    def train_iteration(self, iteration: int) -> Mapping[str, Any]:
        return {}

    def evaluate(self) -> Mapping[str, Any]:
        return {"val/loss": self.values[self.iteration - 1]}


def test_monitor_callback_tracks_best_min_and_max():
    monitor = MonitorCallback(monitor="val/loss", mode="min")
    trainer = ScriptedTrainer([3.0, 2.0, 2.5])

    improvements = []
    trainer.history["val/loss"] = 3.0
    improvements.append(monitor.improved(trainer))
    trainer.history["val/loss"] = 2.0
    improvements.append(monitor.improved(trainer))
    trainer.history["val/loss"] = 2.5
    improvements.append(monitor.improved(trainer))

    assert improvements == [True, True, False]
    assert monitor.best == 2.0

    monitor_max = MonitorCallback(monitor="score", mode="max")
    trainer.history["score"] = 1.0
    assert monitor_max.improved(trainer)
    trainer.history["score"] = 0.5
    assert not monitor_max.improved(trainer)
    assert monitor_max.best == 1.0


def test_monitor_callback_missing_or_bad_key_is_not_improvement():
    monitor = MonitorCallback(monitor="nope")
    trainer = ScriptedTrainer([1.0])
    assert not monitor.improved(trainer)
    trainer.history["nope"] = "not-a-number"
    assert not monitor.improved(trainer)


def test_early_stopping_stops_after_patience():
    stopper = EarlyStopping(monitor="val/loss", mode="min", patience=2)
    trainer = ScriptedTrainer([5.0, 4.0, 4.5, 4.4, 4.3, 1.0], callbacks=[stopper])
    trainer.fit()
    # improvements at 1,2 then no improvement at 3,4 -> stop; 5,6 never run
    assert trainer.iteration == 4
    assert trainer.should_stop


def test_early_stopping_counter_resets_on_improvement():
    stopper = EarlyStopping(monitor="val/loss", mode="min", patience=2)
    trainer = ScriptedTrainer([5.0, 4.9, 4.0, 4.5, 3.0, 2.9], callbacks=[stopper])
    trainer.fit()
    assert trainer.iteration == 6  # never accumulates `patience` misses in a row
    assert not trainer.should_stop


class FakeCheckpointer:
    def __init__(self) -> None:
        self.setup_called = False
        self.saves: list[int] = []
        self.closed = False

    def setup(self, trainer: Trainer) -> None:
        self.setup_called = True

    def save(
        self, trainer: Trainer, iteration: int, metrics: Mapping[str, Any]
    ) -> None:
        self.saves.append(iteration)

    def close(self) -> None:
        self.closed = True


def test_checkpoint_callback_schedule():
    fake = FakeCheckpointer()
    callback = CheckpointCallback(checkpointer=fake, save_freq=2)
    trainer = ScriptedTrainer([1.0] * 5, callbacks=[callback])
    trainer.fit()

    assert fake.setup_called
    # every 2 iterations, plus the final save for iteration 5
    assert fake.saves == [2, 4, 5]
    assert fake.closed


def test_checkpoint_callback_no_double_save_on_aligned_end():
    fake = FakeCheckpointer()
    callback = CheckpointCallback(checkpointer=fake, save_freq=2)
    trainer = ScriptedTrainer([1.0] * 4, callbacks=[callback])
    trainer.fit()
    assert fake.saves == [2, 4]


def test_checkpoint_callback_no_save_when_loop_never_ran():
    fake = FakeCheckpointer()
    callback = CheckpointCallback(checkpointer=fake, save_freq=1)
    trainer = ScriptedTrainer([], callbacks=[callback])
    trainer.fit()
    assert fake.saves == []
    assert fake.closed
