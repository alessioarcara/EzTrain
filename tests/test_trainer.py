from collections.abc import Mapping
from typing import Any

import pytest

from eztrain import Callback, RecordingLogger, Trainer


class ToyTrainer(Trainer):
    """Counts iterations; logs a metric equal to the iteration number."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trained: list[int] = []
        self.evaluated = 0

    def train_iteration(self, iteration: int) -> Mapping[str, Any]:
        self.trained.append(iteration)
        return {"train/loss": float(iteration)}

    def evaluate(self) -> Mapping[str, Any]:
        self.evaluated += 1
        return {"val/loss": float(self.iteration) * 10}


class HookRecorder(Callback):
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.events.append(("train_start", trainer.iteration))

    def on_iteration_end(self, trainer: Trainer, iteration: int) -> None:
        self.events.append(("iteration_end", iteration))

    def on_eval_end(self, trainer: Trainer) -> None:
        self.events.append(("eval_end", trainer.iteration))

    def on_train_end(self, trainer: Trainer) -> None:
        self.events.append(("train_end", trainer.iteration))


def test_loop_runs_all_iterations_and_logs():
    logger = RecordingLogger()
    trainer = ToyTrainer(max_iterations=3, logger=logger)
    trainer.fit()

    assert trainer.trained == [1, 2, 3]
    assert [step for _, step in logger.records] == [1, 2, 3]
    assert logger.records[0][0] == {"train/loss": 1.0}
    assert logger.finished
    assert logger.run is trainer.run


def test_hook_order():
    recorder = HookRecorder()
    trainer = ToyTrainer(max_iterations=2, eval_freq=2, callbacks=[recorder])
    trainer.fit()

    assert recorder.events == [
        ("train_start", 0),
        ("iteration_end", 1),
        ("eval_end", 2),
        ("iteration_end", 2),
        ("train_end", 2),
    ]


def test_eval_freq():
    trainer = ToyTrainer(max_iterations=6, eval_freq=2)
    trainer.fit()
    assert trainer.evaluated == 3

    trainer = ToyTrainer(max_iterations=6)  # eval_freq=0 -> never
    trainer.fit()
    assert trainer.evaluated == 0


def test_start_iteration_set_by_callback_resumes_loop():
    class Restore(Callback):
        def on_train_start(self, trainer: Trainer) -> None:
            trainer.start_iteration = 3

    trainer = ToyTrainer(max_iterations=5, callbacks=[Restore()])
    trainer.fit()
    assert trainer.trained == [4, 5]


def test_should_stop_breaks_loop():
    class StopAtTwo(Callback):
        def on_iteration_end(self, trainer: Trainer, iteration: int) -> None:
            if iteration == 2:
                trainer.should_stop = True

    logger = RecordingLogger()
    trainer = ToyTrainer(max_iterations=10, callbacks=[StopAtTwo()], logger=logger)
    trainer.fit()
    assert trainer.trained == [1, 2]
    assert logger.finished


def test_keyboard_interrupt_still_runs_teardown():
    class InterruptingTrainer(ToyTrainer):
        def train_iteration(self, iteration: int) -> Mapping[str, Any]:
            if iteration == 2:
                raise KeyboardInterrupt
            return super().train_iteration(iteration)

    recorder = HookRecorder()
    logger = RecordingLogger()
    trainer = InterruptingTrainer(
        max_iterations=10, callbacks=[recorder], logger=logger
    )
    trainer.fit()  # must not raise

    assert trainer.trained == [1]
    assert recorder.events[-1][0] == "train_end"
    assert logger.finished


def test_other_exceptions_propagate_but_teardown_runs():
    class FailingTrainer(ToyTrainer):
        def train_iteration(self, iteration: int) -> Mapping[str, Any]:
            raise RuntimeError("boom")

    logger = RecordingLogger()
    trainer = FailingTrainer(max_iterations=3, logger=logger)
    with pytest.raises(RuntimeError):
        trainer.fit()
    assert logger.finished


def test_history_is_fresh_inside_hooks():
    seen: dict[str, Any] = {}

    class Peek(Callback):
        def on_eval_end(self, trainer: Trainer) -> None:
            seen["at_eval"] = dict(trainer.history)

        def on_iteration_end(self, trainer: Trainer, iteration: int) -> None:
            seen["at_iter"] = dict(trainer.history)

    trainer = ToyTrainer(max_iterations=2, eval_freq=2, callbacks=[Peek()])
    trainer.fit()

    # on_eval_end sees the eval metrics of the current iteration
    assert seen["at_eval"]["val/loss"] == 20.0
    # on_iteration_end sees train and eval metrics of the current iteration
    assert seen["at_iter"]["train/loss"] == 2.0
    assert seen["at_iter"]["val/loss"] == 20.0


def test_custom_hooks_dispatch_dynamically():
    class RolloutAware(Callback):
        def __init__(self) -> None:
            self.rollouts: list[int] = []

        def on_rollout_end(self, trainer: Trainer, num_steps: int) -> None:
            self.rollouts.append(num_steps)

    class RLTrainer(ToyTrainer):
        def train_iteration(self, iteration: int) -> Mapping[str, Any]:
            self.call_hook("on_rollout_end", num_steps=iteration * 100)
            return {}

    aware = RolloutAware()
    plain = Callback()  # does not implement the custom hook: must be skipped
    trainer = RLTrainer(max_iterations=2, callbacks=[aware, plain])
    trainer.fit()
    assert aware.rollouts == [100, 200]


def test_log_step_override():
    class GlobalStepTrainer(ToyTrainer):
        batch_size = 32

        def log_step(self) -> int:
            return self.iteration * self.batch_size

    logger = RecordingLogger()
    trainer = GlobalStepTrainer(max_iterations=2, logger=logger)
    trainer.fit()
    assert [step for _, step in logger.records] == [32, 64]


def test_config_is_passed_to_logger():
    logger = RecordingLogger()
    trainer = ToyTrainer(max_iterations=1, logger=logger, config={"lr": 1e-3})
    trainer.fit()
    assert logger.config == {"lr": 1e-3}
