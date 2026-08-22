from collections.abc import Mapping
from typing import Any

from eztrainer import EpochTrainer, Image, MetricCollection, RecordingLogger


class Accuracy:
    """Toy metric over (pred, target) pairs."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, pred: int, target: int) -> None:
        self.correct += int(pred == target)
        self.total += 1

    def compute(self) -> Mapping[str, float]:
        return {"accuracy": self.correct / max(self.total, 1)}


class PlottingMetric(Accuracy):
    def plot(self) -> Mapping[str, Image]:
        return {"scatter": Image(data="fake-figure")}


class ToyEpochTrainer(EpochTrainer):
    """'Learns' a threshold classifier over (x, label) samples."""

    def train_step(self, batch: Any) -> Mapping[str, float]:
        x, label = batch
        return {"loss": abs(x - label)}

    def eval_step(self, batch: Any) -> Mapping[str, float]:
        x, label = batch
        pred = int(x > 0.5)
        self.metrics.update(pred, label)
        return {"loss": abs(x - label)}


def make_trainer(**kwargs: Any) -> ToyEpochTrainer:
    train_data = [(0.0, 0), (1.0, 1)]
    val_data = [(0.2, 0), (0.9, 1), (0.4, 1)]  # 2 of 3 correct
    defaults: dict[str, Any] = dict(
        train_loader=train_data,
        val_loader=val_data,
        metrics=MetricCollection([PlottingMetric()]),
        max_iterations=1,
        eval_freq=1,
    )
    defaults.update(kwargs)
    return ToyEpochTrainer(**defaults)


def test_train_metrics_are_averaged_and_prefixed():
    logger = RecordingLogger()
    trainer = make_trainer(logger=logger)
    trainer.fit()

    logged, step = logger.records[0]
    assert step == 1
    assert logged["train/loss"] == 0.0  # (|0-0| + |1-1|) / 2


def test_eval_merges_losses_metrics_and_plots():
    logger = RecordingLogger()
    trainer = make_trainer(logger=logger)
    trainer.fit()

    logged, _ = logger.records[0]
    assert logged["val/accuracy"] == 2 / 3
    assert isinstance(logged["val/scatter"], Image)
    assert abs(logged["val/loss"] - (0.2 + 0.1 + 0.6) / 3) < 1e-9


def test_metrics_reset_between_evaluations():
    trainer = make_trainer(max_iterations=2, eval_freq=1)
    trainer.fit()
    # if reset didn't happen the second accuracy would average over 6 samples
    assert trainer.history["val/accuracy"] == 2 / 3


def test_no_val_loader_evaluates_to_nothing():
    trainer = make_trainer(val_loader=None)
    trainer.fit()
    assert "val/accuracy" not in trainer.history


def test_evaluate_split_reusable_for_other_loaders():
    trainer = make_trainer()
    results = trainer.evaluate_split([(0.9, 1)], "test")
    assert results["test/accuracy"] == 1.0
