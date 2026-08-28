"""Every public class must be buildable from plain keyword arguments, so an
EzConfy YAML (`_target_type_: eztrain.callbacks:EarlyStopping` +
`_init_args_`) can instantiate it directly."""

import importlib

from eztrain import (
    CheckpointCallback,
    EarlyStopping,
    MetricCollection,
    MonitorCallback,
    NullLogger,
    RecordingLogger,
    WandbLogger,
)
from tests.test_callbacks import FakeCheckpointer


def test_public_classes_instantiable_from_simple_kwargs():
    MonitorCallback(monitor="val/loss", mode="min")
    EarlyStopping(monitor="val/loss", mode="max", patience=5)
    CheckpointCallback(checkpointer=FakeCheckpointer(), save_freq=3)
    MetricCollection(metrics=[])
    NullLogger()
    RecordingLogger()
    WandbLogger(project="p", entity="e", group="g", job_type="train")


def test_base_classes_importable_from_stable_module_paths():
    # these dotted paths are what EzConfy `types:` aliases will reference
    for path in [
        "eztrain.callbacks:Callback",
        "eztrain.metrics:Metric",
        "eztrain.loggers:Logger",
        "eztrain.checkpoint:Checkpointer",
        "eztrain.trainer:Trainer",
        "eztrain.trainer:EpochTrainer",
    ]:
        module_name, _, attr = path.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), path
