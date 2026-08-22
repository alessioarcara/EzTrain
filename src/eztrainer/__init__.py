"""eztrainer: a small, framework-agnostic training-loop library.

The trainer companion to EzConfy: every public class is instantiable from
plain keyword arguments, so it can be built straight from YAML.
"""

from eztrainer.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStopping,
    MonitorCallback,
)
from eztrainer.checkpoint import Checkpointer
from eztrainer.loggers import Logger, NullLogger, RecordingLogger, WandbLogger
from eztrainer.media import Image, Video
from eztrainer.metrics import Metric, MetricCollection
from eztrainer.run import RunInfo, RunType, generate_run_id, resolve_run, run_id_base
from eztrainer.trainer import EpochTrainer, Trainer

__all__ = [
    "Callback",
    "CheckpointCallback",
    "Checkpointer",
    "EarlyStopping",
    "EpochTrainer",
    "Image",
    "Logger",
    "Metric",
    "MetricCollection",
    "MonitorCallback",
    "NullLogger",
    "RecordingLogger",
    "RunInfo",
    "RunType",
    "Trainer",
    "Video",
    "WandbLogger",
    "generate_run_id",
    "resolve_run",
    "run_id_base",
]
