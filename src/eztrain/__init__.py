"""eztrain: a small, framework-agnostic training-loop library.

The trainer companion to EzConfy: every public class is instantiable from
plain keyword arguments, so it can be built straight from YAML.
"""

from importlib.metadata import PackageNotFoundError, version

from eztrain.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStopping,
    MonitorCallback,
)
from eztrain.checkpoint import Checkpointer
from eztrain.loggers import Logger, NullLogger, RecordingLogger, WandbLogger
from eztrain.media import Image, Video
from eztrain.metrics import Metric, MetricCollection
from eztrain.run import RunInfo, RunType, generate_run_id, resolve_run, run_id_base
from eztrain.trainer import EpochTrainer, Trainer

try:
    __version__ = version("eztrain")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

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
