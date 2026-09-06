"""Logger protocol and built-in implementations."""

from __future__ import annotations

import pprint
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger as log

from eztrain.media import Image, Video
from eztrain.run import RunInfo


@runtime_checkable
class Logger(Protocol):
    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None: ...

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...

    def finish(self) -> None: ...


class NullLogger:
    """Discards everything. The default when no logger is given."""

    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None:
        pass

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


class ConsoleLogger:
    """Prints scalar metrics via loguru; non-scalar values are skipped.

    Note: loguru and the trainer's tqdm bars both write to stderr, so each
    log line makes the bar redraw below it — fine at one line per iteration,
    noisy if you log per-batch.
    """

    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None:
        pass

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        scalars = ", ".join(
            f"{k}={self._format(v)}"
            for k, v in metrics.items()
            if isinstance(v, int | float)
        )
        if not scalars:
            return
        if step is None:
            log.info("{}", scalars)
        else:
            log.info("step {}: {}", step, scalars)

    def finish(self) -> None:
        pass

    @staticmethod
    def _format(value: int | float) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)


class RecordingLogger:
    """Keeps every call in memory. Useful in tests and quick scripts."""

    def __init__(self) -> None:
        self.run: RunInfo | None = None
        self.config: Mapping[str, Any] | None = None
        self.records: list[tuple[dict[str, Any], int | None]] = []
        self.finished: bool = False

    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None:
        self.run = run
        self.config = config

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self.records.append((dict(metrics), step))

    def finish(self) -> None:
        self.finished = True


class WandbLogger:
    """Weights & Biases logger. Requires the ``eztrain[wandb]`` extra.

    Maps :class:`~eztrain.run.RunInfo` onto ``wandb.init``: the run id is the
    wandb id (so CONTINUE runs resume the same wandb run, ``resume="must"``)
    and the base name is the display name.
    """

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.group = group
        self.job_type = job_type

    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None:
        import wandb

        wandb.init(
            entity=self.entity,
            project=self.project,
            group=self.group,
            job_type=self.job_type,
            name=run.name,
            id=run.run_id,
            resume=run.resume,
            config=dict(config) if config is not None else None,
        )

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        import wandb

        wandb.log({k: self._convert(v) for k, v in metrics.items()}, step=step)

    def finish(self) -> None:
        import wandb

        wandb.finish()

    @staticmethod
    def _convert(value: Any) -> Any:
        import wandb

        if isinstance(value, Image):
            converted = wandb.Image(value.data)
            # close matplotlib figures without importing matplotlib ourselves
            plt = sys.modules.get("matplotlib.pyplot")
            if plt is not None and hasattr(value.data, "savefig"):
                plt.close(value.data)
            return converted
        if isinstance(value, Video):
            return wandb.Video(value.frames, fps=value.fps, format="mp4")
        return value


class TensorBoardLogger:
    """TensorBoard logger. Requires the ``eztrain[tensorboard]`` extra.

    Uses torch's ``SummaryWriter`` when torch is installed, otherwise falls
    back to the API-compatible ``tensorboardX`` one.

    Routes each value in the metrics dict by type, wandb-style:

    - nested ``Mapping``            -> recursed, keys joined with ``/``
    - :class:`~eztrain.media.Image` -> figure (matplotlib) or HxWxC image
    - :class:`~eztrain.media.Video` -> video (TxCxHxW frames; needs moviepy)
    - anything ``float()`` accepts  -> scalar (numbers, 0-d tensors)
    - everything else               -> dropped

    Events go to ``log_dir / run.run_id``, so CONTINUE runs append to the
    same directory. ``config`` is stored once under the "config" text tab.
    """

    def __init__(self, log_dir: str | Path = "runs") -> None:
        self.log_dir = Path(log_dir)
        self.writer: Any = None

    def start(self, run: RunInfo, config: Mapping[str, Any] | None = None) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(self.log_dir / run.run_id)
        if config is not None:
            self.writer.add_text("config", f"```\n{pprint.pformat(dict(config))}\n```")

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        for tag, value in metrics.items():
            self._log(tag, value, step)

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def _log(self, tag: str, value: Any, step: int | None) -> None:
        w = self.writer
        assert w is not None, "call start() first"

        if isinstance(value, Mapping):
            for k, v in value.items():
                self._log(f"{tag}/{k}", v, step)
        elif isinstance(value, Image):
            if hasattr(value.data, "savefig"):
                w.add_figure(tag, value.data, step, close=True)
            else:
                w.add_image(tag, value.data, step, dataformats="HWC")
        elif isinstance(value, Video):
            # add_video wants NxTxCxHxW; media.Video carries a single clip
            w.add_video(tag, value.frames[None], step, fps=value.fps)
        else:
            try:
                w.add_scalar(tag, float(value), step)
            except (TypeError, ValueError):
                pass  # not numeric and not a known media type
