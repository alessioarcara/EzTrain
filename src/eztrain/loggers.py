"""Logger protocol and built-in implementations.

The trainer owns *when* to log and with which ``step``; loggers only own
*where* the metrics go. ``WandbLogger`` imports wandb lazily so the core
package works without it installed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

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
