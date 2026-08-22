"""Metric protocol and composite collection.

A metric accumulates state across ``update`` calls and produces named scalars
in ``compute``. The trainer only ever calls ``reset``/``compute``/``plot``;
``update`` is called by *your* ``eval_step`` with whatever signature your
metrics need, so the protocol does not fix one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from eztrainer.media import Image, Video


@runtime_checkable
class Metric(Protocol):
    def reset(self) -> None: ...

    def update(self, *args: Any, **kwargs: Any) -> None: ...

    def compute(self) -> Mapping[str, float]: ...

    # Optionally, a metric may also define:
    #   def plot(self) -> Mapping[str, Image | Video]: ...


class MetricCollection:
    """Fans out to a list of metrics and merges their results."""

    def __init__(self, metrics: Sequence[Metric] | None = None) -> None:
        self.metrics: list[Metric] = list(metrics or [])

    def __iter__(self) -> Iterator[Metric]:
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for metric in self.metrics:
            metric.update(*args, **kwargs)

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results

    def plot(self) -> dict[str, Image | Video]:
        visuals: dict[str, Image | Video] = {}
        for metric in self.metrics:
            plot = getattr(metric, "plot", None)
            if plot is not None:
                visuals.update(plot())
        return visuals
