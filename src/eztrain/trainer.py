"""The training loop.

:class:`Trainer` is a small template-method class: it owns the *skeleton*
every training run shares (run identity, hooks, periodic evaluation, history,
logging, graceful interruption) and delegates the *content* of one iteration
to a subclass. An iteration can be anything that returns a mapping of
metrics: an epoch over a dataloader (see :class:`EpochTrainer`), an RL
collect->update cycle, a world-model phase — the core never touches tensors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from loguru import logger as log
from tqdm import tqdm

from eztrain.callbacks import Callback
from eztrain.loggers import Logger, NullLogger
from eztrain.metrics import MetricCollection
from eztrain.run import RunInfo, resolve_run


class Trainer:
    """Base trainer: subclass and implement :meth:`train_iteration`.

    Stable surface exposed to callbacks (safe to rely on):

    - ``run``: :class:`~eztrain.run.RunInfo` (run_id, run_type, restore_dir)
    - ``iteration``: current iteration (0 before the loop starts)
    - ``start_iteration``: writable; a restoring callback sets it in
      ``on_train_start`` and the loop resumes from ``start_iteration + 1``
    - ``history``: dict with the latest value of every logged metric
    - ``should_stop``: writable; checked after every iteration
    - ``checkpointables``: what a checkpointer should persist (override it)
    - ``logger`` and ``call_hook``
    """

    def __init__(
        self,
        *,
        max_iterations: int,
        run_name: str = "run",
        resume_from: str | None = None,
        eval_freq: int = 0,
        callbacks: list[Callback] | None = None,
        logger: Logger | None = None,
        config: Mapping[str, Any] | None = None,
        unit: str = "epoch",
    ) -> None:
        self.run: RunInfo = resolve_run(resume_from, run_name)
        self.max_iterations = max_iterations
        self.eval_freq = eval_freq
        self.callbacks: list[Callback] = callbacks or []
        self.logger: Logger = logger or NullLogger()
        self.config = config
        self.unit = unit

        self.start_iteration: int = 0
        self.iteration: int = 0
        self.history: dict[str, Any] = {}
        self.should_stop: bool = False

        if self.run.restore_dir is not None:
            log.info(
                "{} run '{}' (restoring from {})",
                self.run.run_type.name,
                self.run.run_id,
                self.run.restore_dir,
            )

    # --- override points ---------------------------------------------------

    def train_iteration(self, iteration: int) -> Mapping[str, Any]:
        """One unit of training work (an epoch, an RL update, ...).

        Returns the metrics to log for this iteration.
        """
        raise NotImplementedError

    def evaluate(self) -> Mapping[str, Any]:
        """Periodic evaluation, called every ``eval_freq`` iterations."""
        return {}

    def log_step(self) -> int:
        """The ``step`` passed to the logger. Override e.g. with a global
        environment-step counter in RL trainers."""
        return self.iteration

    @property
    def checkpointables(self) -> Mapping[str, Any]:
        """What a :class:`~eztrain.checkpoint.Checkpointer` should persist
        (e.g. model/optimizer objects, keyed by name). Override it."""
        return {}

    # --- template loop -----------------------------------------------------

    def fit(self) -> None:
        self.logger.start(self.run, self.config)
        try:
            self.call_hook("on_train_start")
            for iteration in tqdm(
                range(self.start_iteration + 1, self.max_iterations + 1),
                desc=self.unit,
                colour="green",
            ):
                self.iteration = iteration

                logs = dict(self.train_iteration(iteration))
                self.history.update(logs)

                if self.eval_freq and iteration % self.eval_freq == 0:
                    eval_logs = dict(self.evaluate())
                    logs.update(eval_logs)
                    self.history.update(eval_logs)
                    self.call_hook("on_eval_end")

                self.logger.log(logs, step=self.log_step())
                self.call_hook("on_iteration_end", iteration=iteration)

                if self.should_stop:
                    log.info("Stop requested, ending training early.")
                    break
        except KeyboardInterrupt:
            log.warning("Training interrupted by user.")
        finally:
            self.call_hook("on_train_end")
            self.logger.finish()

    def call_hook(self, hook: str, **kwargs: Any) -> None:
        """Invoke ``hook`` on every callback that defines it.

        Dispatch is by name, so trainers may introduce custom hooks (e.g.
        ``on_rollout_end``) that only some callbacks implement.
        """
        for callback in self.callbacks:
            fn = getattr(callback, hook, None)
            if fn is not None:
                fn(self, **kwargs)


class EpochTrainer(Trainer):
    """Supervised-style trainer: one iteration = one pass over ``train_loader``.

    Still framework-agnostic: batches are opaque and only handled by your
    ``train_step``/``eval_step``. Per-step metrics are averaged over the
    epoch and prefixed ``train/`` and ``val/``; anything your metrics
    ``compute``/``plot`` is added on evaluation. Your ``eval_step`` is
    responsible for calling ``self.metrics.update(...)`` with whatever
    arguments your metrics expect. For per-batch logging, call
    ``self.logger.log({...})`` inside ``train_step`` (no ``step``).
    """

    def __init__(
        self,
        *,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        test_loader: Iterable[Any] | None = None,
        metrics: MetricCollection | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics or MetricCollection()

    # --- override points ---------------------------------------------------

    def train_step(self, batch: Any) -> Mapping[str, float]:
        """One optimization step. Returns per-step metrics (e.g. losses)."""
        raise NotImplementedError

    def eval_step(self, batch: Any) -> Mapping[str, float]:
        """One evaluation step. Update ``self.metrics`` here and return
        per-step metrics (e.g. losses) to be averaged."""
        raise NotImplementedError

    # --- Trainer implementation --------------------------------------------

    def train_iteration(self, iteration: int) -> Mapping[str, Any]:
        step_logs: defaultdict[str, float] = defaultdict(float)
        num_steps = 0
        for batch in tqdm(
            self.train_loader,
            desc=f"{self.unit} {iteration}",
            leave=False,
            colour="blue",
        ):
            for key, value in self.train_step(batch).items():
                step_logs[key] += float(value)
            num_steps += 1

        if num_steps == 0:
            return {}
        return {f"train/{k}": v / num_steps for k, v in step_logs.items()}

    def evaluate(self) -> Mapping[str, Any]:
        if self.val_loader is None:
            return {}
        return self.evaluate_split(self.val_loader, "val")

    def evaluate_split(self, loader: Iterable[Any], prefix: str) -> dict[str, Any]:
        """Run ``eval_step`` over ``loader``; average step metrics and merge
        ``metrics.compute()``/``metrics.plot()``, all under ``prefix/``."""
        self.metrics.reset()

        step_logs: defaultdict[str, float] = defaultdict(float)
        num_steps = 0
        for batch in tqdm(
            loader, desc=f"evaluating {prefix}", leave=False, colour="red"
        ):
            for key, value in self.eval_step(batch).items():
                step_logs[key] += float(value)
            num_steps += 1

        results: dict[str, Any] = {}
        if num_steps > 0:
            results.update(
                {f"{prefix}/{k}": v / num_steps for k, v in step_logs.items()}
            )
        results.update({f"{prefix}/{k}": v for k, v in self.metrics.compute().items()})
        results.update({f"{prefix}/{k}": v for k, v in self.metrics.plot().items()})
        return results
