# eztrainer

A small, framework-agnostic training-loop library. The trainer companion to
[EzConfy](https://github.com/alessioarcara/EzConfy).

Every ML project rewrites the same trainer: a loop over epochs or updates,
periodic evaluation, callbacks, early stopping, checkpoint scheduling, metric
logging, graceful `Ctrl+C`. **eztrainer** extracts exactly that skeleton and
nothing else:

- **Core never imports torch or jax.** It abstracts the *iteration*, not the
  tensor: an iteration is anything that returns a `Mapping[str, Any]` of
  metrics — an epoch over a dataloader, an RL collect→update cycle, a
  world-model phase.
- **Run lifecycle built in.** A run id (`<name>_<timestamp>`) is shared
  between checkpoints and the experiment tracker, with three resume modes:
  **FRESH** (from scratch), **CONTINUE** (same run: weights + optimizer +
  tracker), **FORK** (new run seeded with old weights).
- **Composition over inheritance** where it matters: `Logger`, `Metric` and
  `Checkpointer` are Protocols you inject; `Callback` hooks are dispatched
  dynamically by name so trainers can invent their own hooks.
- **No hidden defaults.** No optimizer/scheduler construction, no concrete
  metrics, no config system (that's EzConfy's job), no distributed magic.

Core dependencies: `tqdm`, `loguru`. That's it.

## Install

```bash
uv add eztrainer            # core
uv add "eztrainer[wandb]"   # + Weights & Biases logger
```

## Supervised (epoch-based)

```python
from eztrainer import EpochTrainer, EarlyStopping, MetricCollection, WandbLogger

class MyTrainer(EpochTrainer):
    def __init__(self, *, model, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.model, self.optimizer = model, optimizer

    def train_step(self, batch):
        loss = ...                      # your forward/backward/step
        return {"loss": loss.item()}    # averaged over the epoch -> "train/loss"

    def eval_step(self, batch):
        preds, loss = ...
        self.metrics.update(preds, batch.y)   # your metrics, your signature
        return {"loss": loss.item()}          # -> "val/loss"

trainer = MyTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    metrics=MetricCollection([MyAccuracy()]),
    max_iterations=100,                 # epochs
    eval_freq=1,
    callbacks=[EarlyStopping(monitor="val/loss", patience=10)],
    logger=WandbLogger(project="my-project"),
    run_name="baseline",
)
trainer.fit()
```

## RL (update-based)

Subclass `Trainer` directly — one iteration is one update:

```python
from eztrainer import Trainer

class PPOTrainer(Trainer):
    def __init__(self, *, env, agent, num_steps, **kwargs):
        super().__init__(unit="update", **kwargs)
        self.env, self.agent, self.num_steps = env, agent, num_steps
        self.obs = env.reset()

    def train_iteration(self, update):
        segment, self.obs = collect_rollouts(self.env, self.agent, self.num_steps, self.obs)
        advantages, returns = compute_gae(segment)
        return self.agent.learn_from(segment, advantages, returns)

    def evaluate(self):
        return {"eval/reward": evaluate(self.agent)}

    def log_step(self):                       # log by env steps, not updates
        return self.iteration * self.num_steps * self.env.num_envs
```

## Resume and fork

```python
Trainer(run_name="exp-1")                                          # FRESH
Trainer(run_name="exp-1", resume_from="exp-1_20260530_051406")     # CONTINUE
Trainer(run_name="exp-2", resume_from="exp-1_20260530_051406")     # FORK
```

`trainer.run` carries `run_id`, `run_type` and `restore_dir`. A
`CheckpointCallback` restores in `on_train_start` (setting
`trainer.start_iteration`) and saves on schedule; the checkpoint *mechanics*
live in a `Checkpointer` implementation you inject (torch/orbax
implementations ship as extras — coming next). `WandbLogger` reuses the run
id, so a CONTINUE run resumes the same wandb run.

## Callbacks

```python
from eztrainer import Callback

class MyCallback(Callback):
    def on_train_start(self, trainer): ...
    def on_iteration_end(self, trainer, iteration): ...
    def on_eval_end(self, trainer): ...          # trainer.history has fresh metrics
    def on_train_end(self, trainer): ...         # always runs, even on Ctrl+C
```

The stable surface callbacks can rely on: `trainer.run`, `trainer.iteration`,
`trainer.start_iteration` (writable), `trainer.history`,
`trainer.should_stop` (writable), `trainer.checkpointables`,
`trainer.logger`, `trainer.call_hook`. Custom hooks compose freely:
`self.call_hook("on_rollout_end", num_steps=n)` inside your trainer reaches
any callback that defines it.

## With EzConfy

Every public class takes plain keyword arguments, so it can be instantiated
straight from YAML:

```yaml
# schema.yaml
types:
  Callback: eztrainer.callbacks:Callback
  Logger: eztrainer.loggers:Logger
schema:
  trainer:
    callbacks: list[Callback]
    logger: Logger
```

```yaml
# config.yaml
trainer:
  logger:
    _target_type_: eztrainer.loggers:WandbLogger
    _init_args_: { project: my-project, entity: me }
  callbacks:
    - _target_type_: eztrainer.callbacks:EarlyStopping
      _init_args_: { monitor: val/loss, mode: min, patience: 10 }
```

## What's deliberately *not* here

Concrete train steps, models, losses, optimizers/schedulers (inject your
own — a library default becomes a cage), metrics implementations, config
loading, distributed training. Keep those in your project; eztrainer only
owns the loop around them.
