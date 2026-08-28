import sys

from loguru import logger as loguru_logger

from eztrain import (
    ConsoleLogger,
    Logger,
    NullLogger,
    RecordingLogger,
    WandbLogger,
    resolve_run,
)


def test_null_logger_accepts_everything():
    logger = NullLogger()
    logger.start(resolve_run(None, "x"), config={"a": 1})
    logger.log({"loss": 1.0}, step=1)
    logger.log({"loss": 2.0})
    logger.finish()


def test_recording_logger_records():
    logger = RecordingLogger()
    run = resolve_run(None, "x")
    logger.start(run, config={"a": 1})
    logger.log({"loss": 1.0}, step=1)
    logger.log({"loss": 0.5})
    logger.finish()

    assert logger.run is run
    assert logger.config == {"a": 1}
    assert logger.records == [({"loss": 1.0}, 1), ({"loss": 0.5}, None)]
    assert logger.finished


def test_console_logger_formats_scalars_and_skips_the_rest():
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, format="{message}")
    try:
        logger = ConsoleLogger()
        logger.start(resolve_run(None, "x"), config={"a": 1})
        logger.log(
            {"loss": 0.123456, "epoch": 3, "done": True, "plot": object()}, step=2
        )
        logger.log({"batch_loss": 1.0})
        logger.log({"plot": object()}, step=5)  # no scalars -> no output
        logger.finish()
    finally:
        loguru_logger.remove(sink_id)

    assert [m.strip() for m in messages] == [
        "step 2: loss=0.1235, epoch=3, done=True",
        "batch_loss=1.0000",
    ]


def test_builtin_loggers_satisfy_protocol():
    assert isinstance(NullLogger(), Logger)
    assert isinstance(ConsoleLogger(), Logger)
    assert isinstance(RecordingLogger(), Logger)
    assert isinstance(WandbLogger(project="p"), Logger)


def test_wandb_import_is_lazy():
    # constructing the logger must not import wandb; only start/log/finish do
    sys.modules.pop("wandb", None)
    WandbLogger(project="p", entity="e", group="g", job_type="train")
    assert "wandb" not in sys.modules
