import sys
import types

from loguru import logger as loguru_logger

from eztrain import (
    ConsoleLogger,
    Image,
    Logger,
    NullLogger,
    RecordingLogger,
    TensorBoardLogger,
    Video,
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
    assert isinstance(TensorBoardLogger(), Logger)


def test_wandb_import_is_lazy():
    # constructing the logger must not import wandb; only start/log/finish do
    sys.modules.pop("wandb", None)
    WandbLogger(project="p", entity="e", group="g", job_type="train")
    assert "wandb" not in sys.modules


def test_tensorboard_import_is_lazy():
    sys.modules.pop("torch.utils.tensorboard", None)
    sys.modules.pop("tensorboardX", None)
    TensorBoardLogger(log_dir="runs")
    assert "torch.utils.tensorboard" not in sys.modules
    assert "tensorboardX" not in sys.modules


class _FakeWriter:
    """Records every add_* call; stands in for torch's SummaryWriter."""

    def __init__(self, log_dir):
        self.calls = [("init", str(log_dir))]

    def add_text(self, tag, text, global_step=None):
        self.calls.append(("text", tag, text))

    def add_scalar(self, tag, value, global_step=None):
        self.calls.append(("scalar", tag, value, global_step))

    def add_figure(self, tag, figure, global_step=None, close=True):
        self.calls.append(("figure", tag, global_step, close))

    def add_image(self, tag, img, global_step=None, dataformats="CHW"):
        self.calls.append(("image", tag, global_step, dataformats))

    def add_video(self, tag, vid, global_step=None, fps=4):
        self.calls.append(("video", tag, vid, global_step, fps))

    def close(self):
        self.calls.append(("close",))


class _FakeFrames:
    """Supports frames[None] like a numpy/torch array would."""

    def __getitem__(self, key):
        return ("batched", key)


class _FakeFigure:
    def savefig(self, *args, **kwargs):
        pass


def test_tensorboard_logger_routes_by_type(monkeypatch, tmp_path):
    fake_module = types.ModuleType("torch.utils.tensorboard")
    fake_module.SummaryWriter = _FakeWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_module)

    logger = TensorBoardLogger(log_dir=tmp_path)
    run = resolve_run(None, "x")
    logger.start(run, config={"lr": 0.1})
    logger.log(
        {
            "loss": 0.5,
            "val": {"acc": 1},
            "plot": Image(_FakeFigure()),
            "grid": Image(object()),
            "rollout": Video(_FakeFrames(), fps=8),
            "note": object(),  # not numeric, not media -> dropped
        },
        step=3,
    )
    logger.finish()

    writer = logger.writer
    assert writer.calls[0] == ("init", str(tmp_path / run.run_id))
    assert writer.calls[1][:2] == ("text", "config")
    assert "0.1" in writer.calls[1][2]
    assert writer.calls[2:] == [
        ("scalar", "loss", 0.5, 3),
        ("scalar", "val/acc", 1.0, 3),
        ("figure", "plot", 3, True),
        ("image", "grid", 3, "HWC"),
        ("video", "rollout", ("batched", None), 3, 8),
        ("close",),
    ]


def test_tensorboard_logger_falls_back_to_tensorboardx(monkeypatch, tmp_path):
    # None in sys.modules makes the torch import raise ImportError
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    fake_module = types.ModuleType("tensorboardX")
    fake_module.SummaryWriter = _FakeWriter
    monkeypatch.setitem(sys.modules, "tensorboardX", fake_module)

    logger = TensorBoardLogger(log_dir=tmp_path)
    run = resolve_run(None, "x")
    logger.start(run)
    logger.log({"loss": 1.0}, step=0)
    logger.finish()

    assert logger.writer.calls == [
        ("init", str(tmp_path / run.run_id)),
        ("scalar", "loss", 1.0, 0),
        ("close",),
    ]
