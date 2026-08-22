"""Run lifecycle: identity, resume and fork semantics for a training run.

A run id is ``"<base_name>_<timestamp>"`` (e.g. ``"coinrun-exp-1_20260530_051406"``)
and is meant to be shared between the checkpoint directory and the experiment
tracker id (e.g. ``wandb.init(id=...)``), so that resuming a run resumes both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Literal, TypeAlias

ResumePolicy: TypeAlias = Literal["never", "must"]


class RunType(Enum):
    FRESH = auto()  # no checkpoint, train from scratch
    CONTINUE = auto()  # same run: same id/folder, resume tracker + optimizer state
    FORK = auto()  # new run: new id/folder, load weights only, fresh tracker run


_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def generate_run_id(base_name: str) -> str:
    timestamp = datetime.now().strftime(_TIMESTAMP_FMT)
    return f"{base_name}_{timestamp}"


def run_id_base(run_id: str) -> str:
    return _TIMESTAMP_RE.sub("", run_id)


@dataclass(frozen=True)
class RunInfo:
    name: str
    run_id: str
    run_type: RunType
    restore_dir: str | None  # checkpoint folder to load weights from (None = scratch)
    resume: ResumePolicy  # tracker resume semantics ("must" only when continuing)


def resolve_run(resume_from: str | None, name: str) -> RunInfo:
    """3 modes from (resume_from, name):

    - resume_from unset       -> FRESH: new id, train from scratch.
    - resume_from + same name -> CONTINUE: same id/folder, resume tracker.
    - resume_from + new name  -> FORK: new id/folder, load weights, fresh tracker.
    """
    if resume_from is None:
        return RunInfo(name, generate_run_id(name), RunType.FRESH, None, "never")
    if name == run_id_base(resume_from):
        return RunInfo(name, resume_from, RunType.CONTINUE, resume_from, "must")
    return RunInfo(name, generate_run_id(name), RunType.FORK, resume_from, "never")
