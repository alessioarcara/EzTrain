import re

from eztrain.run import RunType, generate_run_id, resolve_run, run_id_base

TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def test_generate_run_id_appends_timestamp():
    run_id = generate_run_id("exp-1")
    assert run_id.startswith("exp-1_")
    assert TIMESTAMP_RE.search(run_id)


def test_run_id_base_strips_timestamp():
    assert run_id_base("exp-1_20260530_051406") == "exp-1"
    assert run_id_base("no-timestamp") == "no-timestamp"


def test_fresh_run():
    run = resolve_run(None, "exp-1")
    assert run.run_type is RunType.FRESH
    assert run.name == "exp-1"
    assert run.run_id.startswith("exp-1_")
    assert run.restore_dir is None
    assert run.resume == "never"


def test_continue_run_same_name():
    run = resolve_run("exp-1_20260530_051406", "exp-1")
    assert run.run_type is RunType.CONTINUE
    assert run.run_id == "exp-1_20260530_051406"
    assert run.restore_dir == "exp-1_20260530_051406"
    assert run.resume == "must"


def test_fork_run_new_name():
    run = resolve_run("exp-1_20260530_051406", "exp-2")
    assert run.run_type is RunType.FORK
    assert run.run_id.startswith("exp-2_")
    assert run.run_id != "exp-1_20260530_051406"
    assert run.restore_dir == "exp-1_20260530_051406"
    assert run.resume == "never"
