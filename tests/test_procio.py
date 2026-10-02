import sys

from tether.config import SandboxConfig
from tether.procio import run_bounded


def _py(code: str, **kw):
    return run_bounded([sys.executable, "-c", code], **{"timeout": 20, "cap": 1000, **kw})


def test_small_output_is_returned_exactly():
    r = _py("import sys; print('héllo'); print('err', file=sys.stderr)")
    assert r.stdout == "héllo\n" and r.stderr == "err\n"
    assert r.returncode == 0 and not r.timed_out


def test_large_output_keeps_head_and_tail_and_names_the_gap():
    r = _py("import sys; sys.stdout.write('A' * 500 + 'B' * 100000 + 'Z' * 500)")
    assert r.stdout.startswith("A" * 500)
    assert r.stdout.endswith("Z" * 500)
    assert "[tether: 100000 bytes of output omitted]" in r.stdout
    assert len(r.stdout) < 1200


def test_output_exactly_at_the_cap_is_not_marked():
    r = _py("import sys; sys.stdout.write('x' * 1000)")
    assert r.stdout == "x" * 1000


def test_timeout_kills_and_keeps_partial_output():
    r = _py("import sys, time; print('started', flush=True); time.sleep(30)", timeout=1)
    assert r.timed_out and r.returncode != 0
    assert "started" in r.stdout


def test_nonzero_exit_code_is_reported():
    assert _py("raise SystemExit(3)").returncode == 3


def test_config_default():
    assert SandboxConfig().max_output_bytes == 8192
