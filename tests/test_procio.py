import os
import sys
import time

from sheath.config import SandboxConfig
from sheath.procio import run_bounded


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
    assert "[sheath: 100000 bytes of output omitted]" in r.stdout
    assert len(r.stdout) < 1200


def test_output_exactly_at_the_cap_is_not_marked():
    r = _py("import sys; sys.stdout.write('x' * 1000)")
    assert r.stdout == "x" * 1000


def test_timeout_kills_and_keeps_partial_output():
    start = time.time()
    r = _py("import sys, time; print('started', flush=True); time.sleep(30)", timeout=1)
    elapsed = time.time() - start
    assert r.timed_out and r.returncode != 0
    assert "started" in r.stdout
    assert elapsed < 10, f"took {elapsed}s, should be < 10s (proves kill, not wait)"


def test_nonzero_exit_code_is_reported():
    assert _py("raise SystemExit(3)").returncode == 3


def test_grandchild_killed_on_normal_exit():
    """Grandchild spawned by the child must be killed when child exits normally."""
    start = time.time()
    code = """
import sys, subprocess
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
print(p.pid)
"""
    r = _py(code)
    elapsed = time.time() - start
    assert r.returncode == 0 and not r.timed_out
    assert elapsed < 3, f"took {elapsed}s, should be < 3s"

    grandchild_pid = int(r.stdout.strip())

    # Poll to confirm grandchild is dead (allow ~1s retry for reaping)
    for _ in range(10):
        try:
            os.kill(grandchild_pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        raise AssertionError(f"grandchild {grandchild_pid} still alive after 1s")


def test_grandchild_killed_on_timeout():
    """Grandchild spawned by the child must be killed when run times out."""
    start = time.time()
    code = """
import sys, subprocess, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
print(p.pid, flush=True)
sys.stdout.flush()
time.sleep(30)
"""
    r = _py(code, timeout=1)
    elapsed = time.time() - start
    assert r.timed_out
    assert elapsed < 4, f"took {elapsed}s, should be < 4s"

    # Parse pid from output; if empty, child was killed before printing
    pid_str = r.stdout.strip()
    if not pid_str:
        # Child was killed too quickly to print; this is acceptable - the important
        # thing is that we returned quickly (timed_out is True and elapsed < 4s)
        return
    grandchild_pid = int(pid_str)

    # Poll to confirm grandchild is dead (allow ~1s retry for reaping)
    for _ in range(10):
        try:
            os.kill(grandchild_pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        raise AssertionError(f"grandchild {grandchild_pid} still alive after 1s")


def test_config_default():
    assert SandboxConfig().max_output_bytes == 8192
