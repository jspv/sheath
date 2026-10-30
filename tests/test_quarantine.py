"""An orphaned container (a timed-out run the runtime could not confirm stopped) quarantines
the session: no further run and no parent-side file work touches the root until the runtime
confirms that exact container is gone. Fake runtime only; no real containers."""

import subprocess

import pytest

from sheath import Session, SheathConfig
from sheath.config import SandboxConfig
from sheath.procio import BoundedRun
from sheath.sandbox import SandboxQuarantinedError


class _FakeRuntime:
    """Stands in for the podman CLI: `run` times out; `rm -f` / `inspect` obey flags."""

    def __init__(self):
        self.rm_ok = False
        self.inspect_gone = False
        self.calls: list[list[str]] = []
        self.run_names: list[str] = []
        self.run_times_out = True
        # When set, both `rm -f` and `container inspect` fail with exactly this text
        # (what a real runtime prints when e.g. its socket is unreachable).
        self.fail_text: str | None = None

    def bounded(self, argv, **kw):
        self.calls.append(argv)
        if argv[1] == "run":
            self.run_names.append(argv[argv.index("--name") + 1])
        if self.run_times_out:
            return BoundedRun("", "", -9, True)
        return BoundedRun("", "", 0, False)

    def run(self, argv, **kw):
        self.calls.append(argv)
        if self.fail_text is not None and (argv[1:3] in (["rm", "-f"],
                                                          ["container", "inspect"])):
            return subprocess.CompletedProcess(argv, 125, "", self.fail_text)
        if argv[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(argv, 0 if self.rm_ok else 125, "",
                                               "" if self.rm_ok else "Error: stuck")
        if argv[1:3] == ["container", "inspect"]:
            if self.inspect_gone:
                return subprocess.CompletedProcess(argv, 125, "",
                                                   "Error: no such container " + argv[3])
            return subprocess.CompletedProcess(argv, 0, "[{}]", "")
        raise AssertionError(f"unexpected runtime call {argv}")

    def rm_calls(self):
        return [a for a in self.calls if a[1:3] == ["rm", "-f"]]


@pytest.fixture
def quarantined(tmp_path, monkeypatch):
    import sheath.container_runtime as cr
    import sheath.sandbox_container as sc
    from sheath.sandbox_container import ContainerSandbox

    fake = _FakeRuntime()
    monkeypatch.setattr(cr, "ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(sc, "run_bounded", fake.bounded)
    monkeypatch.setattr(sc.subprocess, "run", fake.run)
    sess = Session.create(SheathConfig(root_dir=tmp_path / "r"), on_publish=lambda pf: {})
    sb = ContainerSandbox(root=sess.root, store=sess.store, runtime="podman",
                          config=SandboxConfig(backend="container", timeout_s=1))
    sess.sandbox = sb
    sess.io_lock = sb.lock
    (sess.root / "notes.txt").write_text("hello")
    res = sb.run_code("1")                       # times out; rm -f fails -> orphaned
    assert "could not be stopped" in res.error
    return sess, fake


def test_orphaned_run_quarantines_further_runs(quarantined):
    sess, fake = quarantined
    orphan = fake.run_names[0]
    with pytest.raises(SandboxQuarantinedError, match=orphan):
        sess.sandbox.run_code("2")
    with pytest.raises(SandboxQuarantinedError):
        sess.sandbox.run_script(".scripts/x.py")
    assert len(fake.run_names) == 1               # no second container was launched
    # Only the recorded exact name is ever removed.
    assert all(a == ["podman", "rm", "-f", orphan] for a in fake.rm_calls())


def test_quarantine_refuses_parent_file_tools(quarantined):
    sess, fake = quarantined
    tools = {t.__name__: t for t in sess.tools()}
    with pytest.raises(SandboxQuarantinedError):
        tools["write_file"]("outputs/a.txt", "x")
    with pytest.raises(SandboxQuarantinedError):
        tools["read_file"]("notes.txt")
    with pytest.raises(SandboxQuarantinedError):
        tools["edit_file"]("notes.txt", "hello", "bye")
    with pytest.raises(SandboxQuarantinedError):
        tools["list_files"](".")
    with pytest.raises(SandboxQuarantinedError):
        tools["search"]("hello")
    assert not (sess.root / "outputs").exists()
    h = sess.store.put("data", source="t")
    assert "quarantine" in tools["publish_handle"](h.id)["error"]
    assert "quarantine" in tools["publish_file"]("notes.txt")["error"]
    assert "quarantine" in tools["read_document"]("notes.txt",)["error"]
    assert not (sess.root / "outputs").exists()
    with pytest.raises(SandboxQuarantinedError):
        sess.add_input(b"x", input_id="u1", name="a.txt")


def test_quarantine_clears_once_the_runtime_confirms_removal(quarantined):
    sess, fake = quarantined
    tools = {t.__name__: t for t in sess.tools()}
    with pytest.raises(SandboxQuarantinedError):
        tools["read_file"]("notes.txt")
    fake.rm_ok = True
    assert tools["read_file"]("notes.txt") == "hello"
    fake.run_times_out = False
    calls_before = len(fake.rm_calls())
    res = sess.sandbox.run_code("print('ok')")
    assert res.exit_code == 0
    assert len(fake.rm_calls()) == calls_before   # cleared: no further re-check


def test_inspect_saying_no_such_container_also_clears(quarantined):
    # rm -f of an already-removed container can fail (docker); inspect settles it.
    sess, fake = quarantined
    fake.inspect_gone = True
    tools = {t.__name__: t for t in sess.tools()}
    assert tools["read_file"]("notes.txt") == "hello"


def test_false_orphan_when_rm_fails_but_container_is_gone(tmp_path, monkeypatch):
    import sheath.container_runtime as cr
    import sheath.sandbox_container as sc
    from sheath.handles import HandleStore
    from sheath.sandbox_container import ContainerSandbox

    fake = _FakeRuntime()
    fake.inspect_gone = True                      # the container already exited and was --rm'd
    monkeypatch.setattr(cr, "ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(sc, "run_bounded", fake.bounded)
    monkeypatch.setattr(sc.subprocess, "run", fake.run)
    sb = ContainerSandbox(root=tmp_path, store=HandleStore(tmp_path), runtime="podman",
                          config=SandboxConfig(backend="container", timeout_s=1))
    res = sb.run_code("1")
    assert res.killed_by == "timeout" and "could not be stopped" not in (res.error or "")
    sb.check_quarantine()                         # nothing recorded


_SOCKET_ERR = ("Error: unable to connect to Podman socket: Get \"http://d/v5.0.0/libpod/_ping\": "
               "dial unix /run/podman/podman.sock: connect: no such file or directory")


def _container_sandbox(tmp_path, monkeypatch, fake):
    import sheath.container_runtime as cr
    import sheath.sandbox_container as sc
    from sheath.handles import HandleStore
    from sheath.sandbox_container import ContainerSandbox

    monkeypatch.setattr(cr, "ensure_image", lambda *a, **k: None)
    monkeypatch.setattr(sc, "run_bounded", fake.bounded)
    monkeypatch.setattr(sc.subprocess, "run", fake.run)
    return ContainerSandbox(root=tmp_path, store=HandleStore(tmp_path), runtime="podman",
                            config=SandboxConfig(backend="container", timeout_s=1))


def test_unreachable_runtime_at_timeout_is_orphaned_not_gone(tmp_path, monkeypatch):
    # A socket failure mentions "no such file or directory"; it says nothing about the
    # container, which may still be running.
    fake = _FakeRuntime()
    fake.fail_text = _SOCKET_ERR
    sb = _container_sandbox(tmp_path, monkeypatch, fake)
    res = sb.run_code("1")
    assert res.killed_by == "timeout" and "could not be stopped" in (res.error or "")
    with pytest.raises(SandboxQuarantinedError, match=fake.run_names[0]):
        sb.run_code("2")
    assert len(fake.run_names) == 1


def test_unreachable_runtime_does_not_clear_an_existing_quarantine(quarantined):
    sess, fake = quarantined
    fake.fail_text = _SOCKET_ERR
    tools = {t.__name__: t for t in sess.tools()}
    for _ in range(2):
        with pytest.raises(SandboxQuarantinedError):
            tools["read_file"]("notes.txt")
    fake.fail_text = None
    fake.rm_ok = True
    assert tools["read_file"]("notes.txt") == "hello"


@pytest.mark.parametrize("text", [
    'Error: no such container "sheath-abc"',                         # podman
    "Error response from daemon: No such container: sheath-abc",     # docker
])
def test_runtime_no_such_container_message_clears_quarantine(quarantined, text):
    sess, fake = quarantined
    fake.fail_text = text
    tools = {t.__name__: t for t in sess.tools()}
    assert tools["read_file"]("notes.txt") == "hello"


@pytest.mark.parametrize("text", [
    'Error: no such container "sheath-abc"',
    "Error response from daemon: No such container: sheath-abc",
])
def test_no_such_container_at_timeout_is_not_orphaned(tmp_path, monkeypatch, text):
    fake = _FakeRuntime()
    fake.fail_text = text
    sb = _container_sandbox(tmp_path, monkeypatch, fake)
    res = sb.run_code("1")
    assert res.killed_by == "timeout" and "could not be stopped" not in (res.error or "")
    sb.check_quarantine()


def test_local_sandbox_quarantine_check_is_a_noop(tmp_path):
    sess = Session.create(SheathConfig(root_dir=tmp_path / "r"))
    sess.sandbox.check_quarantine()
    with sess.file_io():
        pass
