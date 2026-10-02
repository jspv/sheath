import asyncio

from tether.config import TetherConfig
from tether.session import Session
from tether.status import report_progress


def test_session_creates_root_under_default_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sess = Session.create(TetherConfig())
    assert sess.root.exists()
    assert sess.root.is_dir()
    assert ".tether/runs" in str(sess.root)
    sess.cleanup()


def test_session_uses_explicit_root_dir(tmp_path):
    cfg = TetherConfig(root_dir=tmp_path / "myroot")
    sess = Session.create(cfg)
    assert sess.root == (tmp_path / "myroot").resolve()
    assert sess.root.exists()


def test_session_wires_store_and_sandbox_to_same_root(tmp_path):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"))
    assert sess.store.root == sess.root
    assert sess.sandbox.root == sess.root


def test_session_end_to_end_handle_then_analyze(tmp_path):
    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"))
    h = sess.store.put({"values": [120, 95, 0, 210]}, source="tool:read")
    code = (
        "from tether_sandbox import load, emit\n"
        f"d = load('{h.id}')\n"
        "vals = [v for v in d['values'] if v > 0]\n"
        "emit({'total': sum(vals), 'dropped': len(d['values']) - len(vals)})\n"
    )
    res = sess.sandbox.run_code(code)
    assert res.result == {"total": 425, "dropped": 1}


def test_cleanup_removes_root_when_owned(tmp_path):
    cfg = TetherConfig(root_dir=tmp_path / "r")
    sess = Session.create(cfg)
    assert sess.root.exists()
    sess.cleanup()
    assert not sess.root.exists()


def test_session_handles_and_artifacts(tmp_path):
    from tether import TetherConfig, Session

    sess = Session.create(TetherConfig(root_dir=tmp_path / "r"))
    (sess.root / "report.txt").write_text("hi")
    (sess.root / ".scripts").mkdir(exist_ok=True)
    (sess.root / ".scripts" / "x.py").write_text("# scratch")
    sess.store.put({"a": 1}, source="t")

    assert "report.txt" in sess.artifacts
    assert not any(a.startswith(".scripts") for a in sess.artifacts)
    assert not any(a.startswith("handles") for a in sess.artifacts)
    assert sess.handles  # the put() handle shows up


def test_session_async_context_manager_cleanup(tmp_path):
    from tether import TetherConfig, Session

    async def run():
        cfg = TetherConfig(root_dir=tmp_path / "r", cleanup=True)
        async with Session.create(cfg) as sess:
            root = sess.root
            assert root.exists()
        return root

    root = asyncio.run(run())
    assert not root.exists()  # cleanup=True removed it on exit


def test_session_async_context_manager_no_cleanup_by_default(tmp_path):
    from tether import TetherConfig, Session

    async def run():
        cfg = TetherConfig(root_dir=tmp_path / "r")  # cleanup defaults to False
        async with Session.create(cfg) as sess:
            root = sess.root
            assert root.exists()
        return root

    root = asyncio.run(run())
    assert root.exists()  # root survives because cleanup=False


def test_session_subscribe_receives_events_within_async_context(tmp_path):
    async def run():
        got = []
        async with Session.create(TetherConfig(root_dir=tmp_path / "r")) as session:
            session.subscribe(got.append)
            report_progress("inside the run", tool="x")   # bus is bound by __aenter__
        return got

    got = asyncio.run(run())
    assert len(got) == 1
    assert got[0].message == "inside the run"


def test_report_progress_is_noop_outside_async_context(tmp_path):
    # Session.create without `async with` does NOT bind the bus.
    session = Session.create(TetherConfig(root_dir=tmp_path / "r2"))
    got = []
    session.subscribe(got.append)
    report_progress("nobody bound")                       # no-op
    assert got == []


def test_session_selects_container_backend(tmp_path, monkeypatch):
    from tether.config import TetherConfig, SandboxConfig
    from tether.sandbox import LocalSubprocessSandbox
    from tether.sandbox_container import ContainerSandbox

    # default -> local
    s_local = Session.create(TetherConfig(root_dir=tmp_path / "a"))
    assert isinstance(s_local.sandbox, LocalSubprocessSandbox)

    # backend="container" -> ContainerSandbox (patch detect_runtime so no podman/docker needed)
    import tether.container_runtime as cr
    monkeypatch.setattr(cr, "detect_runtime", lambda override, **kw: "podman")
    cfg = TetherConfig(root_dir=tmp_path / "b", sandbox=SandboxConfig(backend="container"))
    s_cont = Session.create(cfg)
    assert isinstance(s_cont.sandbox, ContainerSandbox)


def test_session_unbinds_bus_on_exit(tmp_path):
    from tether.status import current_bus

    async def run():
        async with Session.create(TetherConfig(root_dir=tmp_path / "r3")):
            assert current_bus() is not None
        assert current_bus() is None

    asyncio.run(run())


def test_failed_sandbox_build_leaves_no_orphan_session_dir(tmp_path, monkeypatch):
    """Session.create used to mkdir the root before _build_sandbox could refuse, so every
    failed start on a runtime-less host left an empty .tether/runs/run-XXXX behind."""
    import pytest

    from tether.config import TetherConfig
    from tether.sandbox import SandboxRuntimeUnavailable
    from tether.session import Session

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tether.session._build_sandbox",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            SandboxRuntimeUnavailable("no runtime")))

    for _ in range(3):
        with pytest.raises(SandboxRuntimeUnavailable):
            Session.create(TetherConfig())

    sessions = tmp_path / ".tether" / "runs"
    assert [p.name for p in sessions.iterdir()] == []


def test_a_pre_existing_pinned_root_survives_a_failed_build(tmp_path, monkeypatch):
    """Cleanup removes only a root this call created -- never the user's own directory."""
    import pytest

    from tether.config import TetherConfig
    from tether.sandbox import SandboxRuntimeUnavailable
    from tether.session import Session

    root = tmp_path / "mine"
    root.mkdir()
    (root / "keepme.txt").write_text("precious", encoding="utf-8")
    monkeypatch.setattr("tether.session._build_sandbox",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            SandboxRuntimeUnavailable("no runtime")))

    with pytest.raises(SandboxRuntimeUnavailable):
        Session.create(TetherConfig(root_dir=root))
    assert (root / "keepme.txt").read_text() == "precious"


def test_default_roots_are_unique_and_outside_the_conversation_namespace(tmp_path, monkeypatch):
    from tether import Session, TetherConfig
    monkeypatch.chdir(tmp_path)
    a = Session.create(TetherConfig())
    b = Session.create(TetherConfig())
    assert a.root != b.root
    for s in (a, b):
        assert s.root.parent == (tmp_path / ".tether" / "runs").resolve()
        assert s.root.name.startswith("run-")
    assert not (tmp_path / ".tether" / "sessions").exists()


def test_default_root_allocation_is_safe_across_threads(tmp_path, monkeypatch):
    import threading
    from tether import Session, TetherConfig
    monkeypatch.chdir(tmp_path)
    roots = []

    def make():
        roots.append(Session.create(TetherConfig()).root)

    threads = [threading.Thread(target=make) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(roots)) == 8


def test_artifacts_exclude_internal_trees_and_links(tmp_path):
    from tether import Session, TetherConfig
    s = Session.create(TetherConfig(root_dir=tmp_path / "root"))
    s.store.put("x", source="t")
    (s.root / "outputs").mkdir()
    (s.root / "outputs" / "a.csv").write_text("1\n")
    (s.root / "link.txt").symlink_to(s.root / "outputs" / "a.csv")
    assert s.artifacts == ["outputs/a.csv"]
