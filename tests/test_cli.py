from sheath.cli import make_status_printer, run_cli
from sheath.status import StatusEvent
from sheath.testing import StubChatClient, text, tool_call


def test_status_printer_formats_event():
    lines = []
    printer = make_status_printer(write=lines.append)
    printer(StatusEvent(tool="read_document", message="converting d.pdf via Docling",
                        current=1, total=4, seq=1, timestamp=1.0))
    assert lines == ["→ read_document: converting d.pdf via Docling [1/4]"]


def test_verbose_prints_tool_status_to_stderr(tmp_path, capsys):
    client = StubChatClient([
        tool_call("run_python", {"code": "from sheath_sandbox import emit\nemit(1)\n"}),
        text("done"),
    ])
    code = run_cli(["go", "-v", "--root", str(tmp_path / "r")], client=client)
    err = capsys.readouterr().err
    assert code == 0
    assert "run_python" in err                            # the instrumented tool reported
    assert "temporarily unavailable" not in err           # old notice is gone


def test_cli_prints_answer_and_session(tmp_path, capsys):
    client = StubChatClient([
        tool_call("list_files", {"path": "."}),
        text("all done"),
    ])
    code = run_cli(
        ["Summarize the workspace.", "--root", str(tmp_path / "r")],
        client=client,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "all done" in out
    assert "[session:" in out


def test_main_prints_the_message_and_exits_2_without_a_runtime(monkeypatch, capsys):
    """The plan's bar: not a traceback, and not a silent local run."""
    import pytest

    from sheath.sandbox import SandboxRuntimeUnavailable

    def unavailable(*a, **kw):
        raise SandboxRuntimeUnavailable("no container runtime found: ...\n\nrun_python ...")

    monkeypatch.setattr("sheath.cli.run_cli", unavailable)
    from sheath.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "no container runtime found" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_runtime_unavailable_message_states_the_opt_out_once(tmp_path, monkeypatch):
    """detect_runtime used to repeat the wrapper's advice in a different spelling, so the
    user read 'install podman or docker ... set backend local' twice in one error."""
    import pytest

    from sheath.config import SandboxConfig
    from sheath.handles import HandleStore
    from sheath.sandbox import SandboxRuntimeUnavailable
    from sheath.session import _build_sandbox

    monkeypatch.setattr("sheath.container_runtime.detect_runtime",
                        lambda override, which=None: (_ for _ in ()).throw(
                            RuntimeError("no container runtime found: neither podman nor "
                                         "docker is on PATH")))
    store = HandleStore(tmp_path / "r")
    with pytest.raises(SandboxRuntimeUnavailable) as excinfo:
        _build_sandbox(tmp_path / "r", store, SandboxConfig(backend="container"))

    message = str(excinfo.value).lower()
    assert message.count("podman") == 1          # named by the probe, not again by the wrapper
    assert message.count("local") == 2           # the opt-out, and its env-var spelling
    assert "no isolation" in message
