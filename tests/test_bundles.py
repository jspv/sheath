from sheath import SheathConfig, Session


def _session(tmp_path):
    return Session.create(SheathConfig(root_dir=tmp_path / "r"))


def test_tools_for_code_bundle_includes_core(tmp_path):
    sess = _session(tmp_path)
    names = {t.__name__ for t in sess.tools("code")}
    assert names == {"inspect_handle", "run_python"}  # core + code


def test_tools_default_is_all_bundles(tmp_path):
    sess = _session(tmp_path)
    names = {t.__name__ for t in sess.tools()}
    assert names == {
        "inspect_handle", "run_python",
        "read_file", "write_file", "list_files", "search",
        "fetch_url", "web_search", "web_extract", "read_document",
    }


def test_read_document_is_in_web_bundle(tmp_path):
    sess = _session(tmp_path)
    assert "read_document" in {t.__name__ for t in sess.tools("web")}
    assert "read_document" not in {t.__name__ for t in sess.tools("files")}


def test_web_instructions_mention_read_document(tmp_path):
    sess = _session(tmp_path)
    assert "read_document" in sess.sheath_instructions("web")


def test_sheath_instructions_compose_by_bundle(tmp_path):
    sess = _session(tmp_path)
    core_only = sess.sheath_instructions()  # always includes core
    assert "handle" in core_only.lower()
    with_code = sess.sheath_instructions("code")
    assert "run_python" in with_code
    assert "load(" in with_code
    # web fragment only appears when web is selected
    assert "web_search" not in sess.sheath_instructions("code")
    assert "web_search" in sess.sheath_instructions("web")


def test_public_exports_for_host_integration():
    import sheath
    for name in ("PublishedFile", "OnPublish", "PublishError", "sandbox_preflight",
                 "PreflightReport", "SandboxRuntimeUnavailable"):
        assert hasattr(sheath, name) and name in sheath.__all__
