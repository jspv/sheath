"""TETHER-4 core: every publication is validated in the parent before the host sees it."""

import hashlib
import os

import pytest

from tether.publish import PublishError, PublishedFile, validate_publication

MAX = 1024


def _write(root, rel, data=b"x,y\n1,2\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_valid_file_is_described(tmp_path):
    data = b"x,y\n1,2\n"
    _write(tmp_path, "outputs/a.csv", data)
    pf = validate_publication(tmp_path, "outputs/a.csv", name=None, description="d",
                              source="tool:publish_file", max_bytes=MAX)
    assert isinstance(pf, PublishedFile)
    assert pf.path == (tmp_path / "outputs/a.csv").resolve()
    assert pf.rel_path == "outputs/a.csv"
    assert pf.name == "a.csv"
    assert pf.size == len(data)
    assert pf.sha256 == hashlib.sha256(data).hexdigest()
    assert pf.content_type == "text/csv"
    assert pf.description == "d" and pf.source == "tool:publish_file"


def test_absolute_path_inside_root_is_accepted(tmp_path):
    _write(tmp_path, "outputs/a.csv")
    pf = validate_publication(tmp_path, str(tmp_path / "outputs/a.csv"), name=None,
                              description=None, source="s", max_bytes=MAX)
    assert pf.rel_path == "outputs/a.csv"


def test_name_is_sanitized(tmp_path):
    _write(tmp_path, "outputs/a.csv")
    pf = validate_publication(tmp_path, "outputs/a.csv", name="../../evil\x00.csv",
                              description=None, source="s", max_bytes=MAX)
    assert pf.name == "evil.csv"


def _reject(root, path, **kw):
    with pytest.raises(PublishError):
        validate_publication(root, path, name=kw.get("name"), description=None, source="s",
                             max_bytes=kw.get("max_bytes", MAX))


def test_rejects_dotdot(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _write(tmp_path, "x")                     # a real file just outside the root
    _reject(root, "../x")


def test_rejects_absolute_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = _write(tmp_path, "secret.txt")
    _reject(root, str(outside))
    _reject(root, "/etc/hosts")


def test_rejects_symlink_pointing_outside(tmp_path):
    root = tmp_path / "root"
    (root / "outputs").mkdir(parents=True)
    outside = _write(tmp_path, "secret.txt")
    os.symlink(outside, root / "outputs" / "link.csv")
    _reject(root, "outputs/link.csv")


def test_rejects_symlink_even_inside_root(tmp_path):
    _write(tmp_path, "outputs/real.csv")
    os.symlink(tmp_path / "outputs/real.csv", tmp_path / "outputs/link.csv")
    _reject(tmp_path, "outputs/link.csv")


def test_rejects_file_under_symlinked_dir_escaping_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _write(tmp_path, "elsewhere/secret.csv")
    os.symlink(tmp_path / "elsewhere", root / "outputs")
    _reject(root, "outputs/secret.csv")


def test_rejects_directory(tmp_path):
    (tmp_path / "outputs").mkdir()
    _reject(tmp_path, "outputs")


def test_rejects_missing(tmp_path):
    _reject(tmp_path, "outputs/nope.csv")


def test_rejects_fifo(tmp_path):
    (tmp_path / "outputs").mkdir()
    os.mkfifo(tmp_path / "outputs" / "p")
    _reject(tmp_path, "outputs/p")


@pytest.mark.parametrize("rel", ["handles/h1.parquet", "handles/_manifest.json",
                                 ".scripts/inline_x.py", "_emit_1_2.json",
                                 "_new_handles_1_2.jsonl", "_publish_1_2.jsonl"])
def test_rejects_internal_files(tmp_path, rel):
    _write(tmp_path, rel)
    _reject(tmp_path, rel)


def test_rejects_oversize(tmp_path):
    _write(tmp_path, "outputs/big.bin", b"x" * (MAX + 1))
    _reject(tmp_path, "outputs/big.bin")


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_rejects_bad_path_values(tmp_path, bad):
    _reject(tmp_path, bad)


def test_escape_errors_do_not_reveal_host_existence(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "exists.txt").write_text("x")
    os.symlink(tmp_path, root / "outputs")          # planted parent-dir symlink
    def message(p):
        with pytest.raises(PublishError) as ei:
            validate_publication(root, p, name=None, description=None, source="s",
                                 max_bytes=MAX)
        return str(ei.value).replace(repr(p), "<p>")

    # Each pair differs only in whether the host path exists; the answer must not.
    assert message(str(tmp_path / "exists.txt")) == message(str(tmp_path / "missing.txt"))
    assert message("outputs/exists.txt") == message("outputs/missing.txt")


def test_symlink_loop_is_a_publish_error(tmp_path):
    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    _reject(tmp_path, "a")


def test_link_through_host_dirs_answers_the_same_whether_or_not_they_exist(tmp_path):
    root = tmp_path / "root"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs" / "f.txt").write_text("x")
    (tmp_path / "hostdir").mkdir()
    os.symlink(f"{tmp_path}/hostdir/../root/outputs", root / "l1")   # host dir exists
    os.symlink(f"{tmp_path}/missing/../root/outputs", root / "l2")   # host dir missing
    results = []
    for p in ("l1/f.txt", "l2/f.txt"):
        try:
            validate_publication(root, p, name=None, description=None, source="s",
                                 max_bytes=MAX)
            results.append("published")
        except PublishError as e:
            results.append(str(e).replace(repr(p), "<p>"))
    assert results[0] == results[1]


def test_symlink_loop_in_a_parent_component_is_a_publish_error(tmp_path):
    os.symlink(tmp_path / "b", tmp_path / "a")
    os.symlink(tmp_path / "a", tmp_path / "b")
    _reject(tmp_path, "a/x")


def test_state_dir_is_not_publishable(tmp_path):
    import pytest
    from tether.publish import PublishError, validate_publication
    (tmp_path / ".tether").mkdir()
    (tmp_path / ".tether" / "manifest.json").write_text("{}")
    with pytest.raises(PublishError, match="internal"):
        validate_publication(tmp_path, ".tether/manifest.json", name=None, description=None,
                             source="t", max_bytes=1000)
