import pandas as pd
import pytest

from tether import TetherConfig, Session
from tether.tools.inspect import inspect_handle


def _session(tmp_path):
    return Session.create(TetherConfig(root_dir=tmp_path / "r"))


def test_inspect_text_handle_returns_more_preview(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put("line\n" * 100, source="s")
    out = inspect_handle(sess, h.id, rows=10)
    assert out["kind"] == "text"
    assert out["preview"].count("line") == 10  # more than the stored summary preview


def test_inspect_dataframe_returns_head_rows(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": range(50)}), source="s")
    out = inspect_handle(sess, h.id, rows=7)
    assert out["kind"] == "dataframe"
    assert out["n_rows"] == 50
    assert len(out["head"]) == 7


def test_inspect_dataframe_with_stats(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": [1, 2, 3, 4]}), source="s")
    out = inspect_handle(sess, h.id, stats=True)
    assert set(out["stats"]["a"]) == {"min", "max", "null_count"}
    assert out["stats"]["a"]["max"] == 4


def test_inspect_unknown_handle_raises(tmp_path):
    sess = _session(tmp_path)
    with pytest.raises(KeyError):
        inspect_handle(sess, "nope")


def test_inspect_text_preview_is_capped_for_one_huge_line(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put("B" * 3_000_000, source="s")
    out = inspect_handle(sess, h.id, rows=10**9)
    assert len(out["preview"]) <= 20_000


def test_inspect_rows_are_clamped(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": range(1000)}), source="s")
    assert len(inspect_handle(sess, h.id, rows=10**9)["head"]) == 100
    assert len(inspect_handle(sess, h.id, rows=-5)["head"]) == 1


def test_inspect_head_is_capped_by_size(tmp_path):
    import json
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": ["v" * 5000] * 100}), source="s")
    out = inspect_handle(sess, h.id, rows=100)
    assert len(json.dumps(out["head"])) <= 20_000 and len(out["head"]) >= 1
    assert out["head_truncated"] is True


def test_inspect_never_materializes_the_dataframe(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    h = sess.store.put(pd.DataFrame({"a": [3, 1, None], "s": ["x", "z", "y"]}), source="s")

    def boom(*a, **k):
        raise AssertionError("the parent must not load a dataframe handle")

    monkeypatch.setattr(pd, "read_parquet", boom)
    out = inspect_handle(sess, h.id, rows=2, stats=True)
    assert len(out["head"]) == 2
    assert out["stats"]["a"] == {"min": 1.0, "max": 3.0, "null_count": 1}
    assert out["stats"]["s"] == {"min": "x", "max": "z", "null_count": 0}


def test_inspect_json_preview_is_bounded_raw_text(tmp_path):
    sess = _session(tmp_path)
    h = sess.store.put({"k": ["v"] * 100_000}, source="s")
    out = inspect_handle(sess, h.id)
    assert out["preview"].startswith('{"k": ["v", "v"') and len(out["preview"]) <= 20_000


def test_inspect_checks_the_digest(tmp_path):
    from tether.handles import HandleTamperedError
    sess = _session(tmp_path)
    h = sess.store.put("abc", source="s")
    (sess.root / h.path).write_text("tampered!")
    with pytest.raises(HandleTamperedError):
        inspect_handle(sess, h.id)


def test_inspect_skips_head_when_first_row_group_is_too_large(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    import tether.handles as handles
    sess = _session(tmp_path)
    monkeypatch.setattr(handles, "_MAX_PREVIEW_GROUP_BYTES", 10)
    h = sess.store.put(pd.DataFrame({"a": list(range(100))}), source="s")

    def boom(*a, **k):
        raise AssertionError("must not decode the first row group")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", boom)
    out = inspect_handle(sess, h.id, rows=5, stats=True)
    assert out["head"] == []
    assert "first row group is" in out["head_omitted"]
    assert out["stats"]["a"]["max"] == 99
