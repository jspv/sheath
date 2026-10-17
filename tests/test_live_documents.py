import importlib.util
import os

import pytest

from sheath import SheathConfig, Session
from sheath.tools.documents import read_document

_RUN = os.environ.get("SHEATH_LIVE_DOCLING") == "1"
_HAS_DOCLING = importlib.util.find_spec("docling") is not None

pytestmark = pytest.mark.skipif(
    not (_RUN and _HAS_DOCLING),
    reason="set SHEATH_LIVE_DOCLING=1 and install the 'docling' extra to run real Docling",
)


def test_real_docling_converts_table_to_markdown(tmp_path):
    sess = Session.create(SheathConfig(root_dir=tmp_path / "r"))
    (sess.root / "t.html").write_text(
        "<html><body><h1>Sales</h1>"
        "<table><tr><th>region</th><th>units</th></tr>"
        "<tr><td>EU</td><td>12</td></tr></table></body></html>",
        encoding="utf-8",
    )
    summary = read_document(sess, "t.html")          # default converter -> real Docling
    assert summary["kind"] == "text"
    markdown = sess.store.get(summary["id"])
    assert "region" in markdown and "units" in markdown
    assert "|" in markdown                           # rendered as a markdown table
