import os

import pytest

from tether.paths import PathEscapesRootError, safe_path


def test_simple_relative_path_resolves_under_root(tmp_path):
    assert safe_path(tmp_path, "data.csv") == (tmp_path / "data.csv").resolve()


def test_subfolder_is_allowed(tmp_path):
    assert safe_path(tmp_path, "sub/dir/data.csv") == (tmp_path / "sub/dir/data.csv").resolve()


def test_root_itself_is_allowed(tmp_path):
    assert safe_path(tmp_path, ".") == tmp_path.resolve()


def test_dotdot_traversal_is_rejected(tmp_path):
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "../escape.txt")


def test_nested_dotdot_escaping_root_is_rejected(tmp_path):
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "sub/../../escape.txt")


def test_absolute_path_outside_root_is_rejected(tmp_path):
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "/etc/passwd")


def test_symlink_pointing_outside_root_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside_secret"
    outside.mkdir()
    link = tmp_path / "link"
    os.symlink(outside, link)
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "link/secret.txt")


def test_dotdot_that_stays_within_root_is_allowed(tmp_path):
    # sub/../data.csv normalizes back to root/data.csv — legal
    assert safe_path(tmp_path, "sub/../data.csv") == (tmp_path / "data.csv").resolve()


def test_absolute_path_inside_root_is_allowed(tmp_path):
    # A model may hand back a fully-qualified path it was previously given.
    inside = tmp_path / "sub" / "file.txt"
    assert safe_path(tmp_path, str(inside)) == inside.resolve()


def test_empty_candidate_is_rejected(tmp_path):
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "")


def test_whitespace_only_candidate_is_rejected(tmp_path):
    with pytest.raises(PathEscapesRootError):
        safe_path(tmp_path, "   ")
