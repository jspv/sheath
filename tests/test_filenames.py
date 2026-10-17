"""validate_segment / safe_filename: untrusted ids and display names → safe single segments."""

import pytest

from sheath.paths import safe_filename, validate_segment


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "a\\b", "/abs", "x\x00y",
                                 "line\nbreak", "tab\tx"])
def test_validate_segment_rejects(bad):
    with pytest.raises(ValueError):
        validate_segment(bad, what="input id")


@pytest.mark.parametrize("ok", ["f1", "upload-123", "a.b", "U_9", "..hidden"])
def test_validate_segment_accepts(ok):
    assert validate_segment(ok, what="input id") == ok


def test_validate_segment_rejects_non_str():
    with pytest.raises(ValueError):
        validate_segment(5, what="input id")   # type: ignore[arg-type]


def test_validate_segment_caps_length():
    with pytest.raises(ValueError):
        validate_segment("x" * 256, what="input id")


@pytest.mark.parametrize("name, expected", [
    ("report.csv", "report.csv"),
    ("../../etc/passwd", "passwd"),
    ("a/b/c.xlsx", "c.xlsx"),
    ("C:\\Users\\x\\data.csv", "data.csv"),
    ("bad\x00na\x1bme\u202e.csv", "badname.csv"),       # control + bidi-override chars stripped
    ("  spaced.csv  ", "spaced.csv"),
])
def test_safe_filename_reduces_to_safe_basename(name, expected):
    assert safe_filename(name, fallback="fallback.bin") == expected


@pytest.mark.parametrize("name", [None, "", "   ", ".", "..", "/", "a/", "\x00\x01"])
def test_safe_filename_falls_back(name):
    assert safe_filename(name, fallback="data.csv") == "data.csv"


def test_safe_filename_fallback_is_sanitized_too():
    assert safe_filename(None, fallback="dir/../x\x00.csv") == "x.csv"


def test_safe_filename_last_resort():
    assert safe_filename("..", fallback="..") == "file"


def test_safe_filename_caps_length_keeping_extension():
    out = safe_filename("a" * 500 + ".csv", fallback="x")
    assert len(out) <= 128 and out.endswith(".csv")
