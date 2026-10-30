# File-Tool Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Record, not a work order.** This plan was written after the work was done, from the steps
> as they were actually executed; every box is checked. Deviations from what a plan written
> first would have said are recorded under each task and in "Execution notes".

**Goal:** Make search → read → edit work as one loop. `search` and `read_file` share 1-based line numbers, a match and its surroundings come back in one call, part of a file can be changed without resending it, files can be found by glob, and searches can answer at file level. All of this without widening the trust boundary.

**Architecture:** Purpose-built, root-confined tools in `sheath/tools/` rather than a shell. Every path goes through `safe_path` under `session.file_io()`. Search context is read by the parent after the scan, so both search backends stay identical. `edit_file` replaces atomically (temp file + `os.replace`) and refuses what `write_file` refuses.

**Tech Stack:** Python 3.12, pytest, ruff, uv, ripgrep 14 (`--json`), Microsoft Agent Framework (tool descriptions come from wrapper docstrings).

**Spec:** `docs/superpowers/specs/2026-10-30-file-tool-parity-design.md`

## Global Constraints

- **Line numbering.** Line numbers are 1-based everywhere. A line ends at `\n`, `\r\n` or a bare `\r`, as `workspace.iter_bounded_lines` splits them.
- **File access.** Every file access holds `session.file_io()` and resolves through `safe_path`. Writes refuse `workspace.is_protected` paths.
- **Result size.** Every tool result stays under `tests/test_tool_ceilings.py`'s 64,000-byte ceiling for hostile input.
- **Backend agreement.** ripgrep and the Python fallback must return identical results. Tests that compare them sort by file, because ripgrep's cross-file order is not fixed.
- **Breaking changes.** `read_file(offset=...)` is replaced, not deprecated; existing callers need not keep working.
- **Naming.** Never name any AI assistant or coding-agent product in tracked files (code, comments, docs, commit messages).
- **Commands.** Run commands with `uv run`. Full suite: `uv run pytest -p no:warnings`. Lint: `uv run ruff check sheath tests`.
- **Baseline before Task 1:** 841 tests; 31 skipped; 4 failures from the environment, not the code:
  - `test_rg_error_text_never_contains_the_host_root`: the tests run as root, so `chmod 0` does not make a file unreadable;
  - the two grandchild-kill tests in `test_procio.py`;
  - `test_session_selects_container_backend`: podman is not installed.

## Review Focus

1. **A model passes a `search` hit's `line` to `read_file`.** The read must start at the hit (Task 1, `test_search_hit_line_feeds_read_file_directly`).
2. **`edit_file`'s `old` matches millions of times.** The error must come back promptly and bounded, not scan the file once per match (Task 4, `test_builtin_tool_results_stay_under_the_ceiling` and `_match_lines`).
3. **A model copies text with the wrong whitespace or case into `old`.** The error must point at the near match by line number (Task 4, `test_edit_file_no_match_points_at_a_near_match`).
4. **`edit_file` is pointed into `handles/`, or through a link out of the root.** It must refuse and leave the target byte-identical (Task 4, `test_edit_file_refuses_protected_directories` and `test_edit_file_refuses_escaping_links_binary_and_non_utf8`).
5. **Adjacent matching lines with `context`.** Each must still show the other as context on both backends (Task 2, `test_search_context_includes_adjacent_matching_lines`).

## Assumptions verified before starting

- ripgrep's `line_number` and the fallback's `enumerate(..., start=1)` are both 1-based; `read_file` skipped `lineno < offset` with `lineno` counted from 0.
- `rg --json --count` prints plain `path:count`, not JSON, so counts must come from the event stream.
- `rg --json --max-count 1` emits one match event per file.
- `safe_path` resolves links; in-root links are followed by the existing file tools and only escapes are refused.

## File Structure

| File | Responsibility |
|---|---|
| `sheath/tools/files.py` | `read_file` (`start_line`, `line_numbers`), `list_files` (`pattern`, `_compile_glob`), `edit_file` and its helpers |
| `sheath/tools/search.py` | `context` (`_add_context`), `output` modes in the ripgrep backend and the dispatcher |
| `sheath/tools/_search_worker.py` | `output` modes in the Python fallback |
| `sheath/tools/registry.py` | Model-facing wrappers and their docstrings |
| `sheath/tools/__init__.py`, `sheath/bundles.py` | Export and bundle `edit_file`; `files` bundle instructions |
| `tests/tools/test_files.py`, `tests/tools/test_search.py` | Unit tests |
| `tests/test_registry.py`, `tests/test_bundles.py`, `tests/test_tool_ceilings.py`, `tests/test_quarantine.py` | Surface, ceiling and quarantine coverage |
| `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md` | Documentation |

---

### Task 1: 1-based `read_file`

**Files:** Modify `sheath/tools/files.py`, `sheath/tools/registry.py`, `README.md`, `CHANGELOG.md`. Test `tests/tools/test_files.py`, `tests/test_tool_ceilings.py`, `tests/tools/test_search.py`.

**Interfaces:**
- Produces: `read_file(session, path, start_line: int = 1, limit: int = 2000, max_chars: int = 50_000) -> str`

- [x] Replace `offset` with `start_line`. Iterate with `enumerate(..., start=1)`, skip `lineno < start_line`, and stop at `start_line + limit`. A `start_line` below 1 raises `ValueError` mentioning "1-based". The truncation note says `start_line/limit`.
- [x] Update the wrapper signature and docstring to state the shared numbering.
- [x] Port the existing tests to 1-based arguments, covering limit and start_line, past EOF, the bare-`\r` file and the CRLF chunk boundary.
- [x] Add `test_search_hit_line_feeds_read_file_directly`.
- [x] README tool table and an **Unreleased / Breaking** CHANGELOG entry.

### Task 2: `search(context=...)`

**Files:** Modify `sheath/tools/search.py`, `sheath/tools/registry.py`. Test `tests/tools/test_search.py`.

**Interfaces:**
- Produces: `search(..., context: int = 0)`. With `context` > 0, each hit has `before: list[str]` and `after: list[str]`.
- Produces: `_add_context(root: Path, hits: list[dict], context: int) -> None` (mutates hits)

- [x] Clamp `context` to `0..10` (`_MAX_CONTEXT`).
- [x] After either backend returns, group hits by file and open each with `safe_path` plus `O_NOFOLLOW | O_NONBLOCK`, regular files only.
- [x] Read each file once with `iter_bounded_lines`, stopping at the last needed line. Bound each neighbour with `_snippet(line, 0)`.
- [x] A file that cannot be opened gets no context rather than an error.
- [x] Apply `_fit` after context, so the result budget includes it.
- [x] Tests:
  - neighbours;
  - file edges;
  - adjacent matching lines;
  - no keys when `context=0`;
  - the cap;
  - long neighbours clipped;
  - backend agreement.

*Executed differently:* the backend-agreement test first compared lists in order and failed, because ripgrep's cross-file order varies. It now sorts by `(file, line)`. The base commit had already fixed the same flake in an older test.

**Checkpoint:** commit `583b4d3` (Tasks 1–2).

### Task 3: `read_file(line_numbers=...)`

**Files:** Modify `sheath/tools/files.py`, `sheath/tools/registry.py`. Test `tests/tools/test_files.py`, `tests/test_tool_ceilings.py`.

- [x] Prefix each kept line with `f"{lineno:>6}\t"` before it counts toward `max_chars`.
- [x] The wrapper docstring says to leave it off when copying text for `edit_file`.
- [x] Test the exact prefix format. Add a ceiling case with `line_numbers=True` on the 5 MB single-line file.

### Task 4: `edit_file`

**Files:** Modify `sheath/tools/files.py`, `sheath/tools/registry.py`, `sheath/tools/__init__.py`, `sheath/bundles.py`. Test `tests/tools/test_files.py`, `tests/test_registry.py`, `tests/test_bundles.py`, `tests/test_tool_ceilings.py`, `tests/test_quarantine.py`.

**Interfaces:**
- Produces: `edit_file(session, path, old: str, new: str, replace_all: bool = False) -> str`
- Internal:
  - `_line_of(text, index) -> int`, using `read_file`'s line breaks;
  - `_match_lines(text, old, limit) -> list[int]`;
  - `_no_match_message(path, text, old) -> str`;
  - `_edit_preview(text, start, length) -> str`;
  - `_replace_atomically(target, data, mode) -> None`.
- Constants: `_MAX_EDIT_BYTES = 10 MiB`, `_EDIT_PREVIEW_LINES = 3`, `_EDIT_PREVIEW_CHARS = 4000`, `_MAX_REPORTED_LINES = 5`.

- [x] Validate first: an empty `old`, and `old == new`, are errors.
- [x] Under `file_io`:
  - `safe_path`;
  - the `is_protected` refusal;
  - `_open_checked` read-only;
  - an `fstat` size cap, re-checked while reading in case the file grows.
- [x] Refuse NUL bytes in the first 8 KiB, and non-UTF-8 content.
- [x] Decode without newline translation.
- [x] For no match, give the case-insensitive line, else the whitespace-insensitive line, else a plain message.
- [x] For several matches without `replace_all`, give the count and the first five distinct lines holding a match.
- [x] Write: `mkstemp` beside the target, `fchmod` to the original mode bits, write, `fsync`, then `os.replace`; unlink the temp file on any failure.
- [x] Reply: the replacement count, then a numbered preview of the first edit with three lines either side, capped at 4,000 characters. Split lines with `_LINE_RE`, not `str.splitlines()`, which also breaks on `\f`, `\x1c` and other separators.
- [x] Wire it in: a registry wrapper (listed after `write_file`), the package export, the `files` bundle, and bundle instructions that steer partial changes to `edit_file`.
- [x] Tests:
  - a unique match and the numbered preview;
  - ambiguous match lines;
  - `replace_all`;
  - deletion;
  - near-match hints;
  - empty or identical text;
  - CRLF and permissions kept, with no temp file left;
  - bare-`\r` line numbers;
  - protected directories;
  - an escaping link, binary and non-UTF-8 files;
  - the size cap;
  - an escaping path;
  - preview bound;
  - the quarantine refusal;
  - the registry and bundle name lists;
  - a ceiling case.

*Executed differently:*
- **Hang.** The first ceiling case hung the suite: `"x"` occurs 5,000,000 times in the hostile file, and the ambiguity error computed a line number per match, each by scanning from the top. `_match_lines` now locates at most five distinct lines and skips to the next line break after each match.
- **Ceiling case.** The case now uses `replace_all=True`, which returns rather than raising, so it measures the large preview path.
- **Link test.** The test originally expected an in-root link to be refused. `safe_path` follows in-root links for every file tool, so the test now asserts that a link escaping the root is refused and the outside file is untouched.

### Task 5: `list_files(pattern=...)`

**Files:** Modify `sheath/tools/files.py`, `sheath/tools/registry.py`. Test `tests/tools/test_files.py`, `tests/test_tool_ceilings.py`.

**Interfaces:**
- Produces: `list_files(session, path=".", pattern: str | None = None) -> list[str]`
- Produces: `_compile_glob(pattern) -> re.Pattern`

- [x] Translate the glob:
  - `**/` → `(?:.*/)?`;
  - `**` → `.*`;
  - `*` → `[^/]*`;
  - `?` → `[^/]`;
  - `[...]` → a class, with `[!` → `[^`;
  - everything else escaped.
- [x] Reject absolute patterns and any `..` segment.
- [x] Apply the pattern with `fullmatch` to each walked path relative to `path`. A single-file `path` matches on its name.
- [x] The overflow note suggests narrowing the pattern.
- [x] Tests:
  - `*` within one folder;
  - `**` across folders;
  - relative to `path`;
  - single file;
  - escaping patterns;
  - a ceiling case.

### Task 6: `search(output=...)`

**Files:** Modify `sheath/tools/search.py`, `sheath/tools/_search_worker.py`, `sheath/tools/registry.py`. Test `tests/tools/test_search.py`, `tests/test_tool_ceilings.py`.

**Interfaces:**
- Produces: `search(..., output: str = "hits")`. `"files"` returns `[{"file"}]` and `"count"` returns `[{"file", "count"}]`.
- Produces: `scan(..., output="hits")` in the worker, and an `output` field in the worker request.

- [x] Reject an unknown `output`, and `context` with any output other than `"hits"`.
- [x] ripgrep:
  - `"files"` adds `--max-count 1`;
  - for the file outputs, count match events per file;
  - on the first event from a new file past the cap, stop early and kill rg;
  - an invalid regex still raises when nothing was found.
- [x] Fallback:
  - count matches per file, and break after the first for `"files"`;
  - after each file, append its entry and stop at the cap.
- [x] Tests:
  - each output;
  - file cap rather than line cap;
  - backend agreement, sorted;
  - argument errors;
  - an invalid regex in count mode;
  - two ceiling cases.

**Checkpoint:** commit `2b07d84` (Tasks 3–6).

### Task 7: Documentation and wrapper coverage

**Files:** Modify `README.md`, `CHANGELOG.md`, `docs/ROADMAP.md`. Test `tests/test_registry.py`.

- [x] README:
  - tool table rows for every changed tool;
  - `edit_file` in the trust-zone, quarantine and bounded-results paragraphs.
- [x] CHANGELOG: an Unreleased entry for the additions.
- [x] ROADMAP:
  - a Delivered entry with the deferrals and the shell decision;
  - a carry-over for bare-`\r` numbering;
  - the review date.
- [x] `test_wrapped_file_tools_forward_every_parameter`: the `build_tools` callables pass `start_line`, `line_numbers`, `pattern`, `context`, `output` and `replace_all` through.

**Checkpoint:** commit `8945975`. This spec and plan follow in their own commit.

## Execution notes

- **Final suite:** 887 tests; 31 skipped; the same 4 environment failures as the baseline. Ruff clean.
- **Planning.** Planning was done in conversation, without the brainstorming and planning skills, and this record was written afterwards.
- **Commit trailers.** The implementation commits first carried assistant attribution trailers, against the naming constraint above. The branch was rewritten to drop them before merge; the hashes in this plan are the rewritten ones.
- **Worktree.** The work was done in a shallow clone, which only tracks `main`. Pushing a feature branch needs an explicit fetch refspec for it before the branch can track its remote.
