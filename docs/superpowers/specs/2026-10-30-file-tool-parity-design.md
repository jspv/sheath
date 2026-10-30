# File-tool parity for find → read → edit — design

Date: 2026-10-30
Status: implemented (written after the fact from the design discussion; see §7)

## 1. Why

The agent's loop is **search → read → analyze**: `search` to locate, `read_file` to load the
right slice, `run_python` to analyze. A review of the file tools against the tools mainstream
coding-agent harnesses give their models found these gaps:

1. **search and read_file disagree on line numbers.** `search` reports 1-based lines (ripgrep's
   `line_number`, and `enumerate(..., start=1)` in the Python fallback); `read_file`'s
   `offset` was 0-based. A hit on line 42 passed as `offset=42` started the read at line 43
   and missed the match. Neither tool description stated a convention.
2. **No targeted edit.** `write_file` truncates and rewrites, so changing one line means
   reading and resending the whole file: expensive in tokens, and a file longer than
   `read_file`'s character cap can be silently cut short by the round trip.
3. **Every hit needs a second call to see its surroundings.** `search` has no context lines.
4. **No file discovery by pattern.** `list_files` lists everything under a path, up to its cap.
5. **No file-level search answers.** "Which files mention X" and "how often" can only be
   answered by pulling line hits and counting them in context.
6. **Line numbers come from the model counting.** `read_file` output is unnumbered, so any
   line-addressed follow-up depends on the model counting lines, which it does badly.

## 2. Goals and non-goals

**Goals**

- G1. `search` and `read_file` share one 1-based line numbering, and a hit's `line` can be
  passed to `read_file` unchanged.
- G2. A part of a file can be changed without resending the whole file, and a failed edit
  never leaves the file changed or half-written.
- G3. A match and its surroundings can be seen in one call.
- G4. Files can be found by glob, and searches can answer at file level.
- G5. Every new parameter and tool keeps the existing guarantees: every path through
  `safe_path` under `file_io`, the protected directories refused for writes, regular files
  only, bounded memory, and a hard ceiling on every result.
- G6. ripgrep and the Python fallback return identical results for every new option.

**Non-goals**

- A general shell tool (§3.1).
- Line-range edits, a file-type filter for `search`, and paging past `max_matches` (§5).
- Backward compatibility for `read_file(offset=...)`. Existing callers do not need to keep
  working; the parameter is replaced, not deprecated.

## 3. Decisions

### 3.1 Purpose-built tools, not a shell

Coding-agent harnesses lean on a shell (`rg`, `sed -n`, `git`, build tools). Here a shell was
rejected:

- The threat model treats the model and fetched content as adversarial, and the model's
  capability set is the tool surface. A parent-side shell bypasses `safe_path` and the
  container tier entirely; quoting, redirects and substitution make allowlisting unreliable.
- What a shell would add for this agent is already covered: data processing (`sort`, `uniq`,
  `wc`, `jq`, `awk`) is a pandas or stdlib one-liner in `run_python`, and keeping data in
  handles rather than piping text into context is the design's central move.
- What coding agents mostly use a shell for (`git`, builds, package installs) is not this
  agent's job.

If a shell is ever wanted, it should be an opt-in tool that runs inside the container tier
with `run_python`'s limits (no network, read-only trust zones, rlimits, bounded output), and
is refused on the local tier.

### 3.2 `read_file(path, start_line=1, limit=2000, line_numbers=False)`

- `start_line` is 1-based and replaces `offset` (breaking). `start_line < 1` is an error.
- `line_numbers=True` prefixes each line with its number right-aligned to six columns and a
  tab, as `cat -n` does. Off by default, because the prefix gets in the way of copying exact
  text into `edit_file`'s `old`.
- Line breaks are unchanged: `\n`, `\r\n` and a bare `\r` each end a line.

### 3.3 `search(..., context=0, output="hits")`

- `context` (clamped to 0–10): each hit gains `before` and `after`, lists of up to that many
  neighbouring lines, each bounded like the hit `text`. The keys are absent when `context`
  is 0, so default results are unchanged.
- Context is read once per file *after* the scan, in the parent, rather than by either
  backend. This keeps ripgrep and the fallback identical: ripgrep's `-C` reports a
  neighbour that is itself a match only as a match, never as context. The read happens
  under `file_io`, so the file cannot change between the scan and this read.
- `output`:
  - `"hits"`: matching lines, as before.
  - `"files"`: each matching file once, as `{"file"}`.
  - `"count"`: each matching file with its matching-line count, as `{"file", "count"}`.

  For the file outputs `max_matches` caps files, not lines. ripgrep runs with `--max-count 1`
  for `"files"`. For `"count"`, match events are counted per file. ripgrep emits each file's
  events together, so stopping on the first file past the cap leaves every earlier count
  complete. `--json` cannot be combined with `--count`, so counting is done from the event
  stream.
- `context` with any output other than `"hits"` is an error.

### 3.4 `edit_file(path, old, new, replace_all=False)`

Anchored on exact text, not line numbers: the text either still matches or the call fails
loudly, while a stale or miscounted line range silently edits the wrong lines.

- `old` must occur exactly once unless `replace_all`. Zero or several occurrences are an
  error and the file is untouched.
  - **Several:** the error gives the count and the first five distinct lines holding a match.
    Only those five are located, because a file can hold millions of matches.
  - **None:** the error points at a case-insensitive match, or else one differing only in
    whitespace, by line number.
- Empty `old`, and `old == new`, are errors. Creating and rewriting files stay with
  `write_file`.
- Same refusals as `write_file`:
  - the protected directories (`handles/`, `inputs/`, `.scripts/`, `.sheath/`);
  - paths escaping the root, including through links.

  Additionally refused:
  - non-regular files and binary files (a NUL byte in the first 8 KiB);
  - non-UTF-8 files;
  - files over 10 MiB, since the whole file is held in memory.
- The file is read as bytes and decoded without newline translation, so its line endings are
  kept. The write goes to a temp file beside the target, `fsync`ed, given the original
  permission bits, then `os.replace`d into place.
- The reply gives the replacement count plus the edited lines with three lines of context,
  numbered as in §3.2 and capped at 4,000 characters. The model can confirm the edit without
  another read.

Arguments considered and rejected:

| Argument | Why not |
|---|---|
| `regex` | Escaping mistakes become silent wrong edits; regex rewrites belong in `run_python`. |
| `start_line`/`end_line` | Deferred, see §5. |
| `create` | `write_file` already creates; one way to create keeps the tools distinct. |
| `dry_run` | Doubles the calls; the numbered reply covers the need. |
| `expected_count` | `replace_all` plus the count in the error covers it with less schema. |
| `encoding` | Workspace text is UTF-8 everywhere else. |

### 3.5 `list_files(path=".", pattern=None)`

A glob parameter on `list_files` rather than a separate tool, to keep the surface small.

- `pattern` is matched against each path relative to `path`.
  - `*` and `?` stay within one path segment.
  - `**/` matches zero or more directories; `**` matches anything.
  - `[...]` is a character class, and `[!...]` its negation.
- The pattern is translated to a regex and applied to paths from the existing walk, so it
  can only narrow what `list_files` already lists. Links and `.sheath/` stay skipped.
- An absolute pattern, or one containing `..`, is an error.

### 3.6 Model-facing text

The `files` bundle instructions name `edit_file`. They also say that a hit's `line` can be
passed to `read_file` as `start_line` and that `context=N` shows the lines around a hit, and
they steer partial changes to `edit_file` rather than `write_file`. Each wrapper's docstring
is its tool description.

## 4. Compatibility

- **Breaking:** `read_file(offset=...)` is gone; use `start_line` (1-based).
- **Additive:** `edit_file` joins the `files` bundle, so hosts that enumerate tool names see a
  new one. `search` gains `context` and `output`. `list_files` gains `pattern`. `read_file`
  gains `line_numbers`. All defaults reproduce the old results.

## 5. Deferred, with the reason

- **Line-range edits.** These would be `start_line`/`end_line` on `edit_file`, verified
  against `old` so stale numbers fail loudly. Worth it only if the agent turns out to edit
  large blocks, where exact-text anchoring costs tokens. Now that `read_file` can number its
  lines, the numbers would come from the tools rather than from the model counting.
- **A file-type filter on `search`.** `glob` covers it, and the Python fallback would need
  its own type table to match ripgrep.
- **Paging past `max_matches`.** ripgrep's order across files is not fixed, so stable pages
  would need `--sort path`, which is single-threaded. Narrowing `path` or `glob`, or using
  `output="files"`, covers the need.

## 6. Testing

- **Unit tests per option:**
  - `start_line` bounds, past EOF, and below 1;
  - `line_numbers`;
  - `context` at file edges, adjacent matching lines, the cap, and long neighbouring lines;
  - `output="files"` and `"count"` with the file cap;
  - glob semantics, a single-file `path`, and escaping patterns;
  - each `edit_file` success and refusal path.
- **Backend agreement:** `context` and both file outputs give identical results on ripgrep
  and the fallback. Comparisons sort by file, because ripgrep's cross-file order is not fixed.
- **Composition:** a `search` hit's `line` fed to `read_file(start_line=...)` starts at the
  hit.
- **Wrappers:** the `build_tools` callables forward every new parameter.
- **Existing safety suites extended:**
  - the result ceiling: numbered reads, `edit_file` with `replace_all` on a 5 MB single-line
    file, a `list_files` glob, and both file outputs;
  - quarantine refuses `edit_file`.

## 7. Residual risks and notes

- **Bare-`\r` line numbering.** `read_file`, `edit_file` and the Python fallback treat a lone
  `\r` as a line break; ripgrep does not. On an old-Mac-style file a ripgrep hit's `line`
  does not match `read_file`. This predates the work; it is listed under the roadmap's
  carry-overs.
- **Edits follow in-root links.** A link inside the root to another file inside the root is
  followed, as `read_file` and `write_file` already do; only escapes are refused.
- **Provenance.** This spec was written after implementation, from the design discussion,
  without the brainstorming step. The plan alongside it records the steps as executed.
