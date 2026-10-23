# docmaint: dependency-ordered documentation maintenance

`docmaint` keeps the repository's Markdown documentation current with one command. It reads a dependency graph of the documents, works out which documents are stale, and asks an agent to update each stale document in topological order. The agent may edit Markdown documents only. Code is never changed.

```sh
python -m tools.docmaint plan        # what is stale, in maintenance order
python -m tools.docmaint run         # maintain every stale document (backend: agy)
python -m tools.docmaint run --commit
```

## The dependency graph

The graph is `doc/docmap.toml`. Each maintained document declares:

| Key | Meaning |
|---|---|
| `path` | The document. It must be a `.md` file inside the repository. |
| `depends_on` | Other documents it builds on. They are maintained first, and an update to one of them makes this document stale. |
| `sources` | Files, folders (a trailing `/`) or glob patterns it describes. A change to any of them makes the document stale. |
| `notes` | Optional guidance for the agent about this document. |
| `max_lines` | Optional size target for the document. |

`[settings]` sets `language` (default `"en"`, professional technical English) and the default `backend`.

`python -m tools.docmaint discover --include doc tutorial ... --write` proposes a manifest from the documents' own links. A link to a Markdown file becomes `depends_on`; a link to a tracked file or folder becomes `sources`. Review the result: discovery drops back edges so that the graph is acyclic. `check` validates the manifest: every path exists, there are no cycles, and every entry is Markdown.

Historical records, such as `evidence/**/README.md`, should not be listed. They document what happened and must not be rewritten.

## Staleness

A document's input fingerprint hashes:
- the blob hashes, at HEAD, of every tracked file its `sources` match;
- the current text of every document in `depends_on`.

After a successful maintenance step, the fingerprint and commit are stored in `doc/docmap.state.json`. A document is stale when its fingerprint differs from the stored one. A document with no entry is stale and gets a full review. `baseline` records every document as current without running an agent; use it to adopt the tool on documentation already known to be correct.

The agent receives the document's inputs, a `git diff --stat`, and a diff file covering its inputs since the commit recorded at its last maintenance.

## Safety

1. **The agent never works in the repository.** Each run exports HEAD with `git archive` into a private scratch directory that has its own throwaway git history. The agent runs there, so it cannot reach the repository's working tree, index, refs or history through git.
2. **One document per agent call.** After each call, the scratch copy's `git status` must show no change except the target document. Any other change (code, other documents, new files, deletions, commits) is reverted, and the document is rejected.
3. **Only the target Markdown file is copied back.** It must be a regular file, `.md`, UTF-8, non-empty and at most 2 MiB. If the repository's copy has local edits, the document is skipped; local work is never overwritten.
4. **The repository is fingerprinted** (HEAD and `git status`) before and after every agent call. If it changed, the run stops.
5. **Commits only on request.** Nothing is committed unless `--commit` is given. Then only the updated documents and the state file are committed.

Documents are maintained against the committed state (HEAD). Commit source changes first.

Prompts, agent logs and a JSON report are written to `~/.cache/docmaint/<repo>/run-<time>/`.

## Writing standard

Every agent call is instructed to:
- write professional technical English, and rewrite text in other languages without losing facts;
- support every statement from the inputs, and never invent results;
- correct or remove outdated content;
- link to upstream documents instead of duplicating them;
- keep links valid;
- leave an accurate document unchanged.

## Backends

| Backend | Status | Invocation |
|---|---|---|
| `agy` | supported | `agy --print <prompt> --dangerously-skip-permissions --mode accept-edits --disable-slash-commands --print-timeout <timeout>s`, run in the scratch copy |
| `codex` | planned | stub in `backends.py` |
| `claude` | planned | stub in `backends.py` |

To add a backend, subclass `Backend` in `backends.py`, implement `command(prompt, workdir, timeout)` (and optionally `available()`), and register it in `BACKENDS`. Isolation, the file guard and the copy-back checks are in the runner, so they apply to every backend unchanged.

## Options

`run` accepts the following options.

| Option | Effect |
|---|---|
| `--only DOC ...` | Maintain only these documents. |
| `--force` | Maintain even when inputs are unchanged. |
| `--dry-run` | Same as `plan`: report which documents would be maintained, without running an agent. |
| `--commit` | Commit the updated documents and the state file. |
| `--backend`, `--model`, `--effort` | Agent selection. |
| `--timeout SECONDS` | Per-document limit (default 1800). |
| `--keep-scratch` | Keep the scratch copy for inspection. |

The exit status is non-zero when any document failed or was rejected.

## Tests

```sh
python -m unittest tools.docmaint.test_docmaint
```

The tests use a scripted fake backend. They cover topological order and cycle detection, propagation of staleness, rejection of edits to code, other documents, new files and commits, protection of local edits, backend failures, and `--commit`.
