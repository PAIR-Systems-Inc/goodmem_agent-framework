# Changelog

## 0.2.1

### Changed

- **The distribution is renamed from `agent-framework-goodmem` to
  `goodmem-agent-framework`.** It moved into the PAIR Systems PyPI
  organisation under the `goodmem-<framework>` naming used by goodmem-adk and
  goodmem-semantic-kernel. Install it with `pip install goodmem-agent-framework`.
  The import package is unchanged (`import agent_framework_goodmem`), and
  the API and behaviour are the same as 0.2.0. `agent-framework-goodmem`
  stays at 0.2.0. Both distributions ship the same `agent_framework_goodmem`
  package, so uninstall the old one first:
  `pip uninstall -y agent-framework-goodmem && pip install goodmem-agent-framework`.
- `__version__` is looked up by the new distribution name (looking it up by
  the import name would find no installed distribution and report `0.0.0`).

## 0.2.0

### Changed (behaviour)

- **Retrieval reports the server's problem statuses.** `retrieve_memories`
  and `goodmem_retrieve_memories` results now carry `partial` and
  `statuses` (`code`, `message`, `details`). A nonexistent reranker or LLM,
  a failed embedder, or any other problem the server streams is no longer
  discarded: hits that came back are still returned, with `partial: true`;
  with no hits, an empty result with `partial: true` is returned. Nothing is
  raised. `FEATURE_DISABLED` and `LLM_CAPABILITY_INFERRED` are informational
  and never set `partial`; unrecognized codes are reported as `UNKNOWN`
  (with `originalCode`) and set `partial`. When the reranker failed,
  `message` says the scores are vector-search scores.
- **`wait_for_indexing` stops polling when the server reports a problem.**
  Previously a nonexistent reranker or LLM on an empty space polled for 60
  seconds and then blamed indexing.
- **Space reuse by name requires a matching embedder.** `create_space` reuses
  a same-name space only if its embedder is the requested one, and reports
  the space's real `embedderId` (it used to echo the requested one). A
  different embedder, or several spaces with the name, now returns
  `success: false` with an error naming the space, its ID and both embedders,
  and creates nothing. With no `embedder_id`, a same-name space is reused
  whatever its embedder. Reused results also carry `embedderIds` and the
  space's real `chunkingConfig`.
- **The name lookup reads every page** of the space listing, using the
  server's `name_filter` and re-checking the exact name (the filter is a
  case-insensitive glob).
- **HTTP errors keep the server's message** (for example `A space with this
  name already exists`, `Embedder not found`, `Space not found: <id>`);
  `GoodMemClient` still raises `httpx.HTTPStatusError`.
- `GoodMemContextProvider` logs a WARNING with the statuses when a retrieval
  is partial.

### Fixed

- `GoodMemContextProvider.before_run` raised `TypeError` whenever it found
  memories, because `Message(text=...)` no longer exists in
  agent-framework-core 1.x. It now builds the message with `contents`.

### Added

- Offline tests replaying NDJSON streams captured from a live server, and
  live tests for the status contract and space reuse.
- CI workflow (`.github/workflows/ci.yml`).

### Removed

- The API key default in `tests/test_goodmem_integration.py`; the live tests
  now skip when `GOODMEM_API_KEY` is not set.

## 0.1.0

- Initial release.
