# goodmem-agent-framework

[GoodMem](https://goodmem.ai) integration for the
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework).

This package gives Agent Framework agents persistent, semantic long-term memory
backed by a GoodMem server. It exposes:

- **`GoodMemClient`** — an async REST client for the GoodMem v1 API.
- **`GoodMemContextProvider`** — a `BaseContextProvider` that automatically
  retrieves relevant memories before each agent run and stores conversations
  afterwards.
- **`create_goodmem_tools`** — a factory that returns ready-to-use function
  tools so the model itself can manage spaces and memories.

> **Renamed on PyPI.** This package was previously published as
> `agent-framework-goodmem` (last version on that name: 0.2.0). It moved into
> the PAIR Systems PyPI organisation under the `goodmem-<framework>` naming
> used by goodmem-adk and goodmem-semantic-kernel. The import name is
> unchanged: `import agent_framework_goodmem` keeps working. Both
> distributions ship the same `agent_framework_goodmem` package, so remove the
> old one first or they overwrite each other's files:
>
> ```bash
> pip uninstall -y agent-framework-goodmem && pip install goodmem-agent-framework
> ```

## Installation

```bash
pip install goodmem-agent-framework
```

For local development:

```bash
pip install -e .
```

## Quickstart

```python
import asyncio
from agent_framework_goodmem import GoodMemClient, create_goodmem_tools

async def main():
    client = GoodMemClient(
        base_url="https://localhost:8080",
        api_key="gm_xxxxxxxxxxxxxxxxxxxxxxxx",
        verify_ssl=False,  # self-signed local server
    )

    embedders = await client.list_embedders()
    embedder_id = embedders[0]["embedderId"]

    space = await client.create_space(name="quickstart", embedder_id=embedder_id)
    space_id = space["spaceId"]

    await client.create_memory(
        space_id=space_id,
        text_content="The capital of France is Paris.",
    )

    results = await client.retrieve_memories(
        query="What is the capital of France?",
        space_ids=[space_id],
        max_results=3,
        wait_for_indexing=True,
    )
    print(results)

    await client.close()

asyncio.run(main())
```

## Available tools

`create_goodmem_tools(client)` returns the following 11 function tools:

| Tool | Description |
|------|-------------|
| `goodmem_list_embedders` | List embedder models available on the server |
| `goodmem_list_spaces` | List all spaces accessible to the API key |
| `goodmem_get_space` | Fetch a space by ID |
| `goodmem_create_space` | Create a space, or reuse a same-name space that uses the same embedder |
| `goodmem_update_space` | Update a space's name/labels/visibility |
| `goodmem_delete_space` | Delete a space |
| `goodmem_create_memory` | Store text or a file as a memory |
| `goodmem_list_memories` | List memories in a space |
| `goodmem_retrieve_memories` | Semantic retrieval, with optional reranker/LLM |
| `goodmem_get_memory` | Fetch a memory by ID (with original content) |
| `goodmem_delete_memory` | Delete a memory |

### Retrieval options

`goodmem_retrieve_memories` (and `GoodMemClient.retrieve_memories`) accept the
GoodMem post-processor parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `reranker_id` | UUID | Reranker model to improve result ordering |
| `llm_id` | UUID | LLM used to generate a contextual abstract reply |
| `relevance_threshold` | 0–1 | Minimum score for including a result |
| `llm_temperature` | 0–2 | Creativity for the LLM post-processor |
| `max_results` | int | Cap on returned chunks |
| `chronological_resort` | bool | Reorder results by memory creation time |

### Retrieval results and server statuses

`retrieve_memories` returns a dict (the tool returns the same dict as JSON):

| Key | Description |
|-----|-------------|
| `success` | `true` unless the request itself failed (see [Errors](#errors)) |
| `results` | Matching chunks: `chunkId`, `chunkText`, `memoryId`, `relevanceScore`, `memoryIndex` |
| `memories` | Memory definitions (when `include_memory_definition` is true) |
| `totalResults` | Number of entries in `results` |
| `resultSetId` | The server's result set ID |
| `abstractReply` | The LLM's reply, when an `llm_id` was given and it succeeded |
| `partial` | `true` when the server reported a problem during this retrieval |
| `statuses` | The problems the server reported, in stream order; empty when `partial` is `false` |
| `message` | A readable summary when `partial` is `true`, or when `wait_for_indexing` gave up after 60 seconds |

Each entry in `statuses` is `{"code", "message", "details"}`, exactly as the
server sent it. These rules decide what counts as a problem:

- `FEATURE_DISABLED` and `LLM_CAPABILITY_INFERRED` are informational (an
  optional feature you did not configure). They are left out of `statuses`
  and never set `partial`.
- A code this package does not recognize is reported with `code: "UNKNOWN"`
  and the server's code in `originalCode`, and sets `partial`. It is never
  dropped and never raises.
- A problem with hits (for example a nonexistent `reranker_id`): the hits are
  returned, with `partial: true` and the statuses.
- A problem with no hits: an empty `results`, with `partial: true` and the
  statuses. Nothing is raised.

With `wait_for_indexing=True`, polling stops as soon as the server reports a
problem, so a nonexistent reranker or LLM is reported at once instead of after
60 seconds. When the reranker fails (`RERANKING_FAILED`, or `NOT_FOUND` naming
the reranker) the server still returns the vector search's hits: their
`relevanceScore` values are vector-search scores, not reranker scores, and
`message` says so.

`GoodMemContextProvider` logs a WARNING with the statuses when a retrieval is
partial, and still uses any chunks that came back.

### Space reuse

`create_space` (and `goodmem_create_space`) looks for a space with exactly the
requested name, across every page of the space listing:

- **No such space:** a new one is created (`reused: false`).
- **One space, same embedder:** it is reused (`reused: true`). `embedderId`
  and `embedderIds` report the space's real embedder (`embedderId` is `null`
  for a space with several), and `chunkingConfig` its real chunking
  configuration (which may differ from the one requested).
- **One space, different embedder:** nothing is created. The result has
  `success: false`, an `error` naming the space, its ID and both embedders,
  plus `existingSpaceId`, `existingEmbedderIds` and `requestedEmbedderId`.
- **Several spaces with that name:** nothing is created. The result has
  `success: false`, an `error` listing each space and its embedders, and
  `existingSpaceIds`.

If no `embedder_id` is given, an existing space with that name is reused
whatever its embedder, and a new space uses the server's first embedder.

### Errors

Tools never raise: a failure is returned as `{"success": false, "error": ...}`.
When the server rejects a request, `GoodMemClient` raises
`httpx.HTTPStatusError` whose message includes the server's own error text,
for example `HTTP 409 Conflict for POST /v1/spaces: A space with this name
already exists`, and the tools pass that text on in `error`.

## Context provider

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_goodmem import GoodMemClient, GoodMemContextProvider

client = GoodMemClient(base_url="https://localhost:8080", api_key="gm_...", verify_ssl=False)

provider = GoodMemContextProvider(
    client=client,
    space_id=space_id,
    max_results=5,
    store_conversations=True,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="memory-agent",
    instructions="You are a helpful assistant with persistent memory.",
    context_providers=[provider],
)
```

## Running the tests

The offline tests replay retrieval streams captured from a live GoodMem server
(`tests/fixtures/`) against a fake server, and need no credentials:

```bash
pip install -e ".[dev]"
pytest -v tests/test_retrieval_statuses.py tests/test_space_reuse.py
```

The live integration tests in `tests/test_goodmem_integration.py` run against
a real server and are skipped when `GOODMEM_API_KEY` is not set. They create
and delete their own spaces. `GOODMEM_EMBEDDER_ID`, `GOODMEM_RERANKER_ID`,
`GOODMEM_LLM_ID` and `GOODMEM_PDF_PATH` pick the models and file they use.

```bash
export GOODMEM_API_KEY=gm_xxxxxxxxxxxxxxxxxxxxxxxx
export GOODMEM_BASE_URL=https://localhost:8080
pip install -e ".[dev]"
pytest -m integration -v tests/test_goodmem_integration.py
```

### Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`,
on Python 3.10, 3.11, 3.12 and 3.13. It installs the package with
`pip install -e ".[dev]"`, compiles every module and imports the public API
(no linter is configured), fails if a GoodMem API key is committed, and runs
`python -m pytest -v` with no API key set, so the live tests are skipped.

## Changes

See [CHANGELOG.md](CHANGELOG.md).
