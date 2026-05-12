# agent-framework-goodmem

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

## Installation

```bash
pip install agent-framework-goodmem
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
| `goodmem_create_space` | Create a space (idempotent by name) |
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

## Running the integration tests

```bash
export GOODMEM_API_KEY=gm_xxxxxxxxxxxxxxxxxxxxxxxx
export GOODMEM_BASE_URL=https://localhost:8080
pip install -e ".[dev]"
pytest -m integration -v tests/test_goodmem_integration.py
```
