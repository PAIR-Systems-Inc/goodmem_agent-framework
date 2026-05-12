# Copyright (c) Microsoft. All rights reserved.

"""Integration tests for the standalone agent-framework-goodmem package.

These tests hit a live GoodMem server and exercise both the low-level
``GoodMemClient`` and the high-level ``create_goodmem_tools`` factory across
all 11 supported operations.

Run with:
    GOODMEM_API_KEY=<key> GOODMEM_BASE_URL=<url> \
        python -m pytest tests/test_goodmem_integration.py -v -s -m integration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

import pytest
import pytest_asyncio

# Allow direct import without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_framework_goodmem import (  # noqa: E402
    GoodMemClient,
    create_goodmem_tools,
)

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "gm_g5xcse2tjgcznlg45c5le4ti5q")
GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
RERANKER_ID = os.environ.get("GOODMEM_RERANKER_ID", "019cfda4-7e2f-743c-9edb-e469a97b95c6")
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "019cfd9f-0963-76f9-b069-4cde19a64ba8")
PREFERRED_EMBEDDER_ID = os.environ.get(
    "GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b"
)
PDF_FILE_PATH = os.environ.get(
    "GOODMEM_PDF_PATH",
    "/home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf",
)

UNIQUE_SUFFIX = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
SPACE_NAME = f"af-goodmem-it-{UNIQUE_SUFFIX}"

_state: dict[str, str] = {}


pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    c = GoodMemClient(base_url=GOODMEM_BASE_URL, api_key=GOODMEM_API_KEY, verify_ssl=False)
    yield c
    await c.close()


@pytest.fixture(scope="module")
def tools(client: GoodMemClient):
    return {t.name: t for t in create_goodmem_tools(client)}


async def _invoke(tool, **kwargs):
    """Invoke an Agent Framework FunctionTool and parse its JSON result."""
    raw = await tool.func(**kwargs) if hasattr(tool, "func") else await tool(**kwargs)
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.integration
class TestGoodMemClient:
    """Direct ``GoodMemClient`` tests across every public method."""

    async def test_01_list_embedders(self, client: GoodMemClient) -> None:
        embedders = await client.list_embedders()
        print(f"\n[list_embedders] count={len(embedders)}")
        assert len(embedders) > 0
        # Prefer the configured embedder, otherwise fall back to last
        chosen = next(
            (e for e in embedders if (e.get("embedderId") or e.get("id")) == PREFERRED_EMBEDDER_ID),
            None,
        )
        embedder_id = (chosen or embedders[-1]).get("embedderId") or (chosen or embedders[-1]).get("id")
        assert embedder_id
        _state["embedder_id"] = embedder_id

    async def test_02_create_space(self, client: GoodMemClient) -> None:
        embedder_id = _state["embedder_id"]
        result = await client.create_space(name=SPACE_NAME, embedder_id=embedder_id)
        print(f"\n[create_space] {json.dumps(result, indent=2, default=str)}")
        assert result["success"]
        assert result.get("spaceId")
        _state["space_id"] = result["spaceId"]

    async def test_03_list_spaces(self, client: GoodMemClient) -> None:
        spaces = await client.list_spaces()
        ids = [s.get("spaceId") for s in spaces]
        print(f"\n[list_spaces] count={len(spaces)} includes_created={(_state['space_id'] in ids)}")
        assert _state["space_id"] in ids

    async def test_04_get_space(self, client: GoodMemClient) -> None:
        result = await client.get_space(_state["space_id"])
        print(f"\n[get_space] name={result['space'].get('name')}")
        assert result["success"]
        assert result["space"]["spaceId"] == _state["space_id"]

    async def test_05_update_space(self, client: GoodMemClient) -> None:
        new_name = SPACE_NAME + "-updated"
        result = await client.update_space(
            _state["space_id"],
            name=new_name,
            replace_labels={"created_by": "pytest"},
        )
        print(f"\n[update_space] {json.dumps(result, indent=2, default=str)}")
        assert result["success"]
        # Confirm via get_space
        again = await client.get_space(_state["space_id"])
        assert again["space"].get("name") == new_name
        _state["space_name"] = new_name

    async def test_06_create_memory_text(self, client: GoodMemClient) -> None:
        result = await client.create_memory(
            space_id=_state["space_id"],
            text_content=(
                "The Microsoft Agent Framework is an open-source library "
                "for building production AI agents in Python and .NET."
            ),
        )
        print(f"\n[create_memory_text] {json.dumps(result, indent=2)}")
        assert result["success"]
        assert result.get("memoryId")
        _state["text_memory_id"] = result["memoryId"]

    async def test_07_create_memory_pdf(self, client: GoodMemClient) -> None:
        if not os.path.isfile(PDF_FILE_PATH):
            pytest.skip(f"PDF not found at {PDF_FILE_PATH}")
        result = await client.create_memory(space_id=_state["space_id"], file_path=PDF_FILE_PATH)
        print(f"\n[create_memory_pdf] {json.dumps(result, indent=2)}")
        assert result["success"]
        assert result.get("contentType") == "application/pdf"
        _state["pdf_memory_id"] = result["memoryId"]

    async def test_08_list_memories(self, client: GoodMemClient) -> None:
        result = await client.list_memories(_state["space_id"])
        print(f"\n[list_memories] count={len(result['memories'])}")
        assert result["success"]
        ids = [m.get("memoryId") for m in result["memories"]]
        assert _state["text_memory_id"] in ids

    async def test_09_retrieve_memories(self, client: GoodMemClient) -> None:
        result = await client.retrieve_memories(
            query="Microsoft Agent Framework for AI agents",
            space_ids=[_state["space_id"]],
            max_results=5,
            wait_for_indexing=True,
        )
        print(f"\n[retrieve_memories] {result.get('totalResults')} chunks")
        assert result["success"]
        assert result["totalResults"] > 0

    async def test_10_retrieve_with_reranker_and_llm(self, client: GoodMemClient) -> None:
        """Exercise the post-processor parameters (reranker + LLM)."""
        result = await client.retrieve_memories(
            query="What does the Agent Framework do?",
            space_ids=[_state["space_id"]],
            max_results=3,
            wait_for_indexing=True,
            reranker_id=RERANKER_ID,
            llm_id=LLM_ID,
            relevance_threshold=0.0,
            llm_temperature=0.3,
            chronological_resort=False,
        )
        print(f"\n[retrieve_with_reranker_and_llm] abstract={bool(result.get('abstractReply'))}, results={result.get('totalResults')}")
        assert result["success"]

    async def test_11_get_memory(self, client: GoodMemClient) -> None:
        result = await client.get_memory(_state["text_memory_id"], include_content=True)
        print(f"\n[get_memory] status={result.get('memory', {}).get('processingStatus')}")
        assert result["success"]
        assert result["memory"]["memoryId"] == _state["text_memory_id"]

    async def test_12_delete_memory(self, client: GoodMemClient) -> None:
        result = await client.delete_memory(_state["text_memory_id"])
        print(f"\n[delete_memory] {json.dumps(result, indent=2)}")
        assert result["success"]
        if _state.get("pdf_memory_id"):
            r2 = await client.delete_memory(_state["pdf_memory_id"])
            assert r2["success"]

    async def test_13_delete_space(self, client: GoodMemClient) -> None:
        result = await client.delete_space(_state["space_id"])
        print(f"\n[delete_space] {json.dumps(result, indent=2)}")
        assert result["success"]


# -- Tool-layer smoke test (one round-trip through create_goodmem_tools) ----

@pytest.mark.integration
class TestGoodMemTools:
    """Smoke test: every registered tool can be invoked end-to-end."""

    async def test_all_tools_present(self, tools) -> None:
        expected = {
            "goodmem_list_embedders",
            "goodmem_list_spaces",
            "goodmem_get_space",
            "goodmem_create_space",
            "goodmem_update_space",
            "goodmem_delete_space",
            "goodmem_create_memory",
            "goodmem_list_memories",
            "goodmem_retrieve_memories",
            "goodmem_get_memory",
            "goodmem_delete_memory",
        }
        present = set(tools.keys())
        missing = expected - present
        assert not missing, f"Missing tools: {missing}"

    async def test_tools_roundtrip(self, tools) -> None:
        # 1. List embedders
        emb = await _invoke(tools["goodmem_list_embedders"])
        assert emb["success"] and emb["count"] > 0
        embedder_id = (
            next(
                (e for e in emb["embedders"] if (e.get("embedderId") or e.get("id")) == PREFERRED_EMBEDDER_ID),
                None,
            )
            or emb["embedders"][-1]
        )
        embedder_id = embedder_id.get("embedderId") or embedder_id.get("id")

        # 2. Create space
        space_name = f"af-goodmem-tools-{UNIQUE_SUFFIX}"
        cs = await _invoke(tools["goodmem_create_space"], name=space_name, embedder_id=embedder_id)
        assert cs["success"], cs
        space_id = cs["spaceId"]

        try:
            # 3. list_spaces / get_space / update_space
            ls = await _invoke(tools["goodmem_list_spaces"])
            assert ls["success"] and any(s["spaceId"] == space_id for s in ls["spaces"])

            gs = await _invoke(tools["goodmem_get_space"], space_id=space_id)
            assert gs["success"]

            us = await _invoke(
                tools["goodmem_update_space"],
                space_id=space_id,
                name=space_name + "-renamed",
                replace_labels_json=json.dumps({"created_by": "tools-test"}),
            )
            assert us["success"], us

            # 4. create_memory / list_memories
            cm = await _invoke(
                tools["goodmem_create_memory"],
                space_id=space_id,
                text_content="Agent Framework tools smoke test memory.",
            )
            assert cm["success"], cm
            memory_id = cm["memoryId"]

            lm = await _invoke(tools["goodmem_list_memories"], space_id=space_id)
            assert lm["success"] and any(m.get("memoryId") == memory_id for m in lm["memories"])

            # 5. retrieve_memories with reranker + LLM
            rm = await _invoke(
                tools["goodmem_retrieve_memories"],
                query="smoke test",
                space_ids=space_id,
                max_results=3,
                wait_for_indexing=True,
                reranker_id=RERANKER_ID,
                llm_id=LLM_ID,
                llm_temperature=0.2,
            )
            assert rm["success"], rm

            # 6. get_memory
            gm = await _invoke(tools["goodmem_get_memory"], memory_id=memory_id)
            assert gm["success"], gm

            # 7. delete_memory
            dm = await _invoke(tools["goodmem_delete_memory"], memory_id=memory_id)
            assert dm["success"], dm
        finally:
            # 8. delete_space (cleanup)
            ds = await _invoke(tools["goodmem_delete_space"], space_id=space_id)
            assert ds["success"], ds
