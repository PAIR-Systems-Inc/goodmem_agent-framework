# Copyright (c) Microsoft. All rights reserved.

"""Integration tests for the standalone goodmem-agent-framework package.

These tests hit a live GoodMem server and exercise both the low-level
``GoodMemClient`` and the high-level ``create_goodmem_tools`` factory across
all 11 supported operations, plus the retrieval status contract (problem
statuses from a nonexistent reranker or LLM) and space reuse by name.
They are skipped when ``GOODMEM_API_KEY`` is not set.

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

import httpx
import pytest
import pytest_asyncio

# Allow direct import without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from goodmem_agent_framework import (  # noqa: E402
    GoodMemClient,
    create_goodmem_tools,
)

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "")
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


BAD_ID = "00000000-0000-0000-0000-000000000000"

pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(not GOODMEM_API_KEY, reason="GOODMEM_API_KEY not set; live tests skipped"),
]


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


# -- Retrieval status contract (live) ------------------------------------------


async def _pick_embedder(client: GoodMemClient) -> str:
    embedders = await client.list_embedders()
    ids = [e.get("embedderId") or e.get("id") for e in embedders]
    return PREFERRED_EMBEDDER_ID if PREFERRED_EMBEDDER_ID in ids else ids[0]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def status_spaces(client: GoodMemClient):
    """(space with an indexed memory, empty space); both deleted afterwards."""
    embedder_id = await _pick_embedder(client)
    created: list[str] = []
    try:
        hits = await client.create_space(name=f"af-goodmem-status-hits-{UNIQUE_SUFFIX}", embedder_id=embedder_id)
        created.append(hits["spaceId"])
        empty = await client.create_space(name=f"af-goodmem-status-empty-{UNIQUE_SUFFIX}", embedder_id=embedder_id)
        created.append(empty["spaceId"])
        await client.create_memory(
            space_id=hits["spaceId"],
            text_content="The Microsoft Agent Framework is an open-source library for building AI agents.",
        )
        indexed = await client.retrieve_memories(
            query="Agent Framework", space_ids=[hits["spaceId"]], max_results=3, wait_for_indexing=True
        )
        assert indexed["totalResults"] > 0, indexed
        yield hits["spaceId"], empty["spaceId"]
    finally:
        for space_id in created:
            await client.delete_space(space_id)


@pytest.mark.integration
class TestRetrievalStatusesLive:
    """A nonexistent reranker or LLM: partial + statuses, hits kept, no indexing wait."""

    async def test_bad_reranker_with_hits(self, client: GoodMemClient, status_spaces) -> None:
        hits_space, _ = status_spaces
        result = await client.retrieve_memories(
            query="Agent Framework", space_ids=[hits_space], max_results=3, reranker_id=BAD_ID
        )
        codes = [s["code"] for s in result["statuses"]]
        print(f"\n[bad_reranker_hits] results={result['totalResults']} partial={result['partial']} codes={codes}")
        assert result["success"] is True
        assert result["totalResults"] > 0
        assert result["partial"] is True
        assert "RERANKING_FAILED" in codes or "NOT_FOUND" in codes
        assert all(s["message"] for s in result["statuses"])
        assert "vector-search scores" in result["message"]

    async def test_bad_reranker_empty_space_returns_promptly(self, client: GoodMemClient, status_spaces) -> None:
        _, empty_space = status_spaces
        started = time.monotonic()
        result = await client.retrieve_memories(
            query="Agent Framework", space_ids=[empty_space], max_results=3, reranker_id=BAD_ID, wait_for_indexing=True
        )
        elapsed = time.monotonic() - started
        print(f"\n[bad_reranker_empty] {elapsed:.2f}s partial={result['partial']} message={result.get('message')!r}")
        assert result["success"] is True
        assert result["results"] == []
        assert result["partial"] is True
        assert result["statuses"]
        assert elapsed < 10
        assert "indexing" not in result["message"]

    async def test_bad_llm_with_hits(self, client: GoodMemClient, status_spaces) -> None:
        hits_space, _ = status_spaces
        result = await client.retrieve_memories(
            query="Agent Framework", space_ids=[hits_space], max_results=3, llm_id=BAD_ID
        )
        codes = [s["code"] for s in result["statuses"]]
        print(f"\n[bad_llm_hits] results={result['totalResults']} partial={result['partial']} codes={codes}")
        assert result["totalResults"] > 0
        assert result["partial"] is True
        assert "SUMMARIZATION_FAILED" in codes or "NOT_FOUND" in codes

    async def test_bad_llm_empty_space_returns_promptly(self, client: GoodMemClient, status_spaces) -> None:
        _, empty_space = status_spaces
        started = time.monotonic()
        result = await client.retrieve_memories(
            query="Agent Framework", space_ids=[empty_space], max_results=3, llm_id=BAD_ID, wait_for_indexing=True
        )
        elapsed = time.monotonic() - started
        print(f"\n[bad_llm_empty] {elapsed:.2f}s partial={result['partial']}")
        assert result["results"] == []
        assert result["partial"] is True
        assert elapsed < 10

    async def test_working_reranker_is_not_partial(self, client: GoodMemClient, status_spaces) -> None:
        # Without an LLM the server streams FEATURE_DISABLED, which is informational.
        hits_space, _ = status_spaces
        result = await client.retrieve_memories(
            query="Agent Framework", space_ids=[hits_space], max_results=3, reranker_id=RERANKER_ID
        )
        print(f"\n[good_reranker] results={result['totalResults']} partial={result['partial']} statuses={result['statuses']}")
        assert result["totalResults"] > 0
        assert result["partial"] is False
        assert result["statuses"] == []

    async def test_tool_reports_partial(self, tools, status_spaces) -> None:
        hits_space, _ = status_spaces
        result = await _invoke(
            tools["goodmem_retrieve_memories"], query="Agent Framework", space_ids=hits_space, reranker_id=BAD_ID
        )
        assert result["success"] is True
        assert result["partial"] is True
        assert result["totalResults"] > 0


# -- Space reuse by name (live) ------------------------------------------------


async def _server_spaces_named(name: str) -> list[dict]:
    """Spaces named exactly *name*, read straight from the server."""
    async with httpx.AsyncClient(base_url=GOODMEM_BASE_URL, headers={"X-API-Key": GOODMEM_API_KEY}, verify=False) as http:
        resp = await http.get("/v1/spaces", params={"name_filter": name})
        resp.raise_for_status()
        return [s for s in resp.json().get("spaces", []) if s.get("name") == name]


@pytest.mark.integration
class TestSpaceReuseLive:
    """Same name is reused only with the same embedder; the real embedder is reported."""

    async def test_reuse_rules(self, client: GoodMemClient, tools) -> None:
        embedder_id = await _pick_embedder(client)
        embedders = await client.list_embedders()
        # Only used for a create request that must be refused; never indexed with.
        other_id = next((e["embedderId"] for e in embedders if e.get("embedderId") != embedder_id), None)
        if other_id is None:
            pytest.skip("only one embedder on the server")
        name = f"af-goodmem-reuse-{UNIQUE_SUFFIX}"
        first = await _invoke(tools["goodmem_create_space"], name=name, embedder_id=embedder_id)
        assert first["success"] and not first["reused"], first
        try:
            mismatch = await _invoke(tools["goodmem_create_space"], name=name, embedder_id=other_id)
            print(f"\n[reuse_mismatch] {mismatch.get('error')}")
            assert mismatch["success"] is False
            for expected in (name, first["spaceId"], embedder_id, other_id):
                assert expected in mismatch["error"]
            assert len(await _server_spaces_named(name)) == 1

            same = await _invoke(tools["goodmem_create_space"], name=name, embedder_id=embedder_id)
            actual = await client.get_space(first["spaceId"])
            real = [e["embedderId"] for e in actual["space"]["spaceEmbedders"]]
            print(f"\n[reuse_same] reused={same.get('reused')} embedderId={same.get('embedderId')} real={real}")
            assert same["success"] is True and same["reused"] is True
            assert same["spaceId"] == first["spaceId"]
            assert [same["embedderId"]] == real
            assert len(await _server_spaces_named(name)) == 1
        finally:
            await client.delete_space(first["spaceId"])

    async def test_server_error_body_is_surfaced(self, tools) -> None:
        result = await _invoke(tools["goodmem_create_space"], name=f"af-goodmem-bad-emb-{UNIQUE_SUFFIX}", embedder_id=BAD_ID)
        print(f"\n[bad_embedder] {result.get('error')}")
        assert result["success"] is False
        assert "Embedder not found" in result["error"]
