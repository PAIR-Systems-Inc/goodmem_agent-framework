# Copyright (c) Microsoft. All rights reserved.

"""Offline tests for space reuse by name and for server error bodies.

A same-name space is reused only when its embedder is the requested one; the
result reports the space's real embedder.  A mismatch or an ambiguous name is
an error that names the space and both embedders, and nothing is created.
The name is looked up across every page of the listing.
"""

from __future__ import annotations

import json

from agent_framework_goodmem import create_goodmem_tools

from ._fakes import EMBEDDER_A, EMBEDDER_B, FakeGoodMem


async def _create_via_tool(fake: FakeGoodMem, **kwargs) -> dict:
    client = fake.client()
    tools = {t.name: t for t in create_goodmem_tools(client)}
    try:
        return json.loads(await tools["goodmem_create_space"].func(**kwargs))
    finally:
        await client.close()


def _spaces_named(fake: FakeGoodMem, name: str) -> list[dict]:
    return [s for s in fake.spaces if s["name"] == name]


async def test_same_name_different_embedder_is_an_error_and_creates_nothing(fake: FakeGoodMem) -> None:
    existing = fake.add_space("notes", EMBEDDER_A)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_B)
    assert result["success"] is False
    for expected in ("notes", existing["spaceId"], EMBEDDER_A, EMBEDDER_B):
        assert expected in result["error"]
    assert result["existingSpaceId"] == existing["spaceId"]
    assert result["existingEmbedderIds"] == [EMBEDDER_A]
    assert result["requestedEmbedderId"] == EMBEDDER_B
    assert len(_spaces_named(fake, "notes")) == 1
    assert fake.requests_to("POST", "/v1/spaces") == []


async def test_same_name_same_embedder_reuses_and_reports_real_embedder(fake: FakeGoodMem) -> None:
    existing = fake.add_space("notes", EMBEDDER_A)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is True
    assert result["reused"] is True
    assert result["spaceId"] == existing["spaceId"]
    assert result["embedderId"] == EMBEDDER_A
    assert result["embedderIds"] == [EMBEDDER_A]
    assert result["chunkingConfig"] == existing["defaultChunkingConfig"]
    assert fake.requests_to("POST", "/v1/spaces") == []


async def test_no_embedder_requested_reuses_and_reports_the_spaces_embedder(fake: FakeGoodMem) -> None:
    # The server's first embedder is EMBEDDER_A; the existing space uses B.
    existing = fake.add_space("notes", EMBEDDER_B)
    result = await _create_via_tool(fake, name="notes")
    assert result["success"] is True
    assert result["reused"] is True
    assert result["spaceId"] == existing["spaceId"]
    assert result["embedderId"] == EMBEDDER_B


async def test_no_embedder_requested_creates_with_the_first_embedder(fake: FakeGoodMem) -> None:
    result = await _create_via_tool(fake, name="fresh")
    assert result["success"] is True
    assert result["reused"] is False
    assert result["embedderId"] == EMBEDDER_A
    created = json.loads(fake.requests_to("POST", "/v1/spaces")[0].content)
    assert created["spaceEmbedders"][0]["embedderId"] == EMBEDDER_A


async def test_space_on_a_later_page_is_found(fake: FakeGoodMem) -> None:
    # One space per page: "notes" is neither on the first unfiltered page nor
    # on the first page of the (case-insensitive) name_filter results.
    fake.page_size = 1
    fake.add_space("notes-0", EMBEDDER_A)
    fake.add_space("Notes", EMBEDDER_A)
    fake.add_space("notes", EMBEDDER_B)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is False
    assert EMBEDDER_B in result["error"]
    assert fake.requests_to("POST", "/v1/spaces") == []


async def test_lookup_uses_name_filter_and_follows_next_token(fake: FakeGoodMem) -> None:
    fake.page_size = 1
    fake.add_space("Notes", EMBEDDER_A)  # the filter is case-insensitive
    target = fake.add_space("notes", EMBEDDER_A)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is True
    assert result["spaceId"] == target["spaceId"]
    listings = fake.requests_to("GET", "/v1/spaces")
    assert all(r.url.params.get("name_filter") == "notes" for r in listings)
    assert [r.url.params.get("nextToken") for r in listings] == [None, "1"]


async def test_case_insensitive_filter_match_is_not_reused(fake: FakeGoodMem) -> None:
    other = fake.add_space("NOTES", EMBEDDER_B)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is True
    assert result["reused"] is False
    assert result["spaceId"] != other["spaceId"]


async def test_glob_characters_in_the_name_are_not_sent_as_a_filter(fake: FakeGoodMem) -> None:
    fake.add_space("notes-2025", EMBEDDER_B)
    result = await _create_via_tool(fake, name="notes-*", embedder_id=EMBEDDER_A)
    assert result["success"] is True
    assert result["reused"] is False
    assert all("name_filter" not in r.url.params for r in fake.requests_to("GET", "/v1/spaces"))


async def test_ambiguous_name_is_an_error_and_creates_nothing(fake: FakeGoodMem) -> None:
    first = fake.add_space("notes", EMBEDDER_A)
    second = fake.add_space("notes", EMBEDDER_B)
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is False
    for expected in ("notes", first["spaceId"], second["spaceId"], EMBEDDER_A, EMBEDDER_B):
        assert expected in result["error"]
    assert result["existingSpaceIds"] == [first["spaceId"], second["spaceId"]]
    assert fake.requests_to("POST", "/v1/spaces") == []


async def test_server_409_body_reaches_the_caller(fake: FakeGoodMem) -> None:
    # The space exists but is hidden from the listing (e.g. created between
    # the lookup and the create); the server rejects the duplicate with 409.
    fake.add_space("notes", EMBEDDER_A)
    fake.hidden.add("notes")
    result = await _create_via_tool(fake, name="notes", embedder_id=EMBEDDER_A)
    assert result["success"] is False
    assert "409" in result["error"]
    assert "A space with this name already exists" in result["error"]


async def test_server_400_body_reaches_the_caller(fake: FakeGoodMem) -> None:
    result = await _create_via_tool(fake, name="bad", embedder_id="00000000-0000-0000-0000-000000000000")
    assert result["success"] is False
    assert "Embedder not found" in result["error"]
