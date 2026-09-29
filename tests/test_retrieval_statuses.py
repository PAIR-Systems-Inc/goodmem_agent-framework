# Copyright (c) Microsoft. All rights reserved.

"""Offline tests for how retrieval reports the server's ``status`` events.

These replay NDJSON streams captured from a live GoodMem server and check the
retrieval status contract:

- Q1: ``FEATURE_DISABLED`` and ``LLM_CAPABILITY_INFERRED`` are informational
  and never mark a result partial.
- Q3: an unrecognized code is surfaced as ``UNKNOWN`` and marks the result
  partial; it never crashes and is never dropped.
- Q4a: a problem with hits returns the hits, ``partial: True`` and the statuses.
- Q4b: a problem with no hits returns an empty result, ``partial: True`` and
  the statuses, without raising and without polling for indexing.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest

from goodmem_agent_framework import GoodMemContextProvider, create_goodmem_tools

from ._fakes import EMBEDDER_A, LLM_ID, RERANKER_ID, FakeGoodMem, load_stream

BAD_ID = "00000000-0000-0000-0000-000000000000"


def _line(event: dict) -> str:
    return json.dumps(event) + "\n"


async def _retrieve(fake: FakeGoodMem, *streams: str, **kwargs):
    space = fake.add_space("statuses", EMBEDDER_A)
    fake.streams = list(streams)
    client = fake.client()
    try:
        started = time.monotonic()
        result = await client.retrieve_memories(query="What does the Agent Framework do?", space_ids=[space["spaceId"]], **kwargs)
        return result, time.monotonic() - started
    finally:
        await client.close()


def _codes(result: dict) -> list[str]:
    return [s["code"] for s in result["statuses"]]


# -- Q4a: problem + hits -------------------------------------------------------


async def test_bad_reranker_with_hits_returns_hits_partial_and_statuses(fake: FakeGoodMem) -> None:
    result, _ = await _retrieve(fake, load_stream("bad_reranker_hits"), reranker_id=BAD_ID)
    assert result["success"] is True
    assert result["totalResults"] == 2
    assert result["partial"] is True
    assert _codes(result) == ["NOT_FOUND", "RERANKING_FAILED"]
    for status in result["statuses"]:
        assert set(status) >= {"code", "message", "details"}
        assert "Reranker not found" in status["message"]
        assert status["details"] == {"reranker_id": BAD_ID}


async def test_bad_llm_with_hits_returns_hits_partial_and_statuses(fake: FakeGoodMem) -> None:
    result, _ = await _retrieve(fake, load_stream("bad_llm_hits"), llm_id=BAD_ID)
    assert result["success"] is True
    assert result["totalResults"] == 2
    assert result["partial"] is True
    assert _codes(result) == ["NOT_FOUND", "SUMMARIZATION_FAILED"]
    assert "LLM not found" in result["statuses"][0]["message"]
    assert "abstractReply" not in result


# -- Q4b: problem + no hits (and no 60 s indexing wait) -------------------------


@pytest.mark.timeout(10)
async def test_bad_reranker_empty_space_returns_promptly_with_statuses(fake: FakeGoodMem) -> None:
    result, elapsed = await _retrieve(fake, load_stream("bad_reranker_empty"), reranker_id=BAD_ID, wait_for_indexing=True)
    assert result["success"] is True
    assert result["results"] == []
    assert result["partial"] is True
    assert _codes(result) == ["NOT_FOUND", "RERANKING_FAILED"]
    assert len(fake.requests_to("POST", "/v1/memories:retrieve")) == 1
    assert elapsed < 2
    assert "indexing" not in result["message"]
    assert "Reranker not found" in result["message"]


@pytest.mark.timeout(10)
async def test_bad_llm_empty_space_returns_promptly_with_statuses(fake: FakeGoodMem) -> None:
    result, elapsed = await _retrieve(fake, load_stream("bad_llm_empty"), llm_id=BAD_ID, wait_for_indexing=True)
    assert result["results"] == []
    assert result["partial"] is True
    assert _codes(result) == ["NOT_FOUND", "SUMMARIZATION_FAILED"]
    assert len(fake.requests_to("POST", "/v1/memories:retrieve")) == 1
    assert elapsed < 2
    assert "LLM not found" in result["message"]


# -- Q1: informational notices -------------------------------------------------


async def test_feature_disabled_is_not_a_problem(fake: FakeGoodMem) -> None:
    # A working reranker with no LLM: the server streams FEATURE_DISABLED.
    assert "FEATURE_DISABLED" in load_stream("good_reranker_hits")
    result, _ = await _retrieve(fake, load_stream("good_reranker_hits"), reranker_id=RERANKER_ID)
    assert result["partial"] is False
    assert result["statuses"] == []
    assert result["totalResults"] == 2
    assert "message" not in result


async def test_llm_capability_inferred_is_not_a_problem(fake: FakeGoodMem) -> None:
    notice = _line({"status": {"code": "LLM_CAPABILITY_INFERRED", "message": "Inferred capabilities", "details": {}}})
    result, _ = await _retrieve(fake, notice + load_stream("good_llm_hits"), llm_id=LLM_ID)
    assert result["partial"] is False
    assert result["statuses"] == []
    assert result["abstractReply"]["text"]


async def test_clean_stream_has_no_statuses(fake: FakeGoodMem) -> None:
    result, _ = await _retrieve(fake, load_stream("plain_hits"))
    assert result["partial"] is False
    assert result["statuses"] == []


@pytest.mark.timeout(10)
async def test_informational_notices_do_not_stop_the_indexing_wait(fake: FakeGoodMem) -> None:
    notice = _line({"status": {"code": "FEATURE_DISABLED", "message": "no LLM configured", "details": {}}})
    empty = notice + load_stream("plain_empty")
    result, _ = await _retrieve(fake, empty, empty, load_stream("good_reranker_hits"), wait_for_indexing=True)
    assert len(fake.requests_to("POST", "/v1/memories:retrieve")) == 3
    assert result["totalResults"] == 2
    assert result["partial"] is False


# -- Q3: unknown codes -----------------------------------------------------------


async def test_unknown_code_is_surfaced_as_unknown_and_marks_partial(fake: FakeGoodMem) -> None:
    unknown = _line({"status": {"code": "QUOTA_SHIFTED", "message": "Something new happened", "details": {"k": "v"}}})
    result, _ = await _retrieve(fake, unknown + load_stream("plain_hits"))
    assert result["totalResults"] == 2
    assert result["partial"] is True
    assert result["statuses"] == [
        {"code": "UNKNOWN", "originalCode": "QUOTA_SHIFTED", "message": "Something new happened", "details": {"k": "v"}}
    ]


@pytest.mark.timeout(10)
async def test_unknown_code_without_hits_returns_promptly(fake: FakeGoodMem) -> None:
    odd = (
        _line({"status": {"message": "no code at all"}})
        + _line({"status": "not an object"})
        + _line({"status": {"code": ["not", "a", "string"], "message": "odd code"}})
    )
    result, elapsed = await _retrieve(fake, odd + load_stream("plain_empty"), wait_for_indexing=True)
    assert result["success"] is True
    assert result["partial"] is True
    assert _codes(result) == ["UNKNOWN", "UNKNOWN", "UNKNOWN"]
    assert [s["message"] for s in result["statuses"]] == ["no code at all", "not an object", "odd code"]
    assert len(fake.requests_to("POST", "/v1/memories:retrieve")) == 1
    assert elapsed < 2


async def test_statuses_in_sse_framing_are_parsed(fake: FakeGoodMem) -> None:
    sse = "".join(f"event: message\ndata: {line}\n\n" for line in load_stream("bad_llm_hits").splitlines())
    result, _ = await _retrieve(fake, sse, llm_id=BAD_ID)
    assert result["totalResults"] == 2
    assert _codes(result) == ["NOT_FOUND", "SUMMARIZATION_FAILED"]


# -- Reranker fallback scores ------------------------------------------------------


async def test_failed_reranker_says_scores_are_vector_scores(fake: FakeGoodMem) -> None:
    result, _ = await _retrieve(fake, load_stream("bad_reranker_hits"), reranker_id=BAD_ID)
    assert "Reranking was not applied" in result["message"]
    assert "vector-search scores" in result["message"]


async def test_reranker_not_found_alone_counts_as_not_reranked(fake: FakeGoodMem) -> None:
    only_not_found = load_stream("bad_reranker_hits").splitlines()[0] + "\n" + load_stream("plain_hits")
    result, _ = await _retrieve(fake, only_not_found, reranker_id=BAD_ID)
    assert _codes(result) == ["NOT_FOUND"]
    assert "Reranking was not applied" in result["message"]


async def test_llm_failure_does_not_claim_reranking_failed(fake: FakeGoodMem) -> None:
    result, _ = await _retrieve(fake, load_stream("bad_llm_hits"), reranker_id=RERANKER_ID, llm_id=BAD_ID)
    assert result["partial"] is True
    assert "Reranking was not applied" not in result["message"]


# -- Tool layer and context provider ---------------------------------------------------


async def test_retrieve_tool_passes_partial_and_statuses_through(fake: FakeGoodMem) -> None:
    space = fake.add_space("tool", EMBEDDER_A)
    fake.streams = [load_stream("bad_reranker_hits")]
    client = fake.client()
    tools = {t.name: t for t in create_goodmem_tools(client)}
    try:
        raw = await tools["goodmem_retrieve_memories"].func(query="x", space_ids=space["spaceId"], reranker_id=BAD_ID)
    finally:
        await client.close()
    result = json.loads(raw)
    assert result["success"] is True
    assert result["totalResults"] == 2
    assert result["partial"] is True
    assert _codes(result) == ["NOT_FOUND", "RERANKING_FAILED"]


async def test_context_provider_warns_on_partial_and_keeps_chunks(fake: FakeGoodMem, caplog) -> None:
    space = fake.add_space("provider", EMBEDDER_A)
    unknown = _line({"status": {"code": "EMBEDDER_FAILED", "message": "embedder 2 of 2 failed", "details": {}}})
    fake.streams = [unknown + load_stream("plain_hits")]
    client = fake.client()
    provider = GoodMemContextProvider(client=client, space_id=space["spaceId"], store_conversations=False)
    added: list = []
    context = SimpleNamespace(
        input_messages=[SimpleNamespace(text="What does the Agent Framework do?")],
        extend_messages=lambda source, messages: added.extend(messages),
    )
    try:
        with caplog.at_level(logging.WARNING, logger="goodmem_agent_framework._context_provider"):
            await provider.before_run(agent=None, session=None, context=context, state={})
    finally:
        await client.close()
    assert "partial" in caplog.text and "EMBEDDER_FAILED" in caplog.text
    assert len(added) == 1 and "Agent Framework" in added[0].text


# -- Server error bodies ---------------------------------------------------------------


async def test_retrieve_http_error_keeps_the_server_message(fake: FakeGoodMem) -> None:
    client = fake.client()
    tools = {t.name: t for t in create_goodmem_tools(client)}
    try:
        raw = await tools["goodmem_retrieve_memories"].func(query="x", space_ids=BAD_ID, wait_for_indexing=False)
    finally:
        await client.close()
    result = json.loads(raw)
    assert result["success"] is False
    assert f"Space not found: {BAD_ID}" in result["error"]
    assert "404" in result["error"]


async def test_context_provider_injects_retrieved_chunks(fake: FakeGoodMem) -> None:
    space = fake.add_space("provider", EMBEDDER_A)
    fake.streams = [load_stream("plain_hits")]
    client = fake.client()
    provider = GoodMemContextProvider(client=client, space_id=space["spaceId"], store_conversations=False)
    added: list = []
    context = SimpleNamespace(
        input_messages=[SimpleNamespace(text="What does the Agent Framework do?")],
        extend_messages=lambda source, messages: added.extend(messages),
    )
    try:
        await provider.before_run(agent=None, session=None, context=context, state={})
    finally:
        await client.close()
    assert len(added) == 1
    assert added[0].role == "user"
    assert added[0].text.startswith(GoodMemContextProvider.DEFAULT_CONTEXT_PROMPT)
    assert "open-source library for building AI agents" in added[0].text
