# Copyright (c) Microsoft. All rights reserved.

"""Fake GoodMem server for the offline tests.

``FakeGoodMem`` stands in for the GoodMem REST API behind an
``httpx.MockTransport``, so the real ``GoodMemClient`` and tool code run
unchanged.  Retrieval responses replay NDJSON streams captured from a live
GoodMem server (``tests/fixtures/*.ndjson``, server v1.0.323).
"""

from __future__ import annotations

import fnmatch
import json
import pathlib
import uuid
from typing import Any

import httpx

from goodmem_agent_framework import GoodMemClient

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

EMBEDDER_A = "019cfd11-6ea9-75c8-925c-6a202a517513"
EMBEDDER_B = "019fa4f8-360c-74f6-a63f-bbf55808c3c3"
RERANKER_ID = "019e6da0-8a5a-72b0-8656-04dddfb25762"
LLM_ID = "019cfd9f-0963-76f9-b069-4cde19a64ba8"


def load_stream(name: str) -> str:
    """Return a captured NDJSON retrieval stream by fixture name."""
    return (FIXTURES / f"{name}.ndjson").read_text()


class FakeGoodMem:
    """In-memory GoodMem server covering the endpoints the client calls.

    Space listing mirrors the live server: ``name_filter`` is a
    case-insensitive glob, results are paged with ``nextToken``, and a
    duplicate name on create is a 409 with the server's JSON error body.
    """

    def __init__(self, *, page_size: int = 2) -> None:
        self.page_size = page_size
        self.embedders: list[dict[str, Any]] = [
            {"embedderId": EMBEDDER_A, "displayName": "OpenAI Text Embedding 3 Small"},
            {"embedderId": EMBEDDER_B, "displayName": "quran-embedder"},
        ]
        self.spaces: list[dict[str, Any]] = []
        self.streams: list[str] = []
        self.hidden: set[str] = set()  # names left out of listings
        self.requests: list[httpx.Request] = []

    # -- setup helpers ---------------------------------------------------------

    def add_space(self, name: str, *embedder_ids: str) -> dict[str, Any]:
        space_id = str(uuid.uuid4())
        space = {
            "spaceId": space_id,
            "name": name,
            "spaceEmbedders": [
                {"spaceId": space_id, "embedderId": e, "defaultRetrievalWeight": 1.0} for e in embedder_ids
            ],
            "defaultChunkingConfig": {"recursive": {"chunkSize": 512, "chunkOverlap": 50}},
        }
        self.spaces.append(space)
        return space

    def requests_to(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def client(self) -> GoodMemClient:
        client = GoodMemClient(base_url="http://goodmem.test", api_key="test-key")
        client._http = httpx.AsyncClient(
            base_url="http://goodmem.test",
            headers={"X-API-Key": "test-key", "Content-Type": "application/json", "Accept": "application/json"},
            transport=httpx.MockTransport(self.handle),
        )
        return client

    # -- request handling ------------------------------------------------------

    @staticmethod
    def _error(status: int, message: str) -> httpx.Response:
        return httpx.Response(status, json={"error": message, "status": status, "timestamp": 1790700558465})

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/v1/embedders":
            return httpx.Response(200, json={"embedders": self.embedders})
        if request.method == "GET" and path == "/v1/spaces":
            return self._list_spaces(request)
        if request.method == "POST" and path == "/v1/spaces":
            return self._create_space(json.loads(request.content))
        if request.method == "POST" and path == "/v1/memories:retrieve":
            return self._retrieve(json.loads(request.content))
        return self._error(404, f"No route for {request.method} {path}")

    def _list_spaces(self, request: httpx.Request) -> httpx.Response:
        pattern = request.url.params.get("name_filter")
        spaces = [s for s in self.spaces if s["name"] not in self.hidden]
        if pattern is not None:
            spaces = [s for s in spaces if fnmatch.fnmatchcase(s["name"].lower(), pattern.lower())]
        offset = int(request.url.params.get("nextToken") or 0)
        page = spaces[offset : offset + self.page_size]
        next_token = str(offset + self.page_size) if offset + self.page_size < len(spaces) else None
        return httpx.Response(200, json={"spaces": page, "nextToken": next_token})

    def _create_space(self, payload: dict[str, Any]) -> httpx.Response:
        if any(s["name"] == payload["name"] for s in self.spaces):
            return self._error(409, "A space with this name already exists")
        embedder_ids = [e["embedderId"] for e in payload["spaceEmbedders"]]
        known = {e["embedderId"] for e in self.embedders}
        if not set(embedder_ids) <= known:
            return self._error(400, "Embedder not found")
        return httpx.Response(201, json=self.add_space(payload["name"], *embedder_ids))

    def _retrieve(self, payload: dict[str, Any]) -> httpx.Response:
        known = {s["spaceId"] for s in self.spaces}
        for key in payload["spaceKeys"]:
            if key["spaceId"] not in known:
                return self._error(404, f"Space not found: {key['spaceId']}")
        # Replay the queued streams in order; the last one repeats.
        text = self.streams.pop(0) if len(self.streams) > 1 else self.streams[0]
        return httpx.Response(200, text=text, headers={"Content-Type": "application/x-ndjson; charset=utf-8"})

