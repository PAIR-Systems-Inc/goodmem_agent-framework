# Copyright (c) Microsoft. All rights reserved.

"""Low-level HTTP client for the GoodMem API.

This module provides ``GoodMemClient``, a thin async wrapper around the
GoodMem REST API.  It handles authentication, URL normalization, and
JSON serialization so that the tool layer can stay focused on schema
definitions and result formatting.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx


# -- MIME helpers --------------------------------------------------------------

_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _guess_mime(extension: str) -> str:
    """Return MIME type for a file extension, falling back to octet-stream."""
    return _MIME_TYPES.get(extension.lower().lstrip("."), "application/octet-stream")


# -- Error helpers -------------------------------------------------------------


def _server_error_detail(resp: httpx.Response) -> str:
    """Extract the server's own error message from an error response body."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text.strip()[:500]
    if isinstance(body, dict):
        if isinstance(body.get("error"), str):
            return body["error"]
        if isinstance(body.get("message"), str):
            return body["message"]
        errors = body.get("errors")
        if isinstance(errors, list):
            parts = []
            for err in errors:
                if isinstance(err, dict):
                    field = err.get("field")
                    message = err.get("message", "")
                    parts.append(f"{field}: {message}" if field else str(message))
                else:
                    parts.append(str(err))
            return "; ".join(parts)
    return json.dumps(body)[:500]


def _raise_for_status(resp: httpx.Response) -> None:
    """Like ``resp.raise_for_status()``, but keep the server's error body.

    The raised ``httpx.HTTPStatusError`` carries the server's own message
    (e.g. ``A space with this name already exists``) so it reaches the caller
    instead of only the status line.
    """
    if resp.is_success:
        return
    request = resp.request
    detail = _server_error_detail(resp)
    message = f"HTTP {resp.status_code} {resp.reason_phrase} for {request.method} {request.url.path}"
    if detail:
        message += f": {detail}"
    raise httpx.HTTPStatusError(message, request=request, response=resp)


# -- Retrieval status helpers --------------------------------------------------

# Informational notices that never mean something the caller asked for is
# missing: FEATURE_DISABLED reports an optional feature the caller did not
# configure, LLM_CAPABILITY_INFERRED a capability the server worked out on its
# own. Decided by code alone (retrieval status contract, Q1).
_INFORMATIONAL_STATUS_CODES = frozenset({"FEATURE_DISABLED", "LLM_CAPABILITY_INFERRED"})

# Status codes this package recognizes.  Anything else is reported as
# ``UNKNOWN`` with the server's code kept in ``originalCode`` (Q3).
_KNOWN_STATUS_CODES = frozenset({
    "GOODMEM_STATUS_CODE_UNSPECIFIED",
    "INVALID_ARGUMENT",
    "NOT_FOUND",
    "PERMISSION_DENIED",
    "FAILED_PRECONDITION",
    "EMBEDDER_FAILED",
    "EMBEDDER_UNAVAILABLE",
    "EMBEDDER_TIMEOUT",
    "VECTOR_SEARCH_FAILED",
    "VECTOR_SEARCH_PARTIAL",
    "VECTOR_SEARCH_TIMEOUT",
    "SPACE_INACCESSIBLE",
    "SPACE_NOT_FOUND",
    "SPACE_NO_EMBEDDERS",
    "CHUNK_NOT_FOUND",
    "MEMORY_LOAD_FAILED",
    "MEMORY_CONTENT_UNAVAILABLE",
    "RERANKING_FAILED",
    "SUMMARIZATION_FAILED",
    "SUMMARIZATION_TIMEOUT",
    "RATE_LIMITED",
    "RESOURCE_EXHAUSTED",
    "CONFIGURATION_ERROR",
}) | _INFORMATIONAL_STATUS_CODES


def _normalize_status(raw: Any) -> dict[str, Any] | None:
    """Turn one ``status`` event into ``{code, message, details}``.

    Returns ``None`` for informational notices (Q1).  A code this package does
    not recognize is reported as ``UNKNOWN`` and never dropped (Q3).
    """
    if not isinstance(raw, dict):
        raw = {"message": str(raw)}
    code = raw.get("code")
    known = isinstance(code, str) and code in _KNOWN_STATUS_CODES
    if known and code in _INFORMATIONAL_STATUS_CODES:
        return None
    details = raw.get("details")
    status: dict[str, Any] = {
        "code": code,
        "message": str(raw.get("message") or ""),
        "details": details if isinstance(details, dict) else {},
    }
    if not known:
        status["code"] = "UNKNOWN"
        status["originalCode"] = code
    return status


def _is_reranker_failure(status: dict[str, Any]) -> bool:
    """True when *status* says the requested reranker was not applied."""
    if status["code"] == "RERANKING_FAILED":
        return True
    if status["code"] != "NOT_FOUND":
        return False
    # A missing reranker arrives as NOT_FOUND naming the reranker; a NOT_FOUND
    # about anything else (an LLM, a space) says nothing about the ranking.
    return "reranker_id" in status["details"] or "reranker" in status["message"].lower()


def _describe_statuses(statuses: list[dict[str, Any]]) -> str:
    """Render statuses as ``CODE: message; CODE: message``."""
    return "; ".join(f"{s['code']}: {s['message']}" if s["message"] else s["code"] for s in statuses)


def _space_embedder_ids(space: dict[str, Any]) -> list[str]:
    """Return the embedder IDs a space was created with."""
    return [e["embedderId"] for e in space.get("spaceEmbedders") or [] if isinstance(e, dict) and e.get("embedderId")]


class GoodMemClient:
    """Async client for the GoodMem REST API.

    Args:
        base_url: Base URL of the GoodMem server (e.g. ``https://api.goodmem.ai``).
        api_key: API key used for ``X-API-Key`` authentication.
    """

    def __init__(self, base_url: str, api_key: str, *, verify_ssl: bool = True) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            verify=verify_ssl,
            timeout=httpx.Timeout(60.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    # -- Spaces ----------------------------------------------------------------

    async def list_spaces(self) -> list[dict[str, Any]]:
        """List all spaces."""
        resp = await self._http.get("/v1/spaces")
        _raise_for_status(resp)
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    async def _find_spaces_by_name(self, name: str) -> list[dict[str, Any]]:
        """Return every space named exactly *name*, reading all pages.

        The server's ``name_filter`` is a case-insensitive glob, not an exact
        match, so the exact name is re-checked here.  A name containing glob
        characters is not sent as a filter at all (it might not match itself).
        """
        params: dict[str, Any] = {}
        if not any(ch in name for ch in "*?[]\\"):
            params["name_filter"] = name
        matches: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        while True:
            resp = await self._http.get("/v1/spaces", params=params)
            _raise_for_status(resp)
            body = resp.json()
            spaces = body if isinstance(body, list) else body.get("spaces", [])
            matches.extend(s for s in spaces if s.get("name") == name)
            next_token = body.get("nextToken") if isinstance(body, dict) else None
            if not next_token or next_token in seen_tokens:
                return matches
            seen_tokens.add(next_token)
            params["nextToken"] = next_token

    async def create_space(
        self,
        name: str,
        embedder_id: str | None,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> dict[str, Any]:
        """Create a new space, or reuse the existing space named *name*.

        An existing space is reused only when its embedder is *embedder_id*
        (or when *embedder_id* is ``None``, meaning no particular embedder was
        requested).  If the existing space uses a different embedder, or more
        than one space has this name, nothing is created and a result with
        ``success: False`` explains the conflict.  When *embedder_id* is
        ``None`` and a new space is needed, the first embedder on the server
        is used.
        """
        embedder_id = embedder_id or None
        existing = await self._find_spaces_by_name(name)
        if len(existing) > 1:
            listing = ", ".join(
                f"{s.get('spaceId')} (embedders: {', '.join(_space_embedder_ids(s)) or 'none'})" for s in existing
            )
            return {
                "success": False,
                "error": (
                    f"{len(existing)} spaces are named '{name}': {listing}. Cannot tell which one to reuse; "
                    "nothing was created. Use a different name or pick a space by ID."
                ),
                "existingSpaceIds": [s.get("spaceId") for s in existing],
            }
        if existing:
            space = existing[0]
            space_embedders = _space_embedder_ids(space)
            if embedder_id is not None and space_embedders != [embedder_id]:
                return {
                    "success": False,
                    "error": (
                        f"A space named '{name}' already exists (spaceId {space.get('spaceId')}) with embedder "
                        f"{', '.join(space_embedders) or 'none'}, but embedder {embedder_id} was requested. "
                        "Reusing it would silently embed with a different model than requested, so nothing was created. "
                        "Use a different name, or pass the existing space's embedder to reuse it."
                    ),
                    "existingSpaceId": space.get("spaceId"),
                    "existingEmbedderIds": space_embedders,
                    "requestedEmbedderId": embedder_id,
                }
            return {
                "success": True,
                "spaceId": space["spaceId"],
                "name": space["name"],
                "embedderId": space_embedders[0] if len(space_embedders) == 1 else None,
                "embedderIds": space_embedders,
                "chunkingConfig": space.get("defaultChunkingConfig"),
                "message": (
                    "Space already exists, reusing existing space"
                    if embedder_id is None
                    else "Space already exists with the same embedder, reusing existing space"
                ),
                "reused": True,
            }

        if embedder_id is None:
            embedders = await self.list_embedders()
            if embedders:
                embedder_id = embedders[0].get("embedderId") or embedders[0].get("id") or None
            if not embedder_id:
                return {"success": False, "error": "No embedder_id provided and no embedders available on the server."}

        payload: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [{"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                },
            },
        }
        resp = await self._http.post("/v1/spaces", json=payload)
        _raise_for_status(resp)
        data = resp.json()
        return {
            "success": True,
            "spaceId": data["spaceId"],
            "name": data["name"],
            "embedderId": embedder_id,
            "embedderIds": _space_embedder_ids(data) or [embedder_id],
            "chunkingConfig": payload["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    async def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a single space by ID."""
        resp = await self._http.get(f"/v1/spaces/{space_id}")
        _raise_for_status(resp)
        return {"success": True, "space": resp.json()}

    async def update_space(
        self,
        space_id: str,
        *,
        name: str | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
        public_read: bool | None = None,
    ) -> dict[str, Any]:
        """Update an existing space.

        The server accepts ``name``, ``publicRead``, ``replaceLabels`` (full
        replacement of the labels map), and ``mergeLabels`` (per-key upsert).
        """
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if replace_labels is not None:
            payload["replaceLabels"] = replace_labels
        if merge_labels is not None:
            payload["mergeLabels"] = merge_labels
        if public_read is not None:
            payload["publicRead"] = public_read
        if not payload:
            return {"success": False, "error": "No fields provided to update."}

        resp = await self._http.put(f"/v1/spaces/{space_id}", json=payload)
        _raise_for_status(resp)
        return {
            "success": True,
            "spaceId": space_id,
            "space": resp.json(),
            "message": "Space updated successfully",
        }

    async def delete_space(self, space_id: str) -> dict[str, Any]:
        """Delete a space by ID."""
        resp = await self._http.delete(f"/v1/spaces/{space_id}")
        _raise_for_status(resp)
        return {"success": True, "spaceId": space_id, "message": "Space deleted successfully"}

    # -- Embedders -------------------------------------------------------------

    async def list_embedders(self) -> list[dict[str, Any]]:
        """List available embedder models."""
        resp = await self._http.get("/v1/embedders")
        _raise_for_status(resp)
        body = resp.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    # -- Memories --------------------------------------------------------------

    async def list_memories(
        self,
        space_id: str,
        *,
        page_size: int | None = None,
        next_token: str | None = None,
    ) -> dict[str, Any]:
        """List memories in a space."""
        params: dict[str, Any] = {}
        if page_size is not None:
            params["pageSize"] = page_size
        if next_token:
            params["nextToken"] = next_token

        resp = await self._http.get(f"/v1/spaces/{space_id}/memories", params=params)
        _raise_for_status(resp)
        body = resp.json()
        return {
            "success": True,
            "spaceId": space_id,
            "memories": body.get("memories", []) if isinstance(body, dict) else body,
            "nextToken": body.get("nextToken") if isinstance(body, dict) else None,
        }


    async def create_memory(
        self,
        space_id: str,
        *,
        text_content: str | None = None,
        file_path: str | None = None,
        file_bytes: bytes | None = None,
        file_extension: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a memory from text or a file (binary uploaded as base64).

        Exactly one of *text_content* or *file_path*/*file_bytes* must be provided.
        """
        payload: dict[str, Any] = {"spaceId": space_id}

        if file_path is not None:
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            mime = _guess_mime(ext)
            with open(file_path, "rb") as fh:
                raw = fh.read()
            if mime.startswith("text/"):
                payload["contentType"] = mime
                payload["originalContent"] = raw.decode("utf-8", errors="replace")
            else:
                payload["contentType"] = mime
                payload["originalContentB64"] = base64.b64encode(raw).decode()
        elif file_bytes is not None:
            ext = file_extension or ""
            mime = _guess_mime(ext)
            if mime.startswith("text/"):
                payload["contentType"] = mime
                payload["originalContent"] = file_bytes.decode("utf-8", errors="replace")
            else:
                payload["contentType"] = mime
                payload["originalContentB64"] = base64.b64encode(file_bytes).decode()
        elif text_content is not None:
            payload["contentType"] = "text/plain"
            payload["originalContent"] = text_content
        else:
            raise ValueError("No content provided. Supply text_content, file_path, or file_bytes.")

        if metadata:
            payload["metadata"] = metadata

        resp = await self._http.post("/v1/memories", json=payload)
        _raise_for_status(resp)
        data = resp.json()
        return {
            "success": True,
            "memoryId": data.get("memoryId"),
            "spaceId": data.get("spaceId"),
            "status": data.get("processingStatus", "PENDING"),
            "contentType": payload["contentType"],
            "message": "Memory created successfully",
        }

    async def retrieve_memories(
        self,
        query: str,
        space_ids: list[str],
        *,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool = False,
    ) -> dict[str, Any]:
        """Retrieve memories via semantic search.

        Problems the server reports during retrieval (an unknown reranker or
        LLM, a failed embedder, ...) are returned in ``statuses`` as
        ``{code, message, details}`` with ``partial: True``.  Any hits the
        server returned alongside them are kept, and an empty result with
        ``partial: True`` is returned rather than raising.  Informational
        notices (``FEATURE_DISABLED``, ``LLM_CAPABILITY_INFERRED``) are not
        problems and are left out.  With *wait_for_indexing*, polling for
        results stops as soon as the server reports a problem.
        """
        space_keys = [{"spaceId": sid} for sid in space_ids if sid]
        if not space_keys:
            return {"success": False, "error": "At least one space must be provided."}

        payload: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        if reranker_id or llm_id:
            config: dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True
            payload["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        max_wait = 60.0 if wait_for_indexing else 0.0
        poll_interval = 0.5
        should_wait = wait_for_indexing
        start = time.monotonic()

        while True:
            headers = {
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            }
            resp = await self._http.post("/v1/memories:retrieve", json=payload, headers=headers)
            _raise_for_status(resp)

            results, memories, result_set_id, abstract_reply, statuses = self._parse_ndjson(resp.text)

            result: dict[str, Any] = {
                "success": True,
                "resultSetId": result_set_id,
                "results": results,
                "memories": memories,
                "totalResults": len(results),
                "query": query,
                "partial": bool(statuses),
                "statuses": statuses,
            }
            if abstract_reply:
                result["abstractReply"] = abstract_reply

            if statuses:
                # The server said what went wrong; polling again cannot fix it.
                if results:
                    message = f"The server reported problems during retrieval; results may be incomplete: {_describe_statuses(statuses)}"
                else:
                    message = f"No results. The server reported problems during retrieval: {_describe_statuses(statuses)}"
                # Decided after the whole stream is read: a failed reranker
                # still returns the vector search's hits, scored on that scale.
                if results and reranker_id and any(_is_reranker_failure(s) for s in statuses):
                    message = message.rstrip(". ") + ". Reranking was not applied; relevanceScore values are vector-search scores."
                result["message"] = message
                return result

            if results or not should_wait:
                return result

            elapsed = time.monotonic() - start
            if elapsed >= max_wait:
                result["message"] = "No results found after waiting 60 seconds for indexing. Memories may still be processing."
                return result

            import asyncio
            await asyncio.sleep(poll_interval)

    async def get_memory(self, memory_id: str, *, include_content: bool = True) -> dict[str, Any]:
        """Fetch a single memory by ID."""
        resp = await self._http.get(f"/v1/memories/{memory_id}")
        _raise_for_status(resp)
        result: dict[str, Any] = {"success": True, "memory": resp.json()}

        if include_content:
            try:
                content_resp = await self._http.get(f"/v1/memories/{memory_id}/content")
                _raise_for_status(content_resp)
                # The content endpoint may return raw text or JSON depending on content type
                try:
                    result["content"] = content_resp.json()
                except Exception:
                    result["content"] = content_resp.text
            except Exception as exc:
                result["contentError"] = f"Failed to fetch content: {exc}"

        return result

    async def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by ID."""
        resp = await self._http.delete(f"/v1/memories/{memory_id}")
        _raise_for_status(resp)
        return {"success": True, "memoryId": memory_id, "message": "Memory deleted successfully"}

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_ndjson(
        text: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any] | None, list[dict[str, Any]]]:
        """Parse NDJSON / SSE response from the retrieve endpoint.

        Returns ``(results, memories, result_set_id, abstract_reply, statuses)``
        where *statuses* holds the problem statuses the server reported, in
        stream order, normalized by :func:`_normalize_status`.
        """
        results: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        result_set_id = ""
        abstract_reply: dict[str, Any] | None = None
        statuses: list[dict[str, Any]] = []

        for line in text.strip().split("\n"):
            json_str = line.strip()
            if not json_str:
                continue
            if json_str.startswith("data:"):
                json_str = json_str[5:].strip()
            if json_str.startswith("event:") or not json_str:
                continue
            try:
                item = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue

            if "status" in item:
                status = _normalize_status(item["status"])
                if status is not None:
                    statuses.append(status)
            elif "resultSetBoundary" in item:
                result_set_id = item["resultSetBoundary"].get("resultSetId", "")
            elif "memoryDefinition" in item:
                memories.append(item["memoryDefinition"])
            elif "abstractReply" in item:
                abstract_reply = item["abstractReply"]
            elif "retrievedItem" in item:
                chunk = item["retrievedItem"].get("chunk", {})
                inner = chunk.get("chunk", {})
                results.append({
                    "chunkId": inner.get("chunkId"),
                    "chunkText": inner.get("chunkText"),
                    "memoryId": inner.get("memoryId"),
                    "relevanceScore": chunk.get("relevanceScore"),
                    "memoryIndex": chunk.get("memoryIndex"),
                })

        return results, memories, result_set_id, abstract_reply, statuses
