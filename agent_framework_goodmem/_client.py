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
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    async def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> dict[str, Any]:
        """Create a new space, or return the existing one if a space with *name* already exists."""
        # Check for existing space with the same name
        spaces = await self.list_spaces()
        for space in spaces:
            if space.get("name") == name:
                return {
                    "success": True,
                    "spaceId": space["spaceId"],
                    "name": space["name"],
                    "embedderId": embedder_id,
                    "message": "Space already exists, reusing existing space",
                    "reused": True,
                }

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
        resp.raise_for_status()
        data = resp.json()
        return {
            "success": True,
            "spaceId": data["spaceId"],
            "name": data["name"],
            "embedderId": embedder_id,
            "chunkingConfig": payload["defaultChunkingConfig"],
            "message": "Space created successfully",
            "reused": False,
        }

    async def get_space(self, space_id: str) -> dict[str, Any]:
        """Fetch a single space by ID."""
        resp = await self._http.get(f"/v1/spaces/{space_id}")
        resp.raise_for_status()
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
        resp.raise_for_status()
        return {
            "success": True,
            "spaceId": space_id,
            "space": resp.json(),
            "message": "Space updated successfully",
        }

    async def delete_space(self, space_id: str) -> dict[str, Any]:
        """Delete a space by ID."""
        resp = await self._http.delete(f"/v1/spaces/{space_id}")
        resp.raise_for_status()
        return {"success": True, "spaceId": space_id, "message": "Space deleted successfully"}

    # -- Embedders -------------------------------------------------------------

    async def list_embedders(self) -> list[dict[str, Any]]:
        """List available embedder models."""
        resp = await self._http.get("/v1/embedders")
        resp.raise_for_status()
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
        resp.raise_for_status()
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
        resp.raise_for_status()
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
        """Retrieve memories via semantic search."""
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
            resp.raise_for_status()

            results, memories, result_set_id, abstract_reply = self._parse_ndjson(resp.text)

            result: dict[str, Any] = {
                "success": True,
                "resultSetId": result_set_id,
                "results": results,
                "memories": memories,
                "totalResults": len(results),
                "query": query,
            }
            if abstract_reply:
                result["abstractReply"] = abstract_reply

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
        resp.raise_for_status()
        result: dict[str, Any] = {"success": True, "memory": resp.json()}

        if include_content:
            try:
                content_resp = await self._http.get(f"/v1/memories/{memory_id}/content")
                content_resp.raise_for_status()
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
        resp.raise_for_status()
        return {"success": True, "memoryId": memory_id, "message": "Memory deleted successfully"}

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_ndjson(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any] | None]:
        """Parse NDJSON / SSE response from the retrieve endpoint."""
        results: list[dict[str, Any]] = []
        memories: list[dict[str, Any]] = []
        result_set_id = ""
        abstract_reply: dict[str, Any] | None = None

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

            if "resultSetBoundary" in item:
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

        return results, memories, result_set_id, abstract_reply
