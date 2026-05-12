# Copyright (c) Microsoft. All rights reserved.

"""GoodMem tools for the Agent Framework.

Each public function in this module is decorated with ``@tool`` so it can be
passed directly to an ``Agent`` instance.  The functions are thin wrappers
around :class:`GoodMemClient` that handle JSON serialization of results.

All tools require a pre-configured ``GoodMemClient`` instance to be injected
via :func:`create_goodmem_tools`.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from agent_framework import tool
from pydantic import Field

from ._client import GoodMemClient


def create_goodmem_tools(client: GoodMemClient) -> list[Any]:
    """Create a list of GoodMem ``FunctionTool`` instances bound to *client*.

    Args:
        client: A configured :class:`GoodMemClient` instance.

    Returns:
        A list of ``FunctionTool`` objects covering embedders, spaces (CRUD),
        and memories (CRUD + retrieve).
    """

    # -- List Embedders --------------------------------------------------------

    @tool(
        name="goodmem_list_embedders",
        description=(
            "List the embedder models available on the GoodMem server. "
            "Use the returned embedderId values when creating new spaces."
        ),
    )
    async def goodmem_list_embedders() -> str:
        """List available embedder models."""
        try:
            embedders = await client.list_embedders()
            return json.dumps({"success": True, "embedders": embedders, "count": len(embedders)})
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- List Spaces -----------------------------------------------------------

    @tool(
        name="goodmem_list_spaces",
        description="List all GoodMem spaces accessible to the current API key.",
    )
    async def goodmem_list_spaces() -> str:
        """List all spaces."""
        try:
            spaces = await client.list_spaces()
            return json.dumps({"success": True, "spaces": spaces, "count": len(spaces)}, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Get Space -------------------------------------------------------------

    @tool(
        name="goodmem_get_space",
        description="Fetch the metadata for a specific GoodMem space by its ID.",
    )
    async def goodmem_get_space(
        space_id: Annotated[str, Field(description="The UUID of the space to fetch.")],
    ) -> str:
        """Fetch a single space by ID."""
        try:
            result = await client.get_space(space_id)
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Create Space ----------------------------------------------------------

    @tool(
        name="goodmem_create_space",
        description=(
            "Create a new GoodMem space or reuse an existing one. "
            "A space is a logical container for organizing related memories, "
            "configured with an embedder that converts text to vector embeddings."
        ),
    )
    async def goodmem_create_space(
        name: Annotated[str, Field(description="A unique name for the space. If a space with this name already exists, its ID will be returned instead of creating a duplicate.")],
        embedder_id: Annotated[str, Field(description="The embedder ID that converts text into vector representations for similarity search. Use goodmem_list_embedders to find available IDs.")] = "",
        chunk_size: Annotated[int, Field(description="Number of characters per chunk when splitting documents.")] = 256,
        chunk_overlap: Annotated[int, Field(description="Number of overlapping characters between consecutive chunks.")] = 25,
        keep_strategy: Annotated[str, Field(description="Where to attach the separator when splitting: KEEP_END, KEEP_START, or DISCARD.")] = "KEEP_END",
        length_measurement: Annotated[str, Field(description="How chunk size is measured: CHARACTER_COUNT or TOKEN_COUNT.")] = "CHARACTER_COUNT",
    ) -> str:
        """Create a new GoodMem space or reuse an existing one."""
        actual_embedder_id = embedder_id
        if not actual_embedder_id:
            embedders = await client.list_embedders()
            if embedders:
                actual_embedder_id = embedders[0].get("embedderId") or embedders[0].get("id", "")
            if not actual_embedder_id:
                return json.dumps({"success": False, "error": "No embedder_id provided and no embedders available on the server."})

        try:
            result = await client.create_space(
                name=name,
                embedder_id=actual_embedder_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_strategy=keep_strategy,
                length_measurement=length_measurement,
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Update Space ----------------------------------------------------------

    @tool(
        name="goodmem_update_space",
        description="Update an existing GoodMem space's name, labels, or public_read flag.",
    )
    async def goodmem_update_space(
        space_id: Annotated[str, Field(description="The UUID of the space to update.")],
        name: Annotated[str, Field(description="Optional new name for the space.")] = "",
        replace_labels_json: Annotated[str, Field(description="Optional JSON object that REPLACES all labels on the space.")] = "",
        merge_labels_json: Annotated[str, Field(description="Optional JSON object whose entries are MERGED into existing labels (upsert per key).")] = "",
        public_read: Annotated[bool, Field(description="If true, makes the space publicly readable. If false, restricts to the owner.")] = False,
        set_public_read: Annotated[bool, Field(description="Whether to apply the public_read value (so the default `false` is not always sent).")] = False,
    ) -> str:
        """Update an existing GoodMem space."""

        def _parse_labels(raw: str, field: str) -> dict[str, str] | None | str:
            if not raw:
                return None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return f"{field} is not valid JSON."
            if not isinstance(parsed, dict):
                return f"{field} must decode to a JSON object."
            return {str(k): str(v) for k, v in parsed.items()}

        replace_labels = _parse_labels(replace_labels_json, "replace_labels_json")
        if isinstance(replace_labels, str):
            return json.dumps({"success": False, "error": replace_labels})
        merge_labels = _parse_labels(merge_labels_json, "merge_labels_json")
        if isinstance(merge_labels, str):
            return json.dumps({"success": False, "error": merge_labels})

        try:
            result = await client.update_space(
                space_id,
                name=name or None,
                replace_labels=replace_labels,
                merge_labels=merge_labels,
                public_read=public_read if set_public_read else None,
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Delete Space ----------------------------------------------------------

    @tool(
        name="goodmem_delete_space",
        description="Permanently delete a GoodMem space and all of its memories.",
    )
    async def goodmem_delete_space(
        space_id: Annotated[str, Field(description="The UUID of the space to delete.")],
    ) -> str:
        """Delete a space by ID."""
        try:
            result = await client.delete_space(space_id)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Create Memory ---------------------------------------------------------

    @tool(
        name="goodmem_create_memory",
        description=(
            "Store a document as a new memory in a GoodMem space. "
            "The memory is processed asynchronously -- chunked into searchable "
            "pieces and embedded into vectors. Accepts plain text or a file path."
        ),
    )
    async def goodmem_create_memory(
        space_id: Annotated[str, Field(description="The ID of the space to store the memory in.")],
        text_content: Annotated[str, Field(description="Plain text content to store as memory. Ignored when file_path is provided.")] = "",
        file_path: Annotated[str, Field(description="Absolute path to a file to store as memory (PDF, DOCX, image, etc.). Takes priority over text_content.")] = "",
        metadata_json: Annotated[str, Field(description="Optional JSON string of key-value metadata to attach to the memory.")] = "",
    ) -> str:
        """Store a document as a new memory in a GoodMem space."""
        metadata: dict[str, Any] | None = None
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                return json.dumps({"success": False, "error": "metadata_json is not valid JSON."})

        try:
            result = await client.create_memory(
                space_id=space_id,
                text_content=text_content or None,
                file_path=file_path or None,
                metadata=metadata,
            )
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- List Memories ---------------------------------------------------------

    @tool(
        name="goodmem_list_memories",
        description="List the memories stored in a specific GoodMem space.",
    )
    async def goodmem_list_memories(
        space_id: Annotated[str, Field(description="The UUID of the space whose memories should be listed.")],
        page_size: Annotated[int, Field(description="Maximum number of memories to return in this page. 0 means use server default.")] = 0,
        next_token: Annotated[str, Field(description="Pagination token returned from a previous call.")] = "",
    ) -> str:
        """List memories in a space."""
        try:
            result = await client.list_memories(
                space_id,
                page_size=page_size or None,
                next_token=next_token or None,
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Retrieve Memories -----------------------------------------------------

    @tool(
        name="goodmem_retrieve_memories",
        description=(
            "Perform similarity-based semantic retrieval across one or more "
            "GoodMem spaces. Returns matching chunks ranked by relevance. "
            "Supports optional reranker and LLM post-processing."
        ),
    )
    async def goodmem_retrieve_memories(
        query: Annotated[str, Field(description="A natural language query used to find semantically similar memory chunks.")],
        space_ids: Annotated[str, Field(description="Comma-separated list of space IDs to search across.")],
        max_results: Annotated[int, Field(description="Maximum number of results to return.")] = 5,
        include_memory_definition: Annotated[bool, Field(description="Include full memory metadata alongside matched chunks.")] = True,
        wait_for_indexing: Annotated[bool, Field(description="Retry for up to 60 seconds when no results are found (useful for recently added memories).")] = True,
        reranker_id: Annotated[str, Field(description="Optional UUID of a reranker model to improve result ordering.")] = "",
        llm_id: Annotated[str, Field(description="Optional UUID of an LLM to generate a contextual abstract reply.")] = "",
        relevance_threshold: Annotated[float, Field(description="Minimum relevance score (0-1) for including a result. 0 disables.")] = 0.0,
        llm_temperature: Annotated[float, Field(description="Creativity setting for the LLM post-processor (0-2). Negative disables.")] = -1.0,
        chronological_resort: Annotated[bool, Field(description="If true, reorder results by memory creation time.")] = False,
    ) -> str:
        """Retrieve memories via semantic search."""
        ids = [s.strip() for s in space_ids.split(",") if s.strip()]
        try:
            result = await client.retrieve_memories(
                query=query,
                space_ids=ids,
                max_results=max_results,
                include_memory_definition=include_memory_definition,
                wait_for_indexing=wait_for_indexing,
                reranker_id=reranker_id or None,
                llm_id=llm_id or None,
                relevance_threshold=relevance_threshold if relevance_threshold > 0 else None,
                llm_temperature=llm_temperature if llm_temperature >= 0 else None,
                chronological_resort=chronological_resort,
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Get Memory ------------------------------------------------------------

    @tool(
        name="goodmem_get_memory",
        description=(
            "Fetch a specific memory record by its ID, including metadata, "
            "processing status, and optionally the original content."
        ),
    )
    async def goodmem_get_memory(
        memory_id: Annotated[str, Field(description="The UUID of the memory to fetch.")],
        include_content: Annotated[bool, Field(description="Also fetch the original document content.")] = True,
    ) -> str:
        """Fetch a specific memory by ID."""
        try:
            result = await client.get_memory(memory_id, include_content=include_content)
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    # -- Delete Memory ---------------------------------------------------------

    @tool(
        name="goodmem_delete_memory",
        description="Permanently delete a memory and its associated chunks and vector embeddings.",
    )
    async def goodmem_delete_memory(
        memory_id: Annotated[str, Field(description="The UUID of the memory to delete.")],
    ) -> str:
        """Delete a memory by ID."""
        try:
            result = await client.delete_memory(memory_id)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    return [
        goodmem_list_embedders,
        goodmem_list_spaces,
        goodmem_get_space,
        goodmem_create_space,
        goodmem_update_space,
        goodmem_delete_space,
        goodmem_create_memory,
        goodmem_list_memories,
        goodmem_retrieve_memories,
        goodmem_get_memory,
        goodmem_delete_memory,
    ]
