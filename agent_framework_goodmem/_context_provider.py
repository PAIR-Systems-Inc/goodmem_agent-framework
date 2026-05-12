# Copyright (c) Microsoft. All rights reserved.

"""GoodMem context provider using BaseContextProvider.

This module provides ``GoodMemContextProvider``, built on the
:class:`BaseContextProvider` hooks pattern.  It automatically retrieves
relevant memories before each agent run and optionally stores
conversations after.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from agent_framework import Message
from agent_framework._sessions import AgentSession, SessionContext

try:
    # agent-framework-core >= 1.2 renamed BaseContextProvider -> ContextProvider.
    from agent_framework._sessions import ContextProvider as _BaseContextProvider
except ImportError:  # pragma: no cover - older versions
    from agent_framework._sessions import BaseContextProvider as _BaseContextProvider  # type: ignore[no-redef]

from ._client import GoodMemClient

if TYPE_CHECKING:
    from agent_framework._agents import SupportsAgentRun

logger = logging.getLogger(__name__)


class GoodMemContextProvider(_BaseContextProvider):
    """GoodMem context provider using the BaseContextProvider hooks pattern.

    Integrates GoodMem for persistent semantic memory, automatically
    searching for relevant memories before each agent run and optionally
    storing conversations after.

    Two integration approaches are supported:

    - **Automatic (context provider)** — attach this provider to an agent
      via ``context_providers=[provider]``. Memories are retrieved and
      stored transparently on every ``agent.run()`` call.
    - **Tool-based** — use :func:`create_goodmem_tools` to let the agent
      decide when to read/write memories via tool calls.

    Args:
        source_id: Unique identifier for this provider instance.
        client: A configured :class:`GoodMemClient` instance.
        space_id: The GoodMem space ID to search and store memories in.
        max_results: Maximum number of memory chunks to retrieve per query.
        context_prompt: Prompt prepended to retrieved memories in the context.
        store_conversations: Whether to store input/response messages as
            new memories after each run.
        wait_for_indexing: Whether to wait for newly created memories to
            be indexed before returning retrieval results.
    """

    DEFAULT_CONTEXT_PROMPT: ClassVar[str] = (
        "## Relevant Memories\n"
        "The following memories were retrieved from long-term storage "
        "and may be relevant to the current conversation:"
    )
    DEFAULT_SOURCE_ID: ClassVar[str] = "goodmem"

    def __init__(
        self,
        client: GoodMemClient,
        space_id: str,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        max_results: int = 5,
        context_prompt: str | None = None,
        store_conversations: bool = True,
        wait_for_indexing: bool = False,
    ) -> None:
        """Initialize the GoodMem context provider.

        Args:
            client: A configured :class:`GoodMemClient` instance.
            space_id: The GoodMem space ID to search and store memories in.
            source_id: Unique identifier for this provider instance.
            max_results: Maximum number of memory chunks to retrieve.
            context_prompt: Prompt prepended to retrieved memories.
            store_conversations: Whether to store conversations after each run.
            wait_for_indexing: Whether to wait for indexing during retrieval.
        """
        super().__init__(source_id)
        self.client = client
        self.space_id = space_id
        self.max_results = max_results
        self.context_prompt = context_prompt or self.DEFAULT_CONTEXT_PROMPT
        self.store_conversations = store_conversations
        self.wait_for_indexing = wait_for_indexing

    # -- Hooks pattern ---------------------------------------------------------

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Search GoodMem for relevant memories and add to the session context.

        Extracts text from the input messages, performs a semantic search
        across the configured space, and injects matching chunks as a
        user message so the model can reference them.
        """
        input_text = "\n".join(
            msg.text for msg in context.input_messages if msg and msg.text and msg.text.strip()
        )
        if not input_text.strip():
            return

        try:
            result = await self.client.retrieve_memories(
                query=input_text,
                space_ids=[self.space_id],
                max_results=self.max_results,
                wait_for_indexing=self.wait_for_indexing,
            )
        except Exception:
            logger.warning("GoodMem retrieval failed", exc_info=True)
            return

        if not result.get("success"):
            logger.warning("GoodMem retrieval returned unsuccessful: %s", result.get("error"))
            return

        chunks = result.get("results", [])
        if not chunks:
            return

        memory_lines = [chunk.get("chunkText", "") for chunk in chunks if chunk.get("chunkText")]
        if not memory_lines:
            return

        memory_text = "\n".join(memory_lines)
        context.extend_messages(
            self,
            [Message(role="user", text=f"{self.context_prompt}\n{memory_text}")],
        )

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Store request/response messages to GoodMem for future retrieval.

        Concatenates input and response messages into a single text block
        and creates a new memory in the configured space.
        """
        if not self.store_conversations:
            return

        messages_to_store: list[Message] = list(context.input_messages)
        if context.response and context.response.messages:
            messages_to_store.extend(context.response.messages)

        def _get_role(role: Any) -> str:
            return role.value if hasattr(role, "value") else str(role)

        lines: list[str] = [
            f"{_get_role(msg.role)}: {msg.text}"
            for msg in messages_to_store
            if _get_role(msg.role) in {"user", "assistant", "system"} and msg.text and msg.text.strip()
        ]

        if not lines:
            return

        conversation_text = "\n".join(lines)

        try:
            await self.client.create_memory(
                space_id=self.space_id,
                text_content=conversation_text,
                metadata={
                    "source": "agent-framework-context-provider",
                    "session_id": session.session_id,
                },
            )
        except Exception:
            logger.warning("GoodMem memory storage failed", exc_info=True)


__all__ = ["GoodMemContextProvider"]
