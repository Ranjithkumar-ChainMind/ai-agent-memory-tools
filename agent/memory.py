"""
memory.py — Mem0 integration for long-term user memory.

Why Mem0? It acts like a smart notepad the agent writes to and reads from.
Unlike conversation history (which resets), Mem0 persists facts about the user
across sessions: "User prefers Python", "User is learning LangGraph", etc.
"""

import os
from mem0 import MemoryClient


def get_memory_client() -> MemoryClient:
    """Return authenticated Mem0 client."""
    return MemoryClient(api_key=os.environ["MEM0_API_KEY"])


def store_memory(client: MemoryClient, user_id: str, messages: list[dict]) -> None:
    """
    Save the current conversation turn to Mem0.
    Mem0 automatically extracts key facts (not raw text) to store efficiently.
    messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    client.add(messages, user_id=user_id)


def retrieve_memories(client: MemoryClient, user_id: str, query: str) -> str:
    """
    Semantic search over stored memories relevant to the query.
    Returns a formatted string the agent can include in its context.
    """
    results = client.search(query, user_id=user_id, limit=5)
    if not results:
        return "No relevant memories found."
    lines = [f"- {item['memory']}" for item in results]
    return "Relevant memories about the user:\n" + "\n".join(lines)
