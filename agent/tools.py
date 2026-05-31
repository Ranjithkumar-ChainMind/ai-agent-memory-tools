"""
tools.py — LangChain-compatible tools the agent can call.

Why this pattern? LangGraph's agent node expects tools as LangChain Tool objects.
Each tool has: name, description (the LLM reads this to decide WHEN to use it), func.

Tools here:
  1. web_search   — Tavily real-time web search
  2. recall_memory — fetch relevant memories about the user
"""

import os
from langchain_core.tools import tool
from tavily import TavilyClient


def make_web_search_tool():
    """Factory: returns a LangChain tool wrapping Tavily search."""
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    @tool
    def web_search(query: str) -> str:
        """
        Search the internet for current information.
        Use this when the user asks about recent events, news, prices, documentation,
        or anything that might not be in the model's training data.
        """
        results = tavily.search(query=query, max_results=4, search_depth="basic")
        if not results.get("results"):
            return "No results found."
        parts = []
        for r in results["results"]:
            parts.append(f"**{r['title']}**\n{r['content']}\nSource: {r['url']}")
        return "\n\n---\n\n".join(parts)

    return web_search


def make_memory_tool(mem0_client, user_id: str):
    """Factory: returns a LangChain tool that retrieves user memories from Mem0."""
    from agent.memory import retrieve_memories

    @tool
    def recall_memory(query: str) -> str:
        """
        Recall relevant facts and preferences about the user from long-term memory.
        Use this when the user references past conversations, personal context,
        or when personalizing the response would help.
        """
        return retrieve_memories(mem0_client, user_id, query)

    return recall_memory
