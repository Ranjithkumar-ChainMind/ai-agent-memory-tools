"""
graph.py — LangGraph ReAct agent graph.

Architecture:
  [START] → agent_node → (tool call?) → tool_node → agent_node → ... → [END]

Why LangGraph over plain LangChain agents?
  - Explicit state: you see exactly what's in memory at each step.
  - Controllable loops: you define when to stop, not the framework.
  - Streaming: every step emits events — perfect for real-time Streamlit UI.
  - Debuggable: each node is a plain Python function.

The pattern here is called "ReAct" (Reason + Act):
  1. Agent reasons about the question.
  2. If it needs info → calls a tool.
  3. Tool result comes back → agent reasons again.
  4. When agent has enough info → outputs final answer.
"""

import os
from typing import Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from agent.prompts import SYSTEM_PROMPT
from agent.tools import make_web_search_tool, make_memory_tool
from agent.memory import get_memory_client


class AgentState(BaseModel):
    """
    The state object passed between every node in the graph.
    `add_messages` is a LangGraph reducer: appends new messages instead of replacing.
    This is how conversation history accumulates across graph steps.
    """
    messages: Annotated[list, add_messages] = []

    class Config:
        arbitrary_types_allowed = True


def build_agent_graph(user_id: str):
    """
    Construct and compile the LangGraph agent.
    Returns a compiled graph ready to stream or invoke.

    Why compile? LangGraph validates the graph structure (no orphan nodes,
    valid edges) and optimizes execution before any messages are processed.
    """
    # --- Setup ---
    mem0_client = get_memory_client()
    web_search = make_web_search_tool()
    recall_memory = make_memory_tool(mem0_client, user_id)
    tools = [web_search, recall_memory]

    # Groq with tool calling enabled — llama-3.3-70b is the best free model for agents
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.1,  # low temp = more consistent tool usage decisions
    ).bind_tools(tools)  # bind_tools tells the LLM what tools exist + their schemas

    # --- Nodes ---

    def agent_node(state: AgentState) -> dict:
        """
        The reasoning node. Prepends system prompt, calls the LLM.
        Returns new messages to append to state.
        If LLM decides to call a tool, the message will contain tool_calls.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state.messages
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    # ToolNode: LangGraph's built-in node that executes whatever tools the LLM called.
    # It reads tool_calls from the last AIMessage, runs each tool, returns ToolMessages.

    # --- Graph wiring ---

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")  # always start at the agent node

    # tools_condition: built-in LangGraph conditional edge.
    # Checks if last message has tool_calls → route to "tools", else → END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  # after tools run, always go back to agent

    return graph.compile(), mem0_client
