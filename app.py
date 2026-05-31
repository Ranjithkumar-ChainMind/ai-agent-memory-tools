"""
app.py — Streamlit UI for the AI Agent with Memory + Tools.

Architecture:
  - st.session_state stores: the compiled graph, mem0 client, conversation history
  - Each user message → streamed through the LangGraph graph
  - Tool calls are shown in real-time as expandable "thinking" sections
  - After each turn, the conversation is saved to Mem0 for long-term memory
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import build_agent_graph
from agent.memory import store_memory

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent — Memory + Tools",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 AI Agent with Memory + Tools")
st.caption("Powered by LangGraph · Groq · Mem0 · Tavily")

# ── Sidebar: user identity ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    user_id = st.text_input(
        "Your User ID",
        value=os.getenv("USER_ID", "user_001"),
        help="Unique ID used to store/retrieve your memories in Mem0.",
    )
    st.divider()
    st.markdown("**Capabilities:**")
    st.markdown("- 🔍 Web search (Tavily)")
    st.markdown("- 🧠 Long-term memory (Mem0)")
    st.markdown("- ⚡ Fast inference (Groq llama-3.3-70b)")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # display history (dicts with role/content)

if "graph" not in st.session_state or st.session_state.get("graph_user_id") != user_id:
    with st.spinner("Initializing agent..."):
        graph, mem0_client = build_agent_graph(user_id)
        st.session_state.graph = graph
        st.session_state.mem0_client = mem0_client
        st.session_state.graph_user_id = user_id
        st.session_state.lc_messages = []  # LangChain message objects for graph state

# ── Display conversation history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to LangChain message history
    st.session_state.lc_messages.append(HumanMessage(content=prompt))

    # ── Stream agent response ─────────────────────────────────────────────────
    with st.chat_message("assistant"):
        full_response = ""
        tool_calls_made = []

        # Stream events from LangGraph graph
        # stream_mode="values" gives us state after each node completes
        events = st.session_state.graph.stream(
            {"messages": st.session_state.lc_messages},
            stream_mode="values",
        )

        response_placeholder = st.empty()

        try:
            for event in events:
                last_msg = event["messages"][-1]

                # Agent decided to call a tool → show in expander
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc["name"]
                        tool_input = tc["args"]
                        tool_calls_made.append((tool_name, tool_input))
                        with st.expander(f"🔧 Using tool: `{tool_name}`", expanded=False):
                            st.json(tool_input)

                # Tool returned a result → show briefly
                elif isinstance(last_msg, ToolMessage):
                    with st.expander(f"📥 Tool result: `{last_msg.name}`", expanded=False):
                        st.markdown(last_msg.content[:800] + ("..." if len(last_msg.content) > 800 else ""))

                # Final AI response (no tool calls)
                elif isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response)

        except Exception as e:
            # Groq sometimes rejects tool calls after long histories — reset and retry
            err_str = str(e)
            if "Failed to call a function" in err_str or "400" in err_str:
                response_placeholder.warning(
                    "The model had trouble with tool formatting. Resetting conversation history and retrying..."
                )
                # Reset lc_messages to just the current user message — clears bad history
                st.session_state.lc_messages = [HumanMessage(content=prompt)]
                try:
                    retry_events = st.session_state.graph.stream(
                        {"messages": st.session_state.lc_messages},
                        stream_mode="values",
                    )
                    for event in retry_events:
                        last_msg = event["messages"][-1]
                        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                            full_response = last_msg.content
                            response_placeholder.markdown(full_response)
                except Exception:
                    response_placeholder.error("Still failing — try clicking '🗑️ Clear conversation' in the sidebar and start fresh.")
            else:
                response_placeholder.error(f"Error: {err_str}")

        # ── Save to Mem0 after response ───────────────────────────────────────
        if full_response:
            store_memory(
                st.session_state.mem0_client,
                user_id,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": full_response},
                ],
            )
            # Update LangChain message history for next turn
            st.session_state.lc_messages.append(AIMessage(content=full_response))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
