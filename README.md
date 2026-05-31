# AI Agent with Memory + Tools

A production-grade conversational AI agent built with **LangGraph**, **Groq**, **Mem0**, and **Tavily**. The agent can search the web in real-time and remembers users across sessions.

## Architecture

```
User Input
    │
    ▼
┌─────────────┐
│  Streamlit  │  ← Chat UI with streaming tool call visibility
│   app.py    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         LangGraph ReAct Agent       │
│                                     │
│  [START] → agent_node               │
│               │                     │
│         (tool needed?)              │
│          yes ↓    no ↓             │
│        tool_node  [END]             │
│           ↓                         │
│        agent_node → [END]           │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐  ┌──────────┐
│ Tavily │  │   Mem0   │
│ Search │  │ Memory   │
└────────┘  └──────────┘
    │             │
    └──────┬──────┘
           ▼
     ┌──────────┐
     │  Groq    │  llama-3.3-70b-versatile
     │  (LLM)   │
     └──────────┘
```

## Setup

### 1. Get API Keys (all free)
| Service | URL | Free Tier |
|---------|-----|-----------|
| Groq | https://console.groq.com | 14,400 req/day |
| Tavily | https://app.tavily.com | 1,000 searches/month |
| Mem0 | https://app.mem0.ai | Free tier available |

### 2. Install dependencies
```bash
cd ai-agent-memory-tools
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run
```bash
streamlit run app.py
```

## Features

- **Real-time web search** — Tavily fetches current info the LLM doesn't know
- **Persistent memory** — Mem0 stores facts about users across sessions
- **Transparent reasoning** — UI shows tool calls and results as expandable sections
- **Multi-turn context** — full conversation history fed to agent each turn
- **Streaming** — LangGraph streams node-by-node, no waiting for full response

## Key Files

| File | Purpose |
|------|---------|
| `agent/graph.py` | LangGraph state machine — the agent "brain" |
| `agent/tools.py` | Web search + memory recall tools |
| `agent/memory.py` | Mem0 read/write integration |
| `agent/prompts.py` | System prompt controlling agent behavior |
| `app.py` | Streamlit UI with streaming |

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io → New app
3. Set secrets: `GROQ_API_KEY`, `TAVILY_API_KEY`, `MEM0_API_KEY`, `USER_ID`
4. Deploy
