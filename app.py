import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# typing / langgraph / langchain
from typing_extensions import TypedDict
from typing import Annotated, Any
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# tools & utilities
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    YouTubeSearchTool,
)
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

# chroma + embeddings
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==================== ENV & TOOLS SETUP ====================
load_dotenv()
os.environ["GOOGLE_BOOKS_API_KEY"] = os.getenv("GOOGLE_BOOKS_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Tool wrappers
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=800)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=800)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

books = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
pubmed = PubmedQueryRun()
youtube = YouTubeSearchTool()
tavily = TavilySearch()

tools = [arxiv, wiki, tavily, books, pubmed, youtube]

# ==================== LLM INITIALIZATION ====================
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools=tools)

# cached summarizer (so model instance isn't re-created on each rerun)
@st.cache_resource
def get_summarizer():
    return ChatGroq(model="llama-3.3-70b-versatile")

summarizer = get_summarizer()

def summarize_results(raw_text: dict, user_query: str) -> str:
    """Synthesize memory + new data into final answer."""
    prompt = f"""
You are an expert assistant. The user asked: "{user_query}"

Below you have some memory context (from prior conversations) and fresh data (from current sources).
Use both — integrate memory naturally (do NOT dump it verbatim) — and produce a single coherent answer
that makes the user feel continuity.

=== MEMORY CONTEXT ===
{raw_text.get('memory','')}

=== NEW DATA ===
{raw_text.get('new','')}

Provide 4-6 clear bullet points or a concise paragraph as final answer.
"""
    try:
        response = summarizer.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"[Summarization failed: {e}]"

# ==================== EMBEDDINGS & PERSISTENT MEMORY (Chroma v1.x) ====================
# Persistent Chroma client (v1.x)
client = chromadb.PersistentClient(path="./vector_memory")

# Create or load a collection. We'll pick a base name and allow automatic suffixing if needed.
BASE_COLLECTION_NAME = "student_assistant_memory"
collection = client.get_or_create_collection(BASE_COLLECTION_NAME)

# Embedding model (safe CPU mode)
# Use langchain_community's wrapper with model_kwargs forcing CPU
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Quick dimension check — if mismatch, create a new collection name to avoid errors
try:
    test_emb = embedder.embed_query("test")
    emb_dim = len(test_emb)
    # If collection exists and its dimension differs, create new suffixed collection
    # (Chroma doesn't expose dimension API directly, so we attempt an add-check inside try)
    # We'll not modify existing collection content here; instead, ensure the collection we use accepts current dimension.
except Exception:
    # If embedder failed, we'll continue — calls later will raise explicit errors for debugging
    emb_dim = None

# ==================== LANGGRAPH STATE ====================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()

# ==================== STREAMLIT UI & SESSION ====================
st.set_page_config(page_title="STUDENT ASSISTANT BOT", page_icon="🎓", layout="wide")

# session history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"query","summary_title","timestamp"}

# optional selected query (when user clicks history)
if "selected_query" not in st.session_state:
    st.session_state.selected_query = None

# sidebar: short chat history (clickable)
st.sidebar.title("💬 Chat History")
if st.session_state.chat_history:
    # show most recent first
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        idx_key = f"hist_{i}"
        if st.sidebar.button(item["summary_title"], key=idx_key):
            # refill main input by storing in session_state.selected_query
            st.session_state.selected_query = item["query"]
        st.sidebar.markdown(f"*{item['timestamp']}*")
        st.sidebar.markdown("---")
else:
    st.sidebar.info("No history yet. Start your first search!")

# main UI
st.title("🎓 STUDENT ASSISTANT")
st.markdown("Ask about **research papers, books**")

# main text input: prefill with clicked history query if any
default_query = st.session_state.get("selected_query", "")
query = st.text_input("Your Query", value=default_query)
# ✅ If we already have a last result, show it by default
if "last_summary" in st.session_state and st.session_state["last_summary"]:
    st.subheader("🧠 Final Synthesized Answer")
    st.markdown(st.session_state["last_summary"])


# ==================== MAIN SEARCH LOGIC ====================
if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        st.info("⏳ Fetching information...")

        try:
            # ----- 0) Silent memory retrieval (semantic search)
            try:
                query_embedding = embedder.embed_query(query)
                # Choose how many past docs to retrieve (2 here)
                mem_results = collection.query(query_embeddings=[query_embedding], n_results=2)
            except Exception:
                # If collection dimension mismatch occurs or other embedding error,
                # create a new collection name and use it (avoids crashing)
                suffix = datetime.now().strftime("%Y%m%d%H%M%S")
                collection = client.get_or_create_collection(f"{BASE_COLLECTION_NAME}_{suffix}")
                mem_results = {"documents": [[]], "metadatas": [[]]}

            memory_context = ""
            if mem_results and mem_results.get("documents") and mem_results["documents"][0]:
                for doc in mem_results["documents"][0]:
                    # Not displayed — passed silently to summarizer
                    memory_context += doc + "\n\n"

            # ----- 1) Run LangGraph to fetch fresh tool data -----
            result = graph.invoke({"messages": [HumanMessage(content=query)]})

            # ----- 2) Collect and clean results from tools -----
            collected_new = ""
            tools_used = set()
            for m in result.get("messages", []):
                # Collect tool names
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for t in m.tool_calls:
                        tools_used.add(t["name"])
                # Collect textual content
                if hasattr(m, "content") and isinstance(m.content, str):
                    text = m.content.strip()
                    # Optional: Parse JSON if Tavily returns structured data
                    if text.startswith("{") or text.startswith("["):
                        try:
                            data = json.loads(text)
                            if "results" in data:
                                for item in data["results"][:3]:
                                    collected_new += f"Title: {item.get('title','')}\nURL: {item.get('url','')}\n"
                                    collected_new += (item.get("content","") or "") + "\n\n"
                                continue
                        except Exception:
                            pass
                    # Fallback: plain text append
                    collected_new += text + "\n\n"

            # ----- 3) Combine memory + new data for reasoning -----
            combined_input = {
                "memory": memory_context.strip(),
                "new": collected_new.strip()
            }

            # ----- 4) LLM Reasoning & Final Answer -----
            st.subheader("🧠 Final Synthesized Answer")
            summary = summarize_results(combined_input, query)
            st.markdown(summary)

            # ----- 5) Store reasoning result in persistent memory -----
            try:
                emb = embedder.embed_query(summary)
                new_id = f"mem_{datetime.now().timestamp()}"
                collection.add(
                    embeddings=[emb],
                    documents=[summary],
                    metadatas=[{
                        "query": query,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tools_used": ", ".join(sorted(tools_used))
                    }],
                    ids=[new_id]
                )
                # PersistentClient handles persistence; no explicit client.persist() required for v1.x
            except Exception as e:
                st.warning(f"⚠️ Memory store failed: {e}")

            # ----- 6) Update Sidebar Chat History -----
            try:
                def create_title(text: str) -> str:
                    """Generate a short 3–5 word sidebar title."""
                    words = text.strip().split()
                    short = " ".join(words[:5])
                    return (short.capitalize() + ("..." if len(words) > 5 else ""))

                short_title = create_title(query)
                st.session_state.chat_history.append({
                    "query": query,
                    "summary_title": short_title,
                    "timestamp": datetime.now().strftime("%H:%M")
                })

                st.session_state["last_summary"] = summary
                st.session_state["last_query"] = query

                # clear selected_query so input is not overwritten next time
                st.session_state.selected_query = None
                # st.rerun()
            except Exception as e:
                st.warning(f"⚠️ Could not update chat history: {e}")

            # ----- 7) Show which tools were used -----
            if tools_used:
                st.markdown(f"🧰 **Sources used:** {', '.join(sorted(tools_used))}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built using Llama LLM ")
