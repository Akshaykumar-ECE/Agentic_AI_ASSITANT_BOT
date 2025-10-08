import streamlit as st
from dotenv import load_dotenv
import os
import json
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    YouTubeSearchTool,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq


# ==================== ENVIRONMENT SETUP ====================
load_dotenv()
os.environ["GOOGLE_BOOKS_API_KEY"] = os.getenv("GOOGLE_BOOKS_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")


# ==================== TOOL INITIALIZATION ====================
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
llm = ChatGroq(model="llama-3.1-8b-instant")  # switched for better rate limits
llm_with_tools = llm.bind_tools(tools=tools)


# ==================== LANGGRAPH STATE ====================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ==================== GRAPH DEFINITION ====================
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()


# ==================== STREAMLIT UI ====================
st.set_page_config(page_title="STUDENT ASSISTANT BOT", page_icon="🎓", layout="wide")

st.title("🎓 STUDENT ASSISTANT BOT")
st.markdown("Ask about **research papers, books, or latest AI news** — get top 5 clean results.")

query = st.text_input("Your Query", placeholder="e.g. latest AI research papers")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        st.info("⏳ Fetching top 5 results from multiple sources...")
        try:
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            st.success("✅ Results fetched successfully!")

            tools_used = set()
            formatted_output = ""

            for m in result["messages"]:
                # Collect which tools were used
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for t in m.tool_calls:
                        tools_used.add(t["name"])

                # Parse and format Tavily JSON results
                if hasattr(m, "content") and isinstance(m.content, str):
                    content = m.content.strip()
                    if content.startswith("{") or content.startswith("["):
                        try:
                            data = json.loads(content)
                            if "results" in data:
                                formatted_output += "### 📰 Tavily — Latest AI News (Top 5)\n"
                                for i, item in enumerate(data["results"][:5], 1):
                                    formatted_output += (
                                        f"{i}. [{item.get('title', 'Untitled')}]({item.get('url', '')})  \n"
                                        f"*{item.get('content', '').strip()}*\n\n"
                                    )
                                formatted_output += "---\n"
                        except Exception:
                            pass

                    # Arxiv or other text-based content
                    elif "Published:" in content or "Title:" in content:
                        formatted_output += "### 📘 Arxiv — AI Research Papers (Top 5)\n"
                        formatted_output += content.replace("=================================", "---") + "\n\n"
                    else:
                        formatted_output += content + "\n\n"

            # Display which tools were used
            if tools_used:
                st.markdown(f"🧰 **Tools used:** {', '.join(tools_used)}")

            # Final formatted readable output
            if formatted_output.strip():
                st.markdown(f"### 🧠 Results\n\n{formatted_output.strip()}")
            else:
                st.warning("⚠️ No readable output returned by the tools.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ by Akshay | Powered by LangChain, LangGraph, and Groq")
