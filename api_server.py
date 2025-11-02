# api_server.py
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# your existing imports (same as in your Streamlit app)
import json
from typing_extensions import TypedDict
from typing import Annotated, Any
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

# load env
load_dotenv()
os.environ["GOOGLE_BOOKS_API_KEY"] = os.getenv("GOOGLE_BOOKS_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# FastAPI app
app = FastAPI(title="Student Assistant Bot API")

# ---- tools & llm ----
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=800)
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=800)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
books = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
pubmed = PubmedQueryRun()
youtube = YouTubeSearchTool()
tavily = TavilySearch()

tools = [arxiv, wiki, tavily, books, pubmed, youtube]

# LLM and tool binding
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools=tools)

# chroma memory
client = chromadb.PersistentClient(path="./vector_memory")
BASE_COLLECTION_NAME = "student_assistant_memory"
collection = client.get_or_create_collection(BASE_COLLECTION_NAME)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# langgraph state graph (same pattern as your app)
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

# request model
class QueryRequest(BaseModel):
    query: str

# ---- endpoints ----
@app.post("/ask")
def ask(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    tools_used = set()
    collected_new = ""

    # Run LangGraph (tools + llm)
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    

    for m in result.get("messages", []):
        if hasattr(m, "tool_calls") and m.tool_calls:
            for t in m.tool_calls:
                tools_used.add(t["name"])
        if hasattr(m, "content") and isinstance(m.content, str):
            collected_new += m.content.strip() + "\n\n"

    # Summarize combined data using LLM (you can tune prompt)
    prompt = f"""
You are an expert assistant. The user asked: "{query}"

Below are retrieved pieces of data from external sources. Summarize the key points in 4-6 clear bullet points and include a short "sources" line listing which tools were used.dont include that 'here are the 6 bullet points like that all'

{collected_new}
"""
    answer = llm.invoke(prompt).content.strip()

    # store in memory
    try:
        emb = embedder.embed_query(answer)
        new_id = f"mem_{datetime.now().timestamp()}"
        collection.add(
            embeddings=[emb],
            documents=[answer],
            metadatas=[{
                "query": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tools_used": ", ".join(sorted(tools_used))
            }],
            ids=[new_id]
        )
    except Exception as e:
        # memory storing failure shouldn't break the API
        print("Memory store failed:", e)

    return {"query": query, "summary": answer, "sources": list(sorted(tools_used))}


@app.get("/history")
def history(limit: int = 10):
    try:
        res = collection.get(limit=limit)
        docs = []
        for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
            docs.append({"summary": doc, "meta": meta})
        return {"items": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
def ping():
    return {"status": "ok"}
