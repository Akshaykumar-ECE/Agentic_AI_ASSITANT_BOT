from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import YouTubeSearchTool
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch


from dotenv import load_dotenv
load_dotenv()
import os

# === Environment Variables ===
os.environ["GOOGLE_BOOKS_API_KEY"] = os.getenv("GOOGLE_BOOKS_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# === Tool Initialization ===
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query Arxiv Paper")
# print(arxiv.name)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# print(wiki.name)

books = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
# print(books.name)

pubmed = PubmedQueryRun()
# print(pubmed.name)

youtube = YouTubeSearchTool()

tavily = TavilySearch()

tools = [arxiv, wiki, tavily, books, pubmed, youtube]

# === LLM (Groq) ===
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools=tools)
llm_with_tools.invoke("ofdm research paper")

# === LangGraph State ===
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# === Graph ===
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

# === Run Query ===
messages = graph.invoke({"messages": "provide me the latest news on ai and provide me the ai research paper"})
for m in messages["messages"]:
    m.pretty_print()
