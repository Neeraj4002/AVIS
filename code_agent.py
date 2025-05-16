# code_agent.py
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any
import streamlit as st
from langchain.schema import BaseMessage, SystemMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from retriever import LangGraphRetrievalSystem

# Constants
QA_LLM_MODEL = "gemini-2.0-flash-lite"
SYSTEM_PROMPT_FILE = "D:\\SNK\\langchain-master\\Langraph agent\\GLang\\langgraph_system_prompt.txt"

# Load base system prompt once
try:
    base_system_prompt = Path(SYSTEM_PROMPT_FILE).read_text()
except Exception:
    base_system_prompt = "You are a helpful AI assistant specializing in LangGraph."

class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    docs: List[Any]

@st.cache_resource
def get_retriever_system(k: int, api_key: str) -> LangGraphRetrievalSystem:
    try:
        return LangGraphRetrievalSystem(k=k, google_api_key=api_key)
    except Exception:
        return LangGraphRetrievalSystem(k=k, google_api_key=api_key)

@st.cache_resource
def get_qa_llm(api_key: str) -> ChatGoogleGenerativeAI:
    try:
        return ChatGoogleGenerativeAI(model=QA_LLM_MODEL, temperature=0.3, google_api_key=api_key)
    except Exception:
        return ChatGoogleGenerativeAI(model=QA_LLM_MODEL, temperature=0.3, google_api_key=api_key)

def rag_qa_node(state: MessagesState, config: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
    api_key = config.get("api_key", "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE")
    k_docs = config.get("k_docs", 5)
    rerank = config.get("rerank", False)

    retriever = get_retriever_system(k=k_docs, api_key=api_key)
    llm = get_qa_llm(api_key=api_key)

    question = state["messages"][-1].content if state["messages"] else ""
    try:
        docs = retriever.retrieve(question, k=k_docs, rerank_llm=rerank)
        context = "\n\n---\n\n".join([d.page_content for d in docs]) or ""
    except Exception:
        context = ""

    system_content = f"{base_system_prompt}\n\n```context\n{context}```"
    sys_msg = SystemMessage(content=system_content)
    msgs = [sys_msg] + state["messages"]

    full = ""
    for chunk in llm.stream(msgs):
        if isinstance(chunk, AIMessageChunk) or hasattr(chunk, 'content'):
            full += chunk.content
    response = {
        "messages": [AIMessage(content=full or "Sorry, no response.")],
        "docs": docs,
    }
    print(f"docs: {docs}")
    return response

@st.cache_resource
def get_compiled_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("rag_qa", rag_qa_node)
    builder.add_edge(START, "rag_qa")
    builder.add_edge("rag_qa", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
