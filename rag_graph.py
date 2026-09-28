"""LangGraph retrieve -> relevance check -> rewrite (once) -> generate."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from config import MIN_SIMILARITY, RETRIEVAL_K
from rag_engine import search_documents
from generation import generate_answer
from hallucination_detector import verify_answer

class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    passages: list[dict]
    attempts: int
    answer: str
    sources: list[dict]
    route: str
    verification: dict

def retrieve(state: RAGState):
    query = state.get("search_query") or state["question"]
    return {"passages": search_documents(query, RETRIEVAL_K)}

def assess(state: RAGState):
    relevant = [p for p in state.get("passages", []) if p["similarity"] >= MIN_SIMILARITY]
    if relevant:
        return {"passages": relevant, "route": "generate"}
    if state.get("attempts", 0) < 1:
        return {"route": "rewrite"}
    return {"passages": [], "route": "no_answer"}

def rewrite(state: RAGState):
    # Deterministic one-time reformulation; replace with an LLM rewrite if needed.
    query = state["question"].strip().rstrip("?!. ")
    return {"search_query": query, "attempts": state.get("attempts", 0) + 1}

def generate(state: RAGState):
    passages = state["passages"]
    return {
        "answer": generate_answer(state["question"], passages),
        "sources": [
            {"source": p["metadata"].get("source"),
             "chapter": p["metadata"].get("chapter"),
             "section": p["metadata"].get("section"),
             "chunk_index": p["metadata"].get("chunk_index"),
             "similarity": p["similarity"]}
            for p in passages
        ],
    }

def verify(state: RAGState):
    result = verify_answer(state["answer"], state["passages"])
    return {"answer": result["answer"], "verification": result,
            "sources": state.get("sources", []) if result["verified"] else []}

def no_answer(state: RAGState):
    return {"answer": "I could not find the answer in the document.", "sources": []}

def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess", assess)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)
    graph.add_node("verify", verify)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess", lambda s: s["route"],
        {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("no_answer", END)
    return graph.compile()

rag_graph = build_graph()

def ask(question: str) -> dict:
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": []}
    return rag_graph.invoke({"question": question, "attempts": 0})
