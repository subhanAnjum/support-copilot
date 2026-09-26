"""Fetches the top-k KB chunks for the ticket."""
from app.rag.vectorstore import get_retriever
from app.config import TOP_K


def retrieve_node(state: dict) -> dict:
    retriever = get_retriever(k=TOP_K)
    docs = retriever.invoke(state["ticket_text"])

    state["retrieved_chunks"] = [d.page_content for d in docs]
    state["retrieved_sources"] = [d.metadata.get("source_doc", "unknown") for d in docs]
    return state
