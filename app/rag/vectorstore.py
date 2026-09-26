"""Chroma helpers."""
from langchain_chroma import Chroma
from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, get_embeddings


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
