"""Loads the KB markdown files, chunks them and writes them to Chroma."""
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from app.config import KB_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.rag.vectorstore import get_vectorstore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kb_documents():
    loader = DirectoryLoader(
        str(KB_DIR), glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def chunk_documents(documents):
    # prefer splitting on headings, then paragraphs, then sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # used for citations in the response
        chunk.metadata["source_doc"] = chunk.metadata.get("source", "unknown").split("/")[-1]
    return chunks


def seed():
    """Rebuilds the collection from scratch."""
    logger.info("Loading KB documents from %s", KB_DIR)
    docs = load_kb_documents()
    logger.info("Loaded %d documents", len(docs))

    chunks = chunk_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    store = get_vectorstore()
    store.reset_collection()
    store.add_documents(chunks)
    logger.info("Ingestion complete")


def seed_if_empty():
    # container disk is ephemeral on Render/Railway, so seed on startup
    if get_vectorstore()._collection.count() == 0:
        seed()


if __name__ == "__main__":
    seed()
