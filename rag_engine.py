"""PDF ingestion and LangChain FAISS retrieval. Model loads are lazy."""
from functools import lru_cache
from pathlib import Path
import json
import threading
import pymupdf4llm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR, EMBEDDING_MODEL, UPLOAD_DIR
from chunking import chunk_document

_LOCK = threading.RLock()
_STORE = None

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

def _save_manifest(source_names):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "sources.json").write_text(json.dumps(sorted(set(source_names)), indent=2))

def indexed_sources():
    manifest = INDEX_DIR / "sources.json"
    return json.loads(manifest.read_text()) if manifest.exists() else []

def load_store():
    global _STORE
    with _LOCK:
        if _STORE is not None:
            return _STORE
        if not (INDEX_DIR / "index.faiss").exists():
            return None
        # Only load a FAISS index created by this application from a trusted directory.
        # LangChain's docstore is pickle-based; never load an untrusted index.
        _STORE = FAISS.load_local(
            str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
        )
        return _STORE

def ingest_pdf(pdf_path: str | Path):
    global _STORE
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("Provide an existing PDF file.")
    # Check PDF magic bytes; extension alone is insufficient.
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("File is not a valid PDF.")
    text = pymupdf4llm.to_markdown(str(path))
    chunks = chunk_document(text, source=path.name, chunk_size=1200, overlap=200)
    if not chunks:
        raise ValueError("No readable text was found in the PDF.")
    docs = [
        Document(page_content=c["text"], metadata={
            "source": c["source"], "chunk_index": c["chunk_index"],
            "chapter": c["chapter"] or "", "section": c["section"] or "",
            "subsection": c["subsection"] or "", "chunk_id": f"{path.name}:{c['id']}",
        }) for c in chunks
    ]
    with _LOCK:
        store = load_store()
        if store is None:
            _STORE = FAISS.from_documents(docs, get_embeddings(), normalize_L2=True)
        else:
            store.add_documents(docs)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _STORE.save_local(str(INDEX_DIR))
        _save_manifest(indexed_sources() + [path.name])
    return len(docs)

def ingest_uploaded_pdf(upload_path):
    """Copy Gradio's temporary file to app storage before indexing."""
    import shutil
    from uuid import uuid4
    source = Path(upload_path)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid4().hex[:8]}_{source.name}"
    dest = UPLOAD_DIR / safe_name
    shutil.copyfile(source, dest)
    try:
        return safe_name, ingest_pdf(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

def search_documents(question: str, top_k: int = 4):
    if not question.strip():
        return []
    with _LOCK:
        store = load_store()
        if store is None:
            return []
        # IndexFlatL2 on normalized vectors: cosine similarity = 1 - squared_L2 / 2.
        matches = store.similarity_search_with_score(question, k=max(1, min(top_k, 20)))
    return [
        {"text": doc.page_content, "metadata": doc.metadata,
         "similarity": round(max(-1.0, min(1.0, 1.0 - float(distance) / 2.0)), 4)}
        for doc, distance in matches
    ]
