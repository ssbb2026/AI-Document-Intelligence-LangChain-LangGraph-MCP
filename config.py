from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "faiss_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_NAME = os.getenv("LLM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "220"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

NLI_MODEL = os.getenv("NLI_MODEL", "facebook/bart-large-mnli")
ENTAILMENT_THRESHOLD = float(os.getenv("ENTAILMENT_THRESHOLD", "0.75"))
