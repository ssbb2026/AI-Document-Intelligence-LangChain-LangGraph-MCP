"""Retrieval metrics; replace sample labels with verified chunk IDs."""
import json
from pathlib import Path
from rag_engine import search_documents

def evaluate_retrieval(items: list[dict], k: int = 5) -> dict:
    if not items:
        raise ValueError("Evaluation data is empty")
    totals = {"hit_at_k": 0., "precision_at_k": 0., "recall_at_k": 0., "mrr": 0.}
    for item in items:
        relevant = set(item["relevant_chunk_ids"])
        if not relevant:
            raise ValueError("Every evaluation question needs relevant_chunk_ids")
        retrieved = [
            p["metadata"]["chunk_id"]
            for p in search_documents(item["question"], k)
        ]
        hits = [i for i, chunk_id in enumerate(retrieved[:k], start=1) if chunk_id in relevant]
        totals["hit_at_k"] += float(bool(hits))
        totals["precision_at_k"] += len(hits) / k
        totals["recall_at_k"] += len(hits) / len(relevant)
        totals["mrr"] += 1 / hits[0] if hits else 0
    return {key: round(value / len(items), 4) for key, value in totals.items()}

if __name__ == "__main__":
    path = Path("evaluation_data.json")
    if not path.exists():
        raise SystemExit("Create evaluation_data.json from the example and label actual chunk IDs.")
    print(json.dumps(evaluate_retrieval(json.loads(path.read_text())), indent=2))
