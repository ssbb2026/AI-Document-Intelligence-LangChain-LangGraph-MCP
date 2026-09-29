from unittest.mock import patch
from evaluation import evaluate_retrieval

def test_retrieval_metrics():
    matches = [
        {"metadata": {"chunk_id": "a"}},
        {"metadata": {"chunk_id": "b"}},
    ]
    with patch("evaluation.search_documents", return_value=matches):
        scores = evaluate_retrieval([{"question": "q", "relevant_chunk_ids": ["b"]}], k=2)
    assert scores == {"hit_at_k": 1., "precision_at_k": .5, "recall_at_k": 1., "mrr": .5}
