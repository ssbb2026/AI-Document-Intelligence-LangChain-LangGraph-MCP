from unittest.mock import patch
import rag_graph

def test_graph_answers_with_retrieved_passages():
    passages = [{"text": "Earth has a crust.", "similarity": 0.9,
                 "metadata": {"source": "earth.pdf", "chunk_index": 0, "chapter": "Earth", "section": ""}}]
    with patch.object(rag_graph, "search_documents", return_value=passages), \
         patch.object(rag_graph, "generate_answer", return_value="Earth has a crust. [Source 1]"), \
         patch.object(rag_graph, "verify_answer", return_value={"verified": True, "answer": "Earth has a crust. [Source 1]", "claims": []}):
        result = rag_graph.ask("What does Earth have?")
    assert "crust" in result["answer"]
    assert result["sources"][0]["source"] == "earth.pdf"

def test_graph_no_results():
    with patch.object(rag_graph, "search_documents", return_value=[]) as search:
        result = rag_graph.ask("Unanswerable question")
    assert "could not find" in result["answer"]
    assert search.call_count == 2
