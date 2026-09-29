import pytest
from chunking import chunk_document

def test_chunking_preserves_metadata_and_links():
    docs = chunk_document("# Earth\nHello world.\n\n## Crust\nThe crust is rocky.", "earth.pdf", 1200, 100)
    assert len(docs) == 2
    assert docs[0]["chapter"] == "Earth"
    assert docs[1]["section"] == "Crust"
    assert docs[0]["next_chunk"] == docs[1]["id"]
    assert docs[1]["previous_chunk"] == docs[0]["id"]

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document("# Earth\nHello", "earth.pdf", 100, 100)
