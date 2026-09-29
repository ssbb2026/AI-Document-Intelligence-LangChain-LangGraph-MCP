from hallucination_detector import verify_answer, ABSTAIN

PASSAGES = [{"text": "Earth has a crust and a mantle."}]

def fake_verifier(inputs, **kwargs):
    claim = inputs["text_pair"].lower()
    supported = "crust" in claim and "mantle" not in claim
    return [{"label": "entailment", "score": 0.95 if supported else 0.03},
            {"label": "neutral", "score": 0.05 if supported else 0.97}]

def test_supported_claim():
    result = verify_answer("Earth has a crust. [Source 1]", PASSAGES, fake_verifier)
    assert result["verified"]
    assert result["claims"][0]["source"] == 1

def test_unsupported_claim_is_withheld():
    result = verify_answer("Earth has a diamond mantle. [Source 1]", PASSAGES, fake_verifier)
    assert not result["verified"]
    assert result["answer"] == ABSTAIN

def test_unknown_citation_is_rejected():
    result = verify_answer("Earth has a crust. [Source 9]", PASSAGES, fake_verifier)
    assert not result["verified"]

def test_mixed_answer_is_withheld():
    result = verify_answer("Earth has a crust. [Source 1]\nEarth has a diamond mantle. [Source 1]", PASSAGES, fake_verifier)
    assert not result["verified"]
