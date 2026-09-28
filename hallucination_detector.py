"""Claim-level NLI verification of generated answers against retrieved PDF passages.

This is a conservative automated check, not a guarantee of factual correctness.
"""
import re
from functools import lru_cache
from config import NLI_MODEL, ENTAILMENT_THRESHOLD

ABSTAIN = "I could not verify the answer from the document."
CITATION_RE = re.compile(r"\[Source\s+(\d+)\]", re.I)

@lru_cache(maxsize=1)
def load_verifier():
    from transformers import pipeline
    # The model is downloaded only when an answer needs verification.
    return pipeline("text-classification", model=NLI_MODEL, tokenizer=NLI_MODEL)


def split_claims(answer: str) -> list[str]:
    """Split short factual answers into claims; discard citation markers."""
    answer = CITATION_RE.sub("", answer)
    return [s.strip(" -\n\t") for s in re.split(r"(?<=[.!?])\s+|\n+", answer)
            if s.strip(" -\n\t")]


def entailment_score(premise: str, hypothesis: str, verifier=None) -> float:
    """Return model-estimated entailment probability for one source/claim pair."""
    model = verifier if verifier is not None else load_verifier()
    # top_k=None requests all labels so entailment is not inferred from the top label.
    predictions = model({"text": premise, "text_pair": hypothesis}, top_k=None,
                        truncation=True)
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    return max((float(p["score"]) for p in predictions
                if p["label"].lower() in ("entailment", "label_2")), default=0.0)


def verify_answer(answer: str, passages: list[dict], verifier=None,
                  threshold: float = ENTAILMENT_THRESHOLD) -> dict:
    """Require each sentence to be supported by at least one retrieved passage.

    Citations, if present, restrict evidence to the cited passage numbers.
    Unverified answers are withheld rather than silently shown as grounded.
    """
    if not answer or not passages:
        return {"verified": False, "answer": ABSTAIN, "claims": []}
    if answer.strip() == "I could not find the answer in the document.":
        return {"verified": False, "answer": answer, "claims": []}

    reports = []
    # Preserve citation association when the model emits one citation per sentence.
    answer = re.sub(r"([.!?])\s*(\[Source\s+\d+\])", r" \2\1", answer, flags=re.I)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    for sentence in sentences:
        cited = [int(n) for n in CITATION_RE.findall(sentence)]
        claim = CITATION_RE.sub("", sentence).strip(" -\n\t")
        if not claim:
            continue
        if cited and any(n < 1 or n > len(passages) for n in cited):
            reports.append({"claim": claim, "supported": False, "score": 0.0, "source": None})
            continue
        candidates = [(n, passages[n-1]) for n in cited] if cited else list(enumerate(passages, 1))
        scores = [(entailment_score(p["text"], claim, verifier), n) for n, p in candidates]
        best_score, best_source = max(scores, default=(0.0, None))
        reports.append({"claim": claim, "supported": best_score >= threshold,
                        "score": round(best_score, 4), "source": best_source})
    verified = bool(reports) and all(c["supported"] for c in reports)
    return {"verified": verified, "answer": answer if verified else ABSTAIN,
            "claims": reports}
