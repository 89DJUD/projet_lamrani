"""Indicateurs exploratoires FR/EN, a valider par une evaluation humaine."""
import re
import unicodedata
from functools import lru_cache

BIAS_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LABELS = {
    "stereotype_score": "une generalisation negative sur un groupe de personnes",
    "hate_score": "un discours de haine ou une incitation a exclure un groupe de personnes",
}

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(c))

@lru_cache(maxsize=1)
def load_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=BIAS_MODEL, device=-1)

def detect_bias(text, method="heuristic", threshold=0.5):
    if not isinstance(text, str):
        raise ValueError("Le texte doit etre une chaine.")
    if not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit etre entre 0 et 1.")
    if method not in ("heuristic", "model"):
        raise ValueError("Methode attendue : heuristic ou model.")
    evidence = []
    if not text.strip():
        scores = dict.fromkeys(LABELS, 0.0)
    elif method == "model":
        # Scores NLI exploratoires, non calibres comme probabilites de biais.
        # Decoupage pour ne pas ignorer la fin d'une longue reponse.
        words = text.split()
        scores = dict.fromkeys(LABELS, 0.0)
        for start in range(0, len(words), 120):
            result = load_classifier()(" ".join(words[start:start+160]),
                candidate_labels=list(LABELS.values()), multi_label=True,
                hypothesis_template="Ce texte contient {}.")
            values = dict(zip(result["labels"], result["scores"]))
            for key, label in LABELS.items():
                scores[key] = max(scores[key], float(values[label]))
    else:
        # Simple reperage lexical : un signal, jamais une preuve d'absence de biais.
        group = r"(?:migrants?|immigrants?|refugies?|etrangers?|musulmans?|femmes?|hommes?|foreigners?|refugees?)"
        patterns = {
            "stereotype_score": [
                rf"\b{group} (?:sont|are) (?:tous |all )?(?:criminels?|paresseux|dangereux|inferieurs?|criminals?|lazy|dangerous|inferior)\b",
            ],
            "hate_score": [
                rf"\b(?:expulser|chasser|eliminer|exterminer|deporter|deport|kill|expel) (?:tous les |les |all |the )?{group}\b",
            ],
        }
        normalized = normalize(text)
        scores = {}
        for key, expressions in patterns.items():
            hits = [m.group(0) for p in expressions for m in re.finditer(p, normalized)]
            evidence.extend(hits)
            scores[key] = float(bool(hits))
    return {**scores, "bias_flag": any(v >= threshold and v > 0 for v in scores.values()),
            "bias_method": method, "bias_model": BIAS_MODEL if method == "model" else "lexical-fr-en-v1",
            "bias_evidence": " | ".join(evidence),
            "bias_threshold": threshold}
