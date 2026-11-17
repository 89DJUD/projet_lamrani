"""BLEU, ROUGE et similarité sémantique. Tous les scores sont entre 0 et 1."""

from functools import lru_cache

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

_rouge = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)


@lru_cache(maxsize=1)
def load_similarity_model():
    # Import local : ce module est lent à charger.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SIMILARITY_MODEL)


def compute_bleu(prediction, reference):
    # sacrebleu renvoie un score entre 0 et 100, parfois 100 + un arrondi.
    score = sacrebleu.sentence_bleu(prediction, [reference]).score / 100
    return min(1.0, score)


def compute_rouge(prediction, reference):
    scores = _rouge.score(reference, prediction)
    return {name: scores[name].fmeasure for name in ROUGE_TYPES}


def compute_semantic_similarity(prediction, reference):
    from sentence_transformers import util

    embeddings = load_similarity_model().encode(
        [prediction, reference],
        convert_to_tensor=True,
    )
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Le cosinus va de -1 à 1 : un score négatif (sans rapport) devient 0.
    return max(0.0, min(1.0, score))


def evaluate_response(prediction, reference):
    return {
        "bleu": compute_bleu(prediction, reference),
        **compute_rouge(prediction, reference),
        "semantic_similarity": compute_semantic_similarity(
            prediction, reference
        ),
    }


def evaluate_test_set(path, generate, on_progress=None):
    """Génère et note une réponse pour chaque ligne du CSV.

    Le CSV doit contenir les colonnes question et reference. La colonne
    context est facultative : vide ou absente, `generate` reçoit None
    (le contexte est alors cherché dans les documents par le RAG).
    `generate(question, context)` est la fonction de génération à évaluer.
    `on_progress(done, total)`, optionnel, est appelé avant chaque question.
    """
    records = pd.read_csv(path).to_dict("records")
    rows = []
    for i, row in enumerate(records):
        if on_progress:
            on_progress(i, len(records))
        context = row.get("context")
        if not isinstance(context, str) or not context.strip():
            context = None
        prediction = generate(row["question"], context)
        rows.append(
            {
                **row,
                "prediction": prediction,
                **evaluate_response(prediction, row["reference"]),
            }
        )
    return pd.DataFrame(rows)
