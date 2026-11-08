"""BLEU, ROUGE et similarité sémantique. Tous les scores sont entre 0 et 1."""

from functools import lru_cache

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

SIMILARITY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

class _FrenchTokenizer:
    def tokenize(self, text):
        import re
        return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


_rouge = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False, tokenizer=_FrenchTokenizer())


@lru_cache(maxsize=1)
def load_similarity_model():
    # Import local : ce module est lent à charger.
    # Reutilise le meme modele multilingue que le module de recherche, sans dupliquer ses poids.
    from rag import load_embedder

    return load_embedder()


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
    if not all(isinstance(t, str) and t.strip() for t in (prediction, reference)):
        raise ValueError("Prediction et reference doivent etre des textes non vides.")
    return {
        "bleu": compute_bleu(prediction, reference),
        **compute_rouge(prediction, reference),
        "semantic_similarity": compute_semantic_similarity(
            prediction, reference
        ),
    }


def evaluate_test_set(path, generate, on_progress=None, bias_method="heuristic",
                      measure_energy=True, country_iso_code="MAR", limit=None):
    """Génère et note une réponse pour chaque ligne du CSV.

    Le CSV doit contenir les colonnes question et reference. La colonne
    context est facultative : vide ou absente, `generate` reçoit None
    (le contexte est alors cherché dans les documents par le RAG).
    `generate(question, context)` renvoie un texte ou un dictionnaire contenant
    prediction et la provenance du contexte. source reste la source de référence ;
    retrieved_sources décrit les passages réellement fournis au modèle.
    `on_progress(done, total)`, optionnel, est appelé avant chaque question.
    """
    from .fairness import detect_bias
    from .footprint import EnergyMonitor
    frame = pd.read_csv(path)
    if frame.empty or not {"question", "reference"}.issubset(frame.columns):
        raise ValueError("CSV non vide attendu avec question et reference.")
    if not frame[["question", "reference"]].map(lambda x: isinstance(x, str) and bool(x.strip())).all().all():
        raise ValueError("Questions ou references manquantes dans le CSV.")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("La limite doit etre un entier positif.")
        frame = frame.head(limit)
    records = frame.to_dict("records")
    rows = []
    for i, row in enumerate(records):
        if on_progress:
            on_progress(i, len(records))
        context = row.get("context")
        if not isinstance(context, str) or not context.strip():
            context = None
        with EnergyMonitor(measure_energy, country_iso_code) as monitor:
            generated = generate(row["question"], context)
        provenance={}
        if isinstance(generated,dict):
            prediction=generated["prediction"]
            provenance={key:generated[key] for key in (
                "context","retrieved_sources","context_origin","generation_skipped","prompt_version") if key in generated}
        else:
            prediction=generated
        rows.append(
            {
                **row,
                **provenance,
                "prediction": prediction,
                **evaluate_response(prediction, row["reference"]),
                **detect_bias(prediction, method=bias_method),
                **monitor.metrics,
                "evaluation_version": "multilingual-v1",
                "similarity_model": SIMILARITY_MODEL,
            }
        )
    if on_progress:
        on_progress(len(records), len(records))
    return pd.DataFrame(rows)
