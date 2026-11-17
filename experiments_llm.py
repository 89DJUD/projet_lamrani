"""Partie B — expériences RAG + Llama 3 pour la partie D (MCDM).

Pour chaque configuration (taille des passages, top-K, température), Llama 3
répond aux questions de test_set.csv avec les passages retrouvés, puis on
calcule BLEU, ROUGE et la similarité sémantique avec la réponse de référence.

Le calcul est long (~5-6 h sur CPU) : chaque réponse est enregistrée tout de
suite dans results/llm_raw.csv. Si le script s'arrête, le relancer reprend là
où il en était.

Lancement : python experiments_llm.py
Résultats :
- results/llm_raw.csv         : une ligne par question et par configuration ;
- results/resultats_pour_D.csv : une ligne par configuration (médianes).
"""

import ctypes
import sys
import time
from pathlib import Path

import pandas as pd

from evaluation import evaluate_response
from llm import LLAMA_MODEL, generate_response
from rag import NO_INFO_ANSWER, format_context, retrieve

# Combinaisons retenues d'après experiments_rag.py (meilleur Hit@K),
# plus 500 mots / top-2 pour comparaison. 100 mots / top-10 a été retiré :
# ~100 s par réponse sur CPU, trop long pour le temps disponible.
CONFIGS = [(50, 2), (100, 5), (500, 2)]
TEMPERATURES = [0.1, 0.5, 0.9]

TEST_SET = Path("test_set.csv")
OUT_DIR = Path("results")
RAW = OUT_DIR / "llm_raw.csv"
SUMMARY = OUT_DIR / "resultats_pour_D.csv"
RETRIEVAL = OUT_DIR / "retrieval_results.csv"
METRICS = ["bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity"]


def keep_pc_awake():
    """Empêche la mise en veille de Windows tant que le script tourne."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def answer(question, chunk_size, top_k, temperature):
    passages = retrieve(question, top_k, chunk_size)
    if not passages:  # sous le seuil de similarité : hors corpus
        return NO_INFO_ANSWER
    return generate_response(question, format_context(passages), LLAMA_MODEL, temperature)


def run():
    questions = pd.read_csv(TEST_SET)
    done = set()
    if RAW.exists():
        old = pd.read_csv(RAW)
        done = set(zip(old.chunk_size, old.top_k, old.temperature, old.question))

    jobs = [
        (c, k, t, row)
        for c, k in CONFIGS
        for t in TEMPERATURES
        for row in questions.itertuples()
        if (c, k, t, row.question) not in done
    ]
    total = len(CONFIGS) * len(TEMPERATURES) * len(questions)
    print(f"{total - len(jobs)}/{total} déjà faits, {len(jobs)} à faire", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    start = time.time()
    for n, (c, k, t, row) in enumerate(jobs, start=1):
        for attempt in range(1, 4):  # 3 essais (délai dépassé, Ollama redémarré...)
            t0 = time.perf_counter()
            try:
                prediction = answer(row.question, c, k, t)
                break
            except Exception as error:
                print(f"  essai {attempt} échoué : {error!r}", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError("Ollama ne répond plus : relancer le script.")
        seconds = time.perf_counter() - t0
        record = {
            "model": LLAMA_MODEL, "chunk_size": c, "top_k": k, "temperature": t,
            "question": row.question, "reference": row.reference,
            "prediction": prediction, "seconds": round(seconds, 1),
            **evaluate_response(prediction, row.reference),
        }
        pd.DataFrame([record]).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
        eta = (time.time() - start) / n * (len(jobs) - n) / 60
        print(f"[{n}/{len(jobs)}] chunk {c} k{k} t{t} {seconds:.0f}s | reste ~{eta:.0f} min", flush=True)


def summarize():
    raw = pd.read_csv(RAW)
    keys = ["model", "chunk_size", "top_k", "temperature"]
    # Seules les configurations prévues et complètes (toutes les questions).
    raw = raw[[(c, k) in CONFIGS for c, k in zip(raw.chunk_size, raw.top_k)]]
    n_questions = len(pd.read_csv(TEST_SET))
    raw = raw.groupby(keys).filter(lambda g: len(g) == n_questions)
    summary = raw.groupby(keys).agg(
        **{m: (m, "median") for m in METRICS},
        seconds_median=("seconds", "median"),
        n_questions=("question", "count"),
    ).reset_index()
    if RETRIEVAL.exists():  # qualité de la recherche seule (Hit@K bonne page)
        hits = pd.read_csv(RETRIEVAL)[["chunk_size", "top_k", "hit_page", "mrr"]]
        summary = summary.merge(hits, on=["chunk_size", "top_k"], how="left")
    summary.insert(0, "config", [
        f"C{i}" for i in range(1, len(summary) + 1)
    ])
    summary.round(4).to_csv(SUMMARY, index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    keep_pc_awake()
    run()
    summarize()
