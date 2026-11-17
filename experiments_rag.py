"""Partie B — évaluation de la recherche (sans LLM).

Pour chaque taille de passage (50 à 1000 mots) et chaque top-K (1 à 10),
on vérifie si la page source attendue (colonne "source" de test_set.csv)
fait partie des passages retrouvés.

Mesures (moyenne sur les questions du corpus, hors question hors sujet) :
- hit_page : la bonne page est dans les K passages (Hit@K) ;
- hit_doc  : le bon document est dans les K passages ;
- mrr      : 1 / rang de la première bonne page (0 si absente du top-K).

Lancement : python experiments_rag.py
Résultat  : results/retrieval_results.csv (une ligne par combinaison)
"""

import re
import time
from pathlib import Path

import pandas as pd

import rag

TEST_SET = Path("test_set.csv")
OUT_DIR = Path("results")


def parse_source(source):
    """'08_x.pdf p.9 et p.4' -> ('08_x.pdf', {9, 4}) ; 'hors sujet' -> (None, set())."""
    match = re.match(r"(\S+\.pdf)", source)
    if not match:
        return None, set()
    return match.group(1), {int(n) for n in re.findall(r"p\.(\d+)", source)}


def main():
    questions = pd.read_csv(TEST_SET)
    gold = [parse_source(s) for s in questions["source"]]
    max_k = max(rag.TOP_K_VALUES)

    rows, off_topic = [], []
    for size in rag.CHUNK_SIZES:
        start = time.time()
        # min_score=0 : on veut le classement complet, sans le seuil.
        results = [rag.retrieve(q, max_k, size, min_score=0.0) for q in questions["question"]]
        seconds = (time.time() - start) / len(questions)

        for k in rag.TOP_K_VALUES:
            hits_page, hits_doc, rr = [], [], []
            for (doc, pages), passages in zip(gold, results):
                if doc is None:
                    continue
                top = passages[:k]
                ranks = [
                    i for i, p in enumerate(top, start=1)
                    if p["source"] == doc and p["page"] in pages
                ]
                hits_page.append(bool(ranks))
                hits_doc.append(any(p["source"] == doc for p in top))
                rr.append(1 / ranks[0] if ranks else 0.0)
            rows.append({
                "chunk_size": size,
                "top_k": k,
                "hit_page": sum(hits_page) / len(hits_page),
                "hit_doc": sum(hits_doc) / len(hits_doc),
                "mrr": sum(rr) / len(rr),
                "seconds_per_query": round(seconds, 3),
            })

        # Score du meilleur passage : questions du corpus vs hors sujet
        # (sert à choisir un seuil de similarité).
        for (doc, _), passages in zip(gold, results):
            off_topic.append({
                "chunk_size": size,
                "hors_sujet": doc is None,
                "top1_score": passages[0]["score"],
            })
        print(f"chunk {size} done ({seconds:.2f}s/question)", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT_DIR / "retrieval_results.csv", index=False)

    scores = pd.DataFrame(off_topic)
    summary = scores.groupby(["chunk_size", "hors_sujet"])["top1_score"].agg(["min", "mean", "max"]).round(3)
    summary.to_csv(OUT_DIR / "top1_scores.csv")

    print(df.to_string(index=False))
    print("\nHit@K (bonne page) — lignes = taille, colonnes = K")
    print(df.pivot(index="chunk_size", columns="top_k", values="hit_page").to_string())
    print("\nScore du 1er passage (corpus vs hors sujet)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
