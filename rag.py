"""Partie B : corpus documentaire et RAG.

1. Importer les PDF du dossier corpus/ (texte page par page).
2. Préparer les données : nettoyage et découpage en passages qui se chevauchent.
3. Calculer les embeddings des passages (modèle multilingue).
4. Les indexer avec FAISS : un index par taille de passage, dans index/chunk_<taille>/.
5. Recherche sémantique : retrouver les k passages les plus proches d'une question.

Hyperparamètres étudiés dans l'article (Castellanos-Nieves et García-Forte, 2025) :
- taille des passages (chunk size) : 50, 100, 300, 500, 1000 — ici en mots ;
- nombre de passages récupérés (top-K) : 1, 2, 5, 8, 10.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

CORPUS_DIR = Path("corpus")
INDEX_ROOT = Path("index")

# Multilingue : français, anglais, espagnol, arabe, etc.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZES = [50, 100, 300, 500, 1000]  # en mots
TOP_K_VALUES = [1, 2, 5, 8, 10]
DEFAULT_CHUNK_SIZE = 300
DEFAULT_TOP_K = 2
OVERLAP_RATIO = 0.2  # 20 % de mots en commun entre deux passages voisins
# Les pages presque vides (titres, sommaires) gênent la recherche : on les écarte.
MIN_CHUNK_WORDS = 25
# Seuil de similarité (cosinus) : en dessous, le passage n'est pas donné au LLM.
# Mesuré avec experiments_rag.py : meilleur passage des questions du corpus
# >= 0,44 ; question hors sujet <= 0,35. Seuil fixé entre les deux.
SIMILARITY_THRESHOLD = 0.40
NO_INFO_ANSWER = "L'information n'est pas disponible dans les documents."


# --- 1. Import des PDF --------------------------------------------------------
def load_pdfs(corpus_dir=CORPUS_DIR):
    """Renvoie une liste de pages : {"source", "page", "text"}."""
    pages = []
    for path in sorted(Path(corpus_dir).glob("*.pdf")):
        reader = PdfReader(path)
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"source": path.name, "page": number, "text": text})
    return pages


# --- 2. Préparation des données ----------------------------------------------
def clean_text(text):
    text = re.sub(r"-\n(\w)", r"\1", text)  # mots coupés en fin de ligne
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, size, overlap):
    """Découpe en passages de `size` mots, avec `overlap` mots en commun."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [
        " ".join(words[start:start + size])
        for start in range(0, len(words) - overlap, step)
    ]


def make_chunks(pages, chunk_size):
    overlap = int(chunk_size * OVERLAP_RATIO)
    min_words = min(MIN_CHUNK_WORDS, chunk_size // 2)
    return [
        {"source": p["source"], "page": p["page"], "text": chunk}
        for p in pages
        for chunk in chunk_text(p["text"], chunk_size, overlap)
        if len(chunk.split()) >= min_words
    ]


# --- 3. Embeddings -------------------------------------------------------------
@lru_cache(maxsize=1)
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts):
    # Vecteurs normalisés : le produit scalaire FAISS = similarité cosinus.
    vectors = load_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


# --- 4. Indexation FAISS -------------------------------------------------------
def index_dir(chunk_size):
    return INDEX_ROOT / f"chunk_{chunk_size}"


def build_index(chunk_size=DEFAULT_CHUNK_SIZE, corpus_dir=CORPUS_DIR, pages=None):
    """Construit et sauvegarde l'index d'une taille de passage. Renvoie des statistiques."""
    if pages is None:
        pages = load_pdfs(corpus_dir)
    chunks = make_chunks(pages, chunk_size)
    if not chunks:
        raise ValueError(
            f"Aucun texte trouvé dans {corpus_dir}/. Ajoutez des PDF dont le "
            "texte est sélectionnable (les scans ne sont pas lus)."
        )

    vectors = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    folder = index_dir(chunk_size)
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(folder / "faiss.index"))
    (folder / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    load_index.cache_clear()

    return {
        "chunk_size": chunk_size,
        "documents": len({c["source"] for c in chunks}),
        "pages": len(pages),
        "chunks": len(chunks),
    }


def build_all_indexes(chunk_sizes=CHUNK_SIZES, corpus_dir=CORPUS_DIR, on_progress=None):
    """Construit un index par taille de passage. Les PDF ne sont lus qu'une fois."""
    pages = load_pdfs(corpus_dir)
    stats = []
    for i, size in enumerate(chunk_sizes):
        if on_progress:
            on_progress(i, len(chunk_sizes), size)
        stats.append(build_index(size, corpus_dir, pages))
    return stats


def available_chunk_sizes():
    return [s for s in CHUNK_SIZES if index_exists(s)]


def index_exists(chunk_size=DEFAULT_CHUNK_SIZE):
    folder = index_dir(chunk_size)
    return (folder / "faiss.index").exists() and (folder / "chunks.json").exists()


@lru_cache(maxsize=len(CHUNK_SIZES))
def load_index(chunk_size=DEFAULT_CHUNK_SIZE):
    folder = index_dir(chunk_size)
    index = faiss.read_index(str(folder / "faiss.index"))
    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


# --- 5. Recherche sémantique ---------------------------------------------------
def retrieve(question, k=DEFAULT_TOP_K, chunk_size=DEFAULT_CHUNK_SIZE,
             min_score=SIMILARITY_THRESHOLD):
    """Renvoie au plus k passages : {"source", "page", "text", "score"}.

    Les passages dont le score est inférieur à min_score sont écartés :
    une liste vide signifie que la question sort du corpus.
    """
    index, chunks = load_index(chunk_size)
    scores, ids = index.search(embed([question]), min(k, index.ntotal))
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1 and s >= min_score
    ]


def format_context(passages):
    """Assemble les passages en un seul texte de contexte pour le LLM."""
    return "\n\n".join(
        f"[{p['source']}, p. {p['page']}] {p['text']}" for p in passages
    )
