"""Chargement du CSV de résultats pour la partie D.

Détecte automatiquement le séparateur et l'encodage, sans jamais
modifier le fichier source.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# results/resultats_pour_D.csv par défaut (partie_D/ -> projet/ -> results/)
DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent / "results" / "resultats_pour_D.csv"
)

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]


def _detect_dialect(path: Path) -> tuple[str, str]:
    """Détecte (séparateur, encodage) en lisant un échantillon du fichier."""
    raw = path.read_bytes()
    text = None
    encoding = "utf-8-sig"
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Encodage non reconnu pour {path}")

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    return delimiter, encoding


def load_results(path: str | Path = DEFAULT_CSV) -> tuple[pd.DataFrame, dict]:
    """Charge le CSV et retourne (DataFrame, métadonnées de chargement)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    delimiter, encoding = _detect_dialect(path)
    df = pd.read_csv(path, sep=delimiter, encoding=encoding)

    meta = {
        "path": str(path),
        "delimiter": delimiter,
        "encoding": encoding,
        "n_rows": len(df),
        "columns": list(df.columns),
    }
    return df, meta
