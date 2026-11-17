"""Préparation des données : matrice de décision, directions, qualité.

Règle fondamentale du projet : toutes les colonnes de scores sont
maximisées, sauf ``seconds_median`` qui est minimisée.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Colonnes identifiant la configuration expérimentale (pas des critères)
IDENTIFIER_COLS = [
    "config",
    "model",
    "chunk_size",
    "top_k",
    "temperature",
    "n_questions",
]

# Critères réellement présents dans resultats_pour_D.csv
CRITERIA_MAXIMIZE = [
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "semantic_similarity",
    "hit_page",
    "mrr",
]
CRITERIA_MINIMIZE = ["seconds_median"]
ALL_CRITERIA = CRITERIA_MAXIMIZE + CRITERIA_MINIMIZE

# Regroupements thématiques utilisés par les scénarios de pondération
GROUP_QUALITY = ["bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity"]
GROUP_RELIABILITY = ["hit_page", "mrr"]          # fiabilité / pertinence RAG
GROUP_SPEED = ["seconds_median"]                # rapidité
GROUP_EQUITY_ENV = []                            # absent du CSV actuel


@dataclass
class DecisionData:
    """Conteneur de la matrice de décision et de son rapport qualité."""

    X: np.ndarray                      # matrice alternatives × critères
    config_names: list[str]            # identifiants des configurations
    criteria: list[str]                # noms des critères (colonnes de X)
    directions: np.ndarray             # +1 = maximiser, -1 = minimiser
    df: pd.DataFrame                   # lignes sources (une par config)
    report: dict = field(default_factory=dict)

    @property
    def n_alternatives(self) -> int:
        return self.X.shape[0]

    @property
    def n_criteria(self) -> int:
        return self.X.shape[1]


def _criterion_directions(criteria: list[str]) -> np.ndarray:
    directions = []
    for c in criteria:
        if c in CRITERIA_MINIMIZE:
            directions.append(-1)
        elif c in CRITERIA_MAXIMIZE:
            directions.append(+1)
        else:
            raise ValueError(
                f"Critère '{c}' sans direction définie. "
                "L'ajouter à CRITERIA_MAXIMIZE ou CRITERIA_MINIMIZE."
            )
    return np.array(directions, dtype=float)


def quality_report(df: pd.DataFrame, criteria: list[str]) -> dict:
    """Rapport de qualité des données pour les colonnes critères."""
    report: dict = {"n_rows": len(df), "criteria": {}}
    for c in criteria:
        s = pd.to_numeric(df[c], errors="coerce")
        report["criteria"][c] = {
            "dtype": str(df[c].dtype),
            "n_missing": int(s.isna().sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
            "is_constant": bool(s.nunique(dropna=True) <= 1),
        }
    # Doublons sur l'identifiant de configuration
    id_col = "config" if "config" in df.columns else None
    if id_col is not None:
        dup = df[id_col].duplicated(keep=False)
        report["duplicate_configs"] = (
            df.loc[dup, id_col].unique().tolist() if dup.any() else []
        )
    return report


def build_decision_matrix(
    df: pd.DataFrame,
    criteria: list[str] | None = None,
    missing_strategy: str = "drop_rows",
    aggregate: str = "mean",
) -> DecisionData:
    """Construit la matrice de décision X depuis le DataFrame brut.

    Paramètres
    ----------
    df : DataFrame issu de ``data_loader.load_results``.
    criteria : critères à utiliser (défaut : tous ceux présents).
    missing_strategy :
        - ``"drop_rows"`` : supprime les lignes avec NA sur un critère ;
        - ``"drop_criteria"`` : retire les critères contenant des NA ;
        - ``"fail"`` : lève une erreur.
    aggregate : si plusieurs lignes partagent le même ``config``,
        agrégation des critères par ``"mean"`` (défaut), ``"median"``
        ou ``"first"``. Les lignes du CSV sont déjà des médianes par
        configuration ; l'agrégation ne sert que de garde-fou.
    """
    if criteria is None:
        criteria = [c for c in ALL_CRITERIA if c in df.columns]
    missing = [c for c in criteria if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes critères absentes du CSV : {missing}")

    work = df.copy()
    for c in criteria:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # --- valeurs manquantes -------------------------------------------
    na_mask = work[criteria].isna().any(axis=1)
    n_na_rows = int(na_mask.sum())
    if n_na_rows:
        if missing_strategy == "fail":
            raise ValueError(f"{n_na_rows} lignes contiennent des NA.")
        elif missing_strategy == "drop_rows":
            work = work.loc[~na_mask].copy()
        elif missing_strategy == "drop_criteria":
            bad = [c for c in criteria if work[c].isna().any()]
            criteria = [c for c in criteria if c not in bad]
            work = work.dropna(subset=criteria).copy()
        else:
            raise ValueError(f"Stratégie inconnue : {missing_strategy}")

    # --- agrégation par configuration ----------------------------------
    id_col = "config" if "config" in work.columns else None
    if id_col is not None and work[id_col].duplicated().any():
        agg_map = {"mean": "mean", "median": "median", "first": "first"}
        if aggregate not in agg_map:
            raise ValueError(f"Agrégation inconnue : {aggregate}")
        work = (
            work.groupby(id_col, as_index=False, sort=False)[criteria]
            .agg(agg_map[aggregate])
        )
        # conserver les colonnes descriptives de la première occurrence
        desc_cols = [c for c in df.columns if c not in criteria + [id_col]]
        if desc_cols:
            first_rows = df.drop_duplicates(subset=[id_col], keep="first")[
                [id_col] + desc_cols
            ]
            work = work.merge(first_rows, on=id_col, how="left")

    if len(work) == 0:
        raise ValueError("Aucune ligne exploitable après traitement des NA.")

    X = work[criteria].to_numpy(dtype=float)
    config_names = (
        work[id_col].astype(str).tolist() if id_col else [f"A{i+1}" for i in range(len(work))]
    )
    directions = _criterion_directions(criteria)

    report = quality_report(work, criteria)
    report.update(
        {
            "missing_strategy": missing_strategy,
            "n_rows_dropped_na": n_na_rows,
            "aggregation": aggregate,
            "n_alternatives": len(work),
            "n_criteria": len(criteria),
        }
    )

    return DecisionData(
        X=X,
        config_names=config_names,
        criteria=criteria,
        directions=directions,
        df=work.reset_index(drop=True),
        report=report,
    )


def to_benefit_space(X: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Transforme X pour que TOUS les critères soient à maximiser.

    Pour un critère à minimiser (direction = -1), on utilise ``-x``.
    """
    return X * directions[np.newaxis, :]
