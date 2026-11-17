"""Analyse de sensibilité et robustesse des classements.

La robustesse est évaluée PAR RAPPORT AUX HYPOTHÈSES DE PONDÉRATION
uniquement — pas à toutes les incertitudes du système (qualité des
métriques, représentativité du test set, matériel de mesure, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..methods.rim import rim
from ..methods.topsis import topsis
from ..methods.vikor import vikor
from ..weighting import random_weights


def _rank_series(result_df: pd.DataFrame, score_col: str, ascending: bool) -> pd.Series:
    """Série config -> rang (1 = meilleur) depuis un DataFrame méthode."""
    s = result_df.set_index("config")[score_col]
    return s.rank(ascending=ascending, method="min")


def run_all_methods(
    X: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray,
    config_names: list[str],
    v: float = 0.5,
) -> dict[str, pd.Series]:
    """Exécute les trois méthodes et retourne les rangs par config."""
    t = topsis(X, weights, directions, config_names)
    vdf = vikor(X, weights, directions, v=v, config_names=config_names)
    r = rim(X, weights, directions, config_names=config_names)
    return {
        "TOPSIS": _rank_series(t, "topsis_score", ascending=False),
        "VIKOR": _rank_series(vdf, "vikor_Q", ascending=True),
        "RIM": _rank_series(r, "rim_score", ascending=False),
    }


def monte_carlo_weights(
    X: np.ndarray,
    directions: np.ndarray,
    config_names: list[str],
    n_iter: int = 500,
    seed: int = 42,
    v: float = 0.5,
) -> dict:
    """Simulation de perturbations des poids (Dirichlet, graine fixe).

    Retourne, par méthode : fréquence de rang 1, rang moyen, écart-type.
    """
    n_crit = X.shape[1]
    samples = random_weights(n_crit, n_iter, seed=seed)

    rank_records = {m: [] for m in ["TOPSIS", "VIKOR", "RIM"]}
    first_counts = {m: {} for m in rank_records}

    for w in samples:
        ranks = run_all_methods(X, w, directions, config_names, v=v)
        for m, series in ranks.items():
            rank_records[m].append(series)
            best = series.idxmin()
            first_counts[m][best] = first_counts[m].get(best, 0) + 1

    summary = {}
    for m, records in rank_records.items():
        arr = pd.DataFrame(records)  # lignes = itérations, colonnes = configs
        summary[m] = {
            "freq_premier": {c: first_counts[m].get(c, 0) / n_iter for c in config_names},
            "rang_moyen": arr.mean(axis=0).to_dict(),
            "rang_std": arr.std(axis=0).to_dict(),
            "rang_min": arr.min(axis=0).to_dict(),
            "rang_max": arr.max(axis=0).to_dict(),
        }
    return {"n_iter": n_iter, "seed": seed, "methods": summary}


def vikor_v_sensitivity(
    X: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray,
    config_names: list[str],
    v_values: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Étudie l'influence du paramètre v de VIKOR sur Q et le rang 1."""
    if v_values is None:
        v_values = np.linspace(0.0, 1.0, 21)
    rows = []
    for v in v_values:
        df = vikor(X, weights, directions, v=float(v), config_names=config_names)
        best = df.iloc[0]
        rows.append(
            {
                "v": float(v),
                "best_config": best["config"],
                "best_Q": float(best["vikor_Q"]),
                **{
                    f"Q_{c}": float(df.set_index("config").loc[c, "vikor_Q"])
                    for c in config_names
                },
            }
        )
    return pd.DataFrame(rows)


def scenario_rankings(
    X: np.ndarray,
    directions: np.ndarray,
    criteria: list[str],
    config_names: list[str],
    scenarios: list[str] | None = None,
    custom_weights: np.ndarray | None = None,
    v: float = 0.5,
) -> dict:
    """Classements pour chaque scénario de pondération.

    Retourne {scénario: {"weights":..., "description":..., "ranks": {...}}}.
    """
    from ..weighting import scenario_weights

    if scenarios is None:
        scenarios = ["uniforme", "qualite", "rapidite", "equite_env"]
    out = {}
    for sc in scenarios:
        w, desc = scenario_weights(criteria, sc, custom_weights=custom_weights)
        ranks = run_all_methods(X, w, directions, config_names, v=v)
        out[sc] = {"weights": w, "description": desc, "ranks": ranks}
    return out
