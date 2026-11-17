"""RIM — Reference Ideal Method.

Référence : Sałabun, Wątróbski & Jankowski (2019), *Reference Ideal
Method*, Advances in Intelligent Systems and Computing 1036 ;
voir aussi Sałabun (2020), *The New Method for Measurement of
Criteria Weight in MCDA Problems*, EAAI.

Étapes :
1. Domaine de chaque critère [min_j, max_j] et intervalle idéal de
   référence [L_j, U_j].
2. Distance de chaque alternative à l'intervalle idéal (0 si dedans).
3. Normalisation par l'étendue du critère, puis pondération.
4. Solution idéale = vecteur nul ; anti-idéale = max par critère.
5. Distances euclidiennes D_i+, D_i- dans l'espace pondéré.
6. Score de proximité  C_i = D_i- / (D_i+ + D_i-)  ∈ [0, 1].

Mode « exploratoire » : sans seuils métier fournis, l'intervalle idéal
est réduit à la MEILLEURE VALEUR OBSERVÉE de chaque critère (point).
C'est un choix documenté, pas un seuil inventé : il est explicitement
signalé dans les résultats et peut être remplacé par des intervalles
fournis par l'utilisateur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import to_benefit_space
from ..weighting import normalize_weights


def _distance_to_interval(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Distance (par critère) à l'intervalle [low, high], 0 si dedans."""
    below = np.maximum(low[np.newaxis, :] - x, 0.0)
    above = np.maximum(x - high[np.newaxis, :], 0.0)
    return below + above


def rim(
    X: np.ndarray,
    weights: np.ndarray | list[float],
    directions: np.ndarray | list[float],
    config_names: list[str] | None = None,
    ideal_intervals: dict[str, tuple[float, float]] | None = None,
    criteria: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule RIM et retourne scores, rangs et distances.

    Paramètres
    ----------
    ideal_intervals : {critère: (borne_inf, borne_sup)} optionnel.
        Les bornes sont exprimées dans l'échelle BRUTE du critère.
        Par défaut (None), mode exploratoire : intervalle = meilleure
        valeur observée (max si maximiser, min si minimiser).
    criteria : noms des critères (requis si ideal_intervals est fourni).
    """
    X = np.asarray(X, dtype=float)
    w = normalize_weights(weights)
    d = np.asarray(directions, dtype=float)
    n_alt, n_crit = X.shape
    if config_names is None:
        config_names = [f"A{i + 1}" for i in range(n_alt)]

    # Espace « tout à maximiser » : les intervalles idéaux sont alors
    # exprimés sur les valeurs transformées.
    Xb = to_benefit_space(X, d)

    if ideal_intervals is not None:
        if criteria is None:
            raise ValueError("criteria requis quand ideal_intervals est fourni.")
        low = np.empty(n_crit)
        high = np.empty(n_crit)
        for j, c in enumerate(criteria):
            if c not in ideal_intervals:
                raise ValueError(f"Intervalle idéal manquant pour '{c}'.")
            lo, hi = ideal_intervals[c]
            # conversion vers l'espace bénéfice
            if d[j] < 0:
                lo_t, hi_t = -hi, -lo
            else:
                lo_t, hi_t = lo, hi
            low[j], high[j] = lo_t, hi_t
        mode = "intervalles_fournis"
    else:
        # Mode exploratoire : point = meilleure valeur observée
        low = Xb.max(axis=0)
        high = Xb.max(axis=0)
        mode = "exploratoire_meilleure_valeur"

    # 2-3. Distances à l'intervalle, normalisation par l'étendue, poids
    D_raw = _distance_to_interval(Xb, low, high)
    span = Xb.max(axis=0) - Xb.min(axis=0)
    span[span == 0] = np.nan
    D_norm = np.nan_to_num(D_raw / span[np.newaxis, :], nan=0.0)
    D_weighted = D_norm * w[np.newaxis, :]

    # 4. Idéale = 0 ; anti-idéale = maximum par critère
    A_plus = np.zeros(n_crit)
    A_minus = D_weighted.max(axis=0)

    # 5. Distances euclidiennes
    D_plus = np.sqrt(((D_weighted - A_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((D_weighted - A_minus[np.newaxis, :]) ** 2).sum(axis=1))

    # 6. Score de proximité
    denom = D_plus + D_minus
    C = np.where(denom > 0, D_minus / np.where(denom == 0, 1, denom), 0.5)

    out = pd.DataFrame(
        {
            "config": config_names,
            "rim_score": C,
            "rank": pd.Series(C).rank(ascending=False, method="min").astype(int),
            "D_plus": D_plus,
            "D_minus": D_minus,
        }
    )
    out.attrs["rim_mode"] = mode
    return out.sort_values(["rim_score", "config"], ascending=[False, True]).reset_index(drop=True)
