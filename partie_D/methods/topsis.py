"""TOPSIS — Technique for Order Preference by Similarity to Ideal Solution.

Référence : Hwang & Yoon (1981), *Multiple Attribute Decision Making*.

Étapes :
1. Normalisation vectorielle  r_ij = x_ij / sqrt(Σ_i x_ij²)
2. Pondération              v_ij = w_j · r_ij
3. Solution idéale positive A+ (max des critères à maximiser, min des
   critères à minimiser) et solution idéale négative A-.
4. Distances euclidiennes D_i+, D_i-.
5. Coefficient de proximité  C_i = D_i- / (D_i+ + D_i-)  ∈ [0, 1].

Implémentation directe des équations, sans bibliothèque dédiée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import to_benefit_space
from ..weighting import normalize_weights


def topsis(
    X: np.ndarray,
    weights: np.ndarray | list[float],
    directions: np.ndarray | list[float],
    config_names: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule TOPSIS et retourne un DataFrame avec scores et rangs.

    Colonnes retournées : config, topsis_score, rank, D_plus, D_minus.
    Le rang 1 est la meilleure configuration (C_i maximal).
    """
    X = np.asarray(X, dtype=float)
    w = normalize_weights(weights)
    d = np.asarray(directions, dtype=float)
    if X.ndim != 2:
        raise ValueError("X doit être une matrice 2D.")
    if X.shape[1] != w.size or X.shape[1] != d.size:
        raise ValueError("Dimensions incohérentes entre X, poids et directions.")
    if config_names is None:
        config_names = [f"A{i + 1}" for i in range(X.shape[0])]

    # 1-2. Normalisation vectorielle puis pondération, dans l'espace
    # « tout à maximiser » (la normalisation commute avec le changement
    # de signe par critère à minimiser).
    Xb = to_benefit_space(X, d)
    norms_b = np.sqrt((Xb ** 2).sum(axis=0))
    norms_b[norms_b == 0] = np.nan
    Rb = np.nan_to_num(Xb / norms_b[np.newaxis, :], nan=0.0)
    Vb = Rb * w[np.newaxis, :]

    A_plus = Vb.max(axis=0)   # solution idéale positive
    A_minus = Vb.min(axis=0)  # solution idéale négative (anti-idéale)

    # 4. Distances euclidiennes
    D_plus = np.sqrt(((Vb - A_plus[np.newaxis, :]) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Vb - A_minus[np.newaxis, :]) ** 2).sum(axis=1))

    # 5. Coefficient de proximité (dénominateur nul -> C = 0.5 neutre)
    denom = D_plus + D_minus
    C = np.where(denom > 0, D_minus / np.where(denom == 0, 1, denom), 0.5)

    out = pd.DataFrame(
        {
            "config": config_names,
            "topsis_score": C,
            "rank": pd.Series(C).rank(ascending=False, method="min").astype(int),
            "D_plus": D_plus,
            "D_minus": D_minus,
        }
    )
    return out.sort_values(["topsis_score", "config"], ascending=[False, True]).reset_index(drop=True)
