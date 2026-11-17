"""VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje.

Référence : Opricovic & Tzeng (2004), *Compromise Solution by MCDM
Methods: A Comparative Analysis of VIKOR and TOPSIS*, EJOR 156(2).

Étapes :
1. Normalisation  f_ij = (x_ij - f_j^-) / (f_j^+ - f_j^-)
   où f_j^+ = meilleure valeur, f_j^- = pire valeur du critère j
   (f_ij = 0 au meilleur, 1 au pire ; on travaille en "espace regret").
2. Utilité collective  S_i = Σ_j w_j f_ij
3. Regret individuel     R_i = max_j w_j f_ij
4. Indice VIKOR         Q_i = v·S_i/S* + (1-v)·R_i/R*
   avec S* = min_i S_i, R* = min_i R_i, v ∈ [0, 1] (v = 0.5 par défaut).

Conditions de compromis (Opricovic & Tzeng) : pour le meilleur Q_1 et
un candidat Q_k, l'avantage est acceptable si Q_1 - Q_k ≥ 1/(n-1) et la
stabilité si R_k - R_1 ≥ 1/(n-1). Sinon, l'ensemble des solutions de
compromis est présenté sans imposer une solution unique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import to_benefit_space
from ..weighting import normalize_weights


def _normalized_regret(Xb: np.ndarray) -> np.ndarray:
    """f_ij ∈ [0,1] : 0 = meilleure valeur du critère, 1 = pire.

    Xb est dans l'espace « tout à maximiser ».
    """
    best = Xb.max(axis=0)
    worst = Xb.min(axis=0)
    span = best - worst
    span[span == 0] = np.nan  # critère constant : pas de regret
    F = (best[np.newaxis, :] - Xb) / span[np.newaxis, :]
    return np.nan_to_num(F, nan=0.0)


def vikor(
    X: np.ndarray,
    weights: np.ndarray | list[float],
    directions: np.ndarray | list[float],
    v: float = 0.5,
    config_names: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule VIKOR. Retourne S, R, Q, rang et indicateurs de compromis.

    Le rang 1 est la configuration avec le plus petit Q_i.
    """
    X = np.asarray(X, dtype=float)
    w = normalize_weights(weights)
    d = np.asarray(directions, dtype=float)
    if not 0.0 <= v <= 1.0:
        raise ValueError("v doit être dans [0, 1].")
    if config_names is None:
        config_names = [f"A{i + 1}" for i in range(X.shape[0])]

    Xb = to_benefit_space(X, d)
    F = _normalized_regret(Xb)

    S = (w[np.newaxis, :] * F).sum(axis=1)
    R = (w[np.newaxis, :] * F).max(axis=1)

    S_star = S.min()
    R_star = R.min()
    # dénominateurs nuls : si toutes les alternatives sont identiques
    denom_S = S_star if S_star != 0 else 1.0
    denom_R = R_star if R_star != 0 else 1.0
    Q = v * S / denom_S + (1.0 - v) * R / denom_R

    n = len(Q)
    order = np.argsort(Q, kind="stable")
    # rangs avec ex æquo (method="min" : les ex æquo partagent le rang)
    ranks = pd.Series(Q).rank(ascending=True, method="min").astype(int).to_numpy()

    # Conditions de compromis VIKOR
    q1 = Q.min()
    r1 = R[order[0]]
    threshold = 1.0 / (n - 1) if n > 1 else 0.0
    acceptable_advantage = Q - q1 >= threshold - 1e-12
    acceptable_stability = R - r1 >= threshold - 1e-12
    is_compromise = acceptable_advantage & acceptable_stability
    # la meilleure solution satisfait toujours les deux conditions
    is_compromise[order[0]] = True

    out = pd.DataFrame(
        {
            "config": config_names,
            "vikor_Q": Q,
            "vikor_S": S,
            "vikor_R": R,
            "rank": ranks,
            "compromise_ok": is_compromise,
        }
    )
    return out.sort_values(["vikor_Q", "config"], ascending=[True, True]).reset_index(drop=True)
