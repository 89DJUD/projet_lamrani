"""Comparaison rigoureuse des classements RIM, TOPSIS et VIKOR.

Les scores bruts des trois méthodes ne sont PAS comparables entre eux
(échelles et sens différents) : on compare les RANGS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

METHODS = ["TOPSIS", "VIKOR", "RIM"]


def ranks_table(
    topsis_df: pd.DataFrame,
    vikor_df: pd.DataFrame,
    rim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tableau config × méthode avec rang et score de chaque méthode."""
    t = topsis_df.set_index("config")[["topsis_score", "rank"]].rename(
        columns={"topsis_score": "score_TOPSIS", "rank": "rang_TOPSIS"}
    )
    v = vikor_df.set_index("config")[["vikor_Q", "rank"]].rename(
        columns={"vikor_Q": "score_VIKOR", "rank": "rang_VIKOR"}
    )
    r = rim_df.set_index("config")[["rim_score", "rank"]].rename(
        columns={"rim_score": "score_RIM", "rank": "rang_RIM"}
    )
    table = t.join(v).join(r)
    table["rang_moyen"] = table[["rang_TOPSIS", "rang_VIKOR", "rang_RIM"]].mean(axis=1)
    return table.sort_values("rang_moyen").reset_index()


def rank_correlations(ranks: pd.DataFrame) -> pd.DataFrame:
    """Corrélations de Spearman et de Kendall entre les rangs des méthodes."""
    cols = {"TOPSIS": "rang_TOPSIS", "VIKOR": "rang_VIKOR", "RIM": "rang_RIM"}
    methods = list(cols)
    spearman = pd.DataFrame(index=methods, columns=methods, dtype=float)
    kendall = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            x = ranks[cols[m1]].to_numpy()
            y = ranks[cols[m2]].to_numpy()
            if m1 == m2:
                spearman.loc[m1, m2] = 1.0
                kendall.loc[m1, m2] = 1.0
            else:
                spearman.loc[m1, m2] = spearmanr(x, y).statistic
                kendall.loc[m1, m2] = kendalltau(x, y).statistic
    return spearman, kendall


def topk_agreement(ranks: pd.DataFrame, k: int = 3) -> dict:
    """Accord sur le Top-k entre méthodes (intersections d'ensembles)."""
    cols = {"TOPSIS": "rang_TOPSIS", "VIKOR": "rang_VIKOR", "RIM": "rang_RIM"}
    tops = {m: set(ranks.loc[ranks[c] <= k, "config"]) for m, c in cols.items()}
    pairs = {}
    ms = list(tops)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            inter = tops[a] & tops[b]
            pairs[f"{a} vs {b}"] = {
                "commun": sorted(inter),
                "taille": len(inter),
                "jaccard": len(inter) / len(tops[a] | tops[b])
                if tops[a] | tops[b]
                else 1.0,
            }
    common_all = set.intersection(*tops.values()) if tops else set()
    return {"top_sets": tops, "pairs": pairs, "top_all_methods": sorted(common_all)}


def best_per_method(
    topsis_df: pd.DataFrame,
    vikor_df: pd.DataFrame,
    rim_df: pd.DataFrame,
) -> dict[str, str]:
    """Meilleure configuration selon chaque méthode (rang 1)."""
    return {
        "TOPSIS": str(topsis_df.iloc[0]["config"]),
        "VIKOR": str(vikor_df.iloc[0]["config"]),
        "RIM": str(rim_df.iloc[0]["config"]),
    }


def disagreement_analysis(ranks: pd.DataFrame) -> pd.DataFrame:
    """Écart-type des rangs par configuration : mesure les désaccords."""
    cols = ["rang_TOPSIS", "rang_VIKOR", "rang_RIM"]
    out = ranks[["config"] + cols].copy()
    out["ecart_type_rangs"] = out[cols].std(axis=1)
    out["etendue_rangs"] = out[cols].max(axis=1) - out[cols].min(axis=1)
    return out.sort_values("ecart_type_rangs", ascending=False).reset_index(drop=True)
