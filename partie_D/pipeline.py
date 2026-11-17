"""Pipeline commun : charge les données et exécute les trois méthodes.

Utilisé par l'application Streamlit (``app.py``) et par le script
d'export (``run_analysis.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import comparison as comp
from .analysis import sensitivity as sens
from .data_loader import DEFAULT_CSV, load_results
from .methods.rim import rim
from .methods.topsis import topsis
from .methods.vikor import vikor
from .preprocessing import DecisionData, build_decision_matrix, to_benefit_space
from .weighting import scenario_weights


@dataclass
class PipelineResult:
    decision: DecisionData
    weights: np.ndarray
    weight_description: str
    topsis_df: pd.DataFrame
    vikor_df: pd.DataFrame
    rim_df: pd.DataFrame
    ranks_df: pd.DataFrame
    spearman: pd.DataFrame
    kendall: pd.DataFrame
    topk: dict
    best: dict
    disagreement: pd.DataFrame
    X_norm: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def min_max_normalize(Xb: np.ndarray) -> np.ndarray:
    """Normalisation min-max par critère vers [0, 1] (1 = meilleur)."""
    mn = Xb.min(axis=0)
    mx = Xb.max(axis=0)
    span = mx - mn
    span[span == 0] = 1.0
    return (Xb - mn[np.newaxis, :]) / span[np.newaxis, :]


def run_pipeline(
    csv_path=str(DEFAULT_CSV),
    scenario: str = "uniforme",
    custom_weights: np.ndarray | list[float] | None = None,
    v: float = 0.5,
    criteria: list[str] | None = None,
    missing_strategy: str = "drop_rows",
) -> PipelineResult:
    df, meta = load_results(csv_path)
    decision = build_decision_matrix(
        df, criteria=criteria, missing_strategy=missing_strategy
    )
    X, directions = decision.X, decision.directions
    names, crit = decision.config_names, decision.criteria

    w, wdesc = scenario_weights(crit, scenario, custom_weights=custom_weights)

    t = topsis(X, w, directions, names)
    vdf = vikor(X, w, directions, v=v, config_names=names)
    r = rim(X, w, directions, config_names=names)

    ranks = comp.ranks_table(t, vdf, r)
    sp, kd = comp.rank_correlations(ranks)
    topk = comp.topk_agreement(ranks, k=min(3, len(names)))
    best = comp.best_per_method(t, vdf, r)
    dis = comp.disagreement_analysis(ranks)

    Xb = to_benefit_space(X, directions)
    X_norm = min_max_normalize(Xb)

    return PipelineResult(
        decision=decision,
        weights=w,
        weight_description=wdesc,
        topsis_df=t,
        vikor_df=vdf,
        rim_df=r,
        ranks_df=ranks,
        spearman=sp,
        kendall=kd,
        topk=topk,
        best=best,
        disagreement=dis,
        X_norm=X_norm,
    )
