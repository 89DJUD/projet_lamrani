"""Point d'entrée en ligne de commande : exécute toute l'analyse D
et exporte les résultats dans ``partie_D/results/``.

Usage :
    python -m partie_D.run_analysis            # depuis la racine du projet
    python partie_D/run_analysis.py            # idem
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partie_D.analysis import sensitivity as sens
from partie_D.data_loader import DEFAULT_CSV
from partie_D.pipeline import run_pipeline
from partie_D.weighting import SCENARIOS, scenario_weights

OUT_DIR = Path(__file__).resolve().parent / "results"


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    print(f"  -> {OUT_DIR / name}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Chargement de", DEFAULT_CSV)
    res = run_pipeline(scenario="uniforme", v=0.5)
    d = res.decision

    print(f"{d.n_alternatives} configurations, {d.n_criteria} critères : "
          f"{d.criteria}")
    print("Rapport qualité :", {k: v for k, v in d.report.items()
                                if k != "criteria"})

    # --- classements ---------------------------------------------------
    _save(res.topsis_df, "classement_topsis.csv")
    _save(res.vikor_df, "classement_vikor.csv")
    _save(res.rim_df, "classement_rim.csv")
    _save(res.ranks_df, "rangs_comparaison.csv")
    _save(res.disagreement, "desaccords_methodes.csv")
    _save(res.spearman.reset_index().rename(columns={"index": "methode"}),
          "correlation_spearman.csv")
    _save(res.kendall.reset_index().rename(columns={"index": "methode"}),
          "correlation_kendall.csv")

    # --- scénarios de pondération --------------------------------------
    scen_rows = []
    for sc in SCENARIOS:
        if sc == "personnalise":
            continue
        w, desc = scenario_weights(d.criteria, sc)
        ranks = sens.run_all_methods(d.X, w, d.directions, d.config_names)
        for m, series in ranks.items():
            scen_rows.append(
                {
                    "scenario": sc,
                    "methode": m,
                    "best_config": series.idxmin(),
                    "description": desc,
                }
            )
    _save(pd.DataFrame(scen_rows), "scenarios_poids.csv")

    # --- Monte-Carlo ----------------------------------------------------
    print("Monte-Carlo (500 itérations, seed=42)...")
    mc = sens.monte_carlo_weights(
        d.X, d.directions, d.config_names, n_iter=500, seed=42
    )
    mc_rows = []
    for m, s in mc["methods"].items():
        for cfg in d.config_names:
            mc_rows.append(
                {
                    "methode": m,
                    "config": cfg,
                    "freq_rang1": s["freq_premier"].get(cfg, 0.0),
                    "rang_moyen": s["rang_moyen"].get(cfg, np.nan),
                    "rang_std": s["rang_std"].get(cfg, np.nan),
                    "rang_min": s["rang_min"].get(cfg, np.nan),
                    "rang_max": s["rang_max"].get(cfg, np.nan),
                }
            )
    _save(pd.DataFrame(mc_rows), "robustesse_montecarlo.csv")

    # --- sensibilité v --------------------------------------------------
    w_uniform, _ = scenario_weights(d.criteria, "uniforme")
    v_sens = sens.vikor_v_sensitivity(
        d.X, w_uniform, d.directions, d.config_names
    )
    _save(v_sens, "sensibilite_v_vikor.csv")

    # --- matrice de décision normalisée ---------------------------------
    X_norm_df = pd.DataFrame(res.X_norm, columns=d.criteria,
                             index=d.config_names)
    X_norm_df.insert(0, "config", d.config_names)
    _save(X_norm_df.reset_index(drop=True), "matrice_normalisee.csv")

    print("Terminé. Résultats dans", OUT_DIR)


if __name__ == "__main__":
    main()
