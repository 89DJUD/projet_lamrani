"""Application Streamlit — Partie D : optimisation multicritère.

Lancement (depuis la racine du projet) :
    streamlit run partie_D/app.py

Application indépendante des modules B et I : elle ne lit que
``results/resultats_pour_D.csv`` et fonctionne hors ligne.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Permet les imports `partie_D.*` quel que soit le répertoire de lancement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from partie_D.analysis import sensitivity as sens
from partie_D.data_loader import DEFAULT_CSV, load_results
from partie_D.pipeline import min_max_normalize, run_pipeline
from partie_D.preprocessing import build_decision_matrix, to_benefit_space
from partie_D.visualization import plots
from partie_D.weighting import SCENARIOS, scenario_weights

st.set_page_config(page_title="Partie D — Optimisation multicritère",
                   layout="wide", page_icon="⚖️")

PAGES = [
    "1. Vue d'ensemble",
    "2. Matrice de décision",
    "3. Méthodes multicritères",
    "4. Comparaison",
    "5. Robustesse",
    "6. Conclusion",
]


@st.cache_data(show_spinner=False)
def _load(csv_path: str):
    return load_results(csv_path)


@st.cache_data(show_spinner="Exécution des trois méthodes...")
def _pipeline(csv_path: str, scenario: str, v: float,
              custom_key: str):
    custom = None
    if scenario == "personnalise" and custom_key:
        custom = [float(x) for x in custom_key.split(",")]
    return run_pipeline(csv_path, scenario=scenario, v=v,
                        custom_weights=custom)


def main() -> None:
    st.title("Optimisation multicritère — RIM, TOPSIS, VIKOR")
    st.caption(
        "Human-Centered AI for Migrant Integration — Partie D. "
        "Classement des configurations RAG + LLM à partir de "
        "`results/resultats_pour_D.csv`."
    )

    # ---------- barre latérale ----------
    with st.sidebar:
        st.header("Paramètres")
        csv_path = st.text_input("Chemin du CSV", value=str(DEFAULT_CSV))
        scenario = st.selectbox("Scénario de pondération", SCENARIOS,
                                format_func=lambda s: {
                                    "uniforme": "Uniforme",
                                    "qualite": "Priorité qualité",
                                    "rapidite": "Priorité rapidité",
                                    "equite_env": "Priorité équité/environnement",
                                    "personnalise": "Personnalisée",
                                }.get(s, s))
        v = st.slider("Paramètre v de VIKOR", 0.0, 1.0, 0.5, 0.05,
                      help="0 = importance maximale au regret individuel, "
                           "1 = à l'utilité collective.")
        custom_key = ""
        if scenario == "personnalise":
            st.info("Saisir les poids séparés par des virgules, "
                    "dans l'ordre des critères affiché page 2.")
            custom_key = st.text_input("Poids personnalisés",
                                       placeholder="ex: 0.2,0.1,...")
        page = st.radio("Navigation", PAGES)

    try:
        df, meta = _load(csv_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de charger le CSV : {exc}")
        return

    try:
        res = _pipeline(csv_path, scenario, v, custom_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur lors de l'analyse : {exc}")
        return

    d = res.decision

    # ---------- Page 1 : Vue d'ensemble ----------
    if page == PAGES[0]:
        st.header("Vue d'ensemble")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Configurations", d.n_alternatives)
        c2.metric("Critères", d.n_criteria)
        c3.metric("Lignes CSV", meta["n_rows"])
        c4.metric("Scénario", scenario)

        st.markdown(
            "**Objectif** : identifier la meilleure configuration RAG + LLM "
            "en recherchant le compromis entre qualité des réponses, "
            "pertinence sémantique, fiabilité, équité, impact "
            "environnemental et temps de réponse — en utilisant "
            "uniquement les critères présents dans le CSV."
        )
        st.markdown(
            f"**Fichier** : `{meta['path']}` — séparateur `{meta['delimiter']}`, "
            f"encodage `{meta['encoding']}`."
        )
        if d.report.get("duplicate_configs"):
            st.warning(f"Configurations en double détectées : "
                       f"{d.report['duplicate_configs']}")

        st.subheader("Critères et sens d'optimisation")
        crit_df = pd.DataFrame(
            {
                "critère": d.criteria,
                "sens": ["maximiser" if x > 0 else "minimiser"
                         for x in d.directions],
                "poids": res.weights,
            }
        )
        st.dataframe(crit_df, use_container_width=True, hide_index=True)
        st.caption(res.weight_description)

        st.subheader("Matrice de décision brute")
        st.dataframe(d.df, use_container_width=True, hide_index=True)

        st.subheader("Classements (pondération actuelle)")
        st.dataframe(res.ranks_df, use_container_width=True, hide_index=True)

    # ---------- Page 2 : Matrice de décision ----------
    elif page == PAGES[1]:
        st.header("Matrice de décision")
        st.subheader("Matrice originale X")
        st.dataframe(
            pd.DataFrame(d.X, columns=d.criteria, index=d.config_names),
            use_container_width=True,
        )

        st.subheader("Sens d'optimisation")
        st.dataframe(
            pd.DataFrame(
                {
                    "critère": d.criteria,
                    "direction": ["MAX" if x > 0 else "MIN" for x in d.directions],
                    "groupe": [
                        "qualité" if c in ("bleu", "rouge1", "rouge2",
                                           "rougeL", "semantic_similarity")
                        else "rapidité" if c == "seconds_median"
                        else "fiabilité" if c in ("hit_page", "mrr")
                        else "autre"
                        for c in d.criteria
                    ],
                }
            ),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Matrice normalisée [0, 1] (1 = meilleur)")
        st.dataframe(
            pd.DataFrame(res.X_norm, columns=d.criteria,
                         index=d.config_names),
            use_container_width=True,
        )
        st.plotly_chart(
            plots.plot_normalized_matrix(
                res.X_norm, d.config_names, d.criteria, d.directions
            ),
            use_container_width=True,
        )

        st.subheader("Pondération")
        wdf = pd.DataFrame(
            {"critère": d.criteria, "poids": res.weights}
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)
        st.caption(res.weight_description)
        st.bar_chart(wdf.set_index("critère"))

        with st.expander("Rapport de qualité des données"):
            st.json(d.report)

    # ---------- Page 3 : Méthodes multicritères ----------
    elif page == PAGES[2]:
        st.header("Méthodes multicritères")
        tab_t, tab_v, tab_r = st.tabs(["TOPSIS", "VIKOR", "RIM"])

        with tab_t:
            st.markdown(
                "**TOPSIS** (Hwang & Yoon, 1981) : normalisation vectorielle, "
                "pondération, distances aux solutions idéale positive A+ et "
                "négative A−, coefficient de proximité "
                "`C_i = D_i− / (D_i+ + D_i−)` ∈ [0,1]."
            )
            st.dataframe(res.topsis_df, use_container_width=True,
                         hide_index=True)
            st.plotly_chart(plots.plot_topsis(res.topsis_df),
                            use_container_width=True)

        with tab_v:
            st.markdown(
                f"**VIKOR** (Opricovic & Tzeng, 2004), v = {v} : "
                "utilité collective `S_i = Σ w_j f_ij`, regret individuel "
                "`R_i = max_j w_j f_ij`, indice "
                "`Q_i = v·S_i/S* + (1−v)·R_i/R*`. Plus petit Q = meilleur."
            )
            st.dataframe(res.vikor_df, use_container_width=True,
                         hide_index=True)
            st.plotly_chart(plots.plot_vikor(res.vikor_df),
                            use_container_width=True)
            ok = res.vikor_df[res.vikor_df["compromise_ok"]]["config"].tolist()
            st.info(f"Solutions de compromis VIKOR (conditions "
                    f"avantage + stabilité) : {ok}")

        with tab_r:
            st.markdown(
                "**RIM — Reference Ideal Method** (Sałabun et al., 2019) : "
                "distance de chaque alternative à un intervalle idéal de "
                "référence par critère, normalisation par l'étendue, "
                "pondération, distances aux solutions idéale/anti-idéale, "
                "score de proximité `C_i = D_i−/(D_i+ + D_i−)`."
            )
            st.warning(
                "Mode **exploratoire** : aucun seuil métier n'étant fourni, "
                "l'intervalle idéal est réduit à la meilleure valeur "
                "observée de chaque critère. Limite : le classement dépend "
                "alors des valeurs extrêmes du jeu de test."
            )
            st.dataframe(res.rim_df, use_container_width=True,
                         hide_index=True)
            st.plotly_chart(plots.plot_rim(res.rim_df),
                            use_container_width=True)

    # ---------- Page 4 : Comparaison ----------
    elif page == PAGES[3]:
        st.header("Comparaison des trois méthodes")
        st.dataframe(res.ranks_df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        for col, (m, cfg) in zip((c1, c2, c3), res.best.items()):
            col.metric(f"Meilleure ({m})", cfg)

        st.subheader("Heatmap des rangs")
        st.plotly_chart(plots.plot_rank_heatmap(res.ranks_df),
                        use_container_width=True)

        st.subheader("Corrélations des rangs")
        st.markdown("**Spearman**")
        st.dataframe(res.spearman.round(3), use_container_width=True)
        st.markdown("**Kendall**")
        st.dataframe(res.kendall.round(3), use_container_width=True)

        st.subheader("Accord Top-3")
        st.json(res.topk)

        st.subheader("Analyse des désaccords")
        st.dataframe(res.disagreement, use_container_width=True,
                     hide_index=True)
        st.caption(
            "Les scores bruts RIM/TOPSIS/VIKOR ne sont pas comparables : "
            "seuls les rangs sont comparés."
        )

        st.subheader("Radar des meilleures configurations")
        top3 = res.ranks_df.head(3)["config"].tolist()
        st.plotly_chart(
            plots.plot_radar(res.X_norm, d.config_names, d.criteria, top3),
            use_container_width=True,
        )

    # ---------- Page 5 : Robustesse ----------
    elif page == PAGES[4]:
        st.header("Robustesse et sensibilité")
        st.caption(
            "La robustesse est évaluée par rapport aux hypothèses de "
            "pondération uniquement — pas à toutes les incertitudes "
            "du système."
        )

        scen = st.multiselect(
            "Scénarios à comparer",
            ["uniforme", "qualite", "rapidite", "equite_env"],
            default=["uniforme", "qualite", "rapidite", "equite_env"],
        )
        scen_data = {}
        for sc in scen:
            w, desc = scenario_weights(d.criteria, sc)
            scen_data[sc] = {
                "weights": w,
                "description": desc,
                "ranks": sens.run_all_methods(d.X, w, d.directions,
                                              d.config_names, v=v),
            }
        if scen_data:
            st.plotly_chart(
                plots.plot_weight_influence(scen_data, d.config_names),
                use_container_width=True,
            )
            for sc, sd in scen_data.items():
                with st.expander(f"Scénario « {sc} » — {sd['description']}"):
                    st.dataframe(
                        pd.DataFrame(
                            {m: sd["ranks"][m] for m in sd["ranks"]}
                        ),
                        use_container_width=True,
                    )

        st.subheader("Simulation Monte-Carlo des poids")
        n_iter = st.slider("Itérations", 100, 2000, 500, 100)
        seed = st.number_input("Graine aléatoire", value=42, step=1)
        if st.button("Lancer la simulation", type="primary"):
            with st.spinner("Simulation en cours..."):
                mc = sens.monte_carlo_weights(
                    d.X, d.directions, d.config_names,
                    n_iter=int(n_iter), seed=int(seed), v=v,
                )
            st.session_state["mc"] = mc
        if "mc" in st.session_state:
            mc = st.session_state["mc"]
            st.plotly_chart(
                plots.plot_first_place_frequency(mc, d.config_names),
                use_container_width=True,
            )
            st.plotly_chart(
                plots.plot_rank_stability(mc, d.config_names),
                use_container_width=True,
            )
            mc_rows = []
            for m, s in mc["methods"].items():
                for cfg in d.config_names:
                    mc_rows.append({
                        "methode": m, "config": cfg,
                        "freq_rang1": s["freq_premier"].get(cfg, 0.0),
                        "rang_moyen": s["rang_moyen"].get(cfg, np.nan),
                        "rang_std": s["rang_std"].get(cfg, np.nan),
                    })
            st.dataframe(pd.DataFrame(mc_rows), use_container_width=True,
                         hide_index=True)

        st.subheader("Sensibilité du paramètre v (VIKOR)")
        v_sens = sens.vikor_v_sensitivity(
            d.X, res.weights, d.directions, d.config_names
        )
        st.plotly_chart(plots.plot_vikor_v(v_sens, d.config_names),
                        use_container_width=True)
        changes = v_sens[v_sens["best_config"] != v_sens["best_config"].iloc[0]]
        if len(changes):
            st.warning(
                f"La meilleure configuration VIKOR change {len(changes)} "
                f"fois quand v varie : "
                f"{v_sens[['v', 'best_config']].drop_duplicates().to_dict('records')}"
            )

    # ---------- Page 6 : Conclusion ----------
    else:
        st.header("Conclusion")
        st.subheader("Meilleures configurations par méthode")
        st.dataframe(
            pd.DataFrame(
                {"méthode": list(res.best.keys()),
                 "configuration": list(res.best.values())}
            ),
            use_container_width=True, hide_index=True,
        )
        consensus = res.ranks_df.head(3)[["config", "rang_moyen"]]
        st.subheader("Top 3 par rang moyen")
        st.dataframe(consensus, use_container_width=True, hide_index=True)

        st.subheader("Compromis qualité / rapidité")
        qcols = [c for c in d.criteria
                 if c in ("bleu", "rouge1", "rouge2", "rougeL",
                          "semantic_similarity")]
        speed = "seconds_median" if "seconds_median" in d.criteria else None
        if qcols and speed:
            qual = d.df[qcols].mean(axis=1)
            comp_df = pd.DataFrame(
                {"config": d.config_names,
                 "qualité_moyenne": qual.values,
                 "seconds_median": d.df[speed].values if speed in d.df
                 else np.nan}
            )
            st.dataframe(comp_df.sort_values("qualité_moyenne",
                                             ascending=False),
                         use_container_width=True, hide_index=True)

        st.subheader("Limites de l'analyse")
        st.markdown(
            "- Les scénarios de pondération non uniformes sont des "
            "**hypothèses d'analyse**, non des préférences validées par "
            "les utilisateurs migrants.\n"
            "- RIM est utilisé en **mode exploratoire** (intervalle idéal "
            "= meilleure valeur observée) faute de seuils métier.\n"
            "- Aucune métrique de biais ou d'impact environnemental "
            "n'est présente dans le CSV : ces dimensions ne sont pas "
            "couvertes par l'optimisation.\n"
            "- `seconds_median` de C4 (79 s) est probablement gonflé par "
            "une perturbation du PC pendant la mesure.\n"
            "- La robustesse est évaluée vis-à-vis des poids uniquement."
        )

        st.subheader("Téléchargements")
        csv_out = res.ranks_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Télécharger les rangs (CSV)", csv_out,
                           "rangs_partie_D.csv", "text/csv")
        report = _synthese(res)
        st.download_button("Télécharger le rapport synthétique (MD)",
                           report.encode("utf-8"),
                           "rapport_partie_D.md", "text/markdown")
        with st.expander("Rapport synthétique"):
            st.markdown(report)


def _synthese(res) -> str:
    d = res.decision
    lines = [
        "# Rapport synthétique — Partie D",
        "",
        f"- Configurations : {d.n_alternatives} | Critères : {d.n_criteria}",
        f"- Pondération : {res.weight_description}",
        "",
        "## Meilleure configuration par méthode",
    ]
    for m, cfg in res.best.items():
        lines.append(f"- **{m}** : {cfg}")
    lines += ["", "## Top 3 (rang moyen)"]
    for _, row in res.ranks_df.head(3).iterrows():
        lines.append(f"- {row['config']} (rang moyen {row['rang_moyen']:.2f})")
    sp = res.spearman.round(3)
    header = "| méthode | " + " | ".join(sp.columns) + " |"
    sep = "|" + "---|" * (len(sp.columns) + 1)
    rows = [
        "| " + idx + " | " + " | ".join(str(v) for v in row) + " |"
        for idx, row in sp.iterrows()
    ]
    lines += [
        "",
        "## Corrélation de Spearman entre rangs",
        "",
        header,
        sep,
        *rows,
        "",
        "## Limites",
        "- Hypothèses de pondération non validées par les utilisateurs.",
        "- RIM en mode exploratoire (pas de seuils métier).",
        "- Pas de métrique biais/environnement dans le CSV.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
