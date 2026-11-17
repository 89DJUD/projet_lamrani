"""Graphiques scientifiques (Plotly) pour la partie D.

Chaque fonction retourne une ``plotly.graph_objects.Figure`` et
s'adapte automatiquement au nombre de configurations et de critères.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palette sobre et lisible
COLORS = {
    "TOPSIS": "#1f77b4",
    "VIKOR": "#d62728",
    "RIM": "#2ca02c",
}


def _ranking_bar(df: pd.DataFrame, score_col: str, title: str,
                 color: str, higher_is_better: bool = True) -> go.Figure:
    """Bar chart horizontal du classement (meilleure config en haut)."""
    d = df.sort_values(score_col, ascending=not higher_is_better).reset_index(drop=True)
    fig = go.Figure(
        go.Bar(
            x=d[score_col],
            y=d["config"],
            orientation="h",
            marker_color=color,
            text=d[score_col].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Configuration",
        yaxis=dict(autorange="reversed"),
        height=max(350, 60 * len(d) + 120),
        margin=dict(l=20, r=60, t=60, b=40),
    )
    return fig


def plot_topsis(topsis_df: pd.DataFrame) -> go.Figure:
    return _ranking_bar(
        topsis_df, "topsis_score",
        "Classement TOPSIS (C_i, plus proche de 1 = meilleur)",
        COLORS["TOPSIS"],
    )


def plot_vikor(vikor_df: pd.DataFrame) -> go.Figure:
    return _ranking_bar(
        vikor_df, "vikor_Q",
        "Classement VIKOR (Q_i, plus petit = meilleur)",
        COLORS["VIKOR"],
        higher_is_better=False,
    )


def plot_rim(rim_df: pd.DataFrame) -> go.Figure:
    return _ranking_bar(
        rim_df, "rim_score",
        "Classement RIM (score de proximité, plus proche de 1 = meilleur)",
        COLORS["RIM"],
    )


def plot_rank_heatmap(ranks_df: pd.DataFrame) -> go.Figure:
    """Heatmap des rangs : lignes = configurations, colonnes = méthodes."""
    cols = {"TOPSIS": "rang_TOPSIS", "VIKOR": "rang_VIKOR", "RIM": "rang_RIM"}
    z = ranks_df[[cols[m] for m in cols]].to_numpy().T
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=ranks_df["config"].tolist(),
            y=list(cols.keys()),
            text=z,
            texttemplate="%{text}",
            colorscale="RdYlGn_r",  # rang 1 = vert, rangs élevés = rouge
            colorbar=dict(title="Rang"),
        )
    )
    fig.update_layout(
        title="Rangs par méthode (1 = meilleure configuration)",
        xaxis_title="Configuration",
        yaxis_title="Méthode",
        height=320,
    )
    return fig


def plot_normalized_matrix(
    X_norm: np.ndarray,
    config_names: list[str],
    criteria: list[str],
    directions: np.ndarray,
) -> go.Figure:
    """Heatmap des critères normalisés [0,1] (1 = meilleur)."""
    fig = go.Figure(
        go.Heatmap(
            z=X_norm.T,
            x=config_names,
            y=criteria,
            colorscale="Viridis",
            colorbar=dict(title="Valeur normalisée"),
        )
    )
    fig.update_layout(
        title="Critères normalisés par configuration (1 = meilleur)",
        xaxis_title="Configuration",
        yaxis_title="Critère",
        height=max(350, 40 * len(criteria) + 150),
    )
    return fig


def plot_radar(
    X_norm: np.ndarray,
    config_names: list[str],
    criteria: list[str],
    top_configs: list[str] | None = None,
) -> go.Figure:
    """Radar des meilleures configurations sur les critères normalisés."""
    idx = {c: i for i, c in enumerate(config_names)}
    if top_configs is None:
        top_configs = config_names[: min(4, len(config_names))]
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for k, cfg in enumerate(top_configs):
        i = idx[cfg]
        values = list(X_norm[i]) + [X_norm[i][0]]  # boucle
        theta = criteria + [criteria[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                name=cfg,
                line_color=palette[k % len(palette)],
            )
        )
    fig.update_layout(
        title="Radar des meilleures configurations (critères normalisés)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480,
    )
    return fig


def plot_weight_influence(
    scenario_ranks: dict,
    config_names: list[str],
) -> go.Figure:
    """Influence des scénarios de poids sur le rang de chaque config."""
    scenarios = list(scenario_ranks.keys())
    methods = ["TOPSIS", "VIKOR", "RIM"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=methods,
                        shared_yaxes=True)
    for col, m in enumerate(methods, start=1):
        for cfg in config_names:
            ranks = [scenario_ranks[s]["ranks"][m].loc[cfg] for s in scenarios]
            fig.add_trace(
                go.Scatter(
                    x=scenarios, y=ranks, mode="lines+markers",
                    name=cfg, legendgroup=cfg,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_layout(
        title="Influence des scénarios de pondération sur les rangs",
        height=420,
    )
    return fig


def plot_first_place_frequency(mc_summary: dict, config_names: list[str]) -> go.Figure:
    """Fréquence de rang 1 lors de la simulation Monte-Carlo."""
    methods = list(mc_summary["methods"].keys())
    fig = make_subplots(rows=1, cols=len(methods), subplot_titles=methods)
    for col, m in enumerate(methods, start=1):
        freq = mc_summary["methods"][m]["freq_premier"]
        fig.add_trace(
            go.Bar(
                x=config_names,
                y=[freq.get(c, 0.0) for c in config_names],
                marker_color=COLORS.get(m, "#7f7f7f"),
                text=[f"{freq.get(c, 0.0):.0%}" for c in config_names],
                textposition="outside",
            ),
            row=1, col=col,
        )
        fig.update_yaxes(title_text="Fréquence rang 1", range=[0, 1],
                         row=1, col=col)
    fig.update_layout(title="Fréquence de première place (Monte-Carlo)",
                      height=380, showlegend=False)
    return fig


def plot_rank_stability(mc_summary: dict, config_names: list[str]) -> go.Figure:
    """Stabilité : écart-type des rangs sur les simulations."""
    methods = list(mc_summary["methods"].keys())
    fig = make_subplots(rows=1, cols=len(methods), subplot_titles=methods)
    for col, m in enumerate(methods, start=1):
        std = mc_summary["methods"][m]["rang_std"]
        fig.add_trace(
            go.Bar(
                x=config_names,
                y=[std.get(c, 0.0) for c in config_names],
                marker_color=COLORS.get(m, "#7f7f7f"),
            ),
            row=1, col=col,
        )
    fig.update_layout(title="Stabilité des rangs (écart-type Monte-Carlo)",
                      height=380, showlegend=False)
    return fig


def plot_vikor_v(v_sens_df: pd.DataFrame, config_names: list[str]) -> go.Figure:
    """Influence du paramètre v sur Q_i et sur la meilleure config."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Q_i selon v", "Meilleure configuration selon v"),
        row_heights=[0.7, 0.3],
    )
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for k, c in enumerate(config_names):
        fig.add_trace(
            go.Scatter(
                x=v_sens_df["v"], y=v_sens_df[f"Q_{c}"],
                mode="lines+markers", name=c,
                line_color=palette[k % len(palette)],
            ),
            row=1, col=1,
        )
    # encode le meilleur rang par une ligne en pointillés
    best_codes = {c: i for i, c in enumerate(config_names)}
    fig.add_trace(
        go.Scatter(
            x=v_sens_df["v"],
            y=[best_codes[c] for c in v_sens_df["best_config"]],
            mode="lines", name="rang 1",
            line=dict(dash="dash", color="black"),
            yaxis="y2",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1,
                     tickvals=list(range(len(config_names))),
                     ticktext=config_names)
    fig.update_xaxes(title_text="v (0 = regret max, 1 = utilité collective)",
                     row=2, col=1)
    fig.update_layout(height=560, legend=dict(orientation="h",
                                              yanchor="bottom", y=1.02))
    return fig
