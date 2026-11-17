"""Système de pondération multicritère.

Cinq scénarios :
1. ``uniforme``     : poids égaux ;
2. ``qualite``      : priorité aux métriques de qualité des réponses ;
3. ``rapidite``     : priorité au temps de réponse ;
4. ``equite_env``   : priorité équité/environnement — absents du CSV,
   repli documenté sur la fiabilité (hit_page, mrr) ;
5. ``personnalise`` : poids choisis par l'utilisateur.

Tous les scénarios non uniformes sont des HYPOTHÈSES D'ANALYSE,
pas des préférences validées par les utilisateurs migrants.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import (
    GROUP_EQUITY_ENV,
    GROUP_QUALITY,
    GROUP_RELIABILITY,
    GROUP_SPEED,
)

SCENARIOS = ["uniforme", "qualite", "rapidite", "equite_env", "personnalise"]

# Poids des GROUPES par scénario (somme = 1). À l'intérieur d'un
# groupe, le poids est réparti uniformément sur ses critères présents.
_SCENARIO_GROUP_WEIGHTS = {
    "uniforme": None,  # cas particulier : 1/n par critère
    "qualite": {
        "quality": 0.60,
        "reliability": 0.20,
        "speed": 0.20,
        "equity_env": 0.0,
    },
    "rapidite": {
        "quality": 0.20,
        "reliability": 0.20,
        "speed": 0.60,
        "equity_env": 0.0,
    },
    "equite_env": {
        # Aucun critère biais/environnement dans le CSV : on documente
        # le repli sur la fiabilité de l'information (hit_page, mrr).
        "quality": 0.30,
        "reliability": 0.50,
        "speed": 0.20,
        "equity_env": 0.0,
    },
}

_GROUP_MEMBERS = {
    "quality": GROUP_QUALITY,
    "reliability": GROUP_RELIABILITY,
    "speed": GROUP_SPEED,
    "equity_env": GROUP_EQUITY_ENV,
}


def normalize_weights(weights: np.ndarray | list[float]) -> np.ndarray:
    """Valide et normalise des poids (somme = 1, tous >= 0, somme > 0)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("Les poids doivent former un vecteur non vide.")
    if np.any(np.isnan(w)):
        raise ValueError("Les poids contiennent des NaN.")
    if np.any(w < 0):
        raise ValueError("Tous les poids doivent être positifs ou nuls.")
    total = w.sum()
    if total <= 0:
        raise ValueError("La somme des poids est nulle : pondération invalide.")
    return w / total


def uniform_weights(n_criteria: int) -> np.ndarray:
    return np.full(n_criteria, 1.0 / n_criteria)


def _group_of(criterion: str) -> str:
    for g, members in _GROUP_MEMBERS.items():
        if criterion in members:
            return g
    return "other"


def scenario_weights(
    criteria: list[str],
    scenario: str = "uniforme",
    custom_weights: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, str]:
    """Retourne (poids normalisés, description) pour un scénario.

    ``custom_weights`` n'est utilisé que pour ``scenario="personnalise"``.
    """
    n = len(criteria)
    if scenario == "uniforme":
        return uniform_weights(n), "Pondération uniforme (1/n par critère)."

    if scenario == "personnalise":
        if custom_weights is None:
            raise ValueError(
                "scenario='personnalise' exige custom_weights."
            )
        w = normalize_weights(custom_weights)
        if w.size != n:
            raise ValueError(
                f"custom_weights a {w.size} valeurs, attendu {n}."
            )
        return w, "Pondération personnalisée fournie par l'utilisateur."

    if scenario not in _SCENARIO_GROUP_WEIGHTS:
        raise ValueError(f"Scénario inconnu : {scenario}. "
                         f"Choix possibles : {SCENARIOS}")

    group_w = _SCENARIO_GROUP_WEIGHTS[scenario]
    raw = np.zeros(n)
    notes = []
    for j, c in enumerate(criteria):
        g = _group_of(c)
        members = [m for m in _GROUP_MEMBERS.get(g, []) if m in criteria]
        if g == "other" or not members:
            # critère hors groupes connus : poids résiduel uniforme
            raw[j] = 0.0
            continue
        raw[j] = group_w[g] / len(members)

    # critères non classés : redistribuer le reliquat uniformément
    unclassified = [j for j, c in enumerate(criteria) if _group_of(c) == "other"]
    if unclassified:
        residual = 1.0 - raw.sum()
        for j in unclassified:
            raw[j] = residual / len(unclassified)

    w = normalize_weights(raw)

    if scenario == "equite_env" and not any(
        _group_of(c) == "equity_env" for c in criteria
    ):
        notes.append(
            "ATTENTION : aucun critère équité/environnement n'est présent "
            "dans le CSV. Le scénario est reporté sur la fiabilité "
            "(hit_page, mrr) — hypothèse d'analyse, non validée."
        )
    desc = (
        f"Scénario '{scenario}' : poids par groupe "
        f"qualité={group_w['quality']:.0%}, "
        f"fiabilité={group_w['reliability']:.0%}, "
        f"rapidité={group_w['speed']:.0%}"
        + (" (+ équité/env si présent)." if group_w["equity_env"] else ".")
    )
    if notes:
        desc += " " + " ".join(notes)
    return w, desc


def random_weights(
    n_criteria: int,
    n_iter: int,
    seed: int = 42,
    alpha: float | None = None,
) -> np.ndarray:
    """Tire ``n_iter`` vecteurs de poids via une loi de Dirichlet.

    ``alpha`` uniforme par défaut (loi de Dirichlet symétrique) :
    couvre tout le simplexe des poids. Graine fixe = reproductibilité.
    """
    rng = np.random.default_rng(seed)
    if alpha is None:
        alpha = np.ones(n_criteria)
    else:
        alpha = np.full(n_criteria, float(alpha))
    samples = rng.dirichlet(alpha, size=n_iter)
    return samples  # shape (n_iter, n_criteria), chaque ligne somme à 1
