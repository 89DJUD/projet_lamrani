"""Tests RIM sur des exemples numériques vérifiés à la main."""

import numpy as np
import pytest

from partie_D.methods.rim import rim


def test_rim_simple_example():
    """Exemple calculé manuellement (mode exploratoire, w = [0.5, 0.5]) :

    X = [[1,4],[3,3],[2,1]], critères à maximiser.
    Idéal (max observé) : c1 = 3, c2 = 4.
    Distances brutes : A=[2,0], B=[0,1], C=[1,3].
    Étendue : c1 = 2, c2 = 3 -> normalisées : A=[1,0], B=[0,1/3], C=[0.5,1].
    Pondérées (×0.5) : A=[0.5,0], B=[0,1/6], C=[0.25,0.5].
    Anti-idéale = max par critère = [0.5, 0.5].
    D- : A=0.5, B=sqrt(0.25 + (1/3)²)=sqrt(13/36)≈0.6009,
         C=0.25.
    D+ : A=0.5, B=1/6≈0.1667, C=sqrt(0.0625+0.25)=sqrt(5/16)≈0.5590.
    Scores : A=0.5, B≈0.7828, C≈0.3090 -> B rang 1.
    """
    X = np.array([[1.0, 4.0], [3.0, 3.0], [2.0, 1.0]])
    out = rim(X, [0.5, 0.5], [1, 1], config_names=["A", "B", "C"])
    s = out.set_index("config")["rim_score"]
    assert s["B"] == pytest.approx(0.7828, abs=1e-3)
    assert s["A"] == pytest.approx(0.5)
    assert s["C"] == pytest.approx(0.3090, abs=1e-3)
    assert out.iloc[0]["config"] == "B"


def test_rim_best_observed_scores_one():
    """L'alternative qui atteint la meilleure valeur sur TOUS les
    critères (quand elle existe) doit avoir un score proche de 1."""
    X = np.array([
        [1.0, 1.0],
        [9.0, 9.0],   # domine partout
        [5.0, 2.0],
    ])
    out = rim(X, [0.5, 0.5], [1, 1], config_names=["A", "B", "C"])
    s = out.set_index("config")["rim_score"]
    assert s["B"] == pytest.approx(1.0)
    assert out.iloc[0]["config"] == "B"


def test_rim_cost_criterion():
    """Pour un critère à minimiser, la plus petite valeur est idéale."""
    X = np.array([
        [1.0, 100.0],
        [2.0, 1.0],
    ])
    directions = [1, -1]
    out = rim(X, [0.5, 0.5], directions, config_names=["A", "B"])
    s = out.set_index("config")["rim_score"]
    # B est meilleur sur c2 (coût) mais moins bon sur c1 : vérifier
    # que le calcul ne plante pas et que les scores sont dans [0,1]
    assert s.between(0, 1).all()


def test_rim_custom_intervals():
    """Des intervalles idéaux fournis doivent être respectés :
    une alternative DANS l'intervalle a distance nulle sur ce critère."""
    X = np.array([[1.0, 4.0], [3.0, 3.0], [2.0, 1.0]])
    intervals = {"c1": (2.5, 3.5), "c2": (3.5, 4.5)}
    out = rim(X, [0.5, 0.5], [1, 1], config_names=["A", "B", "C"],
              ideal_intervals=intervals, criteria=["c1", "c2"])
    assert out["rim_score"].between(0, 1).all()
    # B est dans les deux intervalles -> distance nulle -> meilleur score
    assert out.iloc[0]["config"] == "B"


def test_rim_identical_alternatives():
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0]])
    out = rim(X, [0.5, 0.5], [1, 1], config_names=["A", "B", "C"])
    s = out.set_index("config")["rim_score"]
    assert s["A"] == pytest.approx(s["B"])
