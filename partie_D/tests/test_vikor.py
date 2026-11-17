"""Tests VIKOR sur des exemples numériques vérifiés à la main."""

import numpy as np
import pytest

from partie_D.methods.vikor import vikor


def test_vikor_simple_example():
    """Exemple calculé manuellement (3 alternatives, 2 critères max,
    w = [0.5, 0.5], v = 0.5) :

    X = [[1,4],[2,3],[3,2]] (toutes à maximiser).
    Espace regret f (0 = meilleur, 1 = pire) :
      c1 (min 1, max 3) : A=1, B=0.5, C=0
      c2 (min 2, max 4) : A=0, B=0.5, C=1
    S = 0.5 pour tous ; R : A=0.5, B=0.25, C=0.5.
    S* = 0.5, R* = 0.25.
    Q = 0.5·S/S* + 0.5·R/R* = 0.5 + 2·R  ->  A=1.5, B=1.0, C=1.5.
    => B est le seul rang 1 ; A et C ex æquo.
    """
    X = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
    out = vikor(X, [0.5, 0.5], [1, 1], v=0.5, config_names=["A", "B", "C"])
    q = out.set_index("config")["vikor_Q"]
    assert q["B"] == pytest.approx(1.0)
    assert q["A"] == pytest.approx(1.5)
    assert q["C"] == pytest.approx(1.5)
    assert out.iloc[0]["config"] == "B"
    ranks = out.set_index("config")["rank"]
    assert ranks["A"] == ranks["C"]  # ex æquo


def test_vikor_cost_criterion():
    """Un critère à minimiser doit être avantagé s'il est petit."""
    X = np.array([
        [1.0, 100.0],   # bon sur c1, mauvais sur c2 (coût)
        [2.0, 1.0],     # moyen c1, excellent c2
    ])
    directions = [1, -1]
    out = vikor(X, [0.5, 0.5], directions, v=0.5, config_names=["A", "B"])
    # B a le meilleur compromis : Q_B < Q_A attendu
    q = out.set_index("config")["vikor_Q"]
    assert q["B"] < q["A"]


def test_vikor_v_bounds():
    X = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = vikor(X, [0.5, 0.5], [1, 1], v=v)
        assert len(out) == 3
        assert out["rank"].min() == 1
    with pytest.raises(ValueError):
        vikor(X, [0.5, 0.5], [1, 1], v=1.5)


def test_vikor_compromise_conditions():
    """La meilleure solution satisfait toujours les conditions de
    compromis ; les autres peuvent ou non les satisfaire."""
    X = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
    out = vikor(X, [0.5, 0.5], [1, 1], v=0.5, config_names=["A", "B", "C"])
    best = out.iloc[0]
    assert bool(best["compromise_ok"])
    # A et C sont ex æquo en Q : les deux devraient être compromis
    # ou non de façon cohérente
    ac = out.set_index("config").loc[["A", "C"], "compromise_ok"]
    assert ac["A"] == ac["C"]


def test_vikor_constant_criterion():
    """Un critère constant ne doit pas produire de NaN."""
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    out = vikor(X, [0.5, 0.5], [1, 1], config_names=["A", "B", "C"])
    assert not out["vikor_Q"].isna().any()
    assert out.iloc[0]["config"] == "C"  # c1 max = 3
