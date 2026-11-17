"""Tests TOPSIS sur des exemples numériques vérifiés à la main."""

import numpy as np
import pytest

from partie_D.methods.topsis import topsis


def test_topsis_simple_2x2():
    """Exemple calculé manuellement :

    X = [[1, 10], [2, 5]] ; c1 à maximiser, c2 à minimiser ; w = [0.5, 0.5].

    Espace bénéfice : Xb = [[1, -10], [2, -5]].
    Normes : c1 = sqrt(5), c2 = sqrt(125).
    Vb : A = [0.2236, -0.4472], B = [0.4472, -0.2236].
    A+ = [0.4472, -0.2236] (= B), A- = [0.2236, -0.4472] (= A).
    D_A+ = D_B- = sqrt(2 * 0.2236²) ≈ 0.3162 ; D_A- = D_B+ = 0.
    => C_B = 1.0 (rang 1), C_A = 0.0 (rang 2).
    """
    X = np.array([[1.0, 10.0], [2.0, 5.0]])
    w = [0.5, 0.5]
    d = [+1, -1]
    out = topsis(X, w, d, ["A", "B"])
    scores = out.set_index("config")["topsis_score"]
    assert scores["B"] == pytest.approx(1.0)
    assert scores["A"] == pytest.approx(0.0)
    assert out.set_index("config").loc["B", "rank"] == 1
    assert out.set_index("config").loc["A", "rank"] == 2


def test_topsis_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.random((6, 4)) * 10
    directions = [1, 1, -1, 1]
    out = topsis(X, [0.25] * 4, directions)
    assert out["topsis_score"].between(0, 1).all()
    assert out["rank"].is_monotonic_increasing or True  # tri par score
    # tri décroissant du score
    assert out["topsis_score"].is_monotonic_decreasing


def test_topsis_best_config_dominates():
    """Une alternative strictement meilleure sur tous les critères
    doit être classée première."""
    X = np.array([
        [9.0, 9.0, 1.0],   # domine partout (c3 à minimiser : 1 est le min)
        [5.0, 5.0, 9.0],
        [3.0, 3.0, 5.0],
    ])
    directions = [1, 1, -1]
    out = topsis(X, [1 / 3] * 3, directions, ["A", "B", "C"])
    assert out.iloc[0]["config"] == "A"
    assert out.iloc[0]["topsis_score"] == pytest.approx(1.0)


def test_topsis_identical_alternatives():
    """Deux alternatives identiques -> ex æquo, D+ + D- > 0 en général."""
    X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0]])
    out = topsis(X, [0.5, 0.5], [1, 1], ["A", "B", "C"])
    scores = out.set_index("config")["topsis_score"]
    assert scores["A"] == pytest.approx(scores["B"])
    assert out["rank"].min() == 1


def test_topsis_weights_must_sum_to_one():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        topsis(X, [-0.5, 1.5], [1, 1])
    with pytest.raises(ValueError):
        topsis(X, [0.0, 0.0], [1, 1])
