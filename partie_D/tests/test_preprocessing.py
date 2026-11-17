"""Tests de la préparation des données et de la pondération."""

import numpy as np
import pandas as pd
import pytest

from partie_D.preprocessing import (
    ALL_CRITERIA,
    build_decision_matrix,
    to_benefit_space,
)
from partie_D.weighting import normalize_weights, scenario_weights


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config": ["C1", "C2", "C3"],
            "model": ["m"] * 3,
            "chunk_size": [50, 100, 50],
            "top_k": [2, 5, 2],
            "temperature": [0.1, 0.5, 0.9],
            "bleu": [0.1, 0.2, 0.15],
            "rouge1": [0.3, 0.4, 0.35],
            "rouge2": [0.2, 0.25, 0.22],
            "rougeL": [0.25, 0.3, 0.28],
            "semantic_similarity": [0.6, 0.7, 0.65],
            "seconds_median": [30.0, 25.0, 20.0],
            "n_questions": [20] * 3,
            "hit_page": [0.6, 0.8, 0.7],
            "mrr": [0.5, 0.6, 0.55],
        }
    )


def test_build_decision_matrix_basic():
    dd = build_decision_matrix(_sample_df())
    assert dd.X.shape == (3, len(ALL_CRITERIA))
    assert dd.n_criteria == len(ALL_CRITERIA)
    # seconds_median doit être le seul critère à minimiser
    idx = dd.criteria.index("seconds_median")
    assert dd.directions[idx] == -1
    assert all(dd.directions[j] == 1 for j in range(len(dd.criteria)) if j != idx)


def test_missing_values_drop_rows():
    df = _sample_df()
    df.loc[1, "bleu"] = np.nan
    dd = build_decision_matrix(df, missing_strategy="drop_rows")
    assert dd.n_alternatives == 2
    assert dd.report["n_rows_dropped_na"] == 1


def test_missing_values_fail():
    df = _sample_df()
    df.loc[0, "mrr"] = np.nan
    with pytest.raises(ValueError):
        build_decision_matrix(df, missing_strategy="fail")


def test_constant_criterion_detected():
    df = _sample_df()
    df["bleu"] = 0.15  # constant
    dd = build_decision_matrix(df)
    assert dd.report["criteria"]["bleu"]["is_constant"] is True


def test_duplicate_configs_aggregated():
    df = pd.concat([_sample_df(), _sample_df().iloc[[0]]], ignore_index=True)
    dd = build_decision_matrix(df, aggregate="mean")
    assert dd.n_alternatives == 3


def test_to_benefit_space():
    X = np.array([[1.0, 10.0]])
    d = np.array([1.0, -1.0])
    Xb = to_benefit_space(X, d)
    assert Xb[0, 0] == 1.0
    assert Xb[0, 1] == -10.0


def test_normalize_weights():
    w = normalize_weights([1, 1, 2])
    assert w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_weights([0, 0, 0])
    with pytest.raises(ValueError):
        normalize_weights([1, -1, 1])


def test_scenario_weights_sum_to_one():
    criteria = list(ALL_CRITERIA)
    for sc in ["uniforme", "qualite", "rapidite", "equite_env"]:
        w, desc = scenario_weights(criteria, sc)
        assert w.sum() == pytest.approx(1.0)
        assert len(w) == len(criteria)
        assert desc


def test_custom_weights():
    criteria = ["bleu", "seconds_median"]
    w, _ = scenario_weights(criteria, "personnalise",
                            custom_weights=[3, 1])
    assert w[0] == pytest.approx(0.75)
    assert w[1] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        scenario_weights(criteria, "personnalise")
