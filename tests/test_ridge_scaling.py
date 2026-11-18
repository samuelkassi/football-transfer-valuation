"""Ridge must be fit on standardised inputs (brief issue 4)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src import train as train_mod
from src.train import FeatureSet, SEASON_COL, TARGET_COL, get_feature_cols, make_preprocessor
from test_early_stopping import synthetic_frame


def test_preprocessor_standardises_numeric_columns():
    df = synthetic_frame()
    num, cat = get_feature_cols(df, TARGET_COL, FeatureSet.TRADITIONAL)
    X = make_preprocessor(num, cat).fit_transform(df[num + cat])
    Xn = X[:, : len(num)]
    assert np.allclose(Xn.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Xn.std(axis=0), 1, atol=1e-8)


def test_tuned_ridge_sees_scaled_inputs(monkeypatch):
    real_tune = train_mod.tune_linear_regression
    monkeypatch.setattr(train_mod, "tune_linear_regression",
                        lambda *a, **k: real_tune(*a, **{**k, "n_iter": 3}))
    df = synthetic_frame()
    train, test = df[df[SEASON_COL] < 2025], df[df[SEASON_COL] == 2025]
    res = train_mod.train_and_evaluate_model(train, test, "linear", FeatureSet.TRADITIONAL)

    model = res.model
    assert isinstance(model, Pipeline) and isinstance(model[-1], Ridge)
    num, cat = get_feature_cols(train, TARGET_COL, FeatureSet.TRADITIONAL)
    ridge_input = model[:-1].transform(res.preprocessor.transform(train[num + cat]))
    assert np.allclose(ridge_input.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(ridge_input.std(axis=0), 1, atol=1e-8)
