"""
Baselines and the results report.

Writes results/metrics.md, the SHAP plots and the per-position table from the
saved model bundles in models/ (produced by train.py).

Run on its own:  python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import BUILD_STATS as STATS_PATH, DATASET as DATA_PATH, MODEL_DIR, RESULTS_DIR
from .config import SEASON_COL, TARGET_COL, TEST_SEASON as TEST_SEASON_DEFAULT

# ---------------------------------------------------------------------------
# Trivial baselines, same split as the models
# ---------------------------------------------------------------------------
# Each predicts the log market value from training-season rows only; groups
# never seen in training fall back to the global training mean.
BASELINE_GROUPS = {
    "global_mean": None,
    "position_mean": "position",
    "league_mean": "league",
}


def split_train_test(df: pd.DataFrame, test_season: int = TEST_SEASON_DEFAULT):
    df = df.copy()
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")
    return df[df[SEASON_COL] < test_season].copy(), df[df[SEASON_COL] == test_season].copy()


def predict_group_mean(train_df, test_df, group_col, target_col: str = TARGET_COL) -> np.ndarray:
    global_mean = float(train_df[target_col].mean())
    if group_col is None:
        return np.full(len(test_df), global_mean)
    means = train_df.groupby(group_col, dropna=False)[target_col].mean()
    return test_df[group_col].map(means).fillna(global_mean).to_numpy(dtype=float)


def run_baselines(df: pd.DataFrame, test_season: int = TEST_SEASON_DEFAULT) -> pd.DataFrame:
    train_df, test_df = split_train_test(df, test_season)
    y_test = test_df[TARGET_COL].to_numpy()
    rows = []
    for name, group_col in BASELINE_GROUPS.items():
        pred = predict_group_mean(train_df, test_df, group_col)
        rows.append({
            "Baseline": name,
            "Test R²": float(r2_score(y_test, pred)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "Test MAE": float(mean_absolute_error(y_test, pred)),
        })
    return pd.DataFrame(rows)


def print_baselines(df: pd.DataFrame, test_season: int = TEST_SEASON_DEFAULT) -> pd.DataFrame:
    res = run_baselines(df, test_season)
    print(f"\n{'='*80}\nTRIVIAL BASELINES (fit on train seasons, scored on test season)\n{'='*80}\n")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

MODELS = [("linear", "Ridge"), ("random_forest", "Random Forest"), ("xgboost", "XGBoost")]
FEATURE_SETS = [("traditional", "Traditional"), ("with_modern", "With modern")]

# chart tokens (light surface)
SURFACE, INK, INK_2, SERIES_1, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"


def load_bundles() -> dict:
    return {(m, fs): joblib.load(MODEL_DIR / f"{m}_{fs}.joblib") for m, _ in MODELS for fs, _ in FEATURE_SETS}


def predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = bundle["preprocessor"].transform(df[bundle["num_cols"] + bundle["cat_cols"]])
    return bundle["model"].predict(X)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    eur_err = np.abs(np.expm1(p) - np.expm1(y))
    return {
        "r2": r2_score(y, p),
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "median_abs_err_eur": float(np.median(eur_err)),
    }


def original_feature(name: str, cat_cols: list[str]) -> str:
    """Map a transformed column name back to its source feature."""
    kind, _, rest = name.partition("__")
    if kind == "cat":
        for c in sorted(cat_cols, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c
    return rest


def shap_outputs(bundle: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    pre, model = bundle["preprocessor"], bundle["model"]
    X = pre.transform(test_df[bundle["num_cols"] + bundle["cat_cols"]])
    names = list(pre.get_feature_names_out())
    import shap  # imported here: slow, and only needed for the report
    sv = shap.TreeExplainer(model).shap_values(X)

    # beeswarm, one dot per test transfer
    plt.figure()
    shap.summary_plot(sv, X, feature_names=[n.partition("__")[2] for n in names],
                      max_display=15, show=False, plot_size=(8, 6))
    plt.title("SHAP values, XGBoost with modern metrics (2025 test set)", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150)
    plt.close()

    # mean |SHAP| grouped back to source features (one-hot levels summed)
    per_col = pd.Series(np.abs(sv).mean(axis=0), index=names)
    grouped = per_col.groupby([original_feature(n, bundle["cat_cols"]) for n in names]).sum()
    grouped = grouped.sort_values(ascending=False)

    top = grouped.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(top.index, top.values, color=SERIES_1, height=0.6)
    for y, v in enumerate(top.values):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Mean |SHAP value| (log market value)", color=INK_2)
    ax.set_title("Feature importance: XGBoost with modern metrics", color=INK, loc="left", fontsize=12)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return grouped.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def r2_from_sums(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise R squared for 2-D arrays (one bootstrap resample per row)."""
    ss_res = ((y - p) ** 2).sum(axis=1)
    ss_tot = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1 - ss_res / ss_tot


def bootstrap_r2_differences(y: np.ndarray, preds: dict, pairs: list,
                             n: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Paired bootstrap of test R squared differences.

    Each resample draws len(y) test transfers with replacement; both models in
    a pair are scored on the same resample, so the interval is for the
    difference itself. 95% percentile intervals.
    """
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))
    yb = y[idx]
    rows = []
    for label, a, b in pairs:
        diff = r2_from_sums(yb, preds[a][idx]) - r2_from_sums(yb, preds[b][idx])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({
            "comparison": label,
            "observed": r2_score(y, preds[a]) - r2_score(y, preds[b]),
            "ci_low": lo, "ci_high": hi,
            "share_above_zero": float((diff > 0).mean()),
            "excludes_zero": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


BOOTSTRAP_PAIRS = [
    ("XGBoost modern minus Ridge modern", ("xgboost", "with_modern"), ("linear", "with_modern")),
    ("Ridge: with modern minus traditional", ("linear", "with_modern"), ("linear", "traditional")),
    ("Random Forest: with modern minus traditional", ("random_forest", "with_modern"), ("random_forest", "traditional")),
    ("XGBoost: with modern minus traditional", ("xgboost", "with_modern"), ("xgboost", "traditional")),
]


def fmt_eur(v: float) -> str:
    return f"EUR {v / 1e6:.2f}m" if v >= 1e6 else f"EUR {v / 1e3:.0f}k"


def write_report() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")
    train_df = df[df[SEASON_COL] < TEST_SEASON_DEFAULT]
    test_df = df[df[SEASON_COL] == TEST_SEASON_DEFAULT]
    y = test_df[TARGET_COL].to_numpy()
    bundles = load_bundles()
    stats = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else None

    preds = {k: predict(b, test_df) for k, b in bundles.items()}
    cells = {k: metrics(y, p) for k, p in preds.items()}

    lines = ["# Results", "",
             f"Test set: {len(test_df):,} transfers in the {TEST_SEASON_DEFAULT} season. "
             f"Training set: {len(train_df):,} transfers in earlier seasons. "
             "Target: log(1 + Transfermarkt market value at transfer). "
             "R squared, MAE and RMSE are on that log scale; the median absolute error is in euros.",
             "", "Regenerate with `python -m src.train` (full pipeline) or `python -m src.evaluate` (report only).", "",
             "## 3 x 2 factorial, test set", "",
             "| Model | Features | Test R squared | MAE | RMSE | Median abs. error |",
             "|---|---|---|---|---|---|"]
    for m, mlab in MODELS:
        for fs, flab in FEATURE_SETS:
            c = cells[(m, fs)]
            lines.append(f"| {mlab} | {flab} | {c['r2']:.3f} | {c['mae']:.3f} | {c['rmse']:.3f} | "
                         f"{fmt_eur(c['median_abs_err_eur'])} |")

    base = run_baselines(df)
    lines += ["", "## Trivial baselines (fit on training seasons)", "",
              "| Baseline | Test R squared | MAE | RMSE |", "|---|---|---|---|"]
    labels = {"global_mean": "Global mean", "position_mean": "Per-position mean", "league_mean": "Per-league mean"}
    for _, r in base.iterrows():
        lines.append(f"| {labels[r['Baseline']]} | {r['Test R²']:.3f} | {r['Test MAE']:.3f} | {r['Test RMSE']:.3f} |")

    # per-position
    pos = test_df["position"].fillna("Unknown").to_numpy()
    order = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    lines += ["", "## Test R squared by position", "",
              "| Position | n | " + " | ".join(f"{ml} {fl.lower()}" for _, ml in MODELS for _, fl in FEATURE_SETS) + " |",
              "|---|---|" + "---|" * 6]
    for p in order:
        mask = pos == p
        vals = [r2_score(y[mask], preds[(m, fs)][mask]) for m, _ in MODELS for fs, _ in FEATURE_SETS]
        lines.append(f"| {p} | {int(mask.sum())} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    # bootstrap confidence intervals for the factorial comparisons
    boot = bootstrap_r2_differences(y, preds, BOOTSTRAP_PAIRS)
    lines += ["", "## Are the differences real? Bootstrap confidence intervals", "",
              f"Paired bootstrap: {BOOTSTRAP_RESAMPLES:,} resamples of the {len(y):,} test transfers "
              f"(seed {BOOTSTRAP_SEED}); both models in a comparison are scored on the same resample. "
              "95% percentile intervals for the difference in test R squared.", "",
              "| Comparison | Observed difference | 95% CI | Resamples above zero | Distinguishable from zero |",
              "|---|---|---|---|---|"]
    for _, r in boot.iterrows():
        lines.append(f"| {r['comparison']} | {r['observed']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{r['share_above_zero']:.1%} | {'Yes' if r['excludes_zero'] else 'No'} |")
    print(boot.to_string(index=False))

    # SHAP
    shap_tbl = shap_outputs(bundles[("xgboost", "with_modern")], test_df)
    lines += ["", "## SHAP: XGBoost with modern metrics", "",
              "Mean absolute SHAP value on the test set, one-hot levels summed back to their source feature.", "",
              "| Rank | Feature | Mean abs. SHAP |", "|---|---|---|"]
    for i, r in shap_tbl.head(10).iterrows():
        lines.append(f"| {i + 1} | {r['feature']} | {r['mean_abs_shap']:.3f} |")
    lines += ["", "![Feature importance](feature_importance.png)", "", "![SHAP summary](shap_summary.png)"]

    if stats:
        n = stats["transfers_in_window"]
        lines += ["", "## Data join", "",
                  f"Denominator: all {n:,} Transfermarkt transfers in the 2018 to 2025 seasons.", "",
                  "| Stage | Transfers | Share |", "|---|---|---|"]
        for key, lab in [("name_season_matches", "FBref match on name + previous season"),
                         ("birth_year_matches", "... and birth year within 1 year"),
                         ("verified_matches", "... and unambiguous (match rate)"),
                         ("model_rows", "Model rows (450+ minutes, target and age present)")]:
            lines.append(f"| {lab} | {stats[key]:,} | {stats[key] / n:.2%} |")

    lines += ["", "See [fix_log.md](fix_log.md) for how each fix changed these numbers.", ""]
    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    write_report()
