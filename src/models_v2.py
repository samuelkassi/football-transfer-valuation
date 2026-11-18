from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from enum import Enum

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor


# paths and other constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "final" / "player_market_value_dataset.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "intermediate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Directory where trained (preprocessor, model) bundles are saved so that
# downstream scripts (src/predict.py, notebooks) can score theoretical players
# without retraining. One joblib file per (model_type, feature_set) pair.
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "log_market_value"
SEASON_COL = "season_end_year"
TEST_SEASON_DEFAULT = 2025               # last season in the dataset

# build_dataset.py computes:  log_market_value = np.log1p(market_value_eur_at_transfer)
# so downstream code should invert via np.expm1. Historical default was False but
# that was wrong; flipping to True so inverse_log_value() returns the right euro
# amount if anything else in this file ever uses it.
LOG_TARGET_IS_LOG1P = True


# ============================================================================
# FEATURE SET DEFINITIONS FOR RESEARCH QUESTIONS
# ============================================================================

TRADITIONAL_STATS = [
    # player info
    "age_at_transfer",
    "age_squared",
    "height_in_cm",
    "contract_years_remaining",
    "position",
    "sub_position",
    "foot",
    # league + nationality
    "league",
    "nationality_group",
    # club context -- log1p of per-club, per-year median player MV
    # (from tm_player_valuations; falls back to all-time club median)
    "from_club_level",
    "to_club_level",
    "club_level_delta",
    # playing time
    "minutes",
    "nineties",
    # attacking (traditional)
    "goals",
    "assists",
    "goals_and_assists",
    "goals_p_90",
    "assists_p_90",
    # passing
    "key_passes",
    # defending
    "tackles_won",
    "interceptions",
    "clearances",
    # goalkeeping
    "clean_sheets",
    "saves",
    # --- position x stat interactions for traditional-base stats ---
    # Give the model per-position coefficients for free; especially useful for
    # Ridge, which otherwise has one shared coefficient across all positions.
    "fw_goals", "mf_goals",
    "fw_assists", "mf_assists",
    "mf_key_passes",
    "df_tackles_won",
    "df_interceptions", "mf_interceptions",
    "df_clearances",
    "gk_saves",
    "gk_clean_sheets",
]

MODERN_STATS = [
    "expected_goals",
    "progressive_passes",
    # Additional modern metrics added once the expanded 10-league FBref
    # source made them available:
    "expected_assisted_goals",     # xAG -- chance creation that leads to goals
    "non_penalty_xg",              # npxG -- xG with penalties stripped out
    "non_penalty_xg_plus_xag",     # npxG + xAG -- overall non-penalty creation
    "tackles_plus_interceptions",  # Tkl+Int -- combined defensive activity
    # --- position x stat interactions for modern-base stats ---
    "fw_xg", "mf_xg",
    "fw_xag", "mf_xag",
    "fw_npxg_xag", "mf_npxg_xag",
    "mf_progressive_passes",
    "df_tkl_int", "mf_tkl_int",
]

# ============================================================================
# PHASE 2: ENGINEERED FEATURES
# ============================================================================

# Pruned engineered feature list (Phase 2b).
# Removed features were either:
#   - redundant with raw columns (threshold binaries trees learn automatically,
#     e.g. is_peak_age, is_young, is_starter, is_high_volume, has_long_contract,
#     contract_expiring_soon),
#   - exact duplicates (goal_overperformance == xg_differential),
#   - redundant per-90 ratios (assist_efficiency == assists_p_90,
#     attacking_efficiency == goals_p_90 + assists_p_90),
#   - nearly constant (playing_time_per_game ≈ 90 for everyone because
#     nineties = minutes / 90),
#   - duplicates of one-hot-encoded `position` (is_goalkeeper / is_defender /
#     is_midfielder / is_forward), or
#   - arbitrary hand-weighted composites that injected noise
#     (star_player_score, breakout_potential).
# Keeps only engineered features that carry independent signal and showed up
# in the Phase 2 RF / XGBoost feature importances.
ENGINEERED_STATS = [
    # Performance & Efficiency
    "offensive_contribution",
    "xg_differential",

    # Age & Development
    "age_to_peak",
    "age_potential_score",

    # Defensive Profile
    "defensive_actions",
    "defensive_per_90",
    "clean_sheet_ratio",

    # League & Competition
    # (league_multiplier / league_adjusted_* removed -- redundant with the
    #  one-hot `league` feature which dominated importance on its own.)

    # International Prestige
    "is_top_nation",

    # Passing & Creativity
    "passing_quality",
    "progressive_passing_per_90",

    # Compound Scores
    "overall_impact",
]

ALL_STATS = TRADITIONAL_STATS + MODERN_STATS
ALL_STATS_WITH_ENGINEERED = TRADITIONAL_STATS + MODERN_STATS + ENGINEERED_STATS


class FeatureSet(Enum):
    """Feature sets for comparing research questions"""
    TRADITIONAL = "traditional"
    WITH_MODERN = "with_modern"
    WITH_ENGINEERED = "with_engineered"  # Phase 2: engineered features


# ============================================================================
# RESULTS CONTAINERS
# ============================================================================

@dataclass
class ModelResults:
    """Container for single model evaluation results"""
    model_name: str
    feature_set: str
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    model: Any = None
    preprocessor: Any = None
    
    def to_dict(self) -> dict:
        return {
            "Model": self.model_name,
            "Feature Set": self.feature_set,
            "Train R²": f"{self.train_r2:.4f}",
            "Test R²": f"{self.test_r2:.4f}",
            "Train RMSE": f"{self.train_rmse:.4f}",
            "Test RMSE": f"{self.test_rmse:.4f}",
            "Train MAE": f"{self.train_mae:.4f}",
            "Test MAE": f"{self.test_mae:.4f}",
        }


@dataclass
class ComparisonResults:
    """Container for all research question comparisons"""
    results: list[ModelResults] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easy viewing"""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def compare_linear_vs_ml(self, feature_set: str) -> None:
        """Compare linear vs ML methods for a specific feature set"""
        subset = [r for r in self.results if r.feature_set == feature_set]
        print(f"\n{'='*80}")
        print(f"RESEARCH Q1: Linear vs ML Methods (Feature Set: {feature_set})")
        print(f"{'='*80}")
        
        for result in sorted(subset, key=lambda r: r.test_r2, reverse=True):
            print(f"\n{result.model_name}:")
            print(f"  Test R²:   {result.test_r2:.4f} | Train R²:   {result.train_r2:.4f}")
            print(f"  Test RMSE: {result.test_rmse:.4f} | Train RMSE: {result.train_rmse:.4f}")
            print(f"  Test MAE:  {result.test_mae:.4f} | Train MAE:  {result.train_mae:.4f}")
    
    def compare_traditional_vs_modern(self, model_name: str) -> None:
        """Compare impact of modern stats for a specific model"""
        subset = [r for r in self.results if r.model_name == model_name]
        print(f"\n{'='*80}")
        print(f"RESEARCH Q2: Traditional vs Modern Stats ({model_name})")
        print(f"{'='*80}")
        
        for result in subset:
            print(f"\n{result.feature_set.upper()}:")
            print(f"  Test R²:   {result.test_r2:.4f}")
            print(f"  Test RMSE: {result.test_rmse:.4f}")
            
        if len(subset) == 2:
            trad = [r for r in subset if r.feature_set == "traditional"][0]
            modern = [r for r in subset if r.feature_set == "with_modern"][0]
            r2_gain = modern.test_r2 - trad.test_r2
            rmse_gain = trad.test_rmse - modern.test_rmse
            print(f"\n{'─'*80}")
            print(f"Modern stats contribution:")
            print(f"  R² gain:        {r2_gain:+.4f} ({r2_gain/trad.test_r2*100:+.2f}%)")
            print(f"  RMSE improvement: {rmse_gain:+.4f}")
    
    def summary(self) -> None:
        """Print summary of all results"""
        print(f"\n{'='*80}")
        print(f"FULL RESULTS SUMMARY")
        print(f"{'='*80}\n")
        print(self.to_dataframe().to_string(index=False))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Calculate and print evaluation metrics; return RMSE, MAE, R²"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")
    
    return rmse, mae, r2


def get_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    feature_set: FeatureSet = FeatureSet.WITH_MODERN,
) -> tuple[list[str], list[str]]:
    """
    Split columns into numeric and categorical features.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        feature_set: Which features to include (TRADITIONAL, WITH_MODERN, or WITH_ENGINEERED)
    
    Returns:
        (numeric_cols, categorical_cols)
    """
    drop_cols = {
        target_col,
        "market_value_eur_at_transfer",
        "transfer_date",
        "player",
        SEASON_COL,
        "season_index",
    }
    
    # Determine which stats to use
    if feature_set == FeatureSet.TRADITIONAL:
        allowed_stats = set(TRADITIONAL_STATS)
    elif feature_set == FeatureSet.WITH_MODERN:
        allowed_stats = set(ALL_STATS)
    else:  # WITH_ENGINEERED
        allowed_stats = set(ALL_STATS_WITH_ENGINEERED)
    
    # Get available features
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feature_df = feature_df[[c for c in feature_df.columns if c in allowed_stats]]

    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in categorical_cols]
    
    return numeric_cols, categorical_cols


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def print_feature_importances(
    model,
    preprocessor: ColumnTransformer,
    title: str = "",
    top_k: int = 20,
) -> None:
    """Print feature importances from tree-based models"""
    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        print("\nNo feature importances available.")
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_k} feature importances ({title}):")
    print(fi_df.head(top_k).to_string(index=False))


def save_predictions(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Save predictions with player and transfer info"""
    pred_df = df_test[[SEASON_COL]].copy()
    if "player" in df_test.columns:
        pred_df.insert(0, "player", df_test["player"].values)
    if "transfer_date" in df_test.columns:
        pred_df.insert(1, "transfer_date", df_test["transfer_date"].values)

    pred_df["y_true_log"] = y_true
    pred_df["y_pred_log"] = y_pred

    pred_df.to_parquet(out_path, index=False)
    print(f"\nSaved predictions: {out_path}")


def inverse_log_value(y_log: float) -> float:
    """Convert log-predictions back to euro values"""
    if LOG_TARGET_IS_LOG1P:
        return float(np.expm1(y_log))
    return float(np.exp(y_log))


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def _extract_column_lists(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    """Pull the num / cat column lists back out of a fitted ColumnTransformer.

    A ColumnTransformer stores its transformers as tuples of
    (name, transformer, columns). We built ours with names "num" and "cat" in
    `make_preprocessor`, so this just looks those up.
    """
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for name, _transformer, cols in preprocessor.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols


def save_trained_bundles(
    comparison: ComparisonResults,
    model_dir: Path = MODEL_DIR,
    target_col: str = TARGET_COL,
) -> list[Path]:
    """Persist each fitted (preprocessor, model) pair so predict.py can load them.

    File name convention: ``{model_type}_{feature_set}.joblib`` — e.g.
    ``xgboost_with_modern.joblib``. Each bundle is a dict with enough metadata
    to drive inference end-to-end from a raw player-stats dict.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in comparison.results:
        if result.model is None or result.preprocessor is None:
            print(f"  [skip] {result.model_name} / {result.feature_set}: no fitted objects")
            continue

        num_cols, cat_cols = _extract_column_lists(result.preprocessor)

        bundle = {
            "preprocessor": result.preprocessor,
            "model": result.model,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "target_col": target_col,
            "target_is_log1p": LOG_TARGET_IS_LOG1P,
            "model_type": result.model_name,
            "feature_set": result.feature_set,
            "test_r2": result.test_r2,
            "test_rmse": result.test_rmse,
            "test_mae": result.test_mae,
        }

        out_path = model_dir / f"{result.model_name}_{result.feature_set}.joblib"
        joblib.dump(bundle, out_path, compress=3)
        print(
            f"  [saved] {out_path.name}  "
            f"(test R²={result.test_r2:.4f}, {len(num_cols)} num + {len(cat_cols)} cat)"
        )
        written.append(out_path)

    return written


# ============================================================================
# HYPERPARAMETER TUNING WITH RANDOM SEARCH
# ============================================================================

def tune_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
) -> LinearRegression:
    """Tune Linear Regression with Ridge/Lasso regularization"""
    param_dist = {
        "ridge__alpha": np.logspace(-3, 3, 20),
    }
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV R²: {search.best_score_:.4f}")
    
    return search.best_estimator_


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
) -> RandomForestRegressor:
    """Tune Random Forest hyperparameters"""
    # Tightened grid to reduce overfitting (Phase 1 "overfitting fix"):
    # min_samples_leaf raised from [1, 2, 4] -> [5, 10, 15, 20],
    # min_samples_split raised from [2, 5, 10] -> [5, 10, 20],
    # max_depth tightened from [10, 15, 20, 25, None] -> [8, 12, 16, 20, None].
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [5, 10, 15, 20],
        "max_features": [0.3, 0.5, 0.7, "sqrt"],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV R²: {search.best_score_:.4f}")
    
    return search.best_estimator_


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    X_refit: np.ndarray | None = None,
    y_refit: np.ndarray | None = None,
    n_iter: int = 30,
) -> XGBRegressor:
    """Tune XGBoost hyperparameters with optional early-stopping refit.

    Flow:
      1. Run 5-fold RandomizedSearchCV on (X_train, y_train) *without* early
         stopping to pick the best hyperparameter combination. (Early stopping
         inside CV is not possible here: RandomizedSearchCV doesn't provide a
         per-fold eval_set.)
      2. If (X_eval, y_eval) is provided: refit a probe model with those best
         params plus early stopping against the eval holdout. This determines
         the optimal number of boosting rounds (``best_iteration``).
      3. If (X_refit, y_refit) is also provided: refit on that larger set
         (e.g. the full training set including the eval holdout) for exactly
         ``best_iteration + 1`` rounds and return that model. This lets
         XGBoost use the same data budget as Linear/RF while still benefiting
         from early-stopping-discovered round counts.
      4. Otherwise, return the probe model (or the raw search best_estimator_
         if no eval holdout was given).
    """
    param_dist = {
        "n_estimators": [500, 750, 1000, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [5, 10, 15, 20],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "colsample_bylevel": [0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
    )

    search = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV R²: {search.best_score_:.4f}")

    have_eval = (
        X_eval is not None
        and y_eval is not None
        and len(X_eval) > 0
    )
    if not have_eval:
        return search.best_estimator_

    # Refit with early stopping against the held-out eval set to discover
    # the optimal boosting-round count.
    best_params = dict(search.best_params_)
    best_params["n_estimators"] = 2000  # ceiling; early stopping will cut this

    probe = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
    )
    probe.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    best_iter = (
        probe.best_iteration + 1
        if getattr(probe, "best_iteration", None) is not None
        else probe.n_estimators
    )
    print(f"Early-stopping best_iteration: {best_iter}")

    have_refit = (
        X_refit is not None
        and y_refit is not None
        and len(X_refit) > 0
    )
    if not have_refit:
        return probe

    # Final refit on the full training set (including the eval holdout),
    # for a fixed number of rounds chosen by early stopping above.
    final_params = dict(search.best_params_)
    final_params["n_estimators"] = best_iter
    final_model = XGBRegressor(
        **final_params,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        eval_metric="rmse",
    )
    final_model.fit(X_refit, y_refit, verbose=False)
    return final_model


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str,
    feature_set: FeatureSet,
    target_col: str = TARGET_COL,
    tune_hyperparams: bool = True,
) -> ModelResults:
    """
    Train and evaluate a single model on a feature set.
    
    Args:
        train_df: Training data
        test_df: Test data
        model_type: "linear", "random_forest", or "xgboost"
        feature_set: TRADITIONAL or WITH_MODERN
        target_col: Name of target column
        tune_hyperparams: Whether to use random search for tuning
    
    Returns:
        ModelResults object with all metrics
    """
    num_cols, cat_cols = get_feature_cols(train_df, target_col, feature_set)
    pre = make_preprocessor(num_cols, cat_cols)

    X_train_df = train_df[num_cols + cat_cols]
    y_train = train_df[target_col].to_numpy()

    X_test_df = test_df[num_cols + cat_cols]
    y_test = test_df[target_col].to_numpy()

    X_train = pre.fit_transform(X_train_df)
    X_test = pre.transform(X_test_df)

    # Train appropriate model
    if model_type.lower() == "linear":
        if tune_hyperparams:
            print(f"\nTuning Linear Regression ({feature_set.value})...")
            model = tune_linear_regression(X_train, y_train)
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
    
    elif model_type.lower() == "random_forest":
        if tune_hyperparams:
            print(f"\nTuning Random Forest ({feature_set.value})...")
            model = tune_random_forest(X_train, y_train)
        else:
            model = RandomForestRegressor(
                n_estimators=800,
                max_depth=18,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features=0.5,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
    
    elif model_type.lower() == "xgboost":
        if tune_hyperparams:
            print(f"\nTuning XGBoost ({feature_set.value})...")
            # Temporal eval holdout for early stopping:
            #   tune on seasons < eval_season, early-stop on eval_season,
            #   then refit on the full training set for the discovered number
            #   of rounds so XGBoost gets the same data budget as Linear/RF.
            X_xgb_train = X_train
            y_xgb_train = y_train
            X_xgb_eval = None
            y_xgb_eval = None
            X_xgb_refit = None
            y_xgb_refit = None

            if SEASON_COL in train_df.columns:
                seasons = pd.to_numeric(train_df[SEASON_COL], errors="coerce")
                eval_season = int(seasons.max())
                tune_mask = (seasons < eval_season).to_numpy()
                eval_mask = (seasons == eval_season).to_numpy()

                if tune_mask.sum() > 100 and eval_mask.sum() > 50:
                    X_xgb_train = X_train[tune_mask]
                    y_xgb_train = y_train[tune_mask]
                    X_xgb_eval = X_train[eval_mask]
                    y_xgb_eval = y_train[eval_mask]
                    X_xgb_refit = X_train
                    y_xgb_refit = y_train
                    print(
                        f"  XGBoost tuning split: "
                        f"{tune_mask.sum()} tune / {eval_mask.sum()} eval-holdout "
                        f"(season {eval_season}) / refit on full {len(y_train)}"
                    )

            model = tune_xgboost(
                X_xgb_train,
                y_xgb_train,
                X_eval=X_xgb_eval,
                y_eval=y_xgb_eval,
                X_refit=X_xgb_refit,
                y_refit=y_xgb_refit,
                n_iter=30,
            )
        else:
            model = XGBRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=10,
                subsample=0.7,
                colsample_bytree=0.7,
                colsample_bylevel=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                eval_metric="rmse",
                early_stopping_rounds=100,
                verbose=0,
            )
            model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate on both train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse, train_mae, train_r2 = evaluate(
        f"{model_type} Train ({feature_set.value})",
        y_train,
        y_pred_train,
    )
    test_rmse, test_mae, test_r2 = evaluate(
        f"{model_type} Test ({feature_set.value})",
        y_test,
        y_pred_test,
    )

    # Print feature importances for tree-based models
    if model_type.lower() in ["random_forest", "xgboost"]:
        print_feature_importances(model, pre, title=f"{model_type} ({feature_set.value})")

    return ModelResults(
        model_name=model_type,
        feature_set=feature_set.value,
        train_r2=train_r2,
        test_r2=test_r2,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_mae=train_mae,
        test_mae=test_mae,
        model=model,
        preprocessor=pre,
    )


def run_full_comparison(
    df: pd.DataFrame,
    test_season: int = TEST_SEASON_DEFAULT,
    tune_hyperparams: bool = True,
    feature_sets: list = None,
) -> ComparisonResults:
    """
    Train all models on specified feature sets and return comparison results.
    
    This allows you to answer:
    1. Linear vs ML methods (with traditional stats, then with modern stats)
    2. Impact of modern stats (for each model)
    3. Impact of engineered features (Phase 2)
    
    Args:
        df: Input dataframe
        test_season: Which season to use as test set
        tune_hyperparams: Whether to use random search for hyperparameter tuning
        feature_sets: List of FeatureSet enums to test (default: [TRADITIONAL, WITH_MODERN])
    """
    if feature_sets is None:
        feature_sets = [FeatureSet.TRADITIONAL, FeatureSet.WITH_MODERN]
    
    df = df.copy()
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")

    train_df = df[df[SEASON_COL] < test_season].copy()
    test_df = df[df[SEASON_COL] == test_season].copy()

    print(f"\n{'='*80}")
    print(f"RUNNING FULL COMPARISON")
    print(f"{'='*80}")
    print(f"Train seasons: < {test_season} ({len(train_df)} samples)")
    print(f"Test season:   {test_season} ({len(test_df)} samples)")

    comparison = ComparisonResults()
    models = ["linear", "random_forest", "xgboost"]

    for feature_set in feature_sets:
        print(f"\n{'─'*80}")
        print(f"Feature Set: {feature_set.value.upper()}")
        print(f"{'─'*80}")
        
        for model_type in models:
            result = train_and_evaluate_model(
                train_df,
                test_df,
                model_type=model_type,
                feature_set=feature_set,
                tune_hyperparams=tune_hyperparams,
            )
            comparison.results.append(result)

    return comparison


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(test_season: int = TEST_SEASON_DEFAULT, tune_hyperparams: bool = True) -> ComparisonResults:
    """
    Load data and run full comparison.
    
    Args:
        test_season: Which season to use as test set
        tune_hyperparams: Whether to use random search for hyperparameter tuning
    
    Returns:
        ComparisonResults with all model results
    """
    df = pd.read_parquet(DATA_PATH)
    
    comparison = run_full_comparison(
        df,
        test_season=test_season,
        tune_hyperparams=tune_hyperparams,
    )
    
    # Print comprehensive summary
    print("\n\n")
    comparison.summary()
    
    # Print research question comparisons
    print("\n\n")
    for feature_set in [FeatureSet.TRADITIONAL.value, FeatureSet.WITH_MODERN.value]:
        comparison.compare_linear_vs_ml(feature_set)
    
    print("\n\n")
    for model_type in ["linear", "random_forest", "xgboost"]:
        comparison.compare_traditional_vs_modern(model_type)

    # Persist fitted bundles so src/predict.py can score theoretical players
    # without retraining. One file per (model_type, feature_set) pair.
    print(f"\n{'='*80}")
    print(f"SAVING TRAINED BUNDLES")
    print(f"{'='*80}")
    print(f"Output directory: {MODEL_DIR}")
    save_trained_bundles(comparison)

    return comparison


if __name__ == "__main__":
    # Run with hyperparameter tuning
    comparison = main(tune_hyperparams=True)