"""Score theoretical players with all six trained models.

Flow:
    raw player dict  ->  _derive(...)  ->  one-row DataFrame with all the
    features `build_dataset.py` produces  ->  each saved bundle's preprocessor
    transforms it  ->  each bundle's model predicts in log1p space  ->
    np.expm1 converts back to euros.

Typical use from a notebook::

    from predict import predict_all_models
    player = {
        "position": "FW",
        "age_at_transfer": 24,
        "minutes": 2700,
        "goals": 22, "assists": 8,
        "expected_goals": 19.4, "expected_assisted_goals": 6.1,
        "non_penalty_xg": 16.0,
        "tackles_won": 12, "interceptions": 6, "clearances": 4,
        "saves": 0, "clean_sheets": 0,
        "key_passes": 35, "progressive_passes": 120,
        "height_in_cm": 184, "foot": "right",
        "league": "Premier League", "nationality_group": "France",
        "sub_position": "Centre-Forward",
        "contract_years_remaining": 3,
        "from_club": "Ajax",
        "to_club": "Manchester City",
        "transfer_year": 2024,
    }
    predict_all_models(player)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    # Package-style import (when predict.py is imported as a module)
    from .club_lookup import get_club_level
except ImportError:
    # Script-style import (e.g. running from a notebook with src/ on sys.path)
    from club_lookup import get_club_level


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

# Must match the file-naming scheme used by models_v2.save_trained_bundles():
#   MODEL_DIR / f"{model_type}_{feature_set}.joblib"
MODEL_KEYS: list[tuple[str, str]] = [
    ("linear",        "traditional"),
    ("random_forest", "traditional"),
    ("xgboost",       "traditional"),
    ("linear",        "with_modern"),
    ("random_forest", "with_modern"),
    ("xgboost",       "with_modern"),
]


# ============================================================================
# Bundle loading (cached)
# ============================================================================

@lru_cache(maxsize=16)
def _load_bundle(model_type: str, feature_set: str) -> dict[str, Any]:
    """Load a single (preprocessor, model, column-list) bundle from disk."""
    path = MODEL_DIR / f"{model_type}_{feature_set}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model bundle: {path}\n"
            "Re-run `python src/models_v2.py` to train and save all six models."
        )
    return joblib.load(path)


def available_models() -> list[tuple[str, str]]:
    """Return the list of (model_type, feature_set) pairs found on disk."""
    return [
        (mt, fs)
        for (mt, fs) in MODEL_KEYS
        if (MODEL_DIR / f"{mt}_{fs}.joblib").exists()
    ]


# ============================================================================
# Feature derivation
# ============================================================================

_VALID_POSITIONS = {"GK", "DF", "MF", "FW"}


def _require(d: dict, key: str) -> Any:
    if key not in d or d[key] is None:
        raise ValueError(f"Missing required player attribute: '{key}'")
    return d[key]


def _get(d: dict, key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _derive(player: dict) -> pd.DataFrame:
    """Compute every derived feature that `build_dataset.py` creates.

    The output is a one-row DataFrame. The preprocessor will then pluck out
    whichever columns each bundle actually needs (stored as ``num_cols`` and
    ``cat_cols``).

    Only fields the user can plausibly know for a theoretical player are
    required; everything else is either derived or defaults to 0.
    """
    # ---------- required + basic validation ----------
    position = str(_require(player, "position")).upper()
    if position not in _VALID_POSITIONS:
        raise ValueError(
            f"position must be one of {sorted(_VALID_POSITIONS)}, got {position!r}"
        )
    age = float(_require(player, "age_at_transfer"))
    minutes = float(_require(player, "minutes"))

    # ---------- raw stats with zero defaults ----------
    goals      = float(_get(player, "goals", 0.0))
    assists    = float(_get(player, "assists", 0.0))
    xg         = float(_get(player, "expected_goals", 0.0))
    xag        = float(_get(player, "expected_assisted_goals", 0.0))
    npxg       = float(_get(player, "non_penalty_xg", 0.0))

    tackles_won   = float(_get(player, "tackles_won", 0.0))
    interceptions = float(_get(player, "interceptions", 0.0))
    clearances    = float(_get(player, "clearances", 0.0))
    clean_sheets  = float(_get(player, "clean_sheets", 0.0))
    saves         = float(_get(player, "saves", 0.0))

    key_passes         = float(_get(player, "key_passes", 0.0))
    progressive_passes = float(_get(player, "progressive_passes", 0.0))

    # ---------- non-stat profile fields ----------
    height_in_cm   = _get(player, "height_in_cm")
    foot           = _get(player, "foot")
    league         = _get(player, "league")
    nationality    = _get(player, "nationality_group")
    sub_position   = _get(player, "sub_position")
    contract_years = _get(player, "contract_years_remaining")

    # ---------- derived simple features ----------
    age_squared = age * age
    nineties = minutes / 90.0 if minutes > 0 else 0.0

    # position flags
    is_goalkeeper = int(position == "GK")
    is_defender   = int(position == "DF")
    is_midfielder = int(position == "MF")
    is_forward    = int(position == "FW")

    # aggregates the training pipeline builds
    # (FBref stores these as per-90 — match that semantics)
    goals_p_90   = goals   / nineties if nineties > 0 else 0.0
    assists_p_90 = assists / nineties if nineties > 0 else 0.0
    goals_and_assists = goals_p_90 + assists_p_90  # FBref "Per 90 Minutes_G+A"

    non_penalty_xg_plus_xag = npxg + xag
    tackles_plus_interceptions = tackles_won + interceptions

    # ---------- club-level lookups ----------
    transfer_year = player.get("transfer_year")
    from_club = player.get("from_club") or player.get("from_club_name")
    to_club   = player.get("to_club")   or player.get("to_club_name")

    from_club_level = None
    to_club_level = None
    if from_club:
        from_club_level = get_club_level(from_club, year=transfer_year)
    if to_club:
        to_club_level = get_club_level(to_club, year=transfer_year)

    # If the lookup failed, fall back to NaN — SimpleImputer(median) in the
    # preprocessor will fill it with the training median.
    if from_club_level is None:
        from_club_level = np.nan
    if to_club_level is None:
        to_club_level = np.nan
    club_level_delta = (
        to_club_level - from_club_level
        if (not pd.isna(from_club_level) and not pd.isna(to_club_level))
        else np.nan
    )

    # ---------- Phase-2 engineered features ----------
    goal_overperformance = goals - xg
    assist_efficiency = assists / (minutes / 90.0 + 1.0)
    offensive_contribution = goals + assists + (xg / 2.0)
    xg_differential = goals - xg
    attacking_efficiency = (goals + assists) / (minutes + 1.0)

    age_to_peak = abs(age - 28.5)
    is_peak_age = int(26 <= age <= 31)
    is_young = int(age < 23)
    age_potential_score = (
        goals_p_90 * (1.0 + (25.0 - age) / 20.0) if age < 25 else goals_p_90
    )

    is_starter = int(minutes >= 1500)
    playing_time_per_game = minutes / (nineties + 1.0)
    is_high_volume = int(minutes >= 2500)

    defensive_actions = tackles_won + interceptions + clearances
    defensive_per_90 = defensive_actions / (nineties + 1.0)
    clean_sheet_ratio = clean_sheets / (nineties + 1.0)

    top_nations = {"France", "Germany", "Spain", "England", "Brazil", "Argentina", "Portugal"}
    is_top_nation = int(nationality in top_nations) if nationality else 0

    passing_quality = key_passes / (nineties + 1.0)
    progressive_passing_per_90 = progressive_passes / (nineties + 1.0)

    has_long_contract = (
        int(contract_years >= 3) if contract_years is not None else 0
    )
    contract_expiring_soon = (
        int(contract_years <= 1) if contract_years is not None else 0
    )

    overall_impact = (offensive_contribution + defensive_actions * 0.3) / (nineties + 1.0)

    # ---------- position × stat interactions ----------
    def mask(stat: float, flag: int) -> float:
        return stat * flag

    fw_goals = mask(goals, is_forward)
    mf_goals = mask(goals, is_midfielder)
    fw_assists = mask(assists, is_forward)
    mf_assists = mask(assists, is_midfielder)
    fw_xg = mask(xg, is_forward)
    mf_xg = mask(xg, is_midfielder)
    fw_xag = mask(xag, is_forward)
    mf_xag = mask(xag, is_midfielder)
    fw_npxg_xag = mask(non_penalty_xg_plus_xag, is_forward)
    mf_npxg_xag = mask(non_penalty_xg_plus_xag, is_midfielder)
    mf_key_passes = mask(key_passes, is_midfielder)
    mf_progressive_passes = mask(progressive_passes, is_midfielder)
    df_tackles_won = mask(tackles_won, is_defender)
    df_interceptions = mask(interceptions, is_defender)
    mf_interceptions = mask(interceptions, is_midfielder)
    df_clearances = mask(clearances, is_defender)
    df_tkl_int = mask(tackles_plus_interceptions, is_defender)
    mf_tkl_int = mask(tackles_plus_interceptions, is_midfielder)
    gk_saves = mask(saves, is_goalkeeper)
    gk_clean_sheets = mask(clean_sheets, is_goalkeeper)

    row: dict[str, Any] = {
        # profile
        "age_at_transfer": age,
        "age_squared": age_squared,
        "height_in_cm": height_in_cm,
        "contract_years_remaining": contract_years,
        "position": position,
        "sub_position": sub_position,
        "foot": foot,
        "league": league,
        "nationality_group": nationality,

        # club context
        "from_club_level": from_club_level,
        "to_club_level": to_club_level,
        "club_level_delta": club_level_delta,

        # playing time
        "minutes": minutes,
        "nineties": nineties,

        # attacking
        "goals": goals,
        "assists": assists,
        "expected_goals": xg,
        "goals_and_assists": goals_and_assists,
        "goals_p_90": goals_p_90,
        "assists_p_90": assists_p_90,

        # modern
        "expected_assisted_goals": xag,
        "non_penalty_xg": npxg,
        "non_penalty_xg_plus_xag": non_penalty_xg_plus_xag,

        # passing
        "progressive_passes": progressive_passes,
        "key_passes": key_passes,

        # defending
        "tackles_won": tackles_won,
        "interceptions": interceptions,
        "clearances": clearances,
        "tackles_plus_interceptions": tackles_plus_interceptions,

        # goalkeeping
        "clean_sheets": clean_sheets,
        "saves": saves,

        # engineered
        "goal_overperformance": goal_overperformance,
        "assist_efficiency": assist_efficiency,
        "offensive_contribution": offensive_contribution,
        "xg_differential": xg_differential,
        "attacking_efficiency": attacking_efficiency,
        "age_to_peak": age_to_peak,
        "is_peak_age": is_peak_age,
        "is_young": is_young,
        "age_potential_score": age_potential_score,
        "is_starter": is_starter,
        "playing_time_per_game": playing_time_per_game,
        "is_high_volume": is_high_volume,
        "defensive_actions": defensive_actions,
        "defensive_per_90": defensive_per_90,
        "clean_sheet_ratio": clean_sheet_ratio,
        "is_top_nation": is_top_nation,
        "is_goalkeeper": is_goalkeeper,
        "is_defender": is_defender,
        "is_midfielder": is_midfielder,
        "is_forward": is_forward,
        "passing_quality": passing_quality,
        "progressive_passing_per_90": progressive_passing_per_90,
        "has_long_contract": has_long_contract,
        "contract_expiring_soon": contract_expiring_soon,
        "overall_impact": overall_impact,

        # position × stat interactions
        "fw_goals": fw_goals, "mf_goals": mf_goals,
        "fw_assists": fw_assists, "mf_assists": mf_assists,
        "fw_xg": fw_xg, "mf_xg": mf_xg,
        "fw_xag": fw_xag, "mf_xag": mf_xag,
        "fw_npxg_xag": fw_npxg_xag, "mf_npxg_xag": mf_npxg_xag,
        "mf_key_passes": mf_key_passes,
        "mf_progressive_passes": mf_progressive_passes,
        "df_tackles_won": df_tackles_won,
        "df_interceptions": df_interceptions, "mf_interceptions": mf_interceptions,
        "df_clearances": df_clearances,
        "df_tkl_int": df_tkl_int, "mf_tkl_int": mf_tkl_int,
        "gk_saves": gk_saves, "gk_clean_sheets": gk_clean_sheets,
    }

    return pd.DataFrame([row])


# ============================================================================
# Prediction
# ============================================================================

def _predict_one(bundle: dict[str, Any], feature_row: pd.DataFrame) -> tuple[float, float]:
    """Run one bundle. Returns (log_pred, eur_pred)."""
    num_cols = bundle["num_cols"]
    cat_cols = bundle["cat_cols"]
    pre = bundle["preprocessor"]
    model = bundle["model"]

    # Ensure every column the preprocessor expects is present — if a user omits
    # a stat we didn't specifically require, fill it with NaN and let the
    # median imputer handle it. Categorical missing values become None, which
    # the most_frequent imputer will fill.
    X_row = feature_row.copy()
    for c in num_cols:
        if c not in X_row.columns:
            X_row[c] = np.nan
    for c in cat_cols:
        if c not in X_row.columns:
            X_row[c] = None

    X_row = X_row[num_cols + cat_cols]
    X_transformed = pre.transform(X_row)
    log_pred = float(model.predict(X_transformed)[0])

    # Inverse transform back to euros. The training target is log1p.
    is_log1p = bool(bundle.get("target_is_log1p", True))
    eur_pred = float(np.expm1(log_pred) if is_log1p else np.exp(log_pred))
    return log_pred, eur_pred


def predict_all_models(
    player: dict,
    return_details: bool = False,
) -> pd.DataFrame:
    """Score *player* with every saved (model_type, feature_set) bundle.

    Returns a DataFrame with one row per bundle::

        Model            Feature Set    Predicted MV (EUR)   Log Pred   Test R²
        linear           traditional              3,100,000    14.947    0.3521
        ...

    If ``return_details=True`` an additional column lists columns the derivation
    produced vs. columns each bundle consumed — useful when debugging a
    surprising prediction.
    """
    row = _derive(player)

    records: list[dict[str, Any]] = []
    for model_type, feature_set in MODEL_KEYS:
        try:
            bundle = _load_bundle(model_type, feature_set)
        except FileNotFoundError as e:
            records.append({
                "Model": model_type,
                "Feature Set": feature_set,
                "Predicted MV (EUR)": None,
                "Log Pred": None,
                "Test R²": None,
                "Error": str(e).splitlines()[0],
            })
            continue

        log_pred, eur_pred = _predict_one(bundle, row)
        rec: dict[str, Any] = {
            "Model": model_type,
            "Feature Set": feature_set,
            "Predicted MV (EUR)": eur_pred,
            "Log Pred": log_pred,
            "Test R²": bundle.get("test_r2"),
        }
        if return_details:
            rec["N Features"] = len(bundle["num_cols"]) + len(bundle["cat_cols"])
        records.append(rec)

    df = pd.DataFrame(records)

    # Pretty formatting for the euro column
    if "Predicted MV (EUR)" in df.columns:
        df["Predicted MV (EUR)"] = df["Predicted MV (EUR)"].map(
            lambda v: None if v is None or pd.isna(v) else round(v, 2)
        )
    return df


def derive_features(player: dict) -> pd.DataFrame:
    """Expose the feature-derivation step for debugging. Returns a 1xN DataFrame."""
    return _derive(player)


if __name__ == "__main__":
    # Quick smoke test — requires trained bundles + valuations parquet.
    example_player = {
        "position": "FW",
        "age_at_transfer": 24,
        "minutes": 2700,
        "goals": 22, "assists": 8,
        "expected_goals": 19.4, "expected_assisted_goals": 6.1,
        "non_penalty_xg": 16.0,
        "key_passes": 35, "progressive_passes": 120,
        "tackles_won": 8, "interceptions": 4, "clearances": 2,
        "clean_sheets": 0, "saves": 0,
        "height_in_cm": 184, "foot": "right",
        "league": "Premier League",
        "nationality_group": "France",
        "sub_position": "Centre-Forward",
        "contract_years_remaining": 3.0,
        "from_club": "Ajax",
        "to_club": "Manchester City",
        "transfer_year": 2024,
    }
    print(predict_all_models(example_player).to_string(index=False))
