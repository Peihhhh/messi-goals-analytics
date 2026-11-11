"""Messi-specific xG model: trained on his Barca shots, applied to his PSG shots."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda_utils import RANDOM_SEED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["distance_to_goal", "shot_angle_deg"]
CATEGORICAL_FEATURES = ["body_part", "shot_type", "technique"]
BOOLEAN_FEATURES = ["first_time", "under_pressure"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
TARGET = "goal"
# Each category is compared with Messi's most common case, so coefficients read
# "compared with an open-play, normal, left-foot shot". (drop="first" would pick the
# alphabetically first category, e.g. Backheel with 1 shot, which makes coefficients unreadable.)
REFERENCE_CATEGORIES = {"body_part": "Left Foot", "shot_type": "Open Play", "technique": "Normal"}

N_CV_FOLDS = 5
N_MODEL_BOOTSTRAPS = 500   # refits used to include the model's own uncertainty in the simulation


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_model_data(shots):
    """Non-penalty shots with the target as 0/1 and boolean flags as 0/1."""
    data = shots[~shots["is_penalty"]].copy()
    data[TARGET] = data["is_goal"].astype(int)
    data[BOOLEAN_FEATURES] = data[BOOLEAN_FEATURES].astype(int)
    return data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_xg_pipeline():
    """Same model as the first version: scaled distance/angle + one-hot categories -> logistic regression."""
    reference = [REFERENCE_CATEGORIES[column] for column in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop=reference, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def fit_xg_model(train):
    return build_xg_pipeline().fit(train[FEATURES], train[TARGET])


def predict_xg(model, shots):
    return model.predict_proba(shots[FEATURES])[:, 1]


def cross_validated_predictions(train, group_column="season"):
    """
    Out-of-fold predictions: every shot is predicted by a model that never saw its season.
    Grouping by season stops shots from the same season leaking between train and validation.
    """
    predictions = np.zeros(len(train))
    folds = GroupKFold(n_splits=N_CV_FOLDS)
    for train_index, valid_index in folds.split(train, train[TARGET], groups=train[group_column]):
        model = fit_xg_model(train.iloc[train_index])
        predictions[valid_index] = predict_xg(model, train.iloc[valid_index])
    return predictions


def coefficient_table(model):
    """Coefficients with readable names; odds ratio = exp(coefficient)."""
    preprocessor = model.named_steps["preprocessor"]
    names = (
        NUMERIC_FEATURES
        + preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        + BOOLEAN_FEATURES
    )
    coefficients = model.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["scale"] = np.where(table["feature"].isin(NUMERIC_FEATURES), "mỗi 1 độ lệch chuẩn", "có so với không")
    return table.sort_values("coefficient").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(actual, predicted):
    """Ranking (AUC), probability accuracy (Brier, log loss) and total calibration."""
    return {
        "roc_auc": roc_auc_score(actual, predicted),
        "brier": brier_score_loss(actual, predicted),
        "log_loss": log_loss(actual, predicted),
        "predicted_goals": float(np.sum(predicted)),
        "actual_goals": int(np.sum(actual)),
    }


def bootstrap_auc_difference(actual, predicted_a, predicted_b, n_resamples=2000):
    """95% CI for AUC(a) - AUC(b) on the same shots."""
    rng = np.random.default_rng(RANDOM_SEED)
    actual, predicted_a, predicted_b = map(np.asarray, (actual, predicted_a, predicted_b))
    differences = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(actual), len(actual))
        if actual[picks].min() == actual[picks].max():
            continue
        differences.append(roc_auc_score(actual[picks], predicted_a[picks])
                           - roc_auc_score(actual[picks], predicted_b[picks]))
    return np.percentile(differences, [2.5, 97.5])


def calibration_table(actual, predicted, n_bins=8):
    """Group shots into bins of equal size by predicted xG and compare mean prediction with the goal rate."""
    data = pd.DataFrame({"actual": actual, "predicted": predicted})
    data["bin"] = pd.qcut(data["predicted"], q=n_bins, duplicates="drop")
    return data.groupby("bin", observed=True).agg(
        mean_predicted=("predicted", "mean"), goal_rate=("actual", "mean"), shots=("actual", "size")
    )


# ---------------------------------------------------------------------------
# Counterfactual for PSG (H3)
# ---------------------------------------------------------------------------

def simulate_counterfactual_goals(train, target_shots):
    """
    Goals Messi "should" have scored on target_shots if he finished like in `train`.

    Each round refits the model on a bootstrap sample of `train` (model uncertainty), predicts
    xG for target_shots, then draws a goal / no goal for every shot (finishing luck).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    totals = np.empty(N_MODEL_BOOTSTRAPS, dtype=int)
    for round_index in range(N_MODEL_BOOTSTRAPS):
        resample = train.iloc[rng.integers(0, len(train), len(train))]
        probabilities = predict_xg(fit_xg_model(resample), target_shots)
        totals[round_index] = (rng.random(len(probabilities)) < probabilities).sum()
    return totals


def summarize_counterfactual(label, train, target_shots, simulated_totals):
    actual = int(target_shots[TARGET].sum())
    return {
        "training_data": label,
        "train_shots": len(train),
        "expected_goals": float(predict_xg(fit_xg_model(train), target_shots).sum()),
        "interval_low": float(np.percentile(simulated_totals, 2.5)),
        "interval_high": float(np.percentile(simulated_totals, 97.5)),
        "actual_goals": actual,
        "p_at_most_actual": float((simulated_totals <= actual).mean()),
    }
