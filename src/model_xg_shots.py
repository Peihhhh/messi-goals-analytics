"""
=============================================================================
SHOT-LEVEL EXPECTED GOALS (xG) MODEL FOR LIONEL MESSI
=============================================================================
Algorithm: Logistic Regression (Regularized Classification)
Objective:
  1. Estimate shot conversion probability based on spatial and contextual features.
  2. Test Hypothesis H3: Finishing efficiency comparison between La Liga and Ligue 1.
  3. Benchmark against StatsBomb proprietary xG estimates.
  4. Generate clean shot-level dataset with spatial coordinates and probabilities.
=============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Path configurations
DATA_RAW_DIR = Path("data/raw")
DATA_CLEAN_DIR = Path("data/clean")
DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

shots_path = DATA_RAW_DIR / "shots.csv"
matches_path = DATA_RAW_DIR / "matches.csv"

if not shots_path.exists() or not matches_path.exists():
    raise FileNotFoundError(
        f"Input files not found: {shots_path} or {matches_path}."
    )

shots_df = pd.read_csv(shots_path)
matches_df = pd.read_csv(matches_path)

# 2. Data integration
match_cols = ["match_id", "date", "season", "competition", "opponent", "venue", "club"]
match_cols_available = [c for c in match_cols if c in matches_df.columns]
df = shots_df.merge(matches_df[match_cols_available], on="match_id", how="left")

# 3. Spatial Feature Engineering (StatsBomb 120x80 coordinate system)
# Goal center: (120, 40)
dx = 120 - df["x"]
dy = 40 - df["y"]
df["distance"] = np.sqrt(dx**2 + dy**2)

# Visible goal angle between posts (120, 36) and (120, 44)
v1_x, v1_y = 120 - df["x"], 36 - df["y"]
v2_x, v2_y = 120 - df["x"], 44 - df["y"]
dot_product = v1_x * v2_x + v1_y * v2_y
norm1 = np.sqrt(v1_x**2 + v1_y**2)
norm2 = np.sqrt(v2_x**2 + v2_y**2)
cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
df["angle"] = np.arccos(cos_theta)
df["angle_degrees"] = np.degrees(df["angle"])

# Target variable
df["is_goal"] = (df["outcome"] == "Goal").astype(int)

# Contextual binary indicators
df["first_time"] = df["first_time"].fillna(False).astype(int)
df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)

# 4. Feature sets
num_features = ["distance", "angle"]
cat_features = ["body_part", "shot_type", "technique"]
bool_features = ["first_time", "under_pressure"]
all_features = num_features + cat_features + bool_features

# 5. Dataset partitioning by competition (Hypothesis H3)
train_mask = df["competition"] == "La Liga"
test_mask = df["competition"] == "Ligue 1"

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

# 6. Pipeline construction
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ],
    remainder="passthrough",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Train model
model.fit(train_df[all_features], train_df["is_goal"])

# 7. Probability estimation
df["xg_messi_model"] = model.predict_proba(df[all_features])[:, 1].round(4)
df.rename(columns={"xg": "xg_statsbomb"}, inplace=True)

test_preds = df.loc[test_mask, "xg_messi_model"]
test_actual = df.loc[test_mask, "is_goal"]
test_statsbomb = df.loc[test_mask, "xg_statsbomb"]

train_preds = df.loc[train_mask, "xg_messi_model"]
train_actual = df.loc[train_mask, "is_goal"]

# 8. Evaluation metrics
auc_messi = roc_auc_score(test_actual, test_preds)
auc_sb = roc_auc_score(test_actual, test_statsbomb)
brier_messi = brier_score_loss(test_actual, test_preds)
brier_sb = brier_score_loss(test_actual, test_statsbomb)

auc_train = roc_auc_score(train_actual, train_preds)
actual_goals_l1 = test_actual.sum()
expected_goals_messi = test_preds.sum()
expected_goals_sb = test_statsbomb.sum()

# 9. Clean scientific logging
print("=" * 72)
print("  SHOT-LEVEL EXPECTED GOALS (xG) MODEL - EVALUATION REPORT")
print("=" * 72)
print(f"Dataset Size       : {len(df):,} total shots ({df['is_goal'].sum()} goals, {df['is_goal'].mean()*100:.2f}%)")
print(f"Training Partition : La Liga ({len(train_df):,} shots | {train_df['is_goal'].sum()} goals)")
print(f"Testing Partition  : Ligue 1 ({len(test_df):,} shots | {test_df['is_goal'].sum()} goals)")
print("-" * 72)
print("MODEL PERFORMANCE (TEST SET - LIGUE 1):")
print(f"  - ROC-AUC Score      : {auc_messi:.4f}  (StatsBomb Benchmark: {auc_sb:.4f})")
print(f"  - Brier Score Loss   : {brier_messi:.4f}  (StatsBomb Benchmark: {brier_sb:.4f})")
print(f"  - Training ROC-AUC   : {auc_train:.4f}  (Generalization Gap : {abs(auc_train - auc_messi):.4f})")
print("-" * 72)
print("HYPOTHESIS H3 EMPIRICAL TEST (FINISHING EFFICIENCY):")
print(f"  - Actual Goals Scored in Ligue 1       : {actual_goals_l1:.2f}")
print(f"  - Baseline Expected Goals (StatsBomb)   : {expected_goals_sb:.2f}")
print(f"  - Messi Barcelona Standard Expected xG  : {expected_goals_messi:.2f}")
print(f"  - Finishing Differential (Actual - xG)  : {actual_goals_l1 - expected_goals_messi:+.2f} goals")
print("-" * 72)

# 10. Clean CSV export
output_columns = [
    "match_id",
    "date",
    "season",
    "competition",
    "opponent",
    "venue",
    "period",
    "minute",
    "second",
    "play_pattern",
    "x",
    "y",
    "distance",
    "angle_degrees",
    "body_part",
    "shot_type",
    "technique",
    "first_time",
    "under_pressure",
    "outcome",
    "is_goal",
    "xg_statsbomb",
    "xg_messi_model",
]
output_columns = [c for c in output_columns if c in df.columns]

output_file = DATA_CLEAN_DIR / "CuSut.csv"
df[output_columns].to_csv(output_file, index=False)
print(f"Exported dataset successfully: {output_file} ({len(df):,} records)")
print("=" * 72)
