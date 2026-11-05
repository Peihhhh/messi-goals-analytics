from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

print("==================================================")
print("  CHAY PIPELINE MO HINH PHAN TANG 2 GIAI DOAN (AI)  ")
print("==================================================")

# 1. Dinh nghia duong dan chuan theo cau truc thu muc thuc te
DATA_DIR = Path("data/clean")
matches_path = DATA_DIR / "messi_league_matches.csv"
goals_path = DATA_DIR / "messi_goals_clean.csv"

if not matches_path.exists() or not goals_path.exists():
    raise FileNotFoundError(
        f"Khong tim thay file du lieu trong thu muc {DATA_DIR}. Vui long kiem tra lai!"
    )

matches_df = pd.read_csv(matches_path)
goals_df = pd.read_csv(goals_path)

# 2. Lam giau du lieu (Feature Engineering): Tong hop thong tin tu file chi tiet
goals_agg = (
    goals_df.groupby("match_id")
    .agg(
        total_assists_in_match=("has_assist", "sum"),
        dominant_goal_type=(
            "goal_type",
            lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
        ),
    )
    .reset_index()
)

df = matches_df.merge(goals_agg, on="match_id", how="left")
df["total_assists_in_match"] = df["total_assists_in_match"].fillna(0)
df["dominant_goal_type"] = df["dominant_goal_type"].fillna("None")
print(f"-> Da hop nhat du lieu thanh cong! Tong so tran: {len(df)} dong.")

# ==========================================
# TANG 1: PHAN LOAI (Co ghi ban hay khong?)
# ==========================================
print("\n[TANG 1] Dang huan luyen mo hinh du doan kha nang no sung (Scored)...")
features_stage1 = [
    "age",
    "matchday",
    "venue",
    "started",
    "position_group",
    "opponent",
]
X1 = pd.get_dummies(df[features_stage1], drop_first=True)
y1 = df["scored"].astype(int)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)

model_stage1 = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42
)
model_stage1.fit(X1_train, y1_train)

y1_pred = model_stage1.predict(X1_test)
y1_prob = model_stage1.predict_proba(X1_test)[:, 1]

print(f" -> ROC-AUC Score (Tang 1): {roc_auc_score(y1_test, y1_prob):.4f}")
print(classification_report(y1_test, y1_pred))

# ==========================================
# TANG 2: DU DOAN SO BAN THANG (Neu co ghi ban)
# ==========================================
print("\n[TANG 2] Dang huan luyen mo hinh du doan so luong ban thang...")
scored_matches = df[df["scored"] == True].copy()

features_stage2 = [
    "age",
    "matchday",
    "venue",
    "started",
    "position_group",
    "opponent",
    "shots",
    "xg",
    "total_assists_in_match",
]
X2 = pd.get_dummies(
    scored_matches[[f for f in features_stage2 if f in scored_matches.columns]],
    drop_first=True,
)
y2 = scored_matches["messi_goals"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model_stage2 = RandomForestRegressor(
    n_estimators=150, max_depth=6, random_state=42
)
model_stage2.fit(X2_train, y2_train)

y2_pred = model_stage2.predict(X2_test)

print(f" -> Mean Squared Error (MSE Tang 2): {mean_squared_error(y2_test, y2_pred):.4f}")
print(f" -> R2 Score (Tang 2): {r2_score(y2_test, y2_pred):.4f}")

# ==========================================
# XUAT KET QUA RA FILE PHUC VU POWER BI
# ==========================================
df["pred_scored_prob"] = model_stage1.predict_proba(
    pd.get_dummies(df[features_stage1], drop_first=True)
    .reindex(columns=X1.columns, fill_value=0)
)[
    :, 1
]
output_file = DATA_DIR / "messi_ai_predictions.csv"
df.to_csv(output_file, index=False)
print(f"\n[HOAN TAT] Da luu file ket qua du doan tai: {output_file}")