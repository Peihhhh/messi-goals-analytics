from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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
# KET NOI 2 TANG: CASCADE PREDICTION
# ==========================================
print("\n[CASCADE] Dang ket noi 2 tang de du doan tong hop...")

# Buoc 1: Tang 1 du doan xac suat ghi ban cho TOAN BO tran
X1_all = pd.get_dummies(df[features_stage1], drop_first=True).reindex(
    columns=X1.columns, fill_value=0
)
df["pred_scored_prob"] = model_stage1.predict_proba(X1_all)[:, 1]

# Buoc 2: Tang 2 du doan so ban NHUNG CHI KHI Tang 1 duyet
THRESHOLD = 0.5  # Nguong quyet dinh: P(scored) > 0.5 moi chuyen sang Tang 2

# Chuan bi features cho Tang 2 tren TOAN BO du lieu
features_stage2_safe = [f for f in features_stage2 if f in df.columns]
X2_all = pd.get_dummies(df[features_stage2_safe], drop_first=True).reindex(
    columns=X2.columns, fill_value=0
)

# Tang 2 chi chay khi Tang 1 "duyet" (prob > threshold)
mask_scored = df["pred_scored_prob"] > THRESHOLD
df["pred_goals_if_scored"] = 0.0
df.loc[mask_scored, "pred_goals_if_scored"] = model_stage2.predict(
    X2_all.loc[mask_scored]
)

# Lam tron va gioi han ket qua trong khoang hop ly [1, 4]
df.loc[mask_scored, "pred_goals_if_scored"] = (
    df.loc[mask_scored, "pred_goals_if_scored"].clip(1, 4).round(2)
)

# Buoc 3: Tinh ky vong so ban thang = P(scored) x E[goals | scored]
df["expected_goals"] = (df["pred_scored_prob"] * df["pred_goals_if_scored"]).round(2)

# ==========================================
# HIEN THI KET QUA CASCADE
# ==========================================
print(f"\n{'='*60}")
print(f"  KET QUA DU DOAN PHAN TANG (CASCADE)")
print(f"{'='*60}")

total = len(df)
predicted_scored = mask_scored.sum()
predicted_not_scored = total - predicted_scored

print(f"\n[TANG 1] Phan loai co ghi ban hay khong (Threshold = {THRESHOLD}):")
print(f"   - Du doan GHI BAN:     {predicted_scored}/{total} tran ({predicted_scored/total*100:.1f}%)")
print(f"   - Du doan KHONG ghi:   {predicted_not_scored}/{total} tran ({predicted_not_scored/total*100:.1f}%)")

print(f"\n[TANG 2] Du doan so ban (chi tinh tran duoc Tang 1 duyet):")
scored_preds = df.loc[mask_scored, "pred_goals_if_scored"]
print(f"   - So tran duoc dua vao Tang 2:  {len(scored_preds)}")
print(f"   - Trung binh du doan:           {scored_preds.mean():.2f} ban/tran")
print(f"   - Khoang du doan:               [{scored_preds.min():.2f} - {scored_preds.max():.2f}]")

print(f"\n[TONG HOP] Ky vong so ban thang (E[goals] = P(scored) x E[goals|scored]):")
print(f"   - Trung binh ky vong:           {df['expected_goals'].mean():.2f} ban/tran")
print(f"   - Thuc te trung binh:           {df['messi_goals'].mean():.2f} ban/tran")

# So sanh tren tap test (Tang 1)
print(f"\n[KIEM TRA] So sanh du doan vs thuc te tren tap test Tang 1:")
test_idx = X1_test.index
df_test = df.loc[test_idx].copy()

actual_scored = df_test["scored"].astype(int)
pred_scored = (df_test["pred_scored_prob"] > THRESHOLD).astype(int)
cm = confusion_matrix(actual_scored, pred_scored)
tn, fp, fn, tp = cm.ravel()
print(f"   Confusion Matrix:")
print(f"                      Du doan KHONG  |  Du doan CO")
print(f"   Thuc te KHONG:         {tn:>4}         |     {fp:>4}")
print(f"   Thuc te CO:            {fn:>4}         |     {tp:>4}")

# Vi du cu the
print(f"\n[VI DU] 10 tran dau tien voi du doan cascade:")
print(f"{'':>3} {'Opponent':<20} {'Venue':<6} {'P(scored)':<12} {'Tang2 pred':<12} {'E[goals]':<10} {'Actual':<8}")
print(f"   {'-'*68}")
sample = df.head(10)
for _, row in sample.iterrows():
    status = "-> Tang 2" if row["pred_scored_prob"] > THRESHOLD else "   STOP"
    print(
        f"   {row['opponent']:<20} {row['venue']:<6} "
        f"{row['pred_scored_prob']:<12.3f} "
        f"{row['pred_goals_if_scored']:<12.2f} "
        f"{row['expected_goals']:<10.2f} "
        f"{int(row['messi_goals']):<8} {status}"
    )

# ==========================================
# XUAT FILE CHO POWER BI
# ==========================================
output_cols = [
    "match_id", "date", "season", "age", "club", "opponent", "venue",
    "started", "position_group", "messi_goals", "scored",
    "pred_scored_prob", "pred_goals_if_scored", "expected_goals",
]
output_df = df[[c for c in output_cols if c in df.columns]]

output_file = DATA_DIR / "messi_ai_predictions.csv"
output_df.to_csv(output_file, index=False)
print(f"\n[HOAN TAT] Da luu file ket qua du doan tai: {output_file}")
print(f"   - Cot 'pred_scored_prob':      Xac suat ghi ban (Tang 1)")
print(f"   - Cot 'pred_goals_if_scored':  So ban du doan (Tang 2, = 0 neu Tang 1 khong duyet)")
print(f"   - Cot 'expected_goals':        Ky vong ban thang = P x E[goals|scored]")
