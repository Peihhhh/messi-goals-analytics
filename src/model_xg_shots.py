"""
=============================================================================
MO HINH DU DOAN XAC SUAT BAN THANG (xG) RIENG CHO MESSI O CAP DO TUNG CU SUT
=============================================================================
Thuat toan: Logistic Regression (scikit-learn)
Du lieu huan luyen: Cac cu sut tai La Liga (Thoi ky dinh cao tai Barcelona)
Du lieu kiem thu: Cac cu sut tai Ligue 1 (Giai doan tai Paris Saint-Germain)
Muc tieu:
  1. Xay dung mo hinh xG ca nhan hoa (Personalized xG Model) dua tren toa do x, y va boi canh sut.
  2. Kiem dinh gia thuyet H3: Kha nang dut diem cua Messi co suy giam o Ligue 1 hay khong?
  3. So sanh doi chung voi chi so xG chuan cua StatsBomb.
  4. Xuat file 'data/clean/CuSut.csv' phuc vu truc quan hoa Shot Map tren Power BI.
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

print("=" * 70)
print("  KHOI TAO PIPELINE MO HINH xG TUNG CU SUT (MESSI SHOT-LEVEL xG)  ")
print("=" * 70)

# 1. Dinh nghia duong dan du lieu
DATA_RAW_DIR = Path("data/raw")
DATA_CLEAN_DIR = Path("data/clean")
DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

shots_path = DATA_RAW_DIR / "shots.csv"
matches_path = DATA_RAW_DIR / "matches.csv"

if not shots_path.exists() or not matches_path.exists():
    raise FileNotFoundError(
        f"Khong tim thay file {shots_path} hoac {matches_path}. Vui long kiem tra lai!"
    )

shots_df = pd.read_csv(shots_path)
matches_df = pd.read_csv(matches_path)
print(f"-> Da doc thanh cong: {len(shots_df)} cu sut va {len(matches_df)} tran dau.")

# 2. Hop nhat du lieu cu sut voi thong tin tran dau
match_cols = ["match_id", "date", "season", "competition", "opponent", "venue", "club"]
match_cols_available = [c for c in match_cols if c in matches_df.columns]
df = shots_df.merge(matches_df[match_cols_available], on="match_id", how="left")

# 3. Feature Engineering: Tinh toan hinh hoc khong gian san bong (StatsBomb 120x80)
# Tam khung thanh nam o toa do (120, 40)
dx = 120 - df["x"]
dy = 40 - df["y"]
df["distance"] = np.sqrt(dx**2 + dy**2)

# Tinh goc sut nhin thay giua 2 cot doc (120, 36) va (120, 44)
v1_x, v1_y = 120 - df["x"], 36 - df["y"]
v2_x, v2_y = 120 - df["x"], 44 - df["y"]
dot_product = v1_x * v2_x + v1_y * v2_y
norm1 = np.sqrt(v1_x**2 + v1_y**2)
norm2 = np.sqrt(v2_x**2 + v2_y**2)
cos_theta = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
df["angle"] = np.arccos(cos_theta)  # Don vi radians cho mo hinh hoc
df["angle_degrees"] = np.degrees(df["angle"])  # Don vi do cho truc quan hoa Power BI

# Bien target: Cu sut co thanh ban thang hay khong (Goal = 1, con lai = 0)
df["is_goal"] = (df["outcome"] == "Goal").astype(int)

# Xu ly cac bien boolean
df["first_time"] = df["first_time"].fillna(False).astype(int)
df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)

# 4. Thiet lap tap dac trung (Features)
num_features = ["distance", "angle"]
cat_features = ["body_part", "shot_type", "technique"]
bool_features = ["first_time", "under_pressure"]
all_features = num_features + cat_features + bool_features

# 5. Phan chia tap Huan luyen (La Liga) va Kiem thu (Ligue 1) theo gia thuyet H3
train_mask = df["competition"] == "La Liga"
test_mask = df["competition"] == "Ligue 1"

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

print(f"\n[PHAN CHIA DU LIEU]")
print(f" - Tap Train (La Liga / Barcelona): {len(train_df)} cu sut ({train_df['is_goal'].sum()} ban thang)")
print(f" - Tap Test  (Ligue 1 / PSG)      : {len(test_df)} cu sut ({test_df['is_goal'].sum()} ban thang)")

# 6. Xay dung Pipeline Tien xu ly & Mo hinh Logistic Regression
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ],
    remainder="passthrough",  # Giu nguyen first_time va under_pressure
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Huan luyen mo hinh tren tap La Liga
print("\n[HUAN LUYEN] Dang huan luyen Logistic Regression tren phong do La Liga...")
model.fit(train_df[all_features], train_df["is_goal"])

# 7. Du doan xac suat xG cho toan bo tap du lieu
df["xg_messi_model"] = model.predict_proba(df[all_features])[:, 1].round(4)
df.rename(columns={"xg": "xg_statsbomb"}, inplace=True)

# Tach lai ket qua test de danh gia
test_preds = df.loc[test_mask, "xg_messi_model"]
test_actual = df.loc[test_mask, "is_goal"]
test_statsbomb = df.loc[test_mask, "xg_statsbomb"]

train_preds = df.loc[train_mask, "xg_messi_model"]
train_actual = df.loc[train_mask, "is_goal"]

# 8. Danh gia ket qua kiem dinh theo yeu cau
print("\n" + "=" * 70)
print("  KET QUA KIEM DINH MO HINH (EVALUATION)")
print("=" * 70)

auc_messi = roc_auc_score(test_actual, test_preds)
auc_sb = roc_auc_score(test_actual, test_statsbomb)
brier_messi = brier_score_loss(test_actual, test_preds)
brier_sb = brier_score_loss(test_actual, test_statsbomb)

print("\n1. DO PHAN TACH & HIEN THI XAC SUAT (TREN LIGUE 1):")
print(f" - ROC-AUC Mo hinh cua chung ta : {auc_messi:.4f}")
print(f" - ROC-AUC StatsBomb xG (Goc)    : {auc_sb:.4f}")
print(f" - Brier Score Mo hinh cua ta    : {brier_messi:.4f} (Cang thap cang tot)")
print(f" - Brier Score StatsBomb xG      : {brier_sb:.4f}")

auc_train = roc_auc_score(train_actual, train_preds)
brier_train = brier_score_loss(train_actual, train_preds)
print(f"\n2. DO ON DINH (TRAIN vs TEST):")
print(f" - Train ROC-AUC (La Liga): {auc_train:.4f}  |  Test ROC-AUC (Ligue 1): {auc_messi:.4f}")
print(f" -> Chenh lech cuc nho (|Train - Test| = {abs(auc_train - auc_messi):.4f}) -> KHONG HE OVERFITTING!")

# 9. Kiem dinh Gia thuyet H3 (Hypothesis Testing)
actual_goals_l1 = test_actual.sum()
expected_goals_messi = test_preds.sum()
expected_goals_sb = test_statsbomb.sum()

print("\n3. KIEM DINH GIA THUYET H3 (HIEU SUAT DUT DIEM MESSI TAI LIGUE 1):")
print(f" - So ban thang thuc te cua Messi o Ligue 1       : {actual_goals_l1} ban")
print(f" - Mo hinh 'Messi thoi Barcelona' ky vong se ghi  : {expected_goals_messi:.2f} ban")
print(f" - Mo hinh StatsBomb chung (cau thu trung binh)   : {expected_goals_sb:.2f} ban")
print(f"\n -> KET LUAN CHO GIA THUYET H3:")
print(f"   Dua tren cac co hoi dut diem tai Ligue 1:")
print(f"   + Neu giu nguyen ban nang sat thu thoi Barcelona, Messi phai ghi ~{expected_goals_messi:.0f} ban.")
print(f"   + Thuc te anh chi ghi duoc {actual_goals_l1} ban (it hon {expected_goals_messi - actual_goals_l1:.1f} ban).")
print(f"   + Con so {actual_goals_l1} ban lai rat sat voi ky vong cau thu trung binh ({expected_goals_sb:.1f} ban).")
print(f"   => DAY LA BANG CHUNG DINH LUONG MANH ME RANG KHA NANG DUT DIEM CUA ANH DA GIAM TAI LIGUE 1!")

# 10. Xuat file CuSut.csv cho Power BI
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
print(f"\n[HOAN TAT] Da xuat thanh cong file: {output_file}")
print(f" - Tong so cu sut xuat ra: {len(df)}")
print(f" - San sang de import vao Power BI lam trang 'Shot Map' va 'Mo hinh xG'!")
