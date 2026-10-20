from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. TIỀN XỬ LÝ DỮ LIỆU CHUẨN BỊ CHO ML
current_dir = Path(__file__).parent
matches = pd.read_csv(current_dir / 'matches.csv')
appearances = pd.read_csv(current_dir / 'appearances.csv')
goals = pd.read_csv(current_dir / 'messi_all_goals.csv')

matches['date'] = pd.to_datetime(matches['date'])
goals['date'] = pd.to_datetime(goals['date'])
goal_dates = set(goals['date'].dt.date)

# Biến mục tiêu (target): 1 nếu Messi ghi bàn trong trận, 0 nếu không
matches['scored'] = matches['date'].dt.date.isin(goal_dates).astype(int)
matches['venue_encoded'] = (matches['venue'] == 'home').astype(int)

df_analysis = matches.merge(
    appearances[['match_id', 'started', 'minutes_played']],
    on='match_id',
    how='inner',
)

# Chọn các đặc trưng dự báo (Features)
X = df_analysis[['venue_encoded', 'started', 'minutes_played', 'age']].fillna(0)
y = df_analysis['scored']

# Chia dữ liệu Tập huấn luyện (Train 80%) và Tập kiểm thử (Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('==================================================================')
print('  HUẤN LUYỆN & SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING DỰ ĐOÁN GHI BÀN')
print('==================================================================\n')

# 2. ĐỊNH NGHĨA VÀ ĐÁNH GIÁ ĐA MÔ HÌNH
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_proba)
  cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

  print(f'Mô hình [{name}]:')
  print(f' - 5-Fold Cross-Validation Accuracy: {cv_score * 100:.2f}%')
  print(f' - Test Accuracy: {acc * 100:.2f}%')
  print(f' - Chỉ số ROC-AUC: {auc:.4f}\n')

# 3. ĐÁNH GIÁ ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE - RANDOM FOREST)
rf_model = models['Random Forest']
print('------------------------------------------------------------------')
print('Mức độ ảnh hưởng của từng yếu tố đến khả năng ghi bàn (Feature Importance):')
feature_names = {
    'venue_encoded': 'Sân nhà / Sân khách (Venue)',
    'started': 'Đá chính hay Dự bị (Started)',
    'minutes_played': 'Số phút thi đấu (Minutes Played)',
    'age': 'Độ tuổi cầu thủ (Age)',
}

for col, imp in zip(X.columns, rf_model.feature_importances_):
  print(f' - {feature_names.get(col, col)}: {imp * 100:.2f}%')