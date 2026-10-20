from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# 1. ĐỌC DỮ LIỆU
current_dir = Path(__file__).parent
matches = pd.read_csv(current_dir / 'matches.csv')
appearances = pd.read_csv(current_dir / 'appearances.csv')
goals = pd.read_csv(current_dir / 'messi_all_goals.csv')

print('==================================================')
print('  GIAI ĐOẠN 1: PHÂN TÍCH & KIỂM ĐỊNH THỐNG KÊ (EDA) ')
print('==================================================\n')

# 2. KIỂM ĐỊNH PHÂN PHỐI CHUẨN (Shapiro-Wilk)
stat_shapiro, p_shapiro = stats.shapiro(matches['goals_for'])
print('[1] KIỂM ĐỊNH PHÂN PHỐI CHUẨN (Shapiro-Wilk Test):')
print(f' - W-statistic: {stat_shapiro:.4f}')
print(f' - P-value: {p_shapiro:.4e}')
if p_shapiro < 0.05:
  print(
      ' => Kết luận: Dữ liệu bàn thắng KHÔNG tuân theo phân phối chuẩn (p <'
      ' 0.05).\n'
  )
else:
  print(
      ' => Kết luận: Dữ liệu bàn thắng tuân theo phân phối chuẩn (p >='
      ' 0.05).\n'
  )

# 3. KIỂM ĐỊNH SO SÁNH CÁC GIẢI ĐẤU (ANOVA & Kruskal-Wallis)
top_competitions = matches['competition'].value_counts().head(4).index
comp_data = [
    matches[matches['competition'] == comp]['goals_for']
    for comp in top_competitions
]

f_stat, p_anova = stats.f_oneway(*comp_data)
kw_stat, p_kw = stats.kruskal(*comp_data)

print('[2] KIỂM ĐỊNH KHÁC BIỆT GIỮA CÁC GIẢI ĐẤU (Top 4 Competitions):')
print(f' - Kruskal-Wallis H-statistic: {kw_stat:.4f}, P-value: {p_kw:.4f}')
if p_kw < 0.05:
  print(
      ' => Kết luận: Có sự khác biệt có ý nghĩa thống kê về hiệu suất ghi'
      ' bàn.\n'
  )
else:
  print(' => Kết luận: Không có sự khác biệt có ý nghĩa thống kê.\n')

# 4. MA TRẬN TƯƠNG QUAN (Correlation Heatmap)
matches['date'] = pd.to_datetime(matches['date'])
goals['date'] = pd.to_datetime(goals['date'])
goal_dates = set(goals['date'].dt.date)

matches['scored'] = matches['date'].dt.date.isin(goal_dates).astype(int)
matches['venue_encoded'] = (matches['venue'] == 'home').astype(int)

df_analysis = matches.merge(
    appearances[['match_id', 'started', 'minutes_played']],
    on='match_id',
    how='inner',
)

corr_matrix = df_analysis[
    ['scored', 'venue_encoded', 'started', 'minutes_played', 'age', 'goals_for']
].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    cbar=True,
)
plt.title(
    'Ma trận tương quan Pearson (Correlation Heatmap)',
    fontsize=13,
    weight='bold',
)
heatmap_path = current_dir / 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'[3] Đã xuất biểu đồ Ma trận tương quan: {heatmap_path}\n')