from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. THIẾT LẬP ĐƯỜNG DẪN & ĐỌC DỮ LIỆU
current_dir = Path(__file__).parent
goals = pd.read_csv(current_dir / 'messi_all_goals.csv')

print('==================================================')
print('  PHÂN TÍCH PHÂN PHỐI THỜI ĐIỂM GHI BÀN CỦA MESSI  ')
print('==================================================\n')

# In ra các cột hiện có trong file để kiểm tra
print('Danh sách các cột trong messi_all_goals.csv:', goals.columns.tolist())

# Tự động tìm cột chứa phút ghi bàn (chứa chữ 'min' hoặc 'minute')
minute_col = None
for col in goals.columns:
  if 'min' in col.lower():
    minute_col = col
    break

if minute_col is None:
  raise ValueError(
      'Không tìm thấy cột chứa phút ghi bàn trong file'
      ' messi_all_goals.csv!'
  )

print(f'==> Đã nhận diện cột phút ghi bàn là: "{minute_col}"\n')


# Xử lý cột phút ghi bàn
def clean_minute(val):
  try:
    if pd.isna(val):
      return np.nan
    val_str = str(val).split('+')[0].strip()
    return int(val_str)
  except:
    return np.nan


goals['clean_minute'] = goals[minute_col].apply(clean_minute)
goals = goals.dropna(subset=['clean_minute'])
goals['clean_minute'] = goals['clean_minute'].astype(int)


# Chia khoảng thời gian trận đấu thành các mốc 15 phút chuẩn trong bóng đá
def categorize_time(m):
  if m <= 15:
    return '01-15'
  elif m <= 30:
    return '16-30'
  elif m <= 45:
    return '31-45+'
  elif m <= 60:
    return '46-60'
  elif m <= 75:
    return '61-75'
  else:
    return '76-90+'


goals['time_interval'] = goals['clean_minute'].apply(categorize_time)

# Thống kê số lượng và tỷ lệ phần trăm bàn thắng theo khung giờ
interval_order = ['01-15', '16-30', '31-45+', '46-60', '61-75', '76-90+']
interval_counts = (
    goals['time_interval']
    .value_counts()
    .reindex(interval_order, fill_value=0)
)

total_goals = len(goals)
print('Thống kê số lượng bàn thắng theo từng khung 15 phút:')
for interval, count in interval_counts.items():
  percentage = (count / total_goals) * 100
  print(
      f' - Khung {interval} phút: {count} bàn thắng ({percentage:.2f}% tổng số'
      ' bàn)'
  )

# 2. TRỰC QUAN HÓA BIỂU ĐỒ CỘT (BAR CHART)
plt.figure(figsize=(10, 6))
sns.set_theme(style='whitegrid')

ax = sns.barplot(
    x=interval_counts.index,
    y=interval_counts.values,
    palette='viridis',
    hue=interval_counts.index,
    legend=False,
)

plt.title(
    'Phân phối Thời điểm Ghi bàn của Lionel Messi theo Khung 15 phút',
    fontsize=14,
    weight='bold',
    pad=15,
)
plt.xlabel('Khung thời gian thi đấu (Match Minutes)', fontsize=12)
plt.ylabel('Số lượng bàn thắng (Goals Count)', fontsize=12)

# Gắn nhãn giá trị số bàn thắng lên đỉnh mỗi cột biểu đồ
for p in ax.patches:
  height = p.get_height()
  ax.annotate(
      f'{int(height)}',
      (p.get_x() + p.get_width() / 2.0, height),
      ha='center',
      va='bottom',
      fontsize=10,
      weight='bold',
      xytext=(0, 3),
      textcoords='offset points',
  )

output_path = current_dir / 'goal_timing_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(
    '\n==> Đã xuất biểu đồ Phân phối thời điểm ghi bàn thành công tại:'
    f' {output_path}'
)