from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Tự động xác định đường dẫn thư mục hiện tại và tìm file matches.csv
current_dir = Path(__file__).parent
matches_path = current_dir / 'matches.csv'

# Kiểm tra xem file có tồn tại không trước khi đọc
if not matches_path.exists():
  raise FileNotFoundError(
      f'Không tìm thấy file matches.csv tại đường dẫn: {matches_path}. '
      'Hãy đảm bảo file matches.csv nằm cùng thư mục với visualize.py!'
  )

matches = pd.read_csv(matches_path)
print(f'Đã tải thành công {len(matches)} dòng dữ liệu từ matches.csv')

# 2. Thiết lập giao diện biểu đồ
plt.figure(figsize=(9, 6))
sns.set_theme(style='whitegrid')

# Vẽ biểu đồ hộp (Boxplot) so sánh bàn thắng Sân nhà vs Sân khách
sns.boxplot(
    x='venue', y='goals_for', data=matches, palette=['#3498db', '#e74c3c']
)

plt.title(
    'So sánh Hiệu suất Ghi bàn: Sân nhà (Home) vs Sân khách (Away)',
    fontsize=14,
    weight='bold',
    pad=15,
)
plt.xlabel('Khu vực thi đấu (Venue)', fontsize=12)
plt.ylabel('Số bàn thắng ghi được (Goals For)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Sân khách (Away)', 'Sân nhà (Home)'])

# 3. Lưu file ảnh để chèn vào báo cáo hoặc slide
output_image_path = current_dir / 'home_vs_away_comparison.png'
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()

print(f'==> Đã tạo và lưu biểu đồ thành công tại: {output_image_path}')