# messi-goals-analytics
## H1: Tuổi đỉnh cao ghi bàn của Messi

### Giả thuyết
Đỉnh cao npG/90 của Messi rơi vào độ tuổi 24 đến 26, trùng với độ tuổi đỉnh cao
điển hình của tiền đạo.

### Chỉ số
**npG/90** (non-penalty goals per 90 minutes) là số bàn thắng không tính
penalty, quy về mỗi 90 phút thi đấu:

    npG/90 = (số bàn không penalty / số phút thi đấu) × 90

Bài này dùng npG/90 thay cho tổng số bàn vì hai lý do:
- Chia theo phút giúp so sánh công bằng giữa các mùa có thời lượng thi đấu
  khác nhau (do chấn thương, xoay tua).
- Penalty phản ánh việc ai được phân công đá phạt đền hơn là khả năng ghi bàn.

### Lý do
- Dendir (2016) phân tích điểm đánh giá của WhoScored tại 4 giải hàng đầu
  châu Âu và ước lượng tiền đạo trung bình đạt đỉnh ở khoảng 25 tuổi.
- Vị trí tiền đạo phụ thuộc nhiều vào tốc độ và sức bùng nổ, những yếu tố
  suy giảm sớm. Messi thời trẻ ghi bàn chủ yếu nhờ tăng tốc và rê dắt, nên
  không có lý do rõ ràng để kỳ vọng anh ấy lệch khỏi quy luật này.

**Giả thuyết đối lập:** Dendir cũng lưu ý tuổi đỉnh cao có thể muộn hơn ở
cầu thủ có năng lực cao. Nếu đỉnh của Messi rơi sau tuổi 26, kết quả sẽ
ủng hộ nhận định này thay vì H1.

### Phương pháp
- **Phạm vi:** bàn thắng cấp câu lạc bộ từ 2004/05 đến 2022/23 (Barcelona
  và PSG). Kết quả chính dùng giải vô địch quốc gia; kết quả cho tất cả
  các giải được dùng để kiểm tra lại.
- **Tuổi:** tuổi của mỗi mùa = năm bắt đầu mùa giải trừ 1987.
- **Ngưỡng tối thiểu:** chỉ tính các mùa có từ 900 phút thi đấu trở lên.
- **Định nghĩa đỉnh cao:** 3 mùa liên tiếp có npG/90 trung bình cao nhất.

### Tiêu chí bác bỏ
H1 bị bác bỏ nếu trong 3 mùa liên tiếp có npG/90 trung bình cao nhất,
mùa ở giữa không rơi vào tuổi 24, 25 hoặc 26.

### Hạn chế
- Dendir (2016) đo phong độ bằng điểm đánh giá tổng thể, còn bài này dùng
  npG/90. Hai thước đo có thể đạt đỉnh ở tuổi khác nhau, nên việc so sánh
  chỉ mang tính tham khảo.
- Mốc 25 tuổi được ước lượng từ dữ liệu 2010/11 đến 2014/15, có thể không
  đại diện cho cả giai đoạn 2004 đến 2023.
- Giai đoạn 24 đến 26 tuổi của Messi trùng với thời kỳ đá số 9 ảo trong một
  Barcelona rất mạnh. Phân tích này không tách được ảnh hưởng của tuổi khỏi
  ảnh hưởng của vai trò và đồng đội.
- Kết luận dựa trên một cầu thủ, nên không khái quát được cho tiền đạo
  nói chung.

### Tài liệu tham khảo
Dendir, S. (2016). When do soccer players peak? A note. *Journal of Sports
Analytics*.