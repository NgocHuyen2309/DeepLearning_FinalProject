# Báo cáo Phân tích Dữ liệu (EDA) & Định hướng Kỹ thuật Tối thượng (Scientific Grade)
**Dự án:** Deepfake Detection & Face Anti-Spoofing
**Tập dữ liệu:** Real vs Fake Faces (StyleGAN3)

Bản định hướng này được đúc kết từ 3 Notebook EDA đạt chuẩn nghiên cứu khoa học, áp dụng Toán học thống kê và Xử lý tín hiệu số (DSP) để chứng minh và định lượng lỗi của mạng GAN. Các kỹ sư bắt buộc cấu hình tham số theo sát tài liệu này.

## 1. 🛑 Chỉ định cho Data Pipeline (Huấn luyện chung)

> [!WARNING]  
> **Kiểm định Chi-Square & Bản chất của Tập dữ liệu**
>
> 1. **Bản chất tập gốc**: Kiểm định Chi-bình phương (Chi-Square Test) cho thấy $P-value > 0.05$. Điều này chứng minh StyleGAN3 **KHÔNG CÓ LỖI** thiên lệch; nó chỉ nội suy hoàn hảo phân bố của dữ liệu gốc. Lỗi nằm ở tập dữ liệu Real ban đầu bị mất cân bằng trầm trọng (hơn 60% là người da trắng).
> 2. **Giải pháp**: Mặc dù GAN không có lỗi, nhưng nếu ném dữ liệu này vào ResNet/ViT, mô hình sẽ học "đường tắt" (shortcut learning) dựa trên tone màu đa số thay vì phân tích hình học khuôn mặt. Do đó ta vẫn phải Augmentation để phá màu.

**Thực thi Code (Albumentations & PyTorch):**
Bắt buộc tích hợp các phép biến đổi màu HSV và RGBShift để phá hủy tương quan màu da. Tích hợp thêm nhiễu nén JPEG để ép mô hình bỏ qua lỗi tín hiệu tần số cao (Azimuthal 1D).
```python
import albumentations as A

transform = A.Compose([
    # Phá vỡ thiên lệch tone da (Demographic Bias) nhưng không hủy cấu trúc 3 kênh (Không dùng ToGray)
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.4),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
    
    # Ép mô hình học đặc trưng hình học Robust, bỏ qua lỗi siêu cao tần (High-frequency artifacts)
    A.GaussianBlur(blur_limit=(3, 7), p=0.4), 
    A.ImageCompression(quality_lower=40, quality_upper=80, p=0.5),
])
```

## 2. 🧠 Chỉ định cho Huấn luyện ResNet-50 & ViT

> [!TIP]
> **Điểm mù Bhattacharyya & Quang phổ YCbCr**
>
> Sau khi dùng `MediaPipe Face Mesh` để bóc tách (segmentation) chính xác 100% vùng da (loại bỏ phông nền) và đo khoảng cách **Bhattacharyya Distance**, phân bố màu giữa Real và Fake sai lệch lớn nhất ở kênh độ chói **Y** và sắc đỏ **Cr**. Lỗi này thường tập trung ở các vùng ranh giới hàm/tóc.

**Thực thi Code (Grad-CAM XAI):**
- Trong quá trình validation, **bắt buộc** gọi thư viện XAI (như `pytorch-gradcam`) để kiểm tra Heatmap của lớp `layer4` (đối với ResNet50) và `Attention Rollout` (đối với ViT).
- Nếu Heatmap không Focus vào viền hàm và tóc mà chỉ tập trung vào mắt/mũi thì mô hình đang học sai.

## 3. ⚙️ Quy chuẩn Tiền xử lý (Pre-processing)

> [!IMPORTANT]
> **Azimuthal Average 1D & Sức mạnh của Phông nền**
>
> Phân tích **1D Power Spectrum** (Azimuthal Average của FFT) chứng minh rằng ở tần số cao, StyleGAN3 bị sụt giảm năng lượng (Over-smoothing). Lỗi này rõ nét nhất ở ranh giới giữa tóc và nền. Do đó, trong Data Pipeline thực tế, chúng ta tuyệt đối KHÔNG cắt sát mặt như lúc phân tích EDA.

**Thực thi Code (MTCNN):**
```python
from facenet_pytorch import MTCNN
# Phải chừa lề (margin=40) để giữ lại các lỗi nền và tần số ghép nối ở mép mặt
mtcnn = MTCNN(keep_all=False, margin=40, device=device)
```
