# Hướng dẫn chạy Data Pipeline, Training Hệ thống & Triển khai Ứng dụng

Tài liệu này tổng hợp toàn bộ quy trình từ lúc chuẩn bị dữ liệu, huấn luyện mô hình (AI), cho đến việc thiết lập hệ thống Backend (Spring Boot + FastAPI) và giao diện người dùng Frontend (Next.js Enterprise).

## 1. Môi trường yêu cầu (Prerequisites)

Hãy đảm bảo bạn đã cài đặt các công cụ sau:
- Python 3.12 (Hỗ trợ CUDA cho GPU)
- Node.js (v18 trở lên) & npm
- Java 21 (JDK 21) & Maven

Đối với AI và Python Backend, hãy kích hoạt môi trường ảo (Virtual Environment):
```bash
# Kích hoạt môi trường (Windows)
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---
# PHẦN I: AI PIPELINE & TRAINING

## 2. Giai đoạn 1: Phân tích Dữ liệu (EDA)

- **Thư mục:** `notebooks/`
- **Thao tác:** Mở và chạy file `1_EDA.ipynb`. Quá trình này giúp bạn hiểu phân phối ảnh, phổ màu (Cr/Cb) và các dấu vết bất thường của ảnh Deepfake (do StyleGAN3 tạo ra).

## 3. Giai đoạn 2: Tiền xử lý ngoại tuyến (Offline Preprocessing)

Thay vì cắt khuôn mặt trực tiếp trong lúc nạp dữ liệu, chúng ta chạy kịch bản MTCNN 1 lần duy nhất để tối ưu VRAM.
- **Thực thi lệnh:** `python core/preprocess_data.py`
- **Kết quả:** Sinh ra thư mục `data/cropped_faces/` chứa khuôn mặt đã được cắt và mở rộng viền.

## 4. Giai đoạn 3 & 4: Huấn luyện Mô hình OOP (ResNet-50 & ViT)

Hệ thống đã đóng gói toàn bộ DataModule, Augmentations và Training Pipeline trong `core/pipeline.py`.
Để bắt đầu training, chỉ cần gõ lệnh:

**Huấn luyện Vision Transformer (ViT):**
```bash
python run_experiment.py --model vit --batch_size 8 --epochs 20
```

**Huấn luyện ResNet-50 (CNN):**
```bash
python run_experiment.py --model resnet --batch_size 8 --epochs 20
```

> **Lưu ý:** Gradient Accumulation (`steps=4`) được tự động kích hoạt để chống tràn VRAM. Mô hình tốt nhất sẽ lưu tại `models/`.

## 5. Giai đoạn 5: Đánh giá chéo (Head-to-head Evaluation)

Sau khi training xong, tiến hành test đối kháng cả 2 mô hình trên tập Test:
```bash
python compare_models.py
```
👉 Kết quả (F1-score, biểu đồ Joint ROC Curve) xuất ra thư mục `outputs/comparison/`.

---
# PHẦN II: TRIỂN KHAI HỆ THỐNG (DEPLOYMENT)

Hệ thống được thiết kế theo mô hình Microservices, bao gồm:
1. **AI Worker (Python FastAPI)**
2. **Core Backend (Java Spring Boot 3)**
3. **Frontend Dashboard (Next.js React)**

## 6. Giai đoạn 6: Triển khai Backend Services

### 6.1. FastAPI AI Worker
Chịu trách nhiệm nhận ảnh, chạy qua mô hình PyTorch (ResNet/ViT) và trả về điểm số Fake/Real kèm Heatmap.
- **Thư mục:** `f:\DeepLearning_Final\`
- **Khởi chạy:** `python run_api.py` (Server sẽ chạy ở port `8000`).

### 6.2. Java Spring Boot Core Backend
Chịu trách nhiệm quản lý Session, Lịch sử Quét và giao tiếp WebSocket/REST với Frontend.
- **Thư mục:** `core-backend/`
- **Môi trường:** Đã được nâng cấp lên **Spring Boot 3.3.0** và **Java 21**. Sử dụng kiến trúc `jakarta.*`.
- **Khởi chạy:** Mở `core-backend` bằng IntelliJ IDEA (set Project SDK là Java 21) và chạy class Main, hoặc dùng lệnh Maven: `./mvnw spring-boot:run`.

## 7. Giai đoạn 7: Triển khai Frontend (Next.js Enterprise)

Frontend đã được nâng cấp mạnh mẽ thành một công cụ Forensics chuyên nghiệp (Phiên bản V2).
- **Thư mục:** `frontend/`
- **Tính năng nổi bật:**
  - **Forensics Analysis:** Quét ảnh, hiển thị Heatmap Slider chồng lấp.
  - **XAI Radar Chart:** Bóc tách vùng bị nghi ngờ (Mắt, Miệng, Làn da...).
  - **Export PDF:** Xuất báo cáo pháp y (Forensics Report).
  - **History Dashboard:** Lịch sử quét, Line Chart theo dõi độ ổn định (F1 Score) của mô hình.
  - **Dark Mode:** Tích hợp giao diện Tối/Sáng chuẩn ngầu.
- **Khởi chạy:**
  ```bash
  cd frontend
  npm run dev
  ```
- **Truy cập:** `http://localhost:3000`

---
🎉 **Tiến độ Dự án:** Hoàn thành 100% Pipeline AI, Backend API, và Frontend Dashboard! Sẵn sàng mang đi báo cáo nghiệm thu.
