# DÀN Ý BÁO CÁO MÔN HỌC DEEP LEARNING
**Đề tài:** Xây dựng Hệ thống Phát hiện Deepfake (Deepfake Detection) sử dụng Kiến trúc ResNet-50, Vision Transformer (ViT) và Explainable AI (XAI)

---

## CHƯƠNG 1: MỞ ĐẦU
**1.1. Lý do chọn đề tài**
* Sự bùng nổ của Deepfake và các mô hình sinh (Generative AI) đe dọa an ninh thông tin.
* Nhu cầu cấp thiết về công nghệ giám định đa phương tiện (Forensics).
* Hạn chế của các mô hình hiện tại (tính "Hộp đen" - Black-box, thiếu khả năng giải thích).

**1.2. Mục tiêu nghiên cứu**
* Xây dựng bộ tiền xử lý và phân tích đặc trưng sâu (Deep Features) cho khuôn mặt.
* Huấn luyện, đánh giá và so sánh hai trường phái mạng nơ-ron: CNN (ResNet-50) và Self-Attention (Vision Transformer - ViT).
* Tích hợp XAI (Grad-CAM) để giải thích phán đoán của mô hình.
* Triển khai mô hình AI thành ứng dụng Web thực tế (Microservices).

**1.3. Đối tượng và phạm vi nghiên cứu**
* Đối tượng: Ảnh khuôn mặt người thật và ảnh khuôn mặt được tạo/chỉnh sửa bởi AI (GANs).
* Phạm vi: Ứng dụng bài toán phân loại nhị phân (Binary Classification).

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
**2.1. Tổng quan về Deepfake & Image Forgery**
* Các phương pháp tạo Deepfake (Autoencoders, StyleGAN).
* Các dấu vết để lại (Artifacts): nhiễu tần số (Frequency artifacts), lỗi ghép nối mép (Blending boundaries).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `real_vs_fake_example.png` (Trích xuất từ `notebooks/EDA.ipynb` - 2 bức ảnh minh họa 1 mặt Real và 1 mặt Fake cạnh nhau).

**2.2. Mạng Nơ-ron Tích Chập (CNN) và ResNet-50**
* Cơ chế tích chập trích xuất đặc trưng cục bộ (Local Features).
* Kiến trúc Residual Block giải quyết bài toán Vanishing Gradient.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `resnet50_architecture_diagram.png` (Tải từ Internet hoặc chụp từ tài liệu gốc của ResNet).

**2.3. Vision Transformer (ViT)**
* Cơ chế Self-Attention và việc băm ảnh thành các Patches.
* Khả năng học thông tin toàn cục (Global Context) so với CNN.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `vit_patch_mechanism.png` (Sơ đồ quá trình chia ảnh thành Patch 16x16 của ViT).

**2.4. Trí tuệ Nhân tạo Có thể Giải thích (Explainable AI - XAI)**
* Hạn chế của các mô hình Hộp đen (Black-box).
* Giải thuật Grad-CAM (Gradient-weighted Class Activation Mapping).

---

## CHƯƠNG 3: PHÂN TÍCH KHÁM PHÁ (EDA) VÀ TIỀN XỬ LÝ DỮ LIỆU (Trọng tâm)

**3.1. Thu thập và mô tả dữ liệu**
* Giới thiệu bộ dữ liệu (Real vs Fake Faces StyleGAN3 / FaceForensics++).
* Mô tả siêu dữ liệu (Meta-data) và định dạng.

**3.2. Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)**
* Phân bố nhãn (Class Imbalance): Thống kê số lượng ảnh Real/Fake.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `class_distribution_bar_chart.png` (Lấy từ kết quả chạy `notebooks/EDA.ipynb`).
* Phân tích không gian màu (Color Space): Sự khác biệt trong phân bố kênh RGB/YCbCr.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `rgb_color_distribution_kde.png` (Lấy từ kết quả chạy `notebooks/EDA.ipynb`).
* Phân tích phổ tần số (1D/2D Frequency Spectrum): Phát hiện nhiễu sinh ra từ GAN.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `frequency_1d_spectrum.png` (Render từ dữ liệu của file `notebooks/frequency_1d_spectrum.csv`).

**3.3. Tiền xử lý và Làm sạch Dữ liệu (Data Cleaning & Preprocessing)**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/ml_phase1_data_preprocessing.md`
* **Face Extraction bằng MTCNN:** Sử dụng `margin=40` (chừa lề mép) để giữ lỗi mép (Blending errors).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `mtcnn_margin_40_crop.png` (1 ảnh gốc và 1 ảnh bị crop chừa lề do Team Hoàng/Linh xuất ra).
* Xử lý nhiễu và Ép độ khó (Data Augmentation): Dùng thư viện `albumentations`.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `albumentations_grid.png` (Lưới ảnh chứa: Gốc, Blur, Noise, JPEG Compression).
* Chuẩn hóa (Normalization & Resizing): 224x224, ImageNet Mean/Std.

---

## CHƯƠNG 4: THIẾT KẾ VÀ HUẤN LUYỆN MÔ HÌNH (MODEL ARCHITECTURE)

**4.1. Cấu trúc Pipeline Huấn luyện (Transfer Learning)**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/ml_phase2_model_training.md`
* Tận dụng bộ trọng số ImageNet (Pre-trained Weights).
* Fine-tuning: Thay thế Classification Head thành 2 output (Real/Fake).

**4.2. Kiến trúc Pipeline Phân tích Vùng (Regional XAI Pipeline)**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/ml_phase3_evaluation_xai.md`
* **Thuật toán Mediapipe Face Mesh:** Trích xuất 468 điểm tọa độ (Landmarks).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `mediapipe_468_landmarks_mesh.png` (Ảnh khuôn mặt có phủ lưới chấm đỏ do Team Beo xuất ra).
* Thuật toán ánh xạ (Mapping) Heatmap vào Landmarks để tính Regional Scores.

**4.3. Hàm suy hao và Tối ưu hóa (Loss & Optimizer)**
* Hàm Cross-Entropy Loss và thuật toán AdamW.

---

## CHƯƠNG 5: THỰC NGHIỆM VÀ ĐÁNH GIÁ (EVALUATION)

**5.1. Thiết lập thực nghiệm**
* Cấu hình phần cứng (GPU) và Siêu tham số (Hyperparameters).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `learning_curve_loss_accuracy.png` (Đồ thị TensorBoard hoặc Matplotlib xuất từ quá trình train của Hoàng/Beo).

**5.2. Các độ đo hiệu suất (Evaluation Metrics)**
* Confusion Matrix (TP, TN, FP, FN).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `confusion_matrix_resnet.png` và `confusion_matrix_vit.png`.
* Phân tích ROC Curve và chỉ số AUC.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `roc_curve_comparison.png` (Chứa 2 đường cong so sánh AUC của ResNet và ViT).

**5.3. Đánh giá Đánh đổi (Trade-off Analysis) giữa ResNet và ViT**
* So sánh tốc độ suy luận (FPS), độ chiếm VRAM và độ kháng nhiễu (Robustness).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `tradeoff_fps_vs_vram_bar_chart.png` (Do Linh Lé thực hiện benchmark).

**5.4. Đánh giá tính giải thích (Qualitative XAI Analysis)**
> 👉 **[Gợi ý dán ảnh]:** Tên file: `gradcam_heatmap_example.png` (Ảnh Grad-CAM đỏ rực vùng bị làm giả). 
> 👉 **[Gợi ý dán ảnh]:** Tên file: `nextjs_radar_chart_xai.png` (Chụp màn hình đồ thị Radar Chart Mắt-Miệng-Da từ Frontend `page.tsx`).

---

## CHƯƠNG 6: TRIỂN KHAI HỆ THỐNG THỰC TẾ (SYSTEM DEPLOYMENT)

**6.1. Kiến trúc Hệ thống Microservices**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/backend_architecture.md` và `outputs/architecture/frontend_architecture.md`.
* Frontend Next.js, Core Backend Java Spring Boot, AI Worker Python FastAPI.

**6.2. Cơ chế Đồng bộ và Bảo mật (Inference Lock)**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/ml_phase4_deployment.md`.
* Quản lý `inference_lock` bảo vệ VRAM tránh OOM (Out of Memory).

**6.3. Kiến trúc Cloud Azure**
> 👉 **[Gợi ý dán ảnh]:** Copy sơ đồ Mermaid từ file: `outputs/architecture/cloud_architecture.md`.
* Mô hình Public/Private Subnet (Bastion Host) nâng cao bảo mật hệ thống AI.
> 👉 **[Gợi ý dán ảnh]:** Tên file: `frontend_website_production_demo.png` (Chụp toàn màn hình Website đang hoạt động có tên miền/localhost).

---

## CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
**7.1. Kết luận**
* Tổng kết lại thành quả của đề tài.

**7.2. Hạn chế**
* Điểm mù của mô hình (Trường hợp nào False Positive/False Negative cao?).
> 👉 **[Gợi ý dán ảnh]:** Tên file: `false_positive_failure_case.png` (1 tấm ảnh thật bị chẩn đoán nhầm thành Fake kèm lời giải thích bên dưới).

**7.3. Hướng phát triển**
* Mở rộng nhận diện Video (Temporal Features).
* Xây dựng mạng đa phương thức (Audio-Visual Deepfake Detection).

---
*Tài liệu tham khảo (References)*
