# 🎭 Deepfake Detection & Face Anti-Spoofing Platform (Microservices Architecture)

[![Java Spring Boot](https://img.shields.io/badge/Spring_Boot-3.0+-6DB33F.svg)](https://spring.io/projects/spring-boot)
[![Python FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Đồ án cuối kỳ môn Deep Learning. Dự án xây dựng một hệ thống AI End-to-End có khả năng phân biệt ảnh chụp người thật và ảnh bị thao túng (Deepfake/GAN-generated). Hệ thống so sánh trực tiếp hiệu năng giữa 2 kiến trúc kinh điển: **CNN (ResNet-50)** và **Vision Transformer (ViT)**, đồng thời ứng dụng **Explainable AI (XAI)** kết hợp Mediapipe Face Mesh để trực quan hóa phán đoán của mô hình theo từng vùng khuôn mặt.

Hệ thống được thiết kế theo chuẩn **Microservices (Vi dịch vụ)**, tách biệt hoàn toàn phần xử lý nghiệp vụ (Java Spring Boot) và phần suy luận AI (Python FastAPI Worker), cùng với giao diện Frontend hiện đại xây dựng trên nền tảng Next.js.

---

## ✨ Tính năng Nổi bật (Core Features)

* 🏢 **Microservices Architecture:** 
  * **Core Backend (Java Spring Boot):** Hoạt động như API Gateway, nhận yêu cầu từ Frontend, quản lý File IO và Database thông qua JPA, có tính năng dọn dẹp rác (Garbage Collector) tự động bằng cron job.
  * **AI Worker (Python FastAPI):** Vi dịch vụ xử lý ảnh bất đồng bộ, kết hợp cơ chế `inference_lock` bảo vệ VRAM (GPU/CPU) khi chạy song song nhiều luồng request.
* 🖥️ **Modern Frontend (Next.js):** Giao diện UI/UX tương tác trực quan với các biểu đồ Radar Chart và Progress Bar thông qua Recharts và Zustand State Management.
* 🧠 **Dual-Model Inference:** Đánh giá đồng thời khuôn mặt qua ResNet (bắt lỗi pixel cục bộ) và ViT (bắt lỗi không gian toàn cục).
* 🔍 **Explainable AI (XAI) & Landmarks Analytics:** Cung cấp Heatmap (Grad-CAM) kết hợp 468 điểm tọa độ khuôn mặt của Mediapipe để bóc tách độ uy tín (Confidence) theo từng vùng: Mắt, Miệng, Vùng da.
* 🛡️ **Robust Data Pipeline:** Sử dụng `MTCNN` trích xuất khuôn mặt, kết hợp xử lý song song và catch Exception chi tiết.

---

## 👥 Đội ngũ Phát triển (Team Roles & Alignment)

* **Huỳnh Minh Trí (Cloud Architect & Core Backend):** Thiết lập hạ tầng Azure (Hub-Spoke, Bastion, Blob Storage). Code API Gateway và nghiệp vụ bằng **Java Spring Boot**, quản lý Database qua JPA. Đảm bảo giao tiếp REST mượt mà với AI Worker.
* **Phạm Duy Hoàng (Data Pipeline & ViT Specialist):** Xử lý tiền kỳ ảnh (MTCNN). Xây dựng, huấn luyện mô hình ViT và logic trích xuất Attention Map. Tích hợp pipeline này thành API Endpoint trong **FastAPI**.
* **Hứa Thị Ngọc Huyền (CNN Specialist & Frontend Dev):** Cấu hình và fine-tune ResNet-50. Viết thuật toán Grad-CAM và kết hợp Mediapipe XAI. Tích hợp endpoint vào **FastAPI**. Xây dựng toàn bộ giao diện UI bằng **Next.js** gọi lên server Java.
* **Võ Lê Khánh Linh (Data Analyst & Evaluation Lead):** Phân tích dữ liệu gốc (EDA). Benchmark so sánh Trade-off giữa ViT và ResNet. Vẽ sơ đồ kiến trúc Microservices, Dataflow và tổng hợp báo cáo/slide cuối kỳ.

---

## 🏗️ Kiến Trúc Hệ Thống & ML Pipeline (Architecture & Dataflow)

*Chi tiết biểu đồ xem trong file tại `outputs/architecture/`*

### 1. Kiến Trúc Azure Cloud (Hub-Spoke & Bastion)
```mermaid
graph TD
    User((User / Browser))
    subgraph "Azure Cloud Infrastructure"
        subgraph "Resource Group: Deepfake-RG"
            subgraph "Azure Blob Storage"
                Blob["deepfakefrontend <br> (Static Website)"]
            end
            subgraph "VNet: Deepfake-VNet (10.0.0.0/16)"
                subgraph "Public Subnet (10.0.1.0/24)"
                    ProxyVM["Proxy-VM <br> Bastion / Nginx Reverse Proxy"]
                    PublicIP["Public IP Address"]
                    PublicIP --> ProxyVM
                end
                subgraph "Private Subnet (10.0.2.0/24)"
                    BackendVM["Backend-VM <br> No Public IP"]
                    subgraph "Backend Services"
                        Java["Java Core Backend <br> Port 8080"]
                        Python["Python AI Worker <br> Port 8001"]
                        Postgres[("PostgreSQL")]
                    end
                    BackendVM --> Java
                    BackendVM --> Python
                    BackendVM --> Postgres
                end
            end
        end
    end
    User -- "Access UI (HTTPS)" --> Blob
    Blob -- "API Calls" --> PublicIP
    ProxyVM -- "Reverse Proxy" --> Java
    Java -- "Internal Call" --> Python
    Java -- "JPA" --> Postgres
```

### 2. Machine Learning Pipeline (Deep Learning)

**Phase 1: Data Preprocessing & EDA**
```mermaid
graph TD
    A[("Raw Datasets <br> (FaceForensics++, Celeb-DF)")] --> B("Exploratory Data Analysis (EDA)")
    subgraph Preprocessing ["Tiền xử lý & Trích xuất"]
        B --> C{"MTCNN Face Detection"}
        C -- "Phát hiện mặt" --> D["Crop khuôn mặt"]
        D --> F["Resize (224x224) & Augmentation"]
        F --> H["Chuẩn hóa (Normalization)"]
    end
    Preprocessing --> Output[("Processed Dataset")]
```

**Phase 2: Model Training & Evaluation**
```mermaid
graph TD
    Input[("Processed Dataset")] --> Dataloader["PyTorch DataLoader"]
    subgraph Architectures ["Kiến trúc Mô hình"]
        Dataloader --> ResNet["ResNet-50 (CNN)"]
        Dataloader --> ViT["Vision Transformer (ViT)"]
        ResNet --> FineTune1["Fine-tune (Output: 2 classes)"]
        ViT --> FineTune2["Fine-tune (Output: 2 classes)"]
    end
    subgraph TrainingLoop ["Vòng lặp Huấn luyện"]
        FineTune1 --> Forward["Forward Pass"]
        FineTune2 --> Forward
        Forward --> Loss["Cross Entropy Loss"]
        Loss --> Backward["Backward Pass"]
        Backward --> Optimizer["Adam Optimizer"]
    end
    TrainingLoop --> Checkpoint[("Best Checkpoint (.pth)")]
```

**Phase 3: Explainable AI (XAI)**
```mermaid
graph TD
    Checkpoint[("Model Checkpoint")] --> Inference["Dự đoán trên Test Set"]
    subgraph XAI ["Giải thích Mô hình (XAI)"]
        Inference --> GradCAM["Grad-CAM"]
        GradCAM --> Heatmap["Tạo Heatmap (Vùng tập trung)"]
        Heatmap --> Mediapipe["Mediapipe Face Mesh"]
        Mediapipe --> Landmarks["468 điểm Landmarks"]
        Landmarks --> Mapping["Mapping Heatmap vào từng vùng"]
        Mapping --> RegionalScores["Chấm điểm giả mạo từng khu vực"]
    end
```

---

## 📂 Cấu trúc Dự án (Folder Structure)

```text
DeepLearning_Final/
├── core-backend/    # Mã nguồn Java Spring Boot (Controllers, Services, JPA, Exception Handler)
├── api/             # Mã nguồn Python FastAPI (Nội bộ suy luận AI, XAI, Region Analysis)
├── frontend/        # Mã nguồn giao diện người dùng Next.js (App Router, Tailwind, Zustand)
├── models/          # Load Weights (.pth) cho ResNet và ViT
├── notebooks/       # File Jupyter Notebook phục vụ EDA và test đánh giá Trade-off
├── outputs/         # Thư mục chứa các file đồ thị kiến trúc hệ thống và ML Pipeline
├── data/            # Thư mục lưu trữ Dataset thô (Ignored by Git)
├── .env             # File biến môi trường (Ignored by Git)
├── requirements.txt # Danh sách các thư viện Python
└── README.md
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy (Setup Instructions)

### 1. AI Worker (Python FastAPI)
Tạo môi trường ảo và cài đặt thư viện:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
* Chạy AI Worker: `python run_api.py` (Mở tại port 8001)

### 2. Core Backend (Java Spring Boot)
Yêu cầu: JDK 17+ và Maven. Đảm bảo cấu hình SQL trong `application.yml`.
```bash
cd core-backend
mvn spring-boot:run
```

### 3. Frontend (Next.js)
Yêu cầu: Node.js 18+.
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Bộ dữ liệu (Dataset)
Dự án sử dụng bộ dữ liệu **Real vs Fake Faces (StyleGAN3)**:
🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3/data)
Vui lòng tải dữ liệu về và đặt trong thư mục `data/`.
