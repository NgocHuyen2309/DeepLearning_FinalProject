import io
import os
import cv2
import numpy as np
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

# Thêm path gốc vào sys.path để import dễ dàng (hữu ích nếu chạy uvicorn từ thư mục ngoài)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_classifier import ResNetClassifier
from models.vit_classifier import ViTClassifier
from api.gradcam_utils import generate_gradcam_base64
from api.attention_utils import generate_attention_rollout_base64
from facenet_pytorch import MTCNN
from core.data_module import get_val_transforms
from api.region_utils import RegionalScorer

app = FastAPI(title="Deepfake Detection API Gateway (ResNet & ViT + XAI)")

# ================= GLOBAL STATE =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import threading
inference_lock = threading.Lock()

# 1. Khởi tạo MTCNN (Trap 2 Busted)
# Tạo global instance để tìm khuôn mặt, nạp lên chung device
mtcnn_detector = MTCNN(keep_all=False, device=device)

# 2. Khởi tạo ResNet Classifier (Idle on CPU)
resnet_model = ResNetClassifier(num_classes=2)
resnet_path = os.path.join("models", "resnet_deepfake_best.pth")

if os.path.exists(resnet_path):
    resnet_model.load_state_dict(torch.load(resnet_path, map_location='cpu', weights_only=True))
    resnet_model = resnet_model.to('cpu')
    resnet_model.eval()
    print("[API] Đã nạp thành công ResNet-50 (CPU Idle)")
else:
    print(f"[CẢNH BÁO] Không tìm thấy file {resnet_path}. API ResNet sẽ không thể predict.")

# 3. Khởi tạo ViT Classifier (Idle on CPU)
vit_model = ViTClassifier(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=2)
vit_path = os.path.join("models", "vit_deepfake_best.pth")

if os.path.exists(vit_path):
    vit_model.load_state_dict(torch.load(vit_path, map_location='cpu', weights_only=True))
    vit_model = vit_model.to('cpu')
    vit_model.eval()
    print("[API] Đã nạp thành công ViT (CPU Idle)")
else:
    print(f"[CẢNH BÁO] Không tìm thấy file {vit_path}. API ViT sẽ không thể predict.")

# 4. Khởi tạo Pipeline Tiền xử lý Ảnh (Albumentations) chuẩn ImageNet
# [LỖ HỔNG 1 BUSTED] Import trực tiếp từ T03 để không bị NameError
val_transform = get_val_transforms()

# 5. Khởi tạo MediaPipe Regional Scorer (Singleton)
regional_scorer = RegionalScorer.get_instance()

# ================= ROUTES =================

# [TRAP 1 BUSTED] Dùng 'def' thay vì 'async def' để tránh block Event Loop
@app.post("/api/v1/predict/resnet")
def predict_resnet(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Đọc dữ liệu ảnh gốc
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="File tải lên không phải là định dạng ảnh hợp lệ.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Bước 1: Dùng MTCNN để tìm và crop mặt (Kèm margin như lúc train)
        # Hàm detect trả về boxes, probs. Bọc trong lock để tránh CUDA deadlock.
        with inference_lock:
            boxes, _ = mtcnn_detector.detect(pil_img)
        
        if boxes is None or len(boxes) == 0:
            # Không tìm thấy mặt
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt người trong ảnh. Vui lòng thử ảnh khác.")
            
        # Lấy khuôn mặt đầu tiên (to nhất)
        box = boxes[0]
        x1, y1, x2, y2 = box
        
        # [LỖ HỔNG 2 BUSTED] Ép kiểu int trước khi áp dụng margin để tránh TypeError
        margin = 40
        h, w, _ = img_rgb.shape
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        
        # Đây chính là ảnh nguyên bản (raw) đã crop, KHÔNG chứa chuẩn hóa âm
        cropped_face_np = img_rgb[y1:y2, x1:x2]
        
        # Bước 2: Albumentations Tiền xử lý (đưa vào mạng nơ-ron)
        transformed = val_transform(image=cropped_face_np)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Bước 3 & 4: Đẩy qua mô hình ResNet và Grad-CAM (Bọc trong Lock để tránh Race Condition)
        with inference_lock:
            try:
                # [VRAM RESCUE] Kéo model lên GPU
                global resnet_model
                resnet_model = resnet_model.to(device)
                
                with torch.no_grad():
                    logits = resnet_model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0][1].item() # Chỉ số 1 là Fake
                    
                prediction = "Fake" if fake_prob > 0.5 else "Real"
                confidence = fake_prob if prediction == "Fake" else (1.0 - fake_prob)
                
                # Chạy Grad-CAM XAI
                heatmap_base64, heatmap_gray = generate_gradcam_base64(resnet_model, input_tensor, cropped_face_np)
                
                # Trích xuất điểm số khu vực bằng MediaPipe trên ảnh gốc
                regional_scores = regional_scorer.extract_regional_scores(img_rgb, (x1, y1, x2, y2), heatmap_gray)
                
            finally:
                # [VRAM RESCUE] Ném model về lại CPU và dọn rác bất chấp lỗi
                resnet_model = resnet_model.to('cpu')
                # Xóa tham chiếu thủ công để giải phóng VRAM ngay lập tức
                if 'input_tensor' in locals(): del input_tensor
                if 'logits' in locals(): del logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Bước 5: Trả về JSON Contract
        processing_time_ms = int((time.time() - start_time) * 1000)
        response_data = {
            "model": "ResNet-50",
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "heatmap_base64": heatmap_base64,
            "regional_scores": regional_scores,
            "processing_time_ms": processing_time_ms
        }
        
        return response_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {str(e)}")

# ================= ENDPOINT ViT =================
@app.post("/api/v1/predict/vit")
def predict_vit(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Đọc dữ liệu ảnh gốc
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="File tải lên không phải là định dạng ảnh hợp lệ.")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Bước 1: Dùng MTCNN để tìm và crop mặt
        with inference_lock:
            boxes, _ = mtcnn_detector.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt người trong ảnh. Vui lòng thử ảnh khác.")
            
        box = boxes[0]
        x1, y1, x2, y2 = box
        margin = 40
        h, w, _ = img_rgb.shape
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        
        cropped_face_np = img_rgb[y1:y2, x1:x2]
        
        # Bước 2: Albumentations Tiền xử lý
        transformed = val_transform(image=cropped_face_np)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Bước 3 & 4: Inference và Rollout với Thread Lock
        with inference_lock:
            try:
                # [VRAM RESCUE] Kéo model lên GPU
                global vit_model
                vit_model = vit_model.to(device)
                
                # Bật cờ return_attention=True
                with torch.no_grad():
                    logits, attention_weights = vit_model(input_tensor, return_attention=True)
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0][1].item()
                    
                prediction = "Fake" if fake_prob > 0.5 else "Real"
                confidence = fake_prob if prediction == "Fake" else (1.0 - fake_prob)
                
                # Chạy XAI Attention Rollout
                heatmap_base64, heatmap_gray = generate_attention_rollout_base64(attention_weights, cropped_face_np)
                
                # Trích xuất điểm số khu vực bằng MediaPipe trên ảnh gốc
                regional_scores = regional_scorer.extract_regional_scores(img_rgb, (x1, y1, x2, y2), heatmap_gray)
                
            finally:
                # Xóa mảng attention (ngừa memory leak)
                vit_model.clear_attention()
                # [VRAM RESCUE] Đẩy model về CPU, dọn rác
                vit_model = vit_model.to('cpu')
                # Xóa tham chiếu thủ công để giải phóng VRAM ngay lập tức
                if 'input_tensor' in locals(): del input_tensor
                if 'logits' in locals(): del logits
                if 'attention_weights' in locals(): del attention_weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        # Bước 5: Trả về JSON Contract
        processing_time_ms = int((time.time() - start_time) * 1000)
        response_data = {
            "model": "ViT-Base",
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "heatmap_base64": heatmap_base64,
            "regional_scores": regional_scores,
            "processing_time_ms": processing_time_ms
        }
        
        return response_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {str(e)}")
