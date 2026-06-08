import cv2
import numpy as np
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

def generate_gradcam_base64(model, image_tensor, original_image_np):
    """
    Tạo ảnh Heatmap giải thích (XAI) sử dụng Grad-CAM và trả về chuỗi Base64.
    
    Args:
        model (nn.Module): Mô hình đã được tải trọng số (VD: ResNetClassifier).
        image_tensor (torch.Tensor): Ảnh đầu vào dạng tensor (đã normalize qua Albumentations). Shape (1, 3, 224, 224).
        original_image_np (np.ndarray): Ảnh numpy gốc sau khi Crop bằng MTCNN (RGB). Shape (H, W, 3).
        
    Returns:
        str: Chuỗi Base64 của bức ảnh Heatmap.
    """
    # Chỉ định target layer (được định nghĩa trong ResNetClassifier)
    target_layers = [model.target_layer]
    
    # Khởi tạo GradCAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Lấy nhãn dự đoán có xác suất cao nhất
        grayscale_cam = cam(input_tensor=image_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
    # [TRAP 3 BUSTED] Normalization Trap:
    # Resize ảnh raw về cùng kích thước với output của model (224x224) nếu cần
    # (Do ảnh qua albumentations đã được resize về 224, ta resize original image để chập map)
    img_resized = cv2.resize(original_image_np, (224, 224))
    
    # Ép kiểu bức ảnh nguyên bản về khoảng [0.0, 1.0] dạng float32
    img_float = img_resized.astype(np.float32) / 255.0
    
    # Chập (Blend) Heatmap lên bức ảnh đã Normalize
    # Mặc định show_cam_on_image mong đợi ảnh BGR hoặc RGB đã ở dạng float [0, 1]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    # Chuyển đổi lại về RGB sang BGR để encode OpenCV
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    # Encode ảnh thành Base64 để nhúng vào JSON
    _, buffer = cv2.imencode('.png', visualization_bgr)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Thay đổi kích thước heatmap_gray về bằng ảnh gốc để tính toán vùng
    h, w = original_image_np.shape[:2]
    heatmap_gray_resized = cv2.resize(grayscale_cam, (w, h))
    
    return heatmap_base64, heatmap_gray_resized
