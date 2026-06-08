import cv2
import numpy as np
import torch
import base64

def generate_attention_rollout_base64(attention_weights, original_image_np):
    """
    Tạo ảnh Heatmap XAI từ mảng 12 ma trận Attention của ViT bằng thuật toán Rollout.
    
    Args:
        attention_weights (list of torch.Tensor): Danh sách 12 tensor. 
            Mỗi tensor có shape (B, Heads, N, N) với N = 197 (1 CLS + 196 patches).
        original_image_np (np.ndarray): Ảnh gốc (RGB) kích thước tùy ý đã được crop mặt.
        
    Returns:
        str: Chuỗi Base64 của bức ảnh Heatmap.
    """
    if not attention_weights or len(attention_weights) == 0:
        return ""
        
    # Lấy tensor đầu tiên để xác định shape
    # B = Batch, H = Heads, N = Tokens
    first_layer = attention_weights[0]
    B, H, N, _ = first_layer.shape
    
    # Khởi tạo ma trận kết quả Rollout bằng ma trận đơn vị I
    # Shape: (N, N)
    rollout = torch.eye(N)
    
    for layer_attention in attention_weights:
        # 1. Bóc tách batch (chỉ lấy ảnh đầu tiên trong batch)
        # layer_attention: (B, H, N, N) -> lấy [0] -> (H, N, N)
        layer_attention = layer_attention[0]
        
        # 2. Trung bình hóa các Heads (Head Averaging)
        # a_mean: (N, N)
        a_mean = layer_attention.mean(dim=0)
        
        # 3. Cộng kết nối thặng dư (Residual Connection)
        a_mean = a_mean + torch.eye(N)
        
        # 4. [BẪY CHUẨN HÓA] Bắt buộc chuẩn hóa lại hàng để tổng xác suất bằng 1.0
        # Nếu không chuẩn hóa, giá trị sẽ bùng nổ sau 12 lần nhân.
        a_mean = a_mean / a_mean.sum(dim=-1, keepdim=True)
        
        # 5. Nhân chập ma trận (Matrix Multiplication Order)
        # R = A_i x R
        rollout = torch.matmul(a_mean, rollout)
        
    # 6. Trích xuất thông tin của CLS Token (Hàng 0) đối với các token không gian (từ cột 1 đến hết)
    # rollout[0, 1:] sẽ có shape là (196,)
    cls_attention = rollout[0, 1:]
    
    # 7. Reshape mảng 196 phần tử thành lưới không gian 14x14
    spatial_attention = cls_attention.reshape(14, 14).numpy()
    
    # 8. Chuẩn hóa giá trị spatial_attention về dải [0, 1] để vẽ Heatmap
    spatial_attention = spatial_attention - np.min(spatial_attention)
    spatial_attention = spatial_attention / (np.max(spatial_attention) + 1e-8)
    
    # 9. Lấy kích thước thực tế của khuôn mặt cắt ra để tránh méo bản đồ XAI
    h, w = original_image_np.shape[:2]
    
    # Phóng to lưới 14x14 lên kích thước gốc để chập ảnh (w, h)
    attention_resized = cv2.resize(spatial_attention, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Áp dụng COLORMAP_JET của OpenCV
    rollout_mask_uint8 = np.uint8(255 * attention_resized)
    heatmap = cv2.applyColorMap(rollout_mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Xử lý Normalization Trap cho ảnh gốc (Không resize ép thành hình vuông nữa)
    img_float = original_image_np.astype(np.float32) / 255.0
    
    heatmap_float = heatmap.astype(np.float32) / 255.0
    
    # Chập màu (Blend) Heatmap lên ảnh gốc
    # Tỉ lệ blend: 0.6 phần ảnh gốc, 0.4 phần heatmap
    visualization = 0.6 * img_float + 0.4 * heatmap_float
    # Đưa về dải [0, 255] dạng uint8
    visualization = np.clip(visualization * 255.0, 0, 255).astype(np.uint8)
    
    # Chuyển RGB sang BGR để OpenCV encode
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    # Encode Base64
    _, buffer = cv2.imencode('.png', visualization_bgr)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return heatmap_base64, attention_resized
