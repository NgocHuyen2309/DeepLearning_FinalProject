import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class RegionalScorer:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        # Đường dẫn tới file model landmarker đã tải trong root
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'face_landmarker.task')
        if not os.path.exists(model_path):
            print(f"[CẢNH BÁO] Không tìm thấy {model_path}. Regional Scorer sẽ trả về 0.")
            self.detector = None
            return
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Chỉ mục các điểm trên Face Mesh
        # Mắt trái và phải
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Miệng (viền ngoài)
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        # Đường viền khuôn mặt (Face Oval)
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def extract_regional_scores(self, image_full_np, bbox, heatmap_gray):
        """
        Trích xuất điểm số khu vực dựa trên ảnh gốc (full) và bản đồ XAI (grayscale [0..1]).
        bbox = (x1, y1, x2, y2) của khuôn mặt trên ảnh gốc.
        Returns dict: {eyes, mouth, skin_texture, lighting, background} (scale 0-100)
        """
        default_scores = {"eyes": 0, "mouth": 0, "skin_texture": 0, "lighting": 0, "background": 0}
        
        if self.detector is None:
            return default_scores
            
        h_full, w_full = image_full_np.shape[:2]
        x1, y1, x2, y2 = bbox
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        # Chuyển đổi numpy array sang MediaPipe Image (dùng ascontiguousarray để tránh lỗi)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_full_np))
        
        # Nhận diện Landmarks trên ảnh gốc
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            print("[XAI] MediaPipe không tìm thấy khuôn mặt trên ảnh gốc.")
            return default_scores
            
        landmarks = detection_result.face_landmarks[0]
        
        # Hàm vẽ Polygon từ các chỉ mục Landmark, chiếu tọa độ từ ảnh gốc sang ảnh đã crop
        def create_mask(indices):
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            points = []
            for idx in indices:
                landmark = landmarks[idx]
                # Tọa độ trên ảnh gốc
                x_full = int(landmark.x * w_full)
                y_full = int(landmark.y * h_full)
                # Dịch chuyển về không gian của ảnh crop
                x_crop = x_full - x1
                y_crop = y_full - y1
                points.append([x_crop, y_crop])
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            return mask
            
        # Tạo mask cho từng vùng
        eyes_mask = np.logical_or(create_mask(self.LEFT_EYE), create_mask(self.RIGHT_EYE)).astype(np.uint8)
        mouth_mask = create_mask(self.LIPS)
        face_mask = create_mask(self.FACE_OVAL)
        
        # Vùng da (Skin) = Toàn mặt - (Mắt + Miệng)
        skin_mask = face_mask.copy()
        skin_mask[eyes_mask == 1] = 0
        skin_mask[mouth_mask == 1] = 0
        
        # Vùng nền (Background) = Đảo ngược vùng mặt
        bg_mask = 1 - face_mask
        
        # Tính điểm số trung bình (Mean) cho từng vùng từ heatmap_gray
        # Heatmap giá trị [0, 1] -> chuyển sang % (0-100)
        def get_score(mask):
            if np.sum(mask) == 0: return 0
            # Tính trung bình trên các pixel thuộc mask
            avg_intensity = np.mean(heatmap_gray[mask == 1])
            return int(avg_intensity * 100)
            
        eyes_score = get_score(eyes_mask)
        mouth_score = get_score(mouth_mask)
        skin_score = get_score(skin_mask)
        bg_score = get_score(bg_mask)
        
        # Tính "Lighting" (sự phân bố ánh sáng/độ tương phản do fake artifacts)
        # Bằng độ lệch chuẩn (Std) của Heatmap trên toàn khuôn mặt (được phóng đại)
        if np.sum(face_mask) > 0:
            std_intensity = np.std(heatmap_gray[face_mask == 1])
            lighting_score = int(min(100, std_intensity * 200)) # Scale lên cho dễ nhìn
        else:
            lighting_score = 0
            
        return {
            "eyes": eyes_score,
            "mouth": mouth_score,
            "skin_texture": skin_score,
            "lighting": lighting_score,
            "background": bg_score
        }
