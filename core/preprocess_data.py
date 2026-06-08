import os
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

def crop_and_save_dataset(data_dir, output_dir, margin=40):
    """
    Offline Preprocessing: Sử dụng MTCNN để cắt vùng khuôn mặt (có margin mở rộng)
    từ tập dữ liệu gốc, và lưu sang thư mục mới để tránh tính toán on-the-fly trong DataLoader.
    """
    # Khởi tạo MTCNN
    # Sử dụng keep_all=False để chỉ lấy khuôn mặt tự tin nhất
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=False, device=device)

    subdirs = ['Real faces', 'Fake faces']
    
    for subdir in subdirs:
        input_subdir = os.path.join(data_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Đếm số lượng file để hiển thị tqdm
        if not os.path.exists(input_subdir):
            print(f"Directory not found: {input_subdir}")
            continue
            
        filepaths = [f for f in os.listdir(input_subdir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(filepaths)} images in {subdir}...")
        
        for filename in tqdm(filepaths):
            img_path = os.path.join(input_subdir, filename)
            out_path = os.path.join(output_subdir, filename)
            
            # Đọc ảnh bằng PIL
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
                
            width, height = img.size
            
            # Phát hiện khuôn mặt
            boxes, probs = mtcnn.detect(img)
            
            if boxes is not None and len(boxes) > 0:
                # Lấy box đầu tiên (confident nhất)
                box = boxes[0]
                x_min, y_min, x_max, y_max = box
                
                # Áp dụng margin với clamping
                x_min = max(0, int(x_min - margin))
                y_min = max(0, int(y_min - margin))
                x_max = min(width, int(x_max + margin))
                y_max = min(height, int(y_max + margin))
                
                # Cắt ảnh
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                # Lưu ảnh đã cắt
                cropped_img.save(out_path)
            else:
                # Nếu không tìm thấy khuôn mặt, dùng CenterCrop (cắt 50% ở giữa) để giảm nhiễu background
                target_w, target_h = int(width * 0.6), int(height * 0.6)
                left = (width - target_w) // 2
                upper = (height - target_h) // 2
                right = left + target_w
                lower = upper + target_h
                
                cropped_img = img.crop((left, upper, right, lower))
                cropped_img.save(out_path)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'cropped_faces')
    
    print("Starting Offline MTCNN Preprocessing...")
    crop_and_save_dataset(DATA_DIR, OUTPUT_DIR, margin=40)
    print("Preprocessing completed!")
