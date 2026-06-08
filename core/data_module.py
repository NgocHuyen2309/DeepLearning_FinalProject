import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Augmentation transformations for Training.
    - Add specific augmentations for Deepfake Detection (StyleGAN artifacts).
    """
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), # Tăng cường dữ liệu bằng lật ngang
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.ImageCompression(quality_range=(50, 80), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Transformations for Validation/Test.
    - Only Resize and Normalize.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for Deepfake cropped faces.
    """
    def __init__(self, filepaths, labels, transforms=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transforms = transforms
        
        # Encoding string to int
        self.label_map = {'real': 0, 'fake': 1}
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label_str = self.labels[idx]
        label = self.label_map.get(label_str.lower(), 0)
        
        # Đọc ảnh bằng OpenCV -> Chuyển hệ màu sang RGB vì cv2 mặc định là BGR
        image = cv2.imread(img_path)
        if image is None:
            # Phòng thủ: Báo lỗi để biết file nào hỏng thay vì trả về ảnh rác làm sập hệ thống
            raise ValueError(f"Không thể đọc file ảnh: {img_path}")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Áp dụng Augmentations on-the-fly
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)


class DeepfakeDataModule:
    """
    Standard PyTorch OOP class to manage data splits and DataLoader creation.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = data_dir # Data dir expected to be "data/cropped_faces"
        self.batch_size = batch_size
        
        # CẢNH BÁO CHÍ MẠNG (Đa luồng trên Windows):
        # Khi sử dụng num_workers > 0, tất cả code khởi tạo và chạy DataLoader 
        # BẮT BUỘC phải được đặt trong block: if __name__ == '__main__':
        # Nếu không sẽ văng lỗi RuntimeError: freeze_support().
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        filepaths = []
        labels = []
        
        # Load real and fake cropped paths
        real_dir = os.path.join(self.data_dir, 'Real faces')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(real_dir, f))
                    labels.append('real')
                    
        fake_dir = os.path.join(self.data_dir, 'Fake faces')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(fake_dir, f))
                    labels.append('fake')
                    
        total_samples = len(filepaths)
        if total_samples == 0:
            raise ValueError(f"Khong tim thay du lieu trong: {self.data_dir}")
            
        # Stratified Split: Train 80%, Val 10%, Test 10%
        # Chia 90% (train+val) và 10% (test)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            filepaths, labels, test_size=0.1, stratify=labels, random_state=42
        )
        
        # Tính toán test_size nội bộ để Val chiếm đúng 10% tổng số
        # Val = 10% / 90% của tập train_val ~ 0.1111
        val_ratio = 0.1 / 0.9 
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
        )
        
        # Khởi tạo Datasets
        self.train_dataset = DeepfakeDataset(X_train, y_train, get_train_transforms())
        self.val_dataset = DeepfakeDataset(X_val, y_val, get_val_transforms())
        self.test_dataset = DeepfakeDataset(X_test, y_test, get_val_transforms())
        
        print(f"[DataModule Setup] Hoan tat chia tap - Tong: {total_samples}")
        print(f" Train: {len(X_train)} anh")
        print(f" Val:   {len(X_val)} anh")
        print(f" Test:  {len(X_test)} anh")
        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
                          
    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
                          
    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

# Có thể viết module nhỏ để Un-normalize dùng trong Notebook / Debug
def unnormalize(tensor_img):
    """
    Hàm đảo ngược Normalize của ImageNet để vẽ ảnh chính xác (không bị đen thui/lỗi màu).
    """
    # RGB ImageNet mean, std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Tensor dạng [C, H, W] -> chuyển về dạng NumPy [H, W, C]
    img = tensor_img.numpy().transpose((1, 2, 0))
    img = (img * std) + mean
    img = np.clip(img, 0, 1) # Bắt buộc phải có dòng này
    return img
