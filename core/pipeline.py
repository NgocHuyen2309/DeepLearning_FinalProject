import os
import torch
from core.data_module import DeepfakeDataModule
from models.vit_classifier import ViTClassifier
from models.resnet_classifier import ResNetClassifier
from core.train_engine import ViTTrainer, ResNetTrainer
from core.evaluator import ModelEvaluator

class TrainingPipeline:
    """
    Quy trình huấn luyện chuẩn OOP (Data -> Model -> Train -> Evaluate)
    """
    def __init__(self, model_type="vit", data_dir="data/cropped_faces", batch_size=8):
        self.model_type = model_type.lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"--- KHỞI TẠO PIPELINE: {self.model_type.upper()} ---")
        
        # 1. Khởi tạo DataModule
        self.dm = DeepfakeDataModule(data_dir=data_dir, batch_size=batch_size)
        self.dm.setup()
        self.train_loader = self.dm.get_train_dataloader()
        self.val_loader = self.dm.get_val_dataloader()
        self.test_loader = self.dm.get_test_dataloader()
        
        # 2. Khởi tạo Mô hình và Trainer tương ứng
        if self.model_type == "vit":
            self.model = ViTClassifier(model_name='vit_tiny_patch16_224', pretrained=True)
            self.trainer = ViTTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                patience=3
            )
            self.best_model_path = os.path.join('models', 'vit_deepfake_best.pth')
            
        elif self.model_type == "resnet":
            self.model = ResNetClassifier(num_classes=2)
            self.trainer = ResNetTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                patience=3
            )
            self.best_model_path = os.path.join('models', 'resnet_deepfake_best.pth')
        else:
            raise ValueError(f"Loại mô hình '{model_type}' không được hỗ trợ. Dùng 'vit' hoặc 'resnet'.")
            
        # 3. Khởi tạo Evaluator
        self.evaluator = ModelEvaluator(save_dir=f"outputs/{self.model_type}")

    def run(self, max_epochs=20, accumulation_steps=4):
        """
        Chạy huấn luyện và đánh giá.
        """
        if not os.path.exists(self.best_model_path):
            # Nếu chưa có trọng số thì bắt đầu train
            history = self.trainer.fit(max_epochs=max_epochs, accumulation_steps=accumulation_steps)
            # Vẽ biểu đồ lịch sử
            self.evaluator.plot_history(history, prefix=self.model_type)
        else:
            print(f"[*] Đã tìm thấy file trọng số {self.best_model_path}. Bỏ qua huấn luyện...")
            
        # Đánh giá trên tập Test
        print(f"[*] Tiến hành đánh giá mô hình {self.model_type.upper()}...")
        
        # Tùy thuộc vào thiết kế hàm load_state_dict, đối với ViT nó nằm ở self.model.model, nhưng trong 
        # file `run_train.py` ta thấy load thẳng bằng model.load_state_dict... 
        # Trainer.save lưu state_dict của self.model. Toàn bộ là an toàn.
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        self.model = self.model.to(self.device)
        
        self.evaluator.evaluate_on_test(
            model=self.model, 
            test_loader=self.test_loader, 
            device=self.device, 
            prefix=self.model_type
        )
        print(f"--- HOÀN TẤT PIPELINE {self.model_type.upper()} ---")
