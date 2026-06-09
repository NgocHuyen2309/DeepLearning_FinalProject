import os
import torch
from core.data_module import DeepfakeDataModule
from models.vit_classifier import ViTClassifier
from models.resnet_classifier import ResNetClassifier
from core.evaluator import ModelComparator

def main():
    print("=== DEEPFAKE DETECTION: MODEL COMPARISON ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Khởi tạo DataModule để lấy Test Loader
    dm = DeepfakeDataModule(data_dir="data/cropped_faces", batch_size=16)
    dm.setup()
    test_loader = dm.get_test_dataloader()
    
    # 2. Khởi tạo và nạp trọng số ViT
    vit_path = os.path.join('models', 'vit_deepfake_best.pth')
    if not os.path.exists(vit_path):
        print(f"[LỖI] Không tìm thấy {vit_path}. Vui lòng train ViT trước.")
        return
        
    vit_model = ViTClassifier(model_name='vit_tiny_patch16_224', pretrained=False)
    vit_model.load_state_dict(torch.load(vit_path, map_location=device, weights_only=True))
    vit_model = vit_model.to(device)
    
    # 3. Khởi tạo và nạp trọng số ResNet-50
    resnet_path = os.path.join('models', 'resnet_deepfake_best.pth')
    if not os.path.exists(resnet_path):
        print(f"[LỖI] Không tìm thấy {resnet_path}. Vui lòng train ResNet-50 trước.")
        return
        
    resnet_model = ResNetClassifier(num_classes=2)
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device, weights_only=True))
    resnet_model = resnet_model.to(device)
    
    # 4. Kích hoạt ModelComparator
    comparator = ModelComparator(
        vit_model=vit_model,
        resnet_model=resnet_model,
        test_loader=test_loader,
        device=device,
        save_dir="outputs/comparison"
    )
    
    # 5. So sánh
    comparator.run_comparison()

if __name__ == '__main__':
    main()
