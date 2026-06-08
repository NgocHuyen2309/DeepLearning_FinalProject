import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetClassifier(nn.Module):
    """
    CNN model using ResNet-50 for Deepfake Detection.
    - Sử dụng pre-trained weights IMAGENET1K_V2.
    - Đóng băng từ layer1 đến layer3.
    - Mở khóa (Unfreeze) layer4 và fc để fine-tune mạnh và hỗ trợ Grad-CAM.
    """
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        
        # 1. Load pre-trained ResNet-50
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 2. Đóng băng các layer đầu (layer1, layer2, layer3 và các layer nông hơn)
        layers_to_freeze = [
            self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
            self.model.layer1, self.model.layer2, self.model.layer3
        ]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
                
        # 3. Đảm bảo layer4 được mở khóa (cần thiết cho Grad-CAM và Fine-tuning)
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        # 4. Thay thế lớp Fully Connected (fc) cho bài toán nhị phân
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    @property
    def target_layer(self):
        """
        Trả về lớp Tích chập (Convolutional Layer) cuối cùng để phục vụ cho thuật toán Grad-CAM.
        Với ResNet-50, đó chính là block Bottleneck cuối cùng của layer4.
        """
        return self.model.layer4[-1]

    def forward(self, x):
        # Trả về logits (không qua softmax vì dùng CrossEntropyLoss)
        return self.model(x)
