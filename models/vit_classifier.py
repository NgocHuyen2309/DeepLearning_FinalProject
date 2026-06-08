import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) model for Deepfake Detection.
    - Sử dụng pre-trained weights từ timm.
    - Đóng băng toàn bộ backbone, chỉ fine-tune classifier head.
    - Cung cấp cơ chế thu thập ma trận Attention từ tất cả các khối (blocks) phục vụ XAI.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=2):
        super(ViTClassifier, self).__init__()
        
        # Load pre-trained ViT
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # 1. Freeze toàn bộ tham số của mô hình (Backbone)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 2. Mở khóa (Unfreeze) riêng lớp head cuối cùng (Classifier)
        for param in self.model.head.parameters():
            param.requires_grad = True
            
        # 3. Khởi tạo mảng lưu trữ Attention Weights và các hooks
        self.attention_weights = []
        self.hooks = []
        self.capture_attention = False
        
        # 4. Gắn Hook vào TẤT CẢ các blocks để chuẩn bị cho Attention Rollout
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """
        Duyệt qua 12 khối Transformer Blocks và gắn hook vào lớp attention.
        """
        def hook_fn(module, input, output):
            # Chỉ lưu khi cờ capture_attention = True và detach khỏi graph để tránh Memory Leak
            # Đẩy ma trận sang CPU ngay lập tức để tiết kiệm VRAM cho quá trình XAI
            if self.capture_attention:
                self.attention_weights.append(output.detach().cpu())
            
        # Đảm bảo mô hình có thuộc tính blocks (chuẩn của timm ViT)
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                # Tắt tính năng Fused Attention (SDPA) của PyTorch 2.0+ 
                # Nếu không, PyTorch sẽ bypass lớp attn_drop và hook không hoạt động!
                if hasattr(block, 'attn'):
                    setattr(block.attn, 'fused_attn', False)
                    
                # attn.attn_drop là Dropout layer ngay sau Attention Softmax
                hook = block.attn.attn_drop.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                
    def clear_attention(self):
        """
        Xóa mảng attention weights để tránh tràn RAM (Memory Leak).
        Cần gọi hàm này trước mỗi lần Forward Pass mới nếu muốn lấy Attention.
        """
        self.attention_weights = []
        
    def forward(self, x, return_attention=False):
        # Bẫy an toàn (Senior Polishing): Ngăn chặn XAI bị nhiễu bởi Dropout
        if return_attention and self.training:
            raise RuntimeError("CẢNH BÁO: Không được xuất Attention Map khi đang ở chế độ train()! Hãy gọi model.eval() trước.")
            
        self.capture_attention = return_attention
        # Nếu yêu cầu trả về attention, bắt buộc phải dọn dẹp mảng cũ
        if return_attention:
            self.clear_attention()
            
        logits = self.model(x)
        
        if return_attention:
            # Trả về đồng thời Logits dự đoán và mảng 12 ma trận Attention
            return logits, self.attention_weights
            
        return logits
