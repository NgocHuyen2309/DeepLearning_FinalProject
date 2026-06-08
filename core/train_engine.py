import os
import torch
import torch.nn as nn
from tqdm import tqdm

class ViTTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-4, patience=3):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer: AdamW tối ưu cho Transformer
        # Chỉ truyền vào những parameter có requires_grad = True (lớp head)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, 
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Khởi tạo GradScaler cho AMP (Automatic Mixed Precision)
        # Giúp tiết kiệm VRAM bằng cách cast sang FP16 trong forward pass
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Tham số Early Stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.best_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'vit_deepfake_best.pth')
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        
    def unfreeze_last_blocks(self, num_blocks=2, new_lr=1e-5):
        """
        Cơ chế mở khóa (Unfreeze) vài khối Transformer cuối cùng 
        nếu val_loss bị chững lại để học sâu hơn.
        """
        print(f"\n[Trainer] Unfreezing last {num_blocks} Transformer blocks...")
        
        # Mở khóa các blocks cuối
        blocks = self.model.model.blocks
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
                
        # Cập nhật lại Optimizer với Learning Rate nhỏ hơn
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=new_lr, 
            weight_decay=1e-4
        )
        
    def train_one_epoch(self, epoch, accumulation_steps=4):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Sử dụng AMP Autocast
            with torch.amp.autocast('cuda'):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                # Normalize loss by accumulation steps
                loss = loss / accumulation_steps
                
            # Scale loss và backward
            self.scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Nhân ngược lại để hiển thị đúng loss thực tế
            running_loss += loss.item() * accumulation_steps * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
        
    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Val cũng có thể dùng Autocast để tăng tốc
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
                
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
        
    def fit(self, max_epochs=20, accumulation_steps=4):
        print(f"Bắt đầu huấn luyện ViTClassifier (Max Epochs: {max_epochs}, Patience: {self.patience})")
        
        unfrozen = False
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch, accumulation_steps=accumulation_steps)
            val_loss, val_acc = self.evaluate(epoch)
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Early Stopping Logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                # Lưu checkpoint
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"[Checkpoint] Đã lưu model tốt nhất tại: {self.best_model_path}")
            else:
                self.early_stop_counter += 1
                print(f"[Early Stop] Lần {self.early_stop_counter}/{self.patience} val_loss không giảm.")
                
                # Cân nhắc Unfreeze nếu plateau
                if self.early_stop_counter == 2 and not unfrozen:
                    self.unfreeze_last_blocks(num_blocks=2, new_lr=1e-5)
                    unfrozen = True
                    # Reset counter để cho mô hình thêm thời gian học
                    self.early_stop_counter = 0
                    
                if self.early_stop_counter >= self.patience:
                    print(f"Phát hiện Overfitting! Dừng huấn luyện sớm ở Epoch {epoch}.")
                    break
                    
                    
        print(f"Hoàn tất huấn luyện. Model tốt nhất được lưu tại {self.best_model_path}")
        return history

class ResNetTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', patience=3):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Differential Learning Rates cho ResNet-50
        # layer4 (Unfrozen) học với LR nhỏ để giữ lại filters cấp cao
        # fc học với LR lớn hơn để phân loại
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.model.layer4.parameters(), 'lr': 1e-4},
            {'params': self.model.model.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.patience = patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.best_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'resnet_deepfake_best.pth')
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        
    def train_one_epoch(self, epoch, accumulation_steps=4):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.amp.autocast('cuda'):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        return running_loss / total, 100. * correct / total
        
    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Val")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
                
        return running_loss / total, 100. * correct / total
        
    def fit(self, max_epochs=20, accumulation_steps=4):
        print(f"Bắt đầu huấn luyện ResNetClassifier (Max Epochs: {max_epochs}, Patience: {self.patience})")
        
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch, accumulation_steps=accumulation_steps)
            val_loss, val_acc = self.evaluate(epoch)
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"[Checkpoint] Đã lưu model tốt nhất tại: {self.best_model_path}")
            else:
                self.early_stop_counter += 1
                print(f"[Early Stop] Lần {self.early_stop_counter}/{self.patience} val_loss không giảm.")
                if self.early_stop_counter >= self.patience:
                    print(f"Phát hiện Overfitting! Dừng huấn luyện sớm ở Epoch {epoch}.")
                    break
                    
        print(f"Hoàn tất huấn luyện. Model tốt nhất được lưu tại {self.best_model_path}")
        return history
