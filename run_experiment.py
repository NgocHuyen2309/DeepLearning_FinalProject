import argparse
from core.pipeline import TrainingPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Pipeline")
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet'], 
                        help="Loại mô hình muốn huấn luyện (vit hoặc resnet)")
    parser.add_argument('--batch_size', type=int, default=8, help="Kích thước batch size")
    parser.add_argument('--epochs', type=int, default=20, help="Số epochs tối đa")
    
    args = parser.parse_args()
    
    # Kích hoạt chuẩn OOP
    pipeline = TrainingPipeline(model_type=args.model, batch_size=args.batch_size)
    pipeline.run(max_epochs=args.epochs, accumulation_steps=4)
