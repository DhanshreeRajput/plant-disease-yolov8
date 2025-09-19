# Train YOLOv8 model with plant disease data 
from ultralytics import YOLO
import os
from pathlib import Path
import torch

def check_setup():
    """Check if everything is ready for training"""
    print("🔍 Checking setup...")
    
    # Check dataset
    dataset_path = Path("plant_disease_yolo_dataset/dataset.yaml")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please run data_preprocessing.py first")
        return False
    
    # Check CUDA/GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 0
    else:
        print("💻 No GPU detected, using CPU (training will be slower)")
        device = 'cpu'
    
    return True, device

def train_plant_disease_model():
    """
    Train YOLOv8 model for plant disease detection
    """
    print("🌱 YOLOv8 Plant Disease Detection Training")
    print("=" * 50)
    
    # Check setup
    setup_ok, device = check_setup()
    if not setup_ok:
        return None, None
    
    # Load YOLOv8 model
    print("📥 Loading YOLOv8 model...")
    model = YOLO('yolov8s.pt')  # Downloads automatically if not present
    
    print("🎯 Starting training...")
    print("⏰ This will take 1-3 hours depending on your hardware...")
    
    # Train the model
    results = model.train(
        data='plant_disease_yolo_dataset/dataset.yaml',
        epochs=50,        # Reduced for initial testing
        imgsz=640,        # Image size
        batch=8,          # Small batch size for compatibility
        device=device,    # Auto-detect GPU/CPU
        project='plant_disease_runs',
        name='yolov8_plant_disease',
        save_period=10,   # Save checkpoint every 10 epochs
        patience=15,      # Early stopping patience
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.9,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("\n🎉 Training completed!")
    print(f"📂 Results saved in: {results.save_dir}")
    print(f"🏆 Best model: {results.save_dir}/weights/best.pt")
    print("📊 Training plots saved in the results directory")
    
    return model, results

if __name__ == "__main__":
    model, results = train_plant_disease_model()
    if model:
        print("\n✅ Training finished successfully!")
        print("🧪 Next step: Run 'python test_model.py' to evaluate your model")