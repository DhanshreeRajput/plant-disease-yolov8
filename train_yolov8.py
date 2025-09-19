# Train YOLOv8 model with plant disease data 
from ultralytics import YOLO
import os
from pathlib import Path  # Fixed: was "Paths" instead of "Path"
import torch

def check_setup():
    """Check if everything is ready for training and FORCE GPU usage"""
    print("🔍 Checking setup...")
    
    # Check dataset
    dataset_path = Path("plant_disease_yolo_dataset/dataset.yaml")
    if not dataset_path.exists():
        print("❌ Dataset not found! Please run data_preprocessing.py first")
        return False, None
    
    # Check CUDA/GPU - FORCE GPU USAGE
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU/CUDA detected!")
        print("💡 Solutions:")
        print("   1. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   2. Check if your GPU drivers are installed")
        print("   3. Restart your terminal/IDE after installing CUDA")
        return False, None
    
    # GPU is available - get info and force usage
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 GPU FOUND: {gpu_name}")
    print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
    print(f"📊 CUDA version: {torch.version.cuda}")
    print("✅ FORCING GPU usage for training!")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return True, 0  # Always return GPU device 0

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
    
    print(f"🎯 TRAINING ON GPU: {torch.cuda.get_device_name(0)}")
    print("⏰ GPU training should be much faster...")
    
    # Optimize batch size for GPU
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb >= 8:
        batch_size = 16  # Larger batch for high-memory GPUs
    elif gpu_memory_gb >= 4:
        batch_size = 8   # Medium batch for mid-range GPUs  
    else:
        batch_size = 4   # Small batch for low-memory GPUs
    
    print(f"📊 Optimized batch size for your GPU: {batch_size}")
    
    # Train the model - FORCE GPU
    results = model.train(
        data='plant_disease_yolo_dataset/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=batch_size,     # Optimized batch size
        device=0,            # FORCE GPU device 0
        project='plant_disease_runs',
        name='yolov8_plant_disease',
        save_period=10,
        patience=15,
        
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
        
        # GPU optimization
        amp=True,            # Automatic Mixed Precision for faster training
        cache=True,          # Cache images for faster loading
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