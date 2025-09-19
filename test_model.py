from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

def test_trained_model():
    """
    Test and evaluate the trained YOLOv8 model
    """
    print("🧪 Testing Trained YOLOv8 Model")
    print("=" * 40)
    
    # Find the trained model
    model_path = "plant_disease_runs/yolov8_plant_disease/weights/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Trained model not found!")
        print("Please run train_yolov8.py first to train the model")
        return
    
    print("📥 Loading trained model...")
    model = YOLO(model_path)
    
    # Evaluate on validation set
    print("📊 Evaluating model on validation data...")
    metrics = model.val(data='plant_disease_yolo_dataset/dataset.yaml')
    
    print(f"\n📈 Model Performance:")
    print(f"   🎯 mAP50: {metrics.box.map50:.3f}")
    print(f"   🎯 mAP50-95: {metrics.box.map:.3f}")
    
    # Test on sample images
    test_sample_images(model)

def test_sample_images(model):
    """Test model on sample validation images"""
    print("\n🖼️ Testing on sample images...")
    
    # Get validation images
    val_images_dir = Path("plant_disease_yolo_dataset/val/images")
    if not val_images_dir.exists():
        print("❌ Validation images not found!")
        return
    
    # Get random sample images
    image_files = list(val_images_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No validation images found!")
        return
    
    # Test on 3 random images
    sample_images = random.sample(image_files, min(3, len(image_files)))
    
    for i, img_path in enumerate(sample_images):
        print(f"\n🔍 Testing image {i+1}: {img_path.name}")
        
        # Make prediction
        results = model(str(img_path))
        
        # Process results
        for result in results:
            # Save prediction image
            output_path = f"prediction_result_{i+1}.jpg"
            result.save(output_path)
            print(f"   💾 Saved prediction: {output_path}")
            
            # Print detection details
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = model.names[class_id]
                    print(f"   🦠 Detected: {class_name}")
                    print(f"   📊 Confidence: {confidence:.3f}")
            else:
                print("   ❓ No clear detection")

def predict_custom_image(image_path):
    """
    Predict disease on a custom image
    Usage: predict_custom_image("path/to/your/image.jpg")
    """
    model_path = "plant_disease_runs/yolov8_plant_disease/weights/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Trained model not found! Train the model first.")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Analyzing your image: {Path(image_path).name}")
    
    model = YOLO(model_path)
    results = model(image_path)
    
    for result in results:
        # Save result
        output_path = "your_image_prediction.jpg"
        result.save(output_path)
        print(f"💾 Prediction saved: {output_path}")
        
        # Show results
        boxes = result.boxes
        if boxes is not None:
            print("🎯 Detection Results:")
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                print(f"   🦠 Disease: {class_name}")
                print(f"   📊 Confidence: {confidence:.3f}")
        else:
            print("🌱 No disease detected or low confidence")

if __name__ == "__main__":
    test_trained_model()
    
    # Uncomment below to test your own image
    # predict_custom_image("path/to/your/plant/image.jpg")
    
    print("\n✅ Testing complete!")
    print("🎉 Your plant disease detection model is ready!")