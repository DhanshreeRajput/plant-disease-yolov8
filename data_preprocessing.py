#Convert dataset to YOLO format
import os
import shutil
from sklearn.model_selection import train_test_split
import random
from pathlib import Path

def organize_data_for_yolo(source_dir, output_dir, train_ratio=0.8):
    """
    Organize PlantVillage data into YOLO format
    """
    print("🚀 Starting data organization...")
    
    # Create YOLO directory structure
    datasets_dir = Path(output_dir)
    (datasets_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # Try different possible subdirectories for images
    source_base = Path(source_dir)
    possible_subdirs = ["raw", "color", "segmented", "images"]
    source_path = None
    
    # First check if images are directly in the base directory
    class_dirs_direct = [d for d in source_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    # Check if any of these directories contain image files
    has_images_direct = False
    for d in class_dirs_direct[:3]:  # Check first 3 directories
        if list(d.glob("*.jpg")) or list(d.glob("*.JPG")) or list(d.glob("*.png")) or list(d.glob("*.PNG")):
            has_images_direct = True
            break
    
    if has_images_direct:
        source_path = source_base
        print(f"📁 Found images directly in: {source_path}")
    else:
        # Try subdirectories
        for subdir in possible_subdirs:
            test_path = source_base / subdir
            if test_path.exists():
                source_path = test_path
                print(f"📁 Found images in subdirectory: {source_path}")
                break
    
    if source_path is None or not source_path.exists():
        print(f"❌ Error: Could not find image directory!")
        print("Available directories in PlantVillage-Dataset:")
        for item in source_base.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}")
        
        print(f"\n💡 Searched for images in:")
        for subdir in possible_subdirs:
            print(f"  📁 {source_base / subdir}")
        print(f"  📁 {source_base} (root)")
        return
    
    class_dirs = [d for d in source_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        print(f"❌ No class directories found in {source_path}")
        return
    
    # Create class mapping
    class_names = [d.name for d in class_dirs]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"📊 Found {len(class_names)} disease classes:")
    for i, name in enumerate(class_names[:10]):  # Show first 10
        print(f"   {i}: {name}")
    if len(class_names) > 10:
        print(f"   ... and {len(class_names) - 10} more classes")
    
    total_train = 0
    total_val = 0
    
    # Process each class
    for idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_id = class_to_id[class_name]
        
        # Get all images in this class (support multiple formats)
        image_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            image_files.extend(list(class_dir.glob(ext)))
        
        print(f"🍃 Processing {class_name}: {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"   ⚠️ No images found in {class_name}")
            continue
        
        # Split into train/val
        train_files, val_files = train_test_split(
            image_files, 
            train_size=train_ratio, 
            random_state=42
        )
        
        # Copy training files
        for img_file in train_files:
            # Create unique filename to avoid conflicts
            new_name = f"{class_name}_{img_file.name}"
            dst_img = datasets_dir / "train" / "images" / new_name
            shutil.copy2(img_file, dst_img)
            
            # Create label file (for classification, we'll use whole image)
            label_file = datasets_dir / "train" / "labels" / f"{class_name}_{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                # Format: class_id x_center y_center width height (normalized)
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        # Copy validation files
        for img_file in val_files:
            new_name = f"{class_name}_{img_file.name}"
            dst_img = datasets_dir / "val" / "images" / new_name
            shutil.copy2(img_file, dst_img)
            
            # Create label file
            label_file = datasets_dir / "val" / "labels" / f"{class_name}_{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        # Progress indicator
        print(f"   ✅ {idx+1}/{len(class_dirs)} classes processed")
    
    # Create dataset.yaml file
    yaml_content = f"""path: {datasets_dir.absolute()}
train: train/images
val: val/images

nc: {len(class_names)}
names: {class_names}
"""
    
    with open(datasets_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Also save class names to a separate file
    with open(datasets_dir / "classes.txt", 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"\n🎉 Dataset organized successfully!")
    print(f"📊 Statistics:")
    print(f"   🔹 Total classes: {len(class_names)}")
    print(f"   🔹 Training images: {total_train}")
    print(f"   🔹 Validation images: {total_val}")
    print(f"   🔹 Dataset config: {datasets_dir / 'dataset.yaml'}")
    print(f"   🔹 Class names: {datasets_dir / 'classes.txt'}")

if __name__ == "__main__":
    print("🌱 PlantVillage Dataset Preprocessor")
    print("=" * 50)
    
    source_directory = "PlantVillage-Dataset"
    output_directory = "plant_disease_yolo_dataset"
    
    if not Path(source_directory).exists():
        print(f"❌ Error: {source_directory} not found!")
        print("Please download the PlantVillage dataset first.")
    else:
        organize_data_for_yolo(source_directory, output_directory)
        print("\n✅ Ready for training! Run: python train_yolov8.py")