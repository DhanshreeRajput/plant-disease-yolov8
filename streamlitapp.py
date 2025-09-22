# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import time
import os
import glob

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching."""
    specific_paths = [
        "plant_disease_runs/yolov8_plant_disease3/weights/best.pt",
        "plant_disease_runs/yolov8_plant_disease/weights/best.pt",
    ]
    for path in specific_paths:
        if os.path.exists(path):
            return YOLO(path), path

    # fallback pattern
    models_found = []
    patterns = ["plant_disease_runs/yolov8_plant_disease*/weights/best.pt"]
    for pattern in patterns:
        models_found.extend(glob.glob(pattern))
    if models_found:
        latest_model = max(models_found, key=os.path.getctime)
        return YOLO(latest_model), latest_model

    st.error("No trained YOLO model found!")
    st.stop()

def get_recommendations(class_name, confidence):
    if confidence < 50:
        return "Unclear", [
            "Image quality may be poor",
            "Try a clearer, well-lit image",
            "Ensure the leaf fills most of the frame"
        ]
    
    disease_name = class_name.lower()
    if "healthy" in disease_name:
        return "Healthy", [
            "Plant appears healthy!",
            "Continue current care routine",
            "Monitor regularly",
            "Maintain proper watering and nutrition"
        ]
    else:
        recommendations = [
            f"Disease detected: {class_name.replace('_', ' ').replace('___', ' - ')}",
            "Isolate affected plants",
            "Remove infected parts",
            "Improve air circulation",
            "Adjust watering",
            "Consider applying fungicide"
        ]
        if "blight" in disease_name:
            recommendations += ["Ensure good soil drainage", "Apply copper-based fungicide"]
        elif "rust" in disease_name:
            recommendations += ["Reduce humidity", "Apply sulfur-based treatment"]
        elif "spot" in disease_name or "scab" in disease_name:
            recommendations += ["Remove fallen leaves", "Improve air circulation"]
        return "Disease Detected", recommendations

def predict_image(model, image: Image.Image):
    """Run YOLO inference on PIL image and return results."""
    img_array = np.array(image)
    results = model(img_array)
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            class_name = model.names[class_id]
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            return class_name, confidence, annotated_img_rgb
    return "No detection", 0.0, img_array

def main():
    st.title("🌱 Plant Disease Detection AI")
    st.markdown("Upload a clear leaf image for analysis")

    model, model_path = load_model()
    st.sidebar.header("Model Info")
    st.sidebar.info(f"Path: {model_path}")
    st.sidebar.info(f"Classes: {len(model.names)}")

    uploaded_file = st.file_uploader("Choose image", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                start_time = time.time()
                class_name, confidence, annotated_img = predict_image(model, image)
                processing_time = time.time() - start_time
                health_status, recommendations = get_recommendations(class_name, confidence)

                if "healthy" in class_name.lower():
                    st.success("✅ Plant is Healthy")
                elif confidence < 50:
                    st.warning("❓ Unclear result")
                else:
                    st.error("⚠️ Disease Detected")
                
                st.write(f"**Prediction:** {class_name.replace('_',' ').replace('___',' - ')}")
                st.write(f"**Confidence:** {confidence:.1f}%")
                st.write(f"**Processing time:** {processing_time:.2f}s")
                st.image(annotated_img, caption="Annotated Result", use_column_width=True)

                st.header("💡 Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()
