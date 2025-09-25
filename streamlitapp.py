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
    """Run YOLO inference on PIL image and return results with debug info."""
    img_array = np.array(image)
    results = model(img_array)

    detected_classes = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf) * 100
                class_name = model.names[class_id]
                detected_classes.append((class_name, confidence))

            # Return top prediction
            top_box = boxes[0]
            top_class_id = int(top_box.cls)
            top_confidence = float(top_box.conf) * 100
            top_class_name = model.names[top_class_id]
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            return top_class_name, top_confidence, annotated_img_rgb, detected_classes

    # No detection
    return "No detection", 0.0, img_array, detected_classes

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
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                start_time = time.time()
                class_name, confidence, annotated_img, detected_classes = predict_image(model, image)
                processing_time = time.time() - start_time
                health_status, recommendations = get_recommendations(class_name, confidence)

                # Show all detected classes in the app
                st.subheader("🔍 All Detected Classes")
                if detected_classes:
                    for cls, conf in detected_classes:
                        st.write(f"{cls.replace('_',' ').replace('___',' - ')} : {conf:.1f}%")
                else:
                    st.write("No objects detected.")

                # FIXED DEBUG SECTION
                with st.expander("🔧 Debug Info (Click to expand)"):
                    st.write("**Image Details:**")
                    st.write(f"- Size: {image.size}")
                    st.write(f"- Filename: {uploaded_file.name}")
                    st.write(f"- File size: {len(uploaded_file.getvalue())} bytes")
                    
                    st.write("**Model Output:**")
                    st.write(f"- Raw class name: '{class_name}'")
                    st.write(f"- Top confidence: {confidence:.1f}%")
                    st.write(f"- Total detections: {len(detected_classes)}")
                    
                    if detected_classes:
                        st.write("**All Detections:**")
                        detection_text = ""
                        for i, (cls, conf) in enumerate(detected_classes):
                            detection_text += f"  {i+1}. '{cls}' → {conf:.1f}%\n"
                        st.text(detection_text)
                    else:
                        st.write("- No detections found")
                    
                    # Check for filename vs prediction mismatch
                    filename_lower = uploaded_file.name.lower()
                    prediction_lower = class_name.lower()
                    st.write("**Validation Check:**")
                    if "scab" in filename_lower and "healthy" in prediction_lower:
                        st.error("⚠️ MISMATCH: Filename has 'scab' but model predicts 'healthy'")
                    elif "healthy" in filename_lower and "healthy" not in prediction_lower:
                        st.error("⚠️ MISMATCH: Filename has 'healthy' but model predicts disease")
                    elif ("scab" in filename_lower and "scab" in prediction_lower) or ("healthy" in filename_lower and "healthy" in prediction_lower):
                        st.success("✅ Filename matches prediction")
                    else:
                        st.info("ℹ️ Filename vs prediction comparison inconclusive")

                # Display health status with appropriate styling
                if "healthy" in class_name.lower():
                    st.success("✅ Plant is Healthy")
                elif confidence < 50:
                    st.warning("❓ Unclear result")
                else:
                    st.error("⚠️ Disease Detected")
                
                # Display prediction details
                st.write(f"**Top Prediction:** {class_name.replace('_',' ').replace('___',' - ')}")
                st.write(f"**Confidence:** {confidence:.1f}%")
                st.write(f"**Processing time:** {processing_time:.2f}s")
                
                # Show annotated image
                st.image(annotated_img, caption="Annotated Result", width=400)

                # COMPLETELY FIXED RECOMMENDATIONS SECTION
                st.header("💡 Recommendations")
                if recommendations and len(recommendations) > 0:
                    recommendation_text = ""
                    for idx, rec in enumerate(recommendations, 1):
                        if rec and str(rec).strip():
                            recommendation_text += f"{idx}. {rec}\n"
                    st.markdown(recommendation_text)
                else:
                    st.write("No specific recommendations available.")

if __name__ == "__main__":
    main()