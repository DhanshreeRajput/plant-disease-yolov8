# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io
import base64
import time
import os
import glob

app = FastAPI(title="Plant Disease Detection AI", version="1.0")

# --- Load YOLO model globally ---
def load_model():
    specific_paths = [
        "plant_disease_runs/yolov8_plant_disease3/weights/best.pt",
        "plant_disease_runs/yolov8_plant_disease/weights/best.pt",
    ]
    for path in specific_paths:
        if os.path.exists(path):
            return YOLO(path), path

    # fallback
    models_found = []
    patterns = ["plant_disease_runs/yolov8_plant_disease*/weights/best.pt"]
    for pattern in patterns:
        models_found.extend(glob.glob(pattern))
    if models_found:
        latest_model = max(models_found, key=os.path.getctime)
        return YOLO(latest_model), latest_model

    raise Exception("No trained YOLO model found!")

model, model_path = load_model()

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
            "Continue care routine",
            "Monitor regularly",
            "Maintain proper watering and nutrition"
        ]
    else:
        recommendations = [
            f"Disease detected: {class_name.replace('_',' ').replace('___',' - ')}",
            "Isolate affected plants",
            "Remove infected parts",
            "Improve air circulation",
            "Adjust watering",
            "Consider fungicide"
        ]
        if "blight" in disease_name:
            recommendations += ["Ensure soil drainage", "Apply copper-based fungicide"]
        elif "rust" in disease_name:
            recommendations += ["Reduce humidity", "Apply sulfur treatment"]
        elif "spot" in disease_name or "scab" in disease_name:
            recommendations += ["Remove fallen leaves", "Improve air circulation"]
        return "Disease Detected", recommendations

def predict_image(image: Image.Image):
    img_array = np.array(image)
    results = model(img_array)
    for result in results:
        boxes = result.boxes
        if boxes and len(boxes) > 0:
            box = boxes[0]
            class_id = int(box.cls)
            confidence = float(box.conf) * 100
            class_name = model.names[class_id]
            annotated_img = result.plot()
            annotated_img_rgb = annotated_img[..., ::-1]  # BGR -> RGB
            # Convert to base64
            pil_img = Image.fromarray(annotated_img_rgb)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            img_b64_str = f"data:image/jpeg;base64,{img_b64}"
            return class_name, confidence, img_b64_str
    return "No detection", 0.0, None

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>Plant Disease Detection API is running</h1>"

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        start_time = time.time()
        class_name, confidence, annotated_img = predict_image(image)
        processing_time = time.time() - start_time
        health_status, recommendations = get_recommendations(class_name, confidence)

        return JSONResponse(content={
            "success": True,
            "prediction": {
                "disease": class_name.replace('_',' ').replace('___',' - '),
                "confidence": round(confidence, 1),
                "raw_name": class_name
            },
            "health_status": health_status,
            "recommendations": recommendations,
            "processing_time": f"{processing_time:.2f}s",
            "annotated_image": annotated_img
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
