from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(title="Clothes Classification API")

# ----------------------------
# Load model ONCE at startup
# ----------------------------
MODEL_PATH = "xception_v4_1_13_0.891.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = (299, 299)
CLASS_NAMES = ['dress','hat','longsleeve','outwear','pants','shirt','shoes','shorts','skirt','t-shirt']

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)

    image = np.array(image)
    image = tf.keras.applications.xception.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        predictions = model.predict(image)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])

        return JSONResponse(
            content={
                "class": CLASS_NAMES[predicted_index],
                "confidence": round(confidence, 4)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


