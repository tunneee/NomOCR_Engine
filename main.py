from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

import ray
from ray import serve

from handler.asset import hash_bytes, load_models, retrieve_image
from handler.bbox import generate_initial_drawing, transform_fabric_box, order_boxes4nom, get_patch


# Configure TensorFlow GPU settings
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

app = FastAPI(
    title="NomOCR API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
det_model, rec_model = load_models()

# @app.on_event("startup")
# async def startup_event():
    

@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect bounding boxes
    boxes = det_model.predict_one_page(raw_image)
    
    # Recognize text in each bounding box
    ocr_results = []
    for box in boxes:
        patch = get_patch(raw_image, box)
        nom_text = rec_model.predict_one_patch(patch).strip()
        ocr_results.append({
            'nom_text': nom_text,
            'box': box.tolist()
        })
    
    return JSONResponse(content={"ocr_results": ocr_results})

@app.post("/detect_boxes/")
async def detect_boxes(file: UploadFile = File(...)):
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect bounding boxes
        boxes = det_model.predict_one_page(raw_image)
        
        return JSONResponse(content={"boxes": [box.tolist() for box in boxes]})

@app.post("/detect_custom_boxes/")
async def detect_custom_boxes(file: UploadFile = File(...), custom_boxes: list = None):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if custom_boxes is None:
        # Detect bounding boxes
        boxes = det_model.predict_one_page(raw_image)
    else:
        # Use provided custom boxes
        boxes = [np.array(box) for box in custom_boxes]
    
    return JSONResponse(content={"boxes": [box.tolist() for box in boxes]})

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
