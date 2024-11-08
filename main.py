from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import ray
from ray import serve
from PIL import ImageFont, ImageDraw, Image

from handler.asset import hash_bytes, load_models, retrieve_image
from handler.bbox import generate_initial_drawing, transform_fabric_box, order_boxes4nom, get_patch


# Configure TensorFlow GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from typing import List
from pydantic import BaseModel

class Patch(BaseModel):
    nom: str
    points: str
    height: int
    width: int

class OCRResponse(BaseModel):
    num_boxes: int
    height: int
    width: int
    patches: List[Patch]

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
# Load custom font
font_path = "./fonts/NomNaTong-Regular.ttf"
if not Path(font_path).exists():
    raise FileNotFoundError(f"Font file not found: {font_path}")

font = ImageFont.truetype(font_path, 16)
# Load models
det_model, rec_model = load_models()

# @app.on_event("startup")
# async def startup_event():
    
    
def two_point_to_four_point(box):
    return np.array([
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]]
    ])

@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect bounding boxes
    boxes = det_model(raw_image)[0].boxes.xyxy.to("cpu").numpy()
    boxes = [two_point_to_four_point(box) for box in boxes]
    
    # print(boxes)
    # Recognize text in each bounding box
    ocr_results = []
    for box in boxes:
        patch = get_patch(raw_image, box)
        nom_text = rec_model.predict_one_patch(patch).strip()
        ocr_results.append({
            'nom_text': nom_text,
            'box': box.tolist()
        })
    
    # Draw bounding boxes and text on the image
    for result in ocr_results:
        box = np.array(result['box'], dtype=np.int32)
        nom_text = result['nom_text']
        cv2.polylines(raw_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        img_pil = Image.fromarray(raw_image)
        draw = ImageDraw.Draw(img_pil)
        for i, char in enumerate(nom_text):
            draw.text((box[0][0], box[0][1] - 10 + i * 18), char, font=font, fill=(255, 0, 0, 255))
        raw_image = np.array(img_pil)

    # Save the image with bounding boxes to a temporary folder
    output_path = Path("./tmp/output_image.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, raw_image)
    response = OCRResponse(
        num_boxes=len(boxes),
        height=raw_image.shape[0],
        width=raw_image.shape[1],
        patches=[
            Patch(nom=result['nom_text'], points=str(result['box']), height=patch.shape[0], width=patch.shape[1])
            for result in ocr_results
        ]
    )
    
    return JSONResponse(content=response.dict())

@app.post("/detect_boxes/")
async def detect_boxes(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect bounding boxes
    boxes = det_model(raw_image)[0].boxes.xyxy.to("cpu").numpy()
    boxes = [two_point_to_four_point(box) for box in boxes]
    
    return JSONResponse(content={"boxes": [box.tolist() for box in boxes]})


# @app.post("/recognize_custom_boxes/")
# async def recognize_custom_boxes(custom_boxes: List[List[int]], file: UploadFile = File(...)):
#     # Read image file
#     image_bytes = await file.read()
#     image = Image.open(BytesIO(image_bytes))
#     raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
#     # Use provided custom boxes
#     boxes = [np.array(box) for box in custom_boxes]
    
#     # Recognize text in each bounding box
#     ocr_results = []
#     for box in boxes:
#         patch = get_patch(raw_image, box)
#         nom_text = rec_model.predict_one_patch(patch).strip()
#         ocr_results.append({
#             'nom_text': nom_text,
#             'box': box.tolist()
#         })
    
#     response = OCRResponse(
#         num_boxes=len(boxes),
#         height=raw_image.shape[0],
#         width=raw_image.shape[1],
#         patches=[
#             Patch(nom=result['nom_text'], points=str(result['box']), height=patch.shape[0], width=patch.shape[1])
#             for result in ocr_results
#         ]
#     )
    
#     return JSONResponse(content=response.dict())

@app.post("/recognize_patch/")
async def recognize_patch(file: UploadFile = File(...)):
    # Read patch image file
    image_bytes = await file.read()
    patch_image = Image.open(BytesIO(image_bytes))
    patch = cv2.cvtColor(np.array(patch_image), cv2.COLOR_RGB2BGR)
    
    # Recognize text in the patch
    nom_text = rec_model.predict_one_patch(patch).strip()
    
    return JSONResponse(content={"nom_text": nom_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
