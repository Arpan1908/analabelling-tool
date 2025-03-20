from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import torch
import os
from segment_anything import SamPredictor, sam_model_registry
import cv2
from pathlib import Path
import json


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Configuration
UPLOAD_FOLDER = 'uploads'
LABELS_FOLDER = 'labels'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)

# Store current image filename in memory
current_image = None

# Initialize SAM model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "sam_vit_h_4b8939.pth"

# Check if the checkpoint file exists
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint file not found at: {CHECKPOINT_PATH}")

# Load the SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

@app.get("/")
async def index():
    return FileResponse("templates/anylabeling_template.html")



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_image

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Save with original filename
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        with open(filepath, "wb") as f:
            content = await file.read()  # Async read
            f.write(content)

        # Update current image
        current_image = filename

        return JSONResponse(content={"filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@app.get("/uploads/{filename}")
async def serve_file(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

@app.post("/auto_label")
async def auto_label(request: Request):
     global current_image

     if not current_image:
         raise HTTPException(status_code=400, detail="No image uploaded")

     data = await request.json()
     points = np.array(data['points'])

     # Load current image using the stored filename
     image_path = os.path.join(UPLOAD_FOLDER, current_image)
     image = cv2.imread(image_path)

     if image is None:
         raise HTTPException(status_code=500, detail=f"Failed to load image: {image_path}")

     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     # Generate mask using SAM
     predictor.set_image(image)
     input_points = points.reshape(-1, 2)
     input_labels = np.ones(len(input_points))

     masks, scores, logits = predictor.predict(
         point_coords=input_points,
         point_labels=input_labels,
         multimask_output=True
     )

     # Return the best mask
     best_mask = masks[scores.argmax()].tolist()
     return JSONResponse(content={"mask": best_mask})





@app.post("/save_label")
async def save_label(request: Request):
    global current_image

    if not current_image:
        raise HTTPException(status_code=400, detail="No image selected")

    data = await request.json()
    label_path = os.path.join(LABELS_FOLDER, f'{Path(current_image).stem}.json')

    with open(label_path, 'w') as f:
        json.dump(data, f)

    return JSONResponse(content={"status": "success"})

@app.get("/export_labels")
async def export_labels():
    labels = {}
    try:
        for label_file in os.listdir(LABELS_FOLDER):
            if not label_file.endswith('.json'):
                continue

            label_path = os.path.join(LABELS_FOLDER, label_file)
            try:
                if os.path.getsize(label_path) > 0:
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)
                        labels[label_file] = label_data
                else:
                    print(f"Skipping empty file: {label_file}")
            except json.JSONDecodeError as e:
                print(f"Error reading {label_file}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error reading {label_file}: {str(e)}")
                continue

        return JSONResponse(content=labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting labels: {str(e)}")

# Run the FastAPI app using uvicorn
# Command to run: uvicorn main_2:app --host 0.0.0.0 --port 8000 --reload
