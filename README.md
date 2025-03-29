# Web-Based Image Annotation Tool Backend
## readme_for_flask_app_using_cudecpu_heavy

This is the backend server for a web-based image annotation tool that provides similar functionality to Anylabeling. The server handles image uploads, annotations, and AI-assisted labeling using Facebook's Segment Anything Model (SAM).

## Project Structure

```
backend/
├── app.py              # Main application file using flask
├── uploads/            # Directory for uploaded images
├── labels/            # Directory for saved annotations
└── models/            # Directory for AI models
    └── sam_vit_h_4b8939.pth  # SAM model file
```

## Prerequisites

- Python 3.8 or higher
- Sufficient disk space (approximately 2.5GB for the SAM model)
- CUDA-capable GPU (optional, but recommended for better performance)

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required Python packages:
```bash
pip install flask
pip install pillow
pip install torch
pip install opencv-python
pip install segment-anything
```

3. Create required directories:
```bash
mkdir uploads
mkdir labels
mkdir models
```

4. Download the SAM model:
```bash
cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Or download manually from the link and place in the models directory
cd ..
```
Download the SAM model checkpoint from Meta AI:

Go to https://github.com/facebookresearch/segment-anything
Download the sam_vit_h_4b8939.pth file
Place it in backend and then run the server
```
# Using wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Or using curl
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Configuration

The main configuration variables are at the top of `app.py`:

```python
UPLOAD_FOLDER = 'uploads'      # Directory for uploaded images
LABELS_FOLDER = 'labels'       # Directory for saved annotations
MODEL_TYPE = "vit_h"          # SAM model type
CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"  # Path to SAM model
```

## Running the Server

1. Make sure you're in the backend directory:
```bash
cd backend
```

2. Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Main Routes

- `GET /` - Serves the main application interface
- `POST /upload` - Handles image uploads
- `GET /uploads/<filename>` - Serves uploaded images
- `POST /auto_label` - Performs AI-assisted labeling
- `POST /save_label` - Saves annotation data
- `GET /export_labels` - Exports all annotations

### Request/Response Formats

#### Upload Image
```http
POST /upload
Content-Type: multipart/form-data

file: <image_file>

Response:
{
    "filename": "example.jpg"
}
```

#### Auto Label
```http
POST /auto_label
Content-Type: application/json

{
    "points": [[x1, y1], [x2, y2], ...]
}

Response:
{
    "mask": [[...]] // Binary mask array
}
```

#### Save Label
```http
POST /save_label
Content-Type: application/json

{
    "shapes": [
        {
            "type": "box|polygon|point",
            "points": [...],
            "label": "example_label"
        }
    ]
}

Response:
{
    "status": "success"
}
```

## Features

1. Image Upload and Management
   - Supports common image formats (PNG, JPG, JPEG)
   - Automatic file organization
   - Secure filename handling

2. Annotation Tools
   - Bounding box drawing
   - Polygon drawing
   - Point annotation
   - AI-assisted segmentation using SAM

3. Data Management
   - Local storage of annotations
   - Export functionality
   - Label organization

## Error Handling

The server includes basic error handling for:
- Invalid file uploads
- Missing files
- Invalid JSON data
- Model prediction errors

## Security Considerations

- File upload size is limited
- Secure filename handling is implemented
- Only specific file types are allowed
- No authentication system (add if needed)

## Known Limitations

1. The server uses in-memory storage for some data
2. No built-in authentication
3. Limited to one active image at a time
4. SAM model requires significant memory

## Troubleshooting

Common issues and solutions:

1. **SAM Model Not Loading**
   - Verify the model file is in the correct location
   - Check CUDA availability if using GPU
   - Ensure sufficient RAM/VRAM

2. **Upload Issues**
   - Check folder permissions
   - Verify UPLOAD_FOLDER exists
   - Check file size limits

3. **Performance Issues**
   - Consider reducing image size before processing
   - Use GPU if available
   - Monitor memory usage

## Future Improvements

1. Database integration for label storage
2. User authentication system
3. Support for multiple AI models
4. Batch processing capabilities
5. Advanced export formats
6. Real-time collaboration features



# Image Labeling Tool
##readme_for_frontend_and_backend_using_fastapi_And_js

A full-stack web application for image annotation and labeling, featuring automatic object detection using YOLOv8 and a React-based drawing interface.

## Project Structure

```
project_root/
├── backend/
│   ├── main_2.py   # main app using FastAPI
    ├───early scripts              # earlier scripts containing files
│   ├── uploads/            # Directory for uploaded images
│   ├── annotations.db      # SQLite database for annotations
│   └── yolov8n.pt         # YOLOv8 model file
└── frontend/
    ├── src/
    │   └── components/
    │       └── ImageLabeler.js  # Main React component
    ├── package.json
    └── public/
```

## Backend Setup

### Prerequisites
- Python 3.8 or higher
- SQLite
- GPU (optional, but recommended for better YOLO performance)

### Installation

1. Create and activate a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install sqlalchemy
pip install ultralytics
pip install pillow
pip install opencv-python
```

3. Download YOLOv8 model:
```bash
# The model will be downloaded automatically when first running the application
# Or you can manually download yolov8n.pt from ultralytics
```

4. Create required directories:
```bash
mkdir uploads
```

### Running the Backend

Start the FastAPI server:
```bash
uvicorn app:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

## Frontend Setup

### Prerequisites
- Node.js 14 or higher
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

### Running the Frontend

Start the development server:
```bash
npm start
# or
yarn start
```

The frontend will be available at `http://localhost:3000`

## Features

### Backend Features
1. Image Upload
   - Supports common image formats
   - Automatic file validation
   - Secure file handling

2. Auto-labeling
   - Object detection using YOLOv8
   - Bounding box generation
   - Class prediction with confidence scores

3. Database Integration
   - SQLite database for annotation storage
   - CRUD operations for annotations
   - Automatic schema creation

4. Error Handling
   - Comprehensive error logging
   - Proper HTTP status codes
   - Validation checks

### Frontend Features
1. Interactive Canvas
   - Draw bounding boxes
   - Real-time preview
   - Responsive design

2. Annotation Tools
   - Upload images
   - Manual box drawing
   - Undo/Redo functionality
   - Save annotations

3. UI Components
   - Toolbar with action buttons
   - Status messages
   - Image information display
   - Annotation counter

## API Endpoints

### Image Upload
```http
POST /upload/
Content-Type: multipart/form-data
Body: file=<image_file>

Response:
{
    "filename": "example.jpg",
    "url": "/uploads/example.jpg"
}
```

### Auto Label
```http
POST /auto_label/
Content-Type: application/x-www-form-urlencoded
Body: filename=example.jpg

Response:
{
    "filename": "example.jpg",
    "annotations": [
        {
            "x": float,
            "y": float,
            "width": float,
            "height": float,
            "class_id": int,
            "class_name": string,
            "confidence": float
        }
    ]
}
```

### Save Annotation
```http
POST /save_annotation/
Content-Type: application/x-www-form-urlencoded
Body: 
    filename=example.jpg
    annotation=<json_string>

Response:
{
    "message": "Annotation saved successfully"
}
```

### Get Annotations
```http
GET /annotations/

Response:
[
    {
        "id": int,
        "filename": string,
        "annotation": string
    }
]
```

## Usage Guide

1. Start both backend and frontend servers

2. Using the Web Interface:
   - Click "Upload Image" to select and upload an image
   - Draw boxes manually by clicking and dragging on the image
   - Use Undo/Redo buttons to correct mistakes
   - Click "Save Annotations" to store the annotations
   - View status messages for operation feedback

3. Auto-labeling:
   - Upload an image
   - Click the Play button to trigger auto-labeling
   - Adjust the generated boxes if needed
   - Save the final annotations

## Configuration

### Backend Configuration
```python
# In app.py
DATABASE_URL = "sqlite:///./annotations.db"
UPLOAD_DIR = "uploads"
```

### Frontend Configuration
```javascript
// API endpoint in ImageLabeler.js
const API_URL = 'http://localhost:8000'
```

## Troubleshooting

1. **Image Upload Issues**
   - Check file size limits
   - Verify supported file types
   - Check upload directory permissions
   - Verify CORS settings

2. **Auto-labeling Issues**
   - Ensure YOLOv8 model is downloaded
   - Check GPU availability if performance is slow
   - Verify image preprocessing

3. **Frontend Drawing Issues**
   - Check browser console for errors
   - Verify canvas size calculations
   - Check image CORS settings

## Future Improvements

1. Authentication system
2. Multiple annotation types (polygons, points)
3. Label categories management
4. Batch processing
5. Export functionality
6. Real-time collaboration
7. Custom model integration
