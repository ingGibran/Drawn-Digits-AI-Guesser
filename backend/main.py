from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image 
import io 
import os
import numpy as np
import onnxruntime as ort

app = FastAPI(title="Number Guesser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


'''
Define Transform to manipulate data
'''
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = image.resize((28, 28), Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    return np.expand_dims(img_array, axis=(0, 1))


'''
Load ONNX Model
'''
ort_session = ort.InferenceSession("model/mnist_model.onnx")


'''
Deploy settings
'''
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


'''
API
'''
@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read() 
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess using numpy and PIL
        input_data = preprocess_image(image)
        
        # Inference using onnxruntime
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        output = ort_session.run(None, ort_inputs)[0]
        prediction = int(np.argmax(output, axis=1)[0])
            
        return {"filename": file.filename, "predicted_number": prediction}
    
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error processing image: {str(e)}"})