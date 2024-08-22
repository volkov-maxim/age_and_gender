from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse

from face_detector import FaceDetector
from face_detector import get_image_from_bytes


# Globals.
app = FastAPI()


# Init model.
@app.on_event("startup")
def init_model():
  """
    Init model.
  """
  global face_detector 
  face_detector = FaceDetector("models/yolov8m_face_detection_e50_s640_best.pt")


# Redirect to docs.
@app.get("/")
async def redirect():
  """
    Redirect to docs (Swagger UI).
  """
  return RedirectResponse("/docs")


# Healthcheck.
@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def get_root():
  """
    Simple healthckeck of service.
  """
  return {"healthcheck": "I'm ok!"}


# Detecting face.
@app.post("/face_detection_to_json")
def post_detection(file: bytes = File(...)):
  """
    Detects face on an image.

    Args:
      file (bytes): The image file in bytes format.

    Returns:
      dict: Converted predictions to JSON format.
  """
  # Step 1: Initialize the result dictionary with None values.
  result={'detect_objects': None}

  # Step 2: Convert the image file to an image object.
  input_image = get_image_from_bytes(file)
  
  # Step 3: Predict from model.
  predict = face_detector.detection_to_json(input_image)

  return predict