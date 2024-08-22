from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse

from face_detector import FaceDetector
from face_detector import get_image_from_bytes
from classifiers import GenderClassifier, AgeClassifier


# Globals.
app = FastAPI()


# Init model.
@app.on_event("startup")
def init_model():
  """
    Init models.
  """
  global face_detector 
  face_detector = FaceDetector("models/yolov8m_face_detection_e50_s640_best.pt")
  
  global gender_classifier
  gender_classifier = GenderClassifier(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
  )
  
  global age_classifier
  age_classifier = AgeClassifier(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
  )


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
  # Convert the image file to an image object.
  input_image = get_image_from_bytes(file)
  
  # Predict from model.
  predict = face_detector.detection_to_json(input_image)

  return predict
  

@app.post("/gender_classification_to_json")
def post_gender_classification(file: bytes = File(...))  -> dict:
  """
    Classify gender by image.

    Args:
      file (bytes): The image file in bytes format.

    Returns:
      dict: Converted predictions to JSON format.
  """
  # Convert the image file to an image object.
  input_image = get_image_from_bytes(file)
  
  # Get prediction from detection model.
  detection_predict = face_detector.detection_to_json(input_image)
  
  # Get prediction from classification model.
  predict = gender_classifier.classification_to_json(
    input_image, detection_predict
  )
  
  return predict
  

@app.post("/age_classification_to_json")
def post_age_classification(file: bytes = File(...))  -> dict:
  """
    Classify age by image.

    Args:
      file (bytes): The image file in bytes format.

    Returns:
      dict: Converted predictions to JSON format.
  """
  # Convert the image file to an image object.
  input_image = get_image_from_bytes(file)
  
  # Get prediction from detection model.
  detection_predict = face_detector.detection_to_json(input_image)
  
  # Get prediction from classification model.
  predict = age_classifier.classification_to_json(
    input_image, detection_predict
  )
  
  return predict
  