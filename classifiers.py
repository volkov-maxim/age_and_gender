from PIL import Image
import numpy as np
import cv2 as cv

# Globals.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ("Male", "Female")
AGE_LIST = ("(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)")


class GenderClassifier():
  """
  Class GenderClassifier.
  """  
  def __init__(self, model: str, proto: str):
    """
    Initializing of GenderClassifier instance.
  
    Args:
      model (str): Path to .caffemodel file (model weights).
      proto (str): Path to .prototxt file (model architecture). 
  
    """
    self.model = cv.dnn.readNet(model, proto)


  def classification_to_json(self, input_image: Image, detection_predict: dict) -> dict:
    """
    Makes prediction on image and convert it to JSON.
  
    Args:
      input_image (Image): Input image for prediction.
      detection_predict (dict): Dict with predictions from FaceDetector.
    
    Returns:
      dict: Converted predictions to JSON format.
    """
    input_image = np.array(input_image)
    x1 = int(detection_predict["box"]["x1"])
    y1 = int(detection_predict["box"]["y1"])
    x2 = int(detection_predict["box"]["x2"])
    y2 = int(detection_predict["box"]["y2"])
    face = input_image[y1:y2, x1:x2]

    blob = cv.dnn.blobFromImage(
      face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False
    )
    self.model.setInput(blob)
    gender_preds = self.model.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    return {"Gender": gender}
    
    
class AgeClassifier():
  """
  Class AgeClassifier.
  """  
  def __init__(self, model: str, proto: str):
    """
    Initializing of AgeClassifier instance.
  
    Args:
      model (str): Path to .caffemodel file (model weights).
      proto (str): Path to .prototxt file (model architecture). 
  
    """
    self.model = cv.dnn.readNet(model, proto)


  def classification_to_json(self, input_image: Image, detection_predict: dict) -> dict:
    """
    Makes prediction on image and convert it to JSON compatible dict.
  
    Args:
      input_image (Image): Input image for prediction.
      detection_predict (dict): Dict with predictions from FaceDetector.
    
    Returns:
      dict: Converted predictions to JSON format.
    """
    input_image = np.array(input_image)
    x1 = int(detection_predict["box"]["x1"])
    y1 = int(detection_predict["box"]["y1"])
    x2 = int(detection_predict["box"]["x2"])
    y2 = int(detection_predict["box"]["y2"])
    face = input_image[y1:y2, x1:x2]

    blob = cv.dnn.blobFromImage(
      face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False
    )
    self.model.setInput(blob)
    age_preds = self.model.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    return {"Age": age[1:-1]}