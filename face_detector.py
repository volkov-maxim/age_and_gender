from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import json


def get_image_from_bytes(binary_image: bytes) -> Image:
  """
  Convert image from bytes to PIL RGB format.
  
  Args:
      binary_image (bytes): The binary representation of the image.
  
  Returns:
      PIL.Image: The image in PIL RGB format.
  """
  return Image.open(BytesIO(binary_image)).convert("RGB")


class FaceDetector(YOLO):
  """
  Class FaceDetector.
  """  
  def __init__(self, model_path: str):
    """
    Initializing of FaceDetector instance.
  
    Args:
      model_path: (str): Path to the model.
  
    """
    super(FaceDetector, self).__init__(model_path, task="detect", verbose=False)
    
    
  def detection_to_json(self, input_image: Image, image_size: int = 640, conf: float = 0.5, augment: bool = False) -> dict:
    """
    Makes prediction on image and convert it to JSON compatible dict.
  
    Args:
      input_image (Image): Input image for prediction.
      image_size (int, optional): The size of the image the model will receive. Defaults to 640.
      conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
      augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
  
    Returns:
      dict: Converted predictions to JSON format.
    """
    return json.loads(self.predict(
            source=input_image, 
            imgsz=image_size, 
            conf=conf,
            augment=augment,
          )[0].tojson())[0]
    
