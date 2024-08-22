# Face detection and age and gender classification:
This repository contains face detection and age and gender classification decision that powered by using YOLOv8 (by Ultralytics) and FastAPI. 
The project also includes Docker compose file for easily building, shipping, and running application.

# What's inside:

- YOLOv8: A popular real-time object detection model
- FastAPI: A modern, fast (high-performance) web framework for building APIs
- Docker: A platform for easily building, shipping, and running distributed applications

---
# Getting Started

You have two options to start the application: using Docker or locally on your machine.

## Using Docker
Start the application with the following command:

```
docker compose up
```

## Locally
To start the application locally, follow these steps:

1. Install the required packages:

```
pip install -r requirements.txt
```
2. Start the application:

```
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```  

## FastAPI docs url:
http://0.0.0.0:8001/

---

# Overview of the code
* [main.py](./main.py) - Base FastAPI functions  
* [face_detector.py](./face_detector.py) - Face detector class
* [classifiers.py](./classifiers.py) - Age and gender classes
* [/models](./models) - models folder
