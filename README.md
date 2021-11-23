# Shelf Product Identification
Shelf Product Identification is the solution based on object detection. Two types of modeles, YOLOv5 and One-Stage One-Shot Object Detection(OS2D), are utilized in this solution. 

This solution also provides a simple demo to present the business logic, which contains processes of product photo taking, image inference and displaying output results.
<br/><br/>

# Tutorials of Model Deployment
Link: https://dashccc.github.io/shelf_product_identification/
<br></br>

# Solution Structure
The folder structure of this solution is as belows:
- deploy_models
    - os2d
    - yolov5 
- run_demos
- docs
- test

The two models, OS2D and YOLOv5, are placed in under folder "deploy_models".

The simple business logic demo is placed under folder "run_demos".

The online tutorials are placed under folder "docs", and the test cases are placed in folder "test". There is no need to look at these two folders.
<br/><br/>

# Procedure to run the demo
Prerequisite:
- The modeling endpoint must be deployed in SageMaker Endpoint.
- In line 51 of file "camera_flask_app.py" under folder "run_demos", the right aws profile must be provided.

Steps to run the demo:
1. To run this demo, you should have python, flask and OpenCV installed on your OS sytem. 
2. To start the demo, clone this repo and move to the project directory in the command prompt. 
Go to folder "run-demos" and type: 
python camera_flask_app.py
3. Now, copy-paste http://127.0.0.1:5000/ into your favorite internet browser and that's it.
