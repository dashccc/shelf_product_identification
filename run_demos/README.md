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