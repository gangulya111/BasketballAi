BasketballAI -Hoopers Den

Repository Structure

BasketballAi
  DribbleAPI               
  ShotAPI                   
  travel_detection.py        
  basketballModel.pt         
  yolov8s-pose.pt           
  requirement.txt            
  README.md
  *.MOV                      


Prerequisites 

Python 3.9+
PyTorch + Ultralytics

Installation

git clone https://github.com/gangulya111/BasketballAi.git


Create virtual environment

Install dependencies
pip install -r requirement.txt

If you plan to use YOLOv8-pose directly, ensure Ultralytics is installed:
pip install ultralytics


The repo includes several sample .MOV files to test

Models

- yolov8s-pose.pt: skeletons
- basketballModel.pt: basketball and hoop detection

APIs

- DribbleAPI/ – HTTP endpoints for dribbling metrics (frequency,  control).
- ShotAPI/ – HTTP endpoints for shot metrics ( arc/entry angle, release timing)




Acknowledgements

- YOLOv8 Pose model
