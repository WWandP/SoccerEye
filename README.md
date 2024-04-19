## SoccerEye

---

#### Overview

SoccerEye is a project for analyzing soccer videos, designed to help users detect and track players and the ball during soccer matches.

---

#### Quick Start

1. **Clone the Project**

   ```bash
   git clone [project_url]
   cd SoccerEye
2. **Set Up Environment**  
   You will need a python environment, we recommend installing anaconda or miniconda and configuring the dependent packages
   ```bash
   conda create -n SoccerEye python==3.10
   conda activate SoccerEye
   pip install -r requirements.txt
4. **Download Pretrained Models and Test Video**  
   We provide two pre-trained models for the detector, maskrcnn.pth, an instance segmentation model, and yolov8x_1280.pt, an object detection model trained on yolov8. yolov8x_1280.pt is trained on 1103 custom soccer scene images. maskrcnn is a pre-trained model based on COCO dataset.  
   Download the pretrained model from here.  
   Our example video is a soccer video with a fixed scene.  
   Download the test video from here.  
3. **To start using SoccerEye, run the following command:**
   ```bash
   python run.py --detector yolo --model_path model/yolov8x_1280.pt --video video/video.avi



