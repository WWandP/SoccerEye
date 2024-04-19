## SoccerEye

---

#### Overview
1. **Introduction**  
   SoccerEye is a monocular soccer video processing system that contains a range of functions needed to understand soccer videos. These include player identification, team segmentation, trajectory visualization on bird's eye view, real-time speed display, and advertising maps.
2. **Demo**
   
---

#### Quick Start

1. **Clone the Project**

   ```bash
   git clone https://github.com/NotPostionOldMan/SoccerEye.git
   cd SoccerEye
2. **Set Up Environment**  
   You will need a python environment, we recommend installing anaconda or miniconda and configuring the dependent packages.
   ```bash
   conda create -n SoccerEye python==3.10
   conda activate SoccerEye
   pip install -r requirements.txt
4. **Download Pretrained Models and Test Video**  
   We provide two pre-trained models for the detector, maskrcnn.pth, an instance segmentation model, and yolov8x_1280.pt, an object detection model trained on yolov8. yolov8x_1280.pt is trained on 1103 custom soccer scene images. maskrcnn is a pre-trained model based on COCO dataset.  
Our example video is a soccer video with a fixed scene.  

   * Download the pretrained model from here and put it in the *model/* folder.  
   * Download the test video from here and  put it in the *video/* folder.   
3. **To start using SoccerEye, run the following command:**
   ```bash
   python run.py --detector yolo --model_path model/yolov8x_1280.pt --video video/video.avi
   ```
   To show the bird 's-eye view feature, add the *--bev* command, it looks like this:
   ```bash
   python run.py --detector yolo --model_path model/yolov8x_1280.pt --video video/video.avi --bev
   ```
   Image embedding allows you to place AD images on the field, but it requires an instance segmentation model, which you can use with the following command:
   ```bash
   python run.py --detector maskrcnn --model_path model/maskrcnn.pth --video video/video.avi --bev --ad
   ```
---
#### Custom videos

