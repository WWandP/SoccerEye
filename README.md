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
   We provide two pre-trained models for the detector, yolov8x_1280.pt, an object detection model trained on yolov8 and maskrcnn.pth, an instance segmentation model . yolov8x_1280.pt is trained on 1103 custom soccer scene images. maskrcnn is a pre-trained model based on COCO dataset.  
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
   When using the *--bev* command, the speed display is enabled by default, but if you want to turn it off, add the *--nospeed* command.
---
#### Custom videos
In the case of custom videos, you need to make some adjustments to the code section to suit your needs.  
1.Set the team color filter  
   You need to color the players' clothes and assign a specific team to each jersey. In *inference/filters.py*, you can configure the color of the combined team's jersey and the corresponding team name.
The selection of a wide range of appropriate color filters can improve the classification accuracy.
2.Custom AD projection
You can flexibly customize the placement and transparency of the ads by adjusting the parameters near line 130 in run.py  
```python
# The 1920 x1080 coordinate system is used as the reference , and the origin is in the upper left corner
frame = show_ad ( detections =players_detections , homography =M , img =frame , ad_img = ad , coord =(800 , 400) ,alpha =0.3)
```
3. Considerations under BEV  
We added a Kalman filter to the ground to bird 's-eye view homography matrix to ensure its smoothness, but we did not add recognition for different scene transitions, so we do not recommend using broadcast video with shot transitions. At the same time, soccer videos with fixed viewpoints will have better results.
