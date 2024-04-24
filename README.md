## SoccerEye

---

#### Overview
1. **Introduction**  
SoccerEye is a monocular soccer video processing system that contains a series of functions required for soccer video data visualization.  
Its specific functions include the following:
 * player and football object detection
 * player grouping
 * trajectory visualization on bird's eye view
 * real-time speed display
 * advertising maps
 <div align=center>
   <img src="https://github.com/WWandP/SoccerEye/blob/main/demo/show.png" width="400" height="333">
</div>  
  <p align="center">
  Functional diagram of SoccerEye
  </p> 

 
2. **Demo**
   <div align=center>
   <img src="https://github.com/WWandP/SoccerEye/blob/main/demo/demo.gif" width="480" height="260">
   </div>
   <p align="center">
    Video clips after processing with SoccerEye
   </p>
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

   * Download the pretrained model([yolov8](https://drive.google.com/file/d/1CxbNcKDag-Z4B5Ez8XmlFvaGgZVlFdzi/view?usp=drive_link) or [maskrcnn](https://drive.google.com/file/d/1PpIXoDwLi-FuBFAVUsIwh0Ljm93JiQaH/view?usp=drive_link)) from here and put it in the *model/* folder.  
   * Download the [test video](https://drive.google.com/file/d/1DszEnRSF5E6NpWvgneFxHAlaP8dxFIsm/view?usp=drive_link) from here and  put it in the *video/* folder.   
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
<br>
1.**Set the team color filter**  
   You need to color the players' clothes and assign a specific team to each jersey. In *inference/filters.py*, you can configure the color of the combined team's jersey and the corresponding team name.
The selection of a wide range of appropriate color filters can improve the classification accuracy.    
<br>
2.**Custom AD projection**    
You can flexibly customize the placement and transparency of the ads by adjusting the parameters near line 130 in run.py  
   ```python
   # The 1920 x1080 coordinate system is used as the reference , and the origin is in the upper left corner
   frame = show_ad ( detections =players_detections , homography =M , img =frame , ad_img = ad , coord =(800 , 400) ,alpha =0.3)
   ```

3.**Tips for getting a bird 's-eye view**  
We added a Kalman filter to the ground to bird 's-eye view homography matrix to ensure its smoothness, but we did not add recognition for different scene transitions, so we do not recommend using broadcast video with shot transitions. At the same time, soccer videos with fixed viewpoints will have better results.  
<br>
4.**Customize the minimap**  
SoccerEye integrates the function of using opencv to detect the center circle of the soccer field map. If you want to use a custom bird 's-eye view small map image, please ensure that the center circle is very obvious and the image size is 1920 x1080, which is conducive to accurate advertising images.

---
#### Other downloadable data  
If you are interested in training our yolo model, you can download the dataset we collected [here](https://drive.google.com/file/d/1RHDHztUHho1zP1sKJXts1UxoFtrB2uFz/view?usp=drive_link) (people and balls on the football field only). In addition, we also have 4 yolo models(Not included is based on the COCO dataset) trained with different scales and different number of parameters, which you can download according to your device requirements. Their Training information is as follows:  
| Model    | Size       | Class  | mAP50  | mAP50-95 |
|----------|------------|--------|--------|-------|
| YOLOv8x(COCO)  | 640 × 640  | all    | 0.511  | 0.283 |
|          |            | person | 0.875  | 0.513 |
|          |            | ball   | 0.148  | 0.053 |
| [YOLOv8x](https://drive.google.com/file/d/1hHgq_yD_AA2ioHVIidB4FfMyZYpZxRO1/view?usp=drive_link)  | 640 × 640  | all    | 0.716  | 0.507 |
|          |            | person | 0.978  | 0.793 |
|          |            | ball   | 0.453  | 0.222 |
| [YOLOV8l](https://drive.google.com/file/d/1-VRIjoJcjY4_D_MHOcIGgxHVmB_pid1r/view?usp=drive_link)  | 640 × 640  | all    | 0.773  | 0.529 |
|          |            | person | 0.98   | 0.779 |
|          |            | ball   | 0.565  | 0.278 |
| [YOLOV8m](https://drive.google.com/file/d/1hHgq_yD_AA2ioHVIidB4FfMyZYpZxRO1/view?usp=drive_link)  | 1280 × 1280| all    | 0.795  | 0.59  |
|          |            | person | 0.989  | 0.832 |
|          |            | ball   | 0.601  | 0.348 |
| [YOLOV8l](https://drive.google.com/file/d/1z27p0vS5VHydnWP6opyz_fRn-NnyCpgs/view?usp=drive_link)  | 1280 × 1280| all    | 0.831  | 0.614 |
|          |            | person | 0.981  | 0.833 |
|          |            | ball   | 0.681  | 0.396 |
| [YOLOv8x](https://drive.google.com/file/d/1CxbNcKDag-Z4B5Ez8XmlFvaGgZVlFdzi/view?usp=drive_link)  | 1280 × 1280| all    | 0.847  | 0.625 |
|          |            | person | 0.982  | 0.840 |
|          |            | ball   | 0.713  | 0.411 |  

On another [test set](https://drive.google.com/file/d/1g4QTyTjOm7cx0VP_oAAi0NidJPxwFFdm/view?usp=drive_link) with a different style, their results are as follows  
| Model    | Size       | Class  | mAP50  | mAP50-95 |
|----------|------------|--------|--------|-------|
| YOLOV8l  | 640 × 640  | all    | 0.707  | 0.313 |
|          |            | person | 0.921  | 0.448 |
|          |            | ball   | 0.494  | 0.178 |
| YOLOv8x  | 640 × 640  | all    | 0.721  | 0.343 |
|          |            | person | 0.919  | 0.468 |
|          |            | ball   | 0.524  | 0.217 |
| YOLOv8x(COCO)  | 640 × 640  | all    | 0.754  | 0.367 |
|          |            | person | 0.918  | 0.478 |
|          |            | ball   | 0.59   | 0.257 |
| YOLOV8l  | 1280 × 1280| all    | 0.811  | 0.365 |
|          |            | person | 0.931  | 0.488 |
|          |            | ball   | 0.691  | 0.243 |
| YOLOV8m  | 1280 × 1280| all    | 0.823  | 0.361 |
|          |            | person | 0.926  | 0.474 |
|          |            | ball   | 0.721  | 0.248 |
| YOLOv8x  | 1280 × 1280| all    | 0.823  | 0.359 |
|          |            | person | 0.918  | 0.452 |
|          |            | ball   | 0.729  | 0.266 |

---  
#### Contact
 If you need further information or advice, please contact us:[pww_work@163.com](pww_work@163.com)







