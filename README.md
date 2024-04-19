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
   ```bash
   conda create -n SoccerEye python==3.10
   conda activate SoccerEye
   pip install -r requirements.txt
3. **Download Pretrained Models and Dataset**  
     Download the pretrained model from here.  
     Download the test video from here.  
3. **To start using SoccerEye, run the following command:**
   ```bash
   python run.py --detector yolo --model_path model/yolov8x_1280.pt --video video/video.avi



