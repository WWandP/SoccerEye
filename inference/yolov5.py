from typing import List

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector
from ultralytics import YOLO


class YoloV5(BaseDetector):
    def __init__(
            self,
            model_path: str = None,
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if model_path:
            self.model=YOLO(model_path)
            # self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path,force_reload=True, trust_repo=True)
        else:
            self.model=YOLO(model_path)

            # self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path,force_reload=True, trust_repo=True)

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """
        df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

        results = self.model.predict(source=input_image, save=False, save_txt=False, show_conf=False, show_labels=False,
                                retina_masks=True, iou=0.45, conf=0.1)  # save predictions as labels
        pred_bbox = [box for box in results[0].boxes]

        for box in pred_bbox:
            x = box.xyxy.tolist()[0][0]
            y = box.xyxy.tolist()[0][1]
            x2 = box.xyxy.tolist()[0][2]
            y2 = box.xyxy.tolist()[0][3]
            conf = box.conf.item()
            cls = box.cls.item()
            name = results[0].names[cls]
            df = pd.concat([df, pd.DataFrame({
                'xmin': [x],
                'ymin': [y],
                'xmax': [x2],
                'ymax': [y2],
                'confidence': [conf],
                'class': [cls],
                'name': [name]
            })], ignore_index=True)
        # result = self.model(input_image, size=1280)
        return df

    def predictBall(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        result = self.model(input_image, size=1280)

        return result.pandas().xyxy[32]
