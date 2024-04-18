import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from typing import List
from torchvision.transforms import functional as F
import torchvision

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector
from ultralytics import YOLO

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Detector(BaseDetector):
    def __init__(
            self,
            model_path: str = None,
            detector: str = "yolo"
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """
        self.names = COCO_INSTANCE_CATEGORY_NAMES
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = detector
        if detector == "yolo":
            self.model = YOLO(model_path)
        elif detector == "maskrcnn":
            self.model = maskrcnn_resnet50_fpn(pretrained=False, )
            self.weights = torch.load(model_path)
            self.model.load_state_dict(self.weights)
            self.model.eval()
        self.model = self.model.to(self.device)

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
        mask_list = []
        if self.detector == "maskrcnn":

            frame_tensor = F.to_tensor(input_image).unsqueeze(0).to(self.device)

            # 运行模型推理
            with torch.no_grad():
                prediction = self.model(frame_tensor)
            # 使用 NMS 进行框的后处理
            keep = torchvision.ops.nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=0.35)
            prediction[0]['boxes'] = prediction[0]['boxes'][keep]
            prediction[0]['labels'] = prediction[0]['labels'][keep]
            prediction[0]['scores'] = prediction[0]['scores'][keep]
            prediction[0]['masks'] = prediction[0]['masks'][keep]
            masks = prediction[0]['masks']

            for i in range(masks.shape[0]):
                label = prediction[0]['labels'][i].item()
                conf = prediction[0]['scores'][i].item()
                if label == 1 and conf > 0.5:
                    box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                    mask = prediction[0]['masks'][i, 0].cpu().numpy()
                    x = box[0]
                    y = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    name = "person"
                    df = pd.concat([df, pd.DataFrame({
                        'xmin': [x],
                        'ymin': [y],
                        'xmax': [x2],
                        'ymax': [y2],
                        'confidence': [conf],
                        'class': [label],
                        'name': [name]
                    })], ignore_index=True)
                    mask = mask[y:y2, x:x2]
                    mask_list.append(mask)
                # result = self.model(input_image, size=1280)
            return df, mask_list
        elif self.detector == "yolo":
            results = self.model.predict(source=input_image, save=False, save_txt=False, show_conf=False,
                                         show_labels=False,
                                         retina_masks=True, iou=0.45, conf=0.6)  # save predictions as labels
            pred_bbox = [box for box in results[0].boxes]

            for box in pred_bbox:
                cls = box.cls.item()
                if cls != 0:
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
                mask_list = [np.zeros((1, 1))]
            return df, mask_list



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

        df = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
        mask_list = []
        if self.detector == "maskrcnn":

            frame_tensor = F.to_tensor(input_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(frame_tensor)
            # Post-processing bounding boxes using NMS (Non-Maximum Suppression)
            keep = torchvision.ops.nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=0.35)
            prediction[0]['boxes'] = prediction[0]['boxes'][keep]
            prediction[0]['labels'] = prediction[0]['labels'][keep]
            prediction[0]['scores'] = prediction[0]['scores'][keep]
            prediction[0]['masks'] = prediction[0]['masks'][keep]
            masks = prediction[0]['masks']

            for i in range(masks.shape[0]):
                label = prediction[0]['labels'][i].item()
                conf = prediction[0]['scores'][i].item()
                if label == 37 and conf > 0.4:
                    box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
                    mask = prediction[0]['masks'][i, 0].cpu().numpy()
                    x = box[0]
                    y = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    name = "sports ball"
                    df = pd.concat([df, pd.DataFrame({
                        'xmin': [x],
                        'ymin': [y],
                        'xmax': [x2],
                        'ymax': [y2],
                        'confidence': [conf],
                        'class': [label],
                        'name': [name]
                    })], ignore_index=True)
                    mask = mask[y:y2, x:x2]
                    mask_list.append(mask)
                # result = self.model(input_image, size=1280)
            return df, mask_list
        elif self.detector == "yolo":
            results = self.model.predict(source=input_image, save=False, save_txt=False, show_conf=False,
                                         show_labels=False,
                                         retina_masks=True, iou=0.45, conf=0.4)  # save predictions as labels
            pred_bbox = [box for box in results[0].boxes]

            for box in pred_bbox:
                cls = box.cls.item()
                if cls == 0:
                    x = box.xyxy.tolist()[0][0]
                    y = box.xyxy.tolist()[0][1]
                    x2 = box.xyxy.tolist()[0][2]
                    y2 = box.xyxy.tolist()[0][3]
                    conf = box.conf.item()
                    name = "sports ball"
                    df = pd.concat([df, pd.DataFrame({
                        'xmin': [x],
                        'ymin': [y],
                        'xmax': [x2],
                        'ymax': [y2],
                        'confidence': [conf],
                        'class': [cls],
                        'name': [name]
                    })], ignore_index=True)
                mask_list = [np.zeros((1, 1))]
            return df, mask_list
