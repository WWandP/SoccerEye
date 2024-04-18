from typing import List

import norfair
import numpy as np
from norfair import Detection
from norfair.camera_motion import MotionEstimator

from inference import Converter, Detector
from soccer import Ball, Match
import cv2


def get_ball_detections(
        ball_detector: Detector, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Uses custom Yolov5 detector in order
    to get the predictions of the ball and converts it to
    Norfair.Detection list.

    Parameters
    ----------
    ball_detector : YoloV5
        YoloV5 detector for balls
    frame : np.ndarray
        Frame to get the ball detections from

    Returns
    -------
    List[norfair.Detection]
        List of ball detections
    """
    ball_df, _ = ball_detector.predictBall(frame)
    ball_df = ball_df[ball_df["name"] == 'sports ball']
    ball_df = ball_df[ball_df["confidence"] > 0.35]
    return Converter.DataFrame_to_Detections(ball_df, _)


def get_player_detections(
        person_detector: Detector, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Uses YoloV5 Detector in order to detect the players
    in a match and filter out the detections that are not players
    and have confidence lower than 0.35.

    Parameters
    ----------
    person_detector : YoloV5
        YoloV5 detector
    frame : np.ndarray
        _description_

    Returns
    -------
    List[norfair.Detection]
        List of player detections
    """

    person_df, person_mask = person_detector.predict(frame)
    person_df = person_df[person_df["name"] == "person"]
    person_df = person_df[person_df["confidence"] > 0.25]
    person_detections = Converter.DataFrame_to_Detections(person_df, person_mask)
    return person_detections


def create_mask(frame: np.ndarray, detections: List[norfair.Detection]) -> np.ndarray:
    """

    Creates mask in order to hide detections and goal counter for motion estimation

    Parameters
    ----------
    frame : np.ndarray
        Frame to create mask for.
    detections : List[norfair.Detection]
        Detections to hide.

    Returns
    -------
    np.ndarray
        Mask.
    """

    if not detections:
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    else:
        detections_df = Converter.Detections_to_DataFrame(detections)
        mask = Detector.generate_predictions_mask(detections_df, frame, margin=40)

    # remove goal counter
    mask[69:200, 160:510] = 0

    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an img

    Parameters
    ----------
    img : np.ndarray
        Image to apply the mask to
    mask : np.ndarray
        Mask to apply

    Returns
    -------
    np.ndarray
        img with mask applied
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img


def update_motion_estimator(
        motion_estimator: MotionEstimator,
        detections: List[Detection],
        frame: np.ndarray,
) -> "CoordinatesTransformation":
    """

    Update coordinate transformations every frame

    Parameters
    ----------
    motion_estimator : MotionEstimator
        Norfair motion estimator class
    detections : List[Detection]
        List of detections to hide in the mask
    frame : np.ndarray
        Current frame

    Returns
    -------
    CoordinatesTransformation
        Coordinate transformation for the current frame
    """

    mask = create_mask(frame=frame, detections=detections)
    coord_transformations = motion_estimator.update(frame, mask=mask)
    return coord_transformations


def get_main_ball(detections: List[Detection], match: Match = None) -> Ball:
    """
    Gets the main ball from a list of balls detection

    The match is used in order to set the color of the ball to
    the color of the team in possession of the ball.

    Parameters
    ----------
    detections : List[Detection]
        List of detections
    match : Match, optional
        Match object, by default None

    Returns
    -------
    Ball
        Main ball
    """
    ball = Ball(detection=detections)

    if match:
        ball.set_color(match)

    if detections:
        ball.detection = detections[0]

    return ball


def show_ad(detections: List[Detection], homography: np.ndarray, img: np.ndarray, ad_img: np.ndarray, coord=(800, 400), alpha=0.3):
    img = np.array(img)
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    perspective_matrix_bird_to_scene = np.linalg.inv(homography)
    bird_image = cv2.warpPerspective(img_copy, homography,
                                     (int(img_copy.shape[1]), int(img.shape[0])))
    ad = cv2.cvtColor(ad_img, cv2.COLOR_BGR2BGRA)
    x, y = coord[0], coord[1]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bird_image.shape[1], x1 + ad.shape[1])  # 使用 x1 + ad.shape[1] 而不是 x + ad.shape[1]
    y2 = min(bird_image.shape[0], y1 + ad.shape[0])  # 使用 y1 + ad.shape[0] 而不是 y + ad.shape[0]

    # 裁剪广告图片，使其适应边界
    ad_width = x2 - x1
    ad_height = y2 - y1
    cropped_ad = ad[:ad_height, :ad_width]

    # 放置裁剪后的广告
    bird_image[y1:y2, x1:x2] = cropped_ad

    bird_image = cv2.warpPerspective(bird_image, perspective_matrix_bird_to_scene,
                                     (int(bird_image.shape[1]), int(bird_image.shape[0])))
    ad_image_all = cv2.addWeighted(img_copy, 1, bird_image, alpha, 0)
    for detection in detections:
        mask = detection.data["mask"]
        box_x, box_y, box_x2, box_y2 = detection.points[0][0], detection.points[0][1], detection.points[1][0], \
                                       detection.points[1][1]
        cpro_frame = ad_image_all[box_y:box_y2, box_x:box_x2]
        cpro_ori_frame = img_copy[box_y:box_y2, box_x:box_x2]
        cpro_frame[mask > 0.6] = cpro_ori_frame[mask > 0.6]
        ad_image_all[box_y:box_y2, box_x:box_x2] = cpro_frame
    ad_image_all = cv2.cvtColor(ad_image_all, cv2.COLOR_BGRA2BGR)
    return ad_image_all
