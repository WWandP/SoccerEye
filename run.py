import argparse
from typing import List
from arguments import Arguments
import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from inference import Converter, HSVClassifier, InertiaClassifier, Detector, NNClassifier
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator, show_ad,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.BirdEyeViewDrawer import BirdEyeViewDrawer
from soccer.Perspective_transform import Perspective_transform

args = Arguments().parse()
video = Video(input_path=args.video, output_path=args.output, output_fps=args.output_fps)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
detector = Detector(model_path=args.model_path, detector=args.detector)
print("model init already :)")

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)
# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)
perspective_transform = Perspective_transform()
# Teams and Match
chelsea = Team(
    name="Chelsea",
    abbreviation="CHE",
    color=(255, 0, 0))
man_city = Team(
    name="Man City",
    abbreviation="MNC",
    color=(255, 255, 255))
referee = Team(name='Referee', abbreviation="RER", color=(0, 0, 0))
teams = [chelsea, man_city, referee]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Soccer Field Image
gt_img = cv2.imread('./inference/white2.png')
gt_h, gt_w, _ = gt_img.shape
gt_h_h = int(gt_h / 3)
gt_w_h = int(gt_w / 3)
birdView = BirdEyeViewDrawer(gt_img, gt_w_h, gt_h_h, t1_color=teams[0].color, t2_color=teams[1].color)

# ad img
ad = cv2.imread("Ads.png")
ad = cv2.cvtColor(ad, cv2.COLOR_BGR2BGRA)

for i, frame in enumerate(video):
    print('\nframe_num:', i)
    # Get Detections
    players_detections = get_player_detections(detector, frame)
    ball_detections = get_ball_detections(detector, frame)
    detections = ball_detections + players_detections

    # Output: Homography Matrix and Warped image
    if i % 50 == 0:  # Calculate the homography matrix every 50 frames
        M = perspective_transform.find_homography(frame)

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )
    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections_ = Converter.TrackedObjects_to_Detections(ball_track_objects)
    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball)
    # annotate video frame
    frame = PIL.Image.fromarray(frame)
    frame = Player.draw_players(
        players=players, frame=frame, confidence=False, id=False
    )
    # draw path
    frame = path.draw(
        img=frame,
        ball_detections=ball_detections,
        coord_transformations=coord_transformations,
        color=match.team_possession.color,
    )
    if args.ad:
        # The 1920x1080 coordinate system is used as the reference, and the origin is in the upper left corner
        frame = show_ad(detections=players_detections, homography=M, img=frame, ad_img=ad, coord=(1000, 800), alpha=0.3)
    if args.bev:
        speed_tf = args.nospeed
        if ball_detections:
            coords = (ball_detections[0].points[0] + ball_detections[0].points[1]) / 2
            coords = tuple(coords)
            frame = birdView.bird_eye_view(players,  frame, M, i, coords, speed_tf=speed_tf)
        else:
            frame = birdView.bird_eye_view(players, frame, M, i, speed_tf=speed_tf)

    frame = np.array(frame)
    cv2.putText(frame, str(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    video.write(frame)
