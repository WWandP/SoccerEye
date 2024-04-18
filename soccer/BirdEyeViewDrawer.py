import cv2
import numpy as np
from soccer import Match, Player, Team
from scipy.optimize import linear_sum_assignment


class BirdEyeViewDrawer:
    """
    A class for drawing bird's eye view representation of players and ball on a given frame.
    """

    def __init__(self, gt_img, w, h):
        """
        Initialize the BirdEyeViewDrawer.

        :param gt_img: Bird's eye view map image.
        :param w: Width of the output image.
        :param h: Height of the output image.
        """
        self.gt_img = gt_img  # Birdseye view img
        self.w = w
        self.h = h
        self.kalman_filters = {}
        self.points_set = {}
        self.pix_dist = 18.3 / detect_circle_diameter(gt_img)
        for i in range(0, 30):
            if i < 10:
                point = Point(i, (255, 0, 0), (10, 10))
            else:
                point = Point(i, (255, 255, 255), (10, 10))
            self.points_set[point.id] = point
        self.points_used_id = {}

    def bird_eye_view(self, objects: Player, original_frame, M, frame_i, ball_coords=None, speed_tf=True):
        """
        Generate bird's eye view representation of the players and ball.

        :param objects: List of Player objects.
        :param original_frame: Original frame.
        :param M: Transformation matrix.
        :param frame_i: Frame index.
        :param ball_coords: Coordinates of the ball.
        :return: Bird's eye view representation.
        """
        dict_points = {}
        for i, obj in enumerate(objects):
            bbox = obj.detection.points
            xyxy = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = xyxy[3]
            # 颜色载入
            if obj.team is not None:
                color = obj.detection.data["color"]
            else:
                color = (0, 0, 0)
            dict_points[(x_center, y_center)] = color
        points_set, points_used_id = assign_id(dict_points, self.points_set, self.points_used_id, 200, M)
        # smooth the tracker
        predictions = self.process_frame(points_used_id)
        return self.draw_bird_eye_view(predictions, speed_tf, original_frame, M, frame_i, ball_coords)

    def draw_bird_eye_view(self, objects: dict, speed_tf, original_frame, M, frame_i, ball_coords=None):
        """
        Draw bird's eye view representation of players .
        :param objects: Dictionary containing Kalman filter objects.
        :param speed_tf: Whether to show speed or not
        :param original_frame: Original frame.
        :param M: Transformation matrix.
        :param frame_i: Frame index.
        :param ball_coords: Coordinates of the ball.
        :return: Image with bird's eye view representation.
        """
        original_frame = np.array(original_frame)
        frame = original_frame.copy()
        bg_img = self.gt_img.copy()
        for kf_id, kalman_filter in objects.items():
            center = kalman_filter.get_prediction()
            color = kalman_filter.color
            old_coords = kalman_filter.old_coords
            coords = transform_coordinates(center, M, original_frame, )
            if frame_i % 10 == 0:
                kalman_filter.speed = kalman_filter.dist * self.pix_dist / 0.4
                kalman_filter.dist = 0
            else:
                dist = euclidean_distance(coords, transform_coordinates(old_coords, M, original_frame, ))
                if dist < 150:
                    kalman_filter.dist = dist + kalman_filter.dist
            # draw player in BEV, don’t draw referee
            if color != (0, 0, 0):
                cv2.circle(bg_img, (int(coords[0]), int(coords[1])), 35, color, -1)
                cv2.circle(bg_img, (int(coords[0]), int(coords[1])), 35, (0, 0, 0), 5)
                text_size = cv2.getTextSize(str(kf_id), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 4)[0]
                text_x = int(coords[0] - text_size[0] / 2)
                text_y = int(coords[1] + text_size[1] / 2)
                text_sx = int(kalman_filter.origin_coord[0] - text_size[0] / 2)
                text_sy = int(kalman_filter.origin_coord[1] + text_size[1])
                # draw speed
                if speed_tf and 1 < kalman_filter.speed < 15:
                    str_speed = '%s%s' % (str(int(kalman_filter.speed)), 'm/s')
                    cv2.putText(frame, str_speed, (text_sx, text_sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # draw tracker id
                cv2.putText(bg_img, str(int(kf_id)), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4)
            if ball_coords:
                coords = transform_coordinates(ball_coords, M, original_frame)
                coords = (int(coords[0]), int(coords[1]))
                cv2.circle(bg_img, coords, 20, (102, 0, 102), -1)
                cv2.circle(bg_img, coords, 20, (0, 0, 0), 5)

        # Mixed pixel
        bg_img = cv2.resize(bg_img, (self.w, self.h))
        result = addWeightedSmallImgToLargeImg(frame, 1, bg_img, 0.5, 0)
        result = result[:, :, :3]
        return result

    def update_filters(self, frame_data):
        """
        Update Kalman filters with the given frame data.

        :param frame_data: Dictionary containing frame data.
        """
        for obj_id, point in frame_data.items():
            if obj_id in self.kalman_filters:
                # correct
                if point.obj_fade > 0:
                    self.kalman_filters[obj_id].getLastpred()
                    point.coord = self.kalman_filters[obj_id].get_prediction()
                    self.kalman_filters[obj_id].origin_coord = (-1, -1)
                else:
                    # update
                    self.kalman_filters[obj_id].old_coords = self.kalman_filters[obj_id].get_prediction()
                    self.kalman_filters[obj_id].origin_coord = point.coord
                    self.kalman_filters[obj_id].update(point.coord)
                    self.kalman_filters[obj_id].nodetection = 0
                    self.kalman_filters[obj_id].color = point.color
                    self.kalman_filters[obj_id].predict()
            else:
                # init kf
                self.kalman_filters[obj_id] = KalmanFilter2D(point.coord, obj_id, point.color)
                self.kalman_filters[obj_id].predict()

        # keep when detection fade
        need_del_KF = []
        for existing_id, kalman_filter in self.kalman_filters.items():
            if existing_id not in frame_data:
                need_del_KF.append(existing_id)
        for kfid in need_del_KF:
            deleted_instance = self.kalman_filters.pop(kfid, None)
            if deleted_instance:
                del deleted_instance

    def process_frame(self, frame_data):
        self.update_filters(frame_data)
        return self.kalman_filters


class KalmanFilter2D:
    """
    A class implementing a 2D Kalman filter for tracking objects.
    """

    def __init__(self, initial_measurement, filter_id, color):
        """
        Initialize the KalmanFilter2D.
        :param initial_measurement: Initial measurement for Kalman filter.
        :param filter_id: ID of the filter.
        :param color: Color associated with the filter.
        """

        # Kalman filter parameters
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4

        # Initialize state with the initial measurement
        self.kalman.statePre = np.array([[initial_measurement[0]], [initial_measurement[1]], [0], [0]],
                                        dtype=np.float32)
        self.kalman.correct(np.array([[np.float32(initial_measurement[0])], [np.float32(initial_measurement[1])]]))
        # Filter ID
        self.filter_id = filter_id
        self.nodetection = 0
        self.last_pred = None
        self.color = color
        self.old_coords = (-1, -1)
        self.origin_coord = (-1, -1)
        self.dist = 0
        self.speed = 0

    def predict(self):
        # Predict the next state
        predicted_state = self.kalman.predict()
        self.last_pred = predicted_state
        # Return the predicted coordinates
        return self.last_pred[0, 0], self.last_pred[1, 0]

    def update(self, measurement):
        # Update the state with the new measurement
        self.kalman.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))

    def get_prediction(self):
        """
        Get the current prediction without updating with a measurement.

        :return: Current prediction.
        """

        return self.last_pred[0, 0], self.last_pred[1, 0]

    def getLastpred(self):
        """
        Get the prediction with speed information when there's no detection.

        :return: Prediction with speed information.
        """

        if self.nodetection <= 70:
            predicted_state = self.kalman.predict()
            predicted_x = predicted_state[0, 0] + predicted_state[2, 0] / 4
            predicted_y = predicted_state[1, 0] + predicted_state[3, 0] / 4
            self.last_pred = np.array([[predicted_x], [predicted_y], [0], [0]],
                                      dtype=np.float32)
        self.nodetection += 1
        return self.last_pred, self.nodetection

    def __del__(self):
        print(f"Instance  is being destroyed.")


class Point:
    """
    A class representing a point in the tracking system.
    """

    def __init__(self, id, color, coord, color_max=60):
        self.id = id
        self.color = color
        self.coord = coord
        self.old_coord = None
        self.used = False
        self.frame_used = False
        self.obj_fade = 0
        self.record_color = []
        self.color_max = color_max

    def renew_all(self):
        self.used = False
        self.obj_fade = 0
        self.record_color = []
        self.frame_used = False


    def result_color_rem(self):
        color_wro_num = 0
        for color in self.record_color:
            if color != self.color:
                color_wro_num += 1
        # Color swap if too many mistakes
        if len(self.record_color) >= self.color_max:
            self.record_color.pop(0)
            if color_wro_num >= self.color_max * 2 / 3:
                return False
            else:
                return True
        # 初始化过程中,错误颜色记录超过半数返回错误
        else:
            if color_wro_num >= self.color_max / 2:
                return False
            else:
                return True

    def uppdate_coords(self, new_coords):
        self.old_coord = self.coord
        self.coord = new_coords




def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    :param point1: First point.
    :param point2: Second point.
    :return: Euclidean distance between the points.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def init_track(frame_dict, points_set: dict, points_used_id: dict, init=True):
    """
    Initialize or update tracking of points.

    :param frame_dict: Dictionary containing coordinates and colors of points.
    :param points_set: Dictionary containing existing points.
    :param points_used_id: Dictionary containing points currently being used for tracking.
    :param init: Flag indicating whether it is initialization or update.
    :return: Updated frame_dict, points_set, and points_used_id.
    """
    if init:
        for coord, color in frame_dict.items():
            min_id = None
            min_dist = 1e6
            for id, point in points_set.items():
                # If there are more existing points than used tracking IDs, select a color-matching ID to use
                if point.color == color:
                    dist = euclidean_distance(coord, point.coord)
                    if dist < min_dist:
                        min_dist = dist
                        min_id = id
            if min_id is not None:
                points_used_id[min_id] = points_set[min_id]
                points_set.pop(min_id, None)
                points_used_id[min_id].frame_used = True
                points_used_id[min_id].coord = coord
                points_used_id[min_id].record_color.append(color)

        # Ignore referees
        frame_dict = {}
    else:
        # print(f"New coordinates come in!!:{frame_dict}!!")
        for coord, color in frame_dict.items():
            # if (coord[0] > 1820 or coord[0] < 100) or (coord[1] > 900 or coord[1] < 150):
                min_id = None
                min_dist = 1e6
                for id, point in points_set.items():
                    # If there are more existing points than used tracking IDs, select a color-matching ID to use
                    if point.color == color:
                        dist = euclidean_distance(coord, point.coord)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = id
                if min_id is not None:
                    points_used_id[min_id] = points_set[min_id]
                    points_used_id[min_id].frame_used = True
                    points_used_id[min_id].coord = coord
                    points_used_id[min_id].record_color.append(color)
                    points_set.pop(min_id, None)
        frame_dict = {}
    return frame_dict, points_set, points_used_id


def mahalanobis_distance(xy, xy_pre, c, c_rem):
    """
    Calculate the Mahalanobis distance between two points considering both position and color matching.

    :param xy: Current coordinates.
    :param xy_pre: Previous coordinates.
    :param c: Current color.
    :param c_rem: Color of the point being compared.
    :return: Distance based on position, Distance based on position and color matching.
    """
    # Calculate position difference
    delta_x = xy[0] - xy_pre[0]
    delta_y = xy[1] - xy_pre[1]

    # Color matching
    color_match = 0.1 if c == c_rem else 2
    if c == (0, 0, 0):
        color_match = 100

    # Calculate Euclidean distance
    distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    dist_color = np.sqrt(delta_x ** 2 + delta_y ** 2) * color_match

    return distance, dist_color


def exchange_tracker(points_set, point_input):
    """
    Exchange the color of the input point with another point in the points_set that has the same color.

    :param points_set: Dictionary containing existing points.
    :param point_input: Point whose color needs to be exchanged.
    :return: Updated points_set and point_input.
    """
    min_id = None
    min_dist = 1e6
    for id, point in points_set.items():
        # If there are more existing points than used tracking IDs, select a color-matching ID to use
        if point.color != point_input.color:
            dist = euclidean_distance(point_input.coord, point.coord)
            if dist < min_dist:
                min_dist = dist
                min_id = id
    if min_id is not None:
        # Exchange colors
        t = points_set[min_id].color
        points_set[min_id].color = point_input.color
        point_input.color = t
    print("Color of this ID has been changed:", point_input.id)
    return points_set, point_input


def assign_id(frame_dict: dict, points_set: dict, points_used_id: dict, radius, M):
    """
    Assign IDs to points based on their coordinates in the current frame.

    :param frame_dict: Dictionary containing coordinates of points in the current frame.
    :param points_set: Dictionary containing existing points.
    :param points_used_id: Dictionary containing IDs of points used in tracking.
    :param radius: Radius for matching points.
    :param M: Transformation matrix.
    :return: Updated dictionaries points_set and points_used_id.
    """
    # Initialize and match points
    if not points_used_id:
        frame_dict, points_set, points_used_id = init_track(frame_dict, points_set, points_used_id)
    else:
        cost_matrix = []
        match_points = {}
        no_match_points = {}
        for id, point in points_used_id.items():
            row = []
            for frame_point, color in frame_dict.items():
                dist, dist_c = mahalanobis_distance(point.coord, frame_point, color, point.color)
                if dist <= radius and (frame_point[0] > 0 and frame_point[1] > 0):
                    if dist_c <= radius:
                        row.append(dist_c)
                    else:
                        row.append(1e6)
                else:
                    row.append(1e6)

            if all(row_ == 1e6 for row_ in row):
                point.obj_fade += 1
                if point.obj_fade >= 10:
                    point.renew_all()
                    points_set[id] = point
                    no_match_points[id] = point
            else:
                point.obj_fade = 0
                match_points[point.id] = point
                cost_matrix.append(row)

        if cost_matrix:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            keys_copy = list(frame_dict.keys())  # Create a copy of keys
            for i in range(len(row_ind)):
                if cost_matrix[row_ind[i]][col_ind[i]] != 1e6:
                    coords = keys_copy[col_ind[i]]
                    this_point = match_points[list(match_points.keys())[row_ind[i]]]
                    this_point.uppdate_coords(coords)
                    this_point.record_color.append(frame_dict[coords])
                    if not this_point.result_color_rem():
                        points_set, this_point = exchange_tracker(points_set, this_point)
                    frame_dict.pop(coords)
            if frame_dict:
                frame_dict, points_set, points_used_id = init_track(frame_dict, points_set, points_used_id, init=False)
        # Remove trackers that cannot find targets
        for id, point in no_match_points.items():
            points_used_id.pop(id, None)
            points_set[id]=point
    return points_set, points_used_id


def addWeightedSmallImgToLargeImg(frame, alpha, smallImg, beta, gamma=0.0):
    """
    Add a small image to a larger image with alpha blending.

    :param frame: Large image (destination).
    :param alpha: Weight of the first image (frame).
    :param smallImg: Small image (source).
    :param beta: Weight of the second image (smallImg).
    :param gamma: Scalar added to each sum.
    :return: Resulting image after blending.
    """
    srcW, srcH = frame.shape[1::-1]
    refW, refH = smallImg.shape[1::-1]

    # Place smallImg at the center bottom of frame
    x = (srcW - refW) // 2
    y = (srcH - refH) // 2 + srcH // 3 - 100  # Center bottom position

    if (refW > srcW) or (refH > srcH):
        raise ValueError(
            f"img2's size {smallImg.shape[1::-1]} must be less than or equal to img1's size {frame.shape[1::-1]}")
    else:
        if (x + refW) > srcW:
            x = srcW - refW
        if (y + refH) > srcH:
            y = srcH - refH
        destImg = np.array(frame)
        tmpSrcImg = destImg[y:y + refH, x:x + refW]
        tmpImg = cv2.addWeighted(tmpSrcImg, alpha, smallImg, beta, gamma)
        destImg[y:y + refH, x:x + refW] = tmpImg
        return destImg


def detect_circle_diameter(bevimg):
    """
    Detect the diameter of the largest circle in the bird's eye view image.

    :param bevimg: Bird's eye view image.
    :return: Diameter of the largest circle in pixels.
    """
    # Read the image
    gt_img = bevimg
    gt_h, gt_w, _ = gt_img.shape
    gt_h_h = int(gt_h)
    gt_w_h = int(gt_w)
    # Resize the image
    image = cv2.resize(gt_img, (gt_w_h, gt_h_h))
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # Detect circles
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=100, maxRadius=300)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Find the largest circle
        largest_circle = max(circles, key=lambda x: x[2])
        # Calculate diameter
        diameter_pixels = largest_circle[2] * 2
        return diameter_pixels
    else:
        print("Circle not detected. Please check the image or adjust parameters.")
        return None

# old
# def transform_matrix(matrix, p, vid_shape, gt_shape):
#     p = (p[0] * 1280 / vid_shape[1], p[1] * 720 / vid_shape[0])
#     px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
#         (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
#     py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
#         (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
#     p_after = (int(px * gt_shape[1] / 115), int(py * gt_shape[0] / 74))
#     return p_after


def transform_coordinates(bbox, M, original_frame=None):
    x_center, y_center = bbox[0], bbox[1]
    pts3 = np.array([[x_center, y_center]], dtype=np.float32)  # 将pts3改为形如[[x, y]]的数组
    pts3 = np.array([pts3])  # 包装成包含单个点的数组
    projected_coords = cv2.perspectiveTransform(pts3, M)  # 2D坐标
    newx = projected_coords[0, 0, 0]
    newy = projected_coords[0, 0, 1]
    coords_tradition = (newx, newy)
    # old way
    # coords = self.transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
    return coords_tradition
