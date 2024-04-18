from perspective_transform.camera_pose_estimation.main import calibrate_from_image, display_top_view, \
    homography_top_view
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


class Perspective_transform:
    def __init__(self, guess_fx=4000, guess_rot=np.array([[0.25, 0, 0]]), guess_trans=(0, 0, 80)):
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans
        self.M = np.zeros((3, 3))
        self.last_M = np.zeros((3, 3))
        self.kf = None

    def calibrate_and_display(self, img):
        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, self.guess_fx, self.guess_rot, self.guess_trans
        )
        unskewd = display_top_view(K, to_device_from_world, img)
        cv2.imshow("unskewd", unskewd)
        cv2.waitKey(0)

    def find_homography(self, img):
        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, self.guess_fx, self.guess_rot, self.guess_trans
        )
        if to_device_from_world is not None:
            if np.all(self.last_M == 0):
                # first M
                self.M = homography_top_view(K, to_device_from_world, img)
                self.kf = self.init_kf(self.M)
                self.last_M = self.M
            else:
                M = homography_top_view(K, to_device_from_world, img)
                if self.frobenius_norm(self.M,M) > 100:
                    self.M = self.last_M
                    return self.M
                if self.kf is not None:
                    self.M = self.homog_kf(M)
                else:
                    self.M = M
            print("Homography:\n", self.M)
        else:
            self.M = self.last_M
        return self.M

    def init_kf(self, initial_matrix, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize the Kalman Filter.

        Parameters:
            initial_matrix (numpy.ndarray): The initial homography matrix, a (3x3) numpy array.
            process_noise (float): Standard deviation of process noise for the Kalman Filter.
            measurement_noise (float): Standard deviation of measurement noise for the Kalman Filter.

        Returns:
            kf (filterpy.kalman.KalmanFilter): Initialized Kalman Filter object.
        """
        kf = KalmanFilter(dim_x=9, dim_z=9)
        kf.F = np.eye(9)  # State transition matrix
        kf.H = np.eye(9)  # Measurement matrix
        kf.P *= 10  # Initial covariance matrix
        kf.R *= measurement_noise ** 2  # Measurement noise covariance matrix
        kf.Q *= process_noise ** 2  # Process noise covariance matrix
        kf.x = initial_matrix.flatten()  # Initialize state vector
        return kf

    def homog_kf(self, new_matrix):
        """
        Smooth the homography matrix using Kalman Filter.

        Parameters:
            new_matrix (numpy.ndarray): The new homography matrix, a (3x3) numpy array.
            kf (filterpy.kalman.KalmanFilter): Initialized Kalman Filter object.

        Returns:
            smoothed_matrix (numpy.ndarray): Homography matrix after smoothing.
        """
        # Flatten the homography matrix into a 1D vector
        new_matrix_vector = new_matrix.flatten()

        # Predict and update using the Kalman Filter
        self.kf.predict()
        # Increase the standard deviation of observation noise
        self.kf.R *= 10  # Example: increase by a factor of 10
        self.kf.update(new_matrix_vector)

        # Get the smoothed homography matrix from the Kalman Filter
        smoothed_matrix_vector = self.kf.x
        smoothed_matrix = smoothed_matrix_vector.reshape((3, 3))

        return smoothed_matrix

    def frobenius_norm(self, matrix1, matrix2):
        """
        Calculate the Frobenius norm between two matrices.

        Parameters:
            matrix1 (numpy.ndarray): The first matrix.
            matrix2 (numpy.ndarray): The second matrix.

        Returns:
            norm (float): Frobenius norm between the two matrices.
        """
        diff_matrix = matrix1 - matrix2
        norm = np.linalg.norm(diff_matrix, ord='fro')
        return norm


# 使用示例
if __name__ == "__main__":
    img = cv2.imread("test_bev.png")
    camera_pose = Perspective_transform(4000, np.array([[0.25, 0, 0]]), (0, 0, 80))
    camera_pose.calibrate_and_display(img)
    M = camera_pose.find_homography(img)
    print(M)
