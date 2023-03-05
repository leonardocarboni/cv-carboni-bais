import numpy as np
import cv2 as cv
from utils import *


backgrounds = []

show = False

# edges from camera 4, found manually and saved
edges_cam4 = [(252, 363), (313, 327), (310, 388), (368, 343)]

for camera_i in range(4):
    image_points = []
    object_points = []

    # calibration
    cap = cv.VideoCapture(cameras_videos_info[camera_i][1])
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    exampleFrame = []
    for i in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2):
        retF, img = cap.read()
        if i == 0:
            exampleFrame = img.copy()

        corners = []

        # use interpolation given edges for cam4
        if camera_i == 3:
            corners = interpolate_corners(img, edges_cam4)
        else:
            retC, corners = cv.findChessboardCorners(img, CHESSBOARD_VERTICES)

        # swap order of rows for cam3
        if camera_i == 2:
            corners = corners[::-1]

        image_points.append(corners.squeeze())
        object_points.append(op)

        if show:
            chessboard_copy = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(
                chessboard_copy, CHESSBOARD_VERTICES, corners, retC)
            show_image(chessboard_copy, f"Cam{camera_i} Chessboard Vertices")

    # Calibrate the camera with the intrinsics obtained by the analysis
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, (w, h), None, None)

    image_points = np.array(
        image_points, dtype=np.float32)
    object_points = np.array(
        object_points, dtype=np.float32)

    retval_extr, rvec_extr, tvec_extr = cv.solvePnP(
        object_points[0], image_points[0], camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    print(retval_extr)

    R, _ = cv.Rodrigues(rvec_extr)
    vp_axis, _ = cv.projectPoints(
        axis, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
    frame = draw_axis(exampleFrame, corners.round().astype(
        np.int32), vp_axis.round().astype(np.int32))
    show_image(frame, "Axis on Chessboard")

    # # save the config file for the camera
    # s = cv.FileStorage(f"data/cam{camera_i}/config.xml", cv.FileStorage_WRITE)
    # s.write('camera_matrix', camera_matrix)
    # s.write('dist_coeffs', dist_coeffs)
    # s.write('tvec_extr', tvec_extr)
    # s.write('rvec_extr', rvec_extr)
    # s.write('R_MAT', R)
    # s.release()

    # read bacground video
    cap = cv.VideoCapture(cameras_videos_info[camera_i][0])
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames_background = 40
    imgs_list_background = [cap.read()[1] for i in range(n_frames_background)]
    cap.release()

    # get average of baground
    background = average_images(imgs_list_background)
    backgrounds.append(background)

    if show:
        show_image(background, f"Cam{camera_i+1} Background")
