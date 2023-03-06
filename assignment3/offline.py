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
    print(f"doing camera {camera_i+1}")
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
        image_points.append(corners)
        object_points.append(op)

        if show:
            chessboard_copy = img.copy()
            cv.drawChessboardCorners(
                chessboard_copy, CHESSBOARD_VERTICES, corners, retC)
            show_image(chessboard_copy, f"Cam{camera_i+1} Chessboard Vertices")
        break

    # load intrinsics from assignment 2
    s = cv.FileStorage(
            f"data/cam{camera_i+1}/config.xml", cv.FileStorage_READ)
    camera_matrix = s.getNode('camera_matrix').mat()
    dist_coeffs = s.getNode('dist_coeffs').mat()
    s.release()
    image_points = np.array(
        image_points, dtype=np.float32)
    object_points = np.array(
        object_points, dtype=np.float32)
    
    # get extrinsics
    retval_extr, rvec_extr, tvec_extr = cv.solvePnP(
        object_points[0], image_points[0], camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # drawing axes 
    vp_axis, _ = cv.projectPoints(
        axis, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
    frame = draw_axis(img, corners.round().astype(
        np.int32), vp_axis.round().astype(np.int32))
    if show:
        show_image(frame, "Axis on Chessboard")
    
    R, _ = cv.Rodrigues(rvec_extr)

    

    # save the config file for the camera
    s = cv.FileStorage(f"data/cam{camera_i+1}/config.xml", cv.FileStorage_WRITE)
    s.write('camera_matrix', camera_matrix)
    s.write('dist_coeffs', dist_coeffs)
    s.write('tvec_extr', tvec_extr)
    s.write('rvec_extr', rvec_extr)
    s.write('R_MAT', R) 
    s.release()

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

    # masks extraction
    cap = cv.VideoCapture(cameras_videos_info[camera_i][-1])
    all_masks = []
    for i in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2): # 2724
        retF, frame = cap.read()
        if retF and i == 580:
            w, h, _ = frame.shape
            frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            background_pixels_hsv = cv.cvtColor(background, cv.COLOR_BGR2HSV)
            foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)

            hue, saturation, value = best_masks[str(camera_i+1)]
            best_mask = np.zeros((w, h), dtype=np.uint8)
            for x in range(foreground_hsv.shape[0]):
                for y in range(foreground_hsv.shape[1]):
                    if foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                        best_mask[x, y] = 255
            show_image(best_mask, "MASK")

            # best_mask = cv.morphologyEx(
            #     best_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
            best_mask = cv.morphologyEx(
                best_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
            output = cv.bitwise_and(frame, frame, mask=best_mask)
            all_masks.append(best_mask)
            show_image(best_mask, "MASK")
            np.savez(f"data/cam{camera_i+1}/masks", masks=all_masks)
            break
    cap.release()
    # np.savez(f"data/cam{camera_i+1}/masks", masks=all_masks)
        
