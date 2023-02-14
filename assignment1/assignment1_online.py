import numpy as np
import cv2 as cv

for i in range(1, 31):
    img = cv.imread(
        f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
    # alpha = 0.9
    # matIntr2, _ = cv.getOptimalNewCameraMatrix(
    #     matIntr_run2, distCoeff_run2, (w, h), alpha)
    # img_ret = cv.undistort(img, matIntr_run2, distCoeff_run2, None, matIntr2)

    # points = np.array([punti_test], np.float32)
    # dPt = cv.undistortPoints(points, matIntr, distCoeff, None, matIntr2)
    # dPt = dPt.round().astype(int)
    # cv.polylines(img_ret, [dPt], True, (0, 0, 255))
    tmp = img.copy()
    _, rvec, tvec, _ = cv.solvePnPRansac(
        punti_oggetto[i-1], corners_list[i-1], matIntr_run1, distCoeff_run1)
    vp, _ = cv.projectPoints(cube_vertices(
        0, 0, 0, 2), rvec, tvec, matIntr_run1, distCoeff_run1)
    res = draw_cube(tmp, vp.round().astype(np.int32))
    show_image(res)