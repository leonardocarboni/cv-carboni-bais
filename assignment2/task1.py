import cv2 as cv
import numpy as np

CHESSBOARD_VERTICES = (8, 6)

corners_list = []
object_points = []
op = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

for camera_i in range(1, 5):
    path = f"./data/cam{camera_i}/intrinsics.avi"
    cap = cv.VideoCapture(path)
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while cap.isOpened():
        retF, frame = cap.read()
        if retF:
            retC, corners = cv.findChessboardCorners(
                frame, CHESSBOARD_VERTICES)
            if retC:
                corners_list.append(corners)
                object_points.append(op)
            print(i)
            i+=1
        else:
            print("aaaa")
            break
    cap.release()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    object_points, corners_list, (w, h), None, None)

np.savez('camera_info', ret=ret, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
