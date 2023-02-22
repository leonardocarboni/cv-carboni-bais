import numpy as np
import cv2 as cv

for camera_i in range(1, 5):
    s = cv.FileStorage(f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
    camera_matrix = s.getNode('camera_matrix').mat()
    dist_coeffs = s.getNode('dist_coeffs').mat()
    tvec_extr = s.getNode('tvec_extr').mat()
    R = s.getNode('R_MAT').mat()

    s.release()