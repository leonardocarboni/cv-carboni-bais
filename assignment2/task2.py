import numpy as np
import cv2 as cv

background_frames = 100



background_pixels = np.zeros((486, 644, 3), dtype = np.float32)
for camera_i in range(1, 5):
    with np.load(f'./data/cam{camera_i}/config.npz') as file:
        camera_matrix, dist_coeffs, rvec_extr, tvec_extr, R = [file[i] for i in ['camera_matrix', 'dist_coeffs', 'rvec_extr', 'tvec_extr', 'R']]
        cap = cv.VideoCapture(f"./data/cam{camera_i}/background.avi")
        n_frame = 0
        while n_frame < background_frames:
            retF, frame = cap.read()
            if retF:
                height, width, channels = frame.shape
                #print(height, width)
                for x in range(0, width) :
                    for y in range(0, height) :
                        background_pixels[x, y, 0] = frame[x,y,0] #B Channel Value
                        background_pixels[x, y, 1] = frame[x,y,1] #G Channel Value
                        background_pixels[x, y, 2] = frame[x,y,2] #R Channel Value
