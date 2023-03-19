import cv2 as cv
import numpy as np

centers_list = np.load('centers_list.npz')['centers_list']
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

paths = np.zeros((500, 500, 3), dtype=np.uint8)
paths1 = np.zeros((500, 500, 3), dtype=np.uint8)
paths2 = np.zeros((500, 500, 3), dtype=np.uint8)
paths3 = np.zeros((500, 500, 3), dtype=np.uint8)
paths4 = np.zeros((500, 500, 3), dtype=np.uint8)
paths5 = np.zeros((500, 500, 3), dtype=np.uint8)

old_c = []
for c_in_frame in centers_list:
    for label, c in enumerate(c_in_frame):
        # normalize the coordinates to be in a 500 x 500 image
        x = (c[0] + 40) / 128 * 700
        y = (c[1] + 50) / 128 * 700
        # normalize the coordinates to be in a 500 x 500 image

        if len(old_c) > 0:
            old_x = (old_c[label][0] + 40) / 128 * 700
            old_y = (old_c[label][1] + 50) / 128 * 700
            dist = np.sqrt((x - old_x) ** 2 + (y - old_y) ** 2)
            if dist < 150:
                cv.line(paths5, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)

        cv.circle(paths, (int(x), int(y)), 3, labels_to_color[label], -1)

        if label == 0:
            cv.circle(paths1, (int(x), int(y)), 3, labels_to_color[label], -1)
        elif label == 1:
            cv.circle(paths2, (int(x), int(y)), 3, labels_to_color[label], -1)
        elif label == 2:
            cv.circle(paths3, (int(x), int(y)), 3, labels_to_color[label], -1)
        elif label == 3:
            cv.circle(paths4, (int(x), int(y)), 3, labels_to_color[label], -1)

            # if len(old_c) > 0:
        #     # calculate the distance between the current and the previous position
        #     dist = np.sqrt((x - old_x)**2 + (y - old_y)**2)
        #
        #     if dist < 10:
        #         cv.line(paths2, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)
        #     if dist < 20:
        #         cv.line(paths3, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)
        #     if dist < 8:
        #         cv.line(paths4, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)
    old_c = c_in_frame

cv.imwrite("paths_v2.png", paths)
cv.imwrite("paths_only_1.png", paths1)
cv.imwrite("paths_only_2.png", paths2)
cv.imwrite("paths_only_3.png", paths3)
cv.imwrite("paths_only_4.png", paths4)
cv.imwrite("paths_v3.png", paths5)