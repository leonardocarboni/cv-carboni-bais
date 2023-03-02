import numpy as np
import cv2 as cv

num_frames = 50

background_pixels = np.zeros((486, 644, 3), dtype=np.float32)

# known values for best threshold after search, just to make execution go faster
best_masks = {'1': (2, 0, 15), '2': (10, 0, 30),
              '3': (4, 0, 30), '4': (1, 0, 25)}

find = input(
    "Do you want to run the algorithm to find the best values (will take time) [y/n]: ")


def find_best_mask(ground_truth, foreground_hsv):
    best_differences = ground_truth.shape[0] * ground_truth.shape[1]
    best_mask = np.zeros((w, h), dtype=np.uint8)
    for hue in range(0, 11):
        for saturation in range(0, 6, 5):
            for value in range(0, 31, 5):
                mask = np.zeros((w, h), dtype=np.uint8)
                for x in range(foreground_hsv.shape[0]):
                    for y in range(foreground_hsv.shape[1]):
                        if foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                            mask[x, y] = 255
                differences = np.sum(cv.bitwise_xor(ground_truth, mask)) / 255
                if differences < best_differences:
                    print(hue, saturation, value)
                    best_mask = mask
                    best_differences = differences
    print(best_differences)
    return best_mask


for camera_i in range(1, 5):
    background_pixels = np.zeros((486, 644, 3), dtype=np.float32)
    with np.load(f'./data/cam{camera_i}/config.npz') as file:
        camera_matrix, dist_coeffs, rvec_extr, tvec_extr, R = [file[i] for i in [
            'camera_matrix', 'dist_coeffs', 'rvec_extr', 'tvec_extr', 'R']]
        cap = cv.VideoCapture(f"./data/cam{camera_i}/background.avi")
        for n_frame in range(num_frames):
            retF, frame = cap.read()
            if retF:
                height, width, channels = frame.shape
                for y in range(0, height):
                    for x in range(0, width):
                        background_pixels[y, x] += frame[y, x]

        cap.release()

        background_pixels = np.array(
            background_pixels / num_frames, dtype=np.uint8)

        cap = cv.VideoCapture(f"./data/cam{camera_i}/video.avi")

        ground_truth = cv.imread(
            f"./data/cam{camera_i}/ground_truth.jpg", cv.IMREAD_GRAYSCALE)
        all_masks = []
        for n_frame in range(num_frames):
            retF, frame = cap.read()
            if retF:
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                w, h, _ = frame.shape
                background_pixels_hsv = cv.cvtColor(
                    background_pixels, cv.COLOR_BGR2HSV)
                foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)

                # call to use known value, uncomment to go faster
                best_mask = np.zeros((w, h), dtype=np.uint8)

                if find == "y":
                    best_mask = find_best_mask(ground_truth, foreground_hsv)
                else:
                    hue, saturation, value = best_masks[str(camera_i)]
                    for x in range(foreground_hsv.shape[0]):
                        for y in range(foreground_hsv.shape[1]):
                            if foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                                best_mask[x, y] = 255

                # cv.imshow('contours', best_mask)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                best_mask = cv.morphologyEx(
                    best_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
                best_mask = cv.morphologyEx(
                    best_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))

                # leave only biggest connected component in best mask
                nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(
                    best_mask)
                biggest_component = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1
                final_mask = np.zeros((w, h), dtype=np.uint8)
                final_mask[labels == biggest_component] = 255

                # cv.imshow('final mask', final_mask)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                output = cv.bitwise_and(frame, frame, mask=final_mask)
                # cv.imshow('output', output)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                all_masks.append(final_mask)
        np.savez(f"data/cam{camera_i}/masks", masks=all_masks)
                
