import numpy as np
import cv2 as cv

# i = ncamera --> background video, extrinsics, video
cameras_videos_info = np.array([
    ["./4persons/background/Take26.54389819.20141124164130.avi", "./4persons/extrinsics/Take25.54389819.20141124164119.avi",
        "./4persons/video/Take30.54389819.20141124164749.avi"],
    ["./4persons/background/Take26.59624062.20141124164130.avi", "./4persons/extrinsics/Take25.59624062.20141124164119.avi",
        "./4persons/video/Take30.59624062.20141124164749.avi"],
    ["./4persons/background/Take26.60703227.20141124164130.avi", "./4persons/extrinsics/Take25.60703227.20141124164119.avi",
        "./4persons/video/Take30.60703227.20141124164749.avi"],
    ["./4persons/background/Take26.62474905.20141124164130.avi", "./4persons/extrinsics/Take25.62474905.20141124164119.avi",
        "./4persons/video/Take30.62474905.20141124164749.avi"]
])


def show_image(img, name="chessboard"):
    """
    It takes an image and displays it in a window

    :param img: The image to be processed
    :param name: The name of the window, defaults to chessboard (optional)
    """
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_best_mask(ground_truth, foreground_hsv, w, h):
    """
    It iterates over all possible combinations of hue, saturation and value thresholds, and returns the
    mask that has the least amount of differences with the ground truth

    :param ground_truth: the ground truth mask
    :param foreground_hsv: the HSV image of the foreground
    :return: The best mask for the foreground.
    """
    best_differences = ground_truth.shape[0] * ground_truth.shape[1]
    best_mask = np.zeros((w, h), dtype=np.uint8)
    for hue in range(0, 11):
        for saturation in range(0, 11, 5):
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

backgrounds = []

for camera_i in range(4):
    # read bacground video
    cap = cv.VideoCapture(cameras_videos_info[camera_i][0])

    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    background_pixels = np.zeros((h, w, 3), dtype=np.float32)


    n_frames_background = 40

    for frame_i in range(n_frames_background):
        if cap.isOpened():
            retF, frame = cap.read()

            if not retF:
                print(f"ERROR! Cap closed after {frame_i} frames")

            background_pixels += np.array(frame, dtype=np.float32)

    cap.release()
    background = np.array(
        background_pixels / n_frames_background, dtype=np.uint8)

    show_image(background, f"Cam{camera_i+1} Background")
    backgrounds.append(background)
    
