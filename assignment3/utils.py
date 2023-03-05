import numpy as np
import cv2 as cv

CHESSBOARD_VERTICES = (8, 6)

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

dim_square = 115

# axis coordinates
axis = np.float32([[dim_square * 3, 0, 0], [0, dim_square * 3,
                  0], [0, 0, -dim_square * 3]]).reshape(-1, 3)


op = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32) * dim_square


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


def draw_axis(img, corners, vp):
    """
    It draws the three axes of the plane on the image

    :param img: The image to draw the axis on
    :param corners: The corners of the chessboard
    :param vp: vanishing points
    :return: The image with the axis
    """
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(vp[0].ravel()), (255, 255, 0), 2)
    img = cv.line(img, corner, tuple(vp[1].ravel()), (0, 255, 255), 2)
    img = cv.line(img, corner, tuple(vp[2].ravel()), (255, 0, 255), 2)
    return img


def average_images(imgs_list):
    # background_pixels = np.zeros((h, w, 3), dtype=np.float32)
    return np.array(np.sum(np.array(imgs_list, dtype=np.float32), axis=0) / len(imgs_list), dtype=np.uint8)
    # for img in imgs_list:
    #     background_pixels += np.array(img, dtype=np.float32)

    # return np.array(
    #     background_pixels / len(imgs_list), dtype=np.uint8)


def interpolate_corners(image, image_edges):
    """
    It estimates internal corner positions through linear interpolation

    :param image: the image 
    :param image_edges: the edges of the chessboard in the original image
    :return: the coordinates of the corners of the chessboard in the original image.
    """

    # fouth camera
    horizontal_squares = CHESSBOARD_VERTICES[0] - 1
    vertical_squares = CHESSBOARD_VERTICES[1] - 1

    # size of the rectified image
    dst_size = (horizontal_squares * dim_square,
                vertical_squares * dim_square)

    # Define the corners of the rectified image
    dst_corners = np.array([[0, 0], [dst_size[0], 0], [0, dst_size[1]], [
                           dst_size[0], dst_size[1]]], dtype=np.float32)

    rectified_corners = np.array([[0, 0], [dst_size[0], 0], [
                                 dst_size[0], dst_size[1]], [0, dst_size[1]]], dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv.getPerspectiveTransform(
        np.array(image_edges, dtype=np.float32), dst_corners)

    # all corners coordinates in the rectified image
    transformed_vertices = np.zeros(
        (CHESSBOARD_VERTICES[1], CHESSBOARD_VERTICES[0], 2))

    # for each row of the chessboard
    for i in range(CHESSBOARD_VERTICES[1]):
        # for each column of the chessboard
        for j in range(CHESSBOARD_VERTICES[0]):
            # store the estimated x coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j,
                                 0] = rectified_corners[0][0] + j * dim_square
            # store the estimated y coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j,
                                 1] = rectified_corners[0][1] + i * dim_square

    # extract the inverse transformation matrix
    M_inv = np.linalg.inv(M)

    # invert the rectified corners to map in the original image
    original_vertices = np.float32(cv.perspectiveTransform(
        transformed_vertices.reshape(1, -1, 2), M_inv))
    original_vertices = original_vertices.reshape(-1, 2)

    return np.array(original_vertices, dtype=np.float32)


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
