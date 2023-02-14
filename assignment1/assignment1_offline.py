import cv2 as cv
import numpy as np

# number of horizontal and vertical edges
CHESSBOARD_VERTICES = (9, 6)

# termination criteria for subpixel corner detection algorithm
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# list of all detected corners in the chessboard
corners_list = []

# 3D coordinates of all corners for camera calibration
punti_oggetto = []
po = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

# callback function to manually annotate the four outmost corner points 
def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN and len(edges) < 4:
        edges.append((x, y))
        print(f"Edge set: ({x}, {y}), Select {4-len(edges)} more corners")

# function to automatically detect all corners of the chessboard, with augmented precision of subpixel detection
def find_chessboard(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to gray scale
    ret, corners = cv.findChessboardCorners(gray_img, CHESSBOARD_VERTICES) # find all corners automatically
    if ret: # corners found
        corners = cv.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA) # augment precision of corners
    return ret, corners


def cube_vertices(x, y, z, s):
    """
    It takes the x, y, and z coordinates of the cube's center, and the cube's size, and returns the
    coordinates of the cube's vertices

    :param x: x coordinate of the cube
    :param y: y-coordinate of the center of the cube
    :param z: the z-coordinate of the cube
    :param s: size of the cube
    :return: The vertices of a cube.
    """
    return np.float32([[x, y, z], [x+s, y, z], [x+s, y+s, z], [x, y+s, z],
                       [x, y, z-s], [x+s, y, z-s], [x+s, y+s, z-s], [x, y+s, z-s]])


def draw_cube(img, v):
    """
    It draws a cube on the image

    :param img: The image to draw on
    :param v: the vertices of the cube
    """
    image = np.copy(img)
    cv.polylines(image, [v[:4]], True, (0, 255, 0), 5)
    cv.polylines(image, np.array([v[i::4]
                 for i in range(4)]), False, (0, 0, 255), 5)
    cv.polylines(image, [v[4:8]], True, (255, 0, 0), 5)
    return image

# function to display an image
def show_image(img, name="chessboard"):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# function to estimate internal corner positions through linear interpolation
def interpolate_corners(image, edges):

    horizontal_squares = CHESSBOARD_VERTICES[0] - 1 # number of squares for each row
    vertical_squares = CHESSBOARD_VERTICES[1] - 1 # number of squares for each column

    # size of the rectified image
    dst_size = (horizontal_squares * 200, vertical_squares * 200)

    # Define the corners of the rectified image
    dst_corners = np.array([[0, 0], [dst_size[0], 0], [0, dst_size[1]], [
                           dst_size[0], dst_size[1]]], dtype=np.float32)
    #TODO what is the point of having two?
    rectified_corners = np.array([[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]], dtype=np.float32)
    
    # Compute the perspective transformation matrix
    M = cv.getPerspectiveTransform(
        np.array(edges, dtype=np.float32), dst_corners)

    # all corners coordinates in the rectified image
    transformed_vertices = np.zeros(
        (CHESSBOARD_VERTICES[1], CHESSBOARD_VERTICES[0], 2))

    for i in range(CHESSBOARD_VERTICES[1]): # for each row of the chessboard
        for j in range(CHESSBOARD_VERTICES[0]): # for each column of the chessboard
            transformed_vertices[i, j, 0] = rectified_corners[0][0] + j * 200 # store the estimated x coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j, 1] = rectified_corners[0][1] + i * 200 # store the estimated y coordinate of the (i,j) edge using linear interpolation

    # extract the inverse transformation matrix
    M_inv = np.linalg.inv(M) 

    # invert the rectified corners to map in the original image
    original_vertices = cv.perspectiveTransform(
        transformed_vertices.reshape(1, -1, 2), M_inv)
    original_vertices = original_vertices.reshape(-1, 2)

    # TODO: Make this work
    # original_vertices = cv.cornerSubPix(
    #         image, original_vertices, (11, 11), (-1, -1), TERMINATION_CRITERIA)
    
    return np.array(original_vertices, dtype=np.float32)


# corner detection phase
for i in range(1, 31): # for each training image
    img = cv.imread(
        f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY) # read image
    h, w = img.shape[:2] # extract height and with

    ret, corners = find_chessboard(img) # get corners

    # per ora teniamo
    # if i == 23:
    #     cornerizzata = img.copy()
    #     cv.drawChessboardCorners(
    #         cornerizzata, CHESSBOARD_VERTICES, corners, True)
    #     show_image(cornerizzata)
    #     print(corners)

    if not ret: # if the corners were not automatically detected
        edges = [] # list of outmost corners to be annotated
        print(f"Corners not in image {i} found, Select them manually")
        cv.namedWindow("chessboard", cv.WINDOW_NORMAL)
        cv.imshow("chessboard", img)
        cv.setMouseCallback('chessboard', click_event) # initiate manual annotation
        cv.waitKey(0)
        cv.destroyAllWindows()
        corners = np.array(interpolate_corners(img, edges))[:, np.newaxis] # interpolate the internal corners given the 4 manually annotated ones
        # corners = interpolate_corners(edges)

    print(f"{i}: n corners = {len(corners)}")
    punti_oggetto.append(po) # store 3d coordinates for calibration
    corners_list.append(corners) # store detected corners for calibration


# Training Phase

# Run 1 (all images)
err_run1, matIntr_run1, distCoeff_run1, rotEstr_run1, traEstr_run1 = cv.calibrateCamera(
    punti_oggetto, corners_list, (w, h), None, None)
np.savez('camera_matrix_Run1',mtx=matIntr_run1,dist=distCoeff_run1) # store parameters for online phase
# Run 2 (only 10)
err_run2, matIntr_run2, distCoeff_run2, rotEstr_run2, traEstr_run2 = cv.calibrateCamera(
    punti_oggetto[:10], corners_list[:10], (w, h), None, None)
np.savez('camera_matrix_Run2',mtx=matIntr_run2,dist=distCoeff_run2) # store parameters for online phase
# Run 3 (only 5)
err_run3, matIntr_run3, distCoeff_run3, rotEstr_run3, traEstr_run3 = cv.calibrateCamera(
    punti_oggetto[:5], corners_list[:5], (w, h), None, None)
np.savez('camera_matrix_Run3',mtx=matIntr_run3,dist=distCoeff_run3) # store parameters for online phase


