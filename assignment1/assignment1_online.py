import numpy as np
import cv2 as cv

# function to display an image
def show_image(img, name="chessboard"):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
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


with np.load('corners.npz') as file:
    corners_list, punti_oggetto=[file[i] for i in ['corners','punti_oggetto']]

with np.load('camera_matrix_Run1.npz') as file:
    matIntr_run1, distCoeff_run1=[file[i] for i in ['mtx','dist']]

with np.load('camera_matrix_Run2.npz') as file:
    matIntr_run2, distCoeff_run2=[file[i] for i in ['mtx','dist']]

with np.load('camera_matrix_Run3.npz') as file:
    matIntr_run3, distCoeff_run3=[file[i] for i in ['mtx','dist']]


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