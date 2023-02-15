import numpy as np
import cv2 as cv
CHESSBOARD_VERTICES = (9, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
po = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

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

def show_image(img, name="chessboard"):
    """
    It takes an image and displays it in a window

    :param img: the image to be processed
    :param name: The name of the window in which image to be displayed, defaults to chessboard
    (optional)
    """
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


print("Select traning: ")
print("1. All images")
print("2. 10 images where corners are found automatically")
print("3. 5 images where corners are found automatically")
training_i = input("Select the training: ")

mtx, dist = [], []

if training_i == "1" or training_i == "2" or training_i == "3":
    with np.load(f'camera_matrix_Run{training_i}.npz') as file:
        mtx, dist = [file[i] for i in ['mtx', 'dist']]

print("Select action: ")
print("1. Webcam with cube")
print("2. Webcam with cube and corners")
print("3. Images with cube without correction")
print("4. Images with cube and corners without correction")
print("5. Images with cube with correction")
print("6. Images with cube and corners with correction")
action_i = input("Select what to do: ")


def find_chessboard(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to gray scale
    ret, corners = cv.findChessboardCorners(
        gray_img, CHESSBOARD_VERTICES)  # find all corners automatically
    if ret:  # corners found
        corners = cv.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)  # augment precision of corners
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


if action_i == "1" or action_i == "2":
    vid = cv.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        retC, corners = find_chessboard(frame)
        if retC:
            _, rvec, tvec, _ = cv.solvePnPRansac(
                po, corners, mtx, dist)
            vp, _ = cv.projectPoints(cube_vertices(
                0, 0, 0, 2), rvec, tvec, mtx, dist)
            cv.imshow('frame', draw_cube(frame, vp.round().astype(np.int32)))
        else:
            cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()

if action_i == "3" or action_i == "4" or action_i == "5" or action_i == "6":
    for i in range(1, 31):
        img = cv.imread(
            f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # non sono sicuro di questo perché le foto vengono undistortate tutte nello stesso modo
        if action_i == "5" or action_i == "6":
            alpha = 0.9
            mtx2, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha)
            img_ret = cv.undistort(img, mtx, dist, None, mtx2)

            test = np.zeros((10, 1, 2), dtype=np.float32)
            dPt = cv.undistortPoints(test, mtx, dist, None, mtx2)
            dPt = dPt.round().astype(int)
            cv.polylines(img_ret, [dPt], True, (0, 0, 255))
            show_image(img_ret)

        # _, rvec, tvec, _ = cv.solvePnPRansac(
        #     punti_oggetto[i-1], corners_list[i-1], matIntr_run1, distCoeff_run1)
        # vp, _ = cv.projectPoints(cube_vertices(
        #     0, 0, 0, 2), rvec, tvec, matIntr_run1, distCoeff_run1)
        # res = draw_cube(tmp, vp.round().astype(np.int32))
        # show_image(res)
