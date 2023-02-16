import numpy as np
import cv2 as cv
from scipy.stats import sem

CHESSBOARD_VERTICES = (9, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
po = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

print("Select traning: ")
print("1. All images")
print("2. 10 images where corners are found automatically")
print("3. 5 images where corners are found automatically")
training_i = input("Select the training: ")

mtx, dist = [], []

with np.load(f'corners.npz') as file:
    corners_list, punti_oggetto = [file[i]
                                   for i in ['corners', 'punti_oggetto']]

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
print("7. Test the three runs with video")
print("8. Shoot new video for video test")
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


def show_image(img, name="chessboard"):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


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

            if action_i == "2":
                cv.drawChessboardCorners(
                    frame, CHESSBOARD_VERTICES, corners, True)
            cv.imshow('frame', draw_cube(frame, vp.round().astype(np.int32)))
        else:
            cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()

if action_i == "3" or action_i == "4" or action_i == "5" or action_i == "6":
    for i in range(1, 31):
        img = cv.imread(
            f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # non sono sicuro di questo perché le foto vengono undistortate tutte nello stesso modo
        if action_i == "5" or action_i == "6":
            imgcopy = img.copy()
            if action_i == "6":
                cv.drawChessboardCorners(
                    imgcopy, CHESSBOARD_VERTICES, corners_list[i-1], True)
            _, rvec, tvec, _ = cv.solvePnPRansac(
                po, corners_list[i-1], mtx, dist)
            vp, _ = cv.projectPoints(cube_vertices(
                0, 0, 0, 2), rvec, tvec, mtx, dist)
            print(vp)
            print(corners_list[i-1])
            imgcopy = draw_cube(imgcopy, vp.round().astype(np.int32))
            # test = np.zeros((10, 1, 2), dtype=np.float32)
            # dPt = cv.undistortPoints(test, mtx, dist, None, mtx2)
            # dPt = dPt.round().astype(int)
            # cv.polylines(imgcopy, [dPt], True, (0, 0, 255))

            show_image(imgcopy)

        # _, rvec, tvec, _ = cv.solvePnPRansac(
        #     punti_oggetto[i-1], corners_list[i-1], matIntr_run1, distCoeff_run1)
        # vp, _ = cv.projectPoints(cube_vertices(
        #     0, 0, 0, 2), rvec, tvec, matIntr_run1, distCoeff_run1)
        # res = draw_cube(tmp, vp.round().astype(np.int32))
        # show_image(res)

runs = [[]]
if action_i == "7":
    errors= np.zeros((3, 325, 4, 2))
    nans = 0
    n_frame = 0
    cap = cv.VideoCapture("./outpy.avi")
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            retC, corners = find_chessboard(frame)
            if retC:
                for filei in range(1, 4):
                    with np.load(f'camera_matrix_Run{filei}.npz') as file:
                        mtx, dist = [file[i] for i in ['mtx', 'dist']]
                        _, rvec, tvec, _ = cv.solvePnPRansac(
                        po, corners, mtx, dist)
                        vp, _ = cv.projectPoints(cube_vertices(
                            0, 0, 0, 2), rvec, tvec, mtx, dist)
                        first, second, third, fourth = (corners[0, 0], corners[2, 0], corners[18, 0], corners[20, 0])
                        first_vp, second_vp, third_vp, fourth_vp = (vp[0, 0], vp[1, 0], vp[3, 0], vp[2, 0])
                        
                        cv.imshow("frame", draw_cube(frame, vp.round().astype(np.int32)))
                        cv.waitKey(0)
                        
                        #print(f"frame {n_frame}, run {filei}:" + f"{np.abs(first_vp - first)} {np.abs(second_vp-second)} {np.abs(third_vp - third)} {np.abs(fourth_vp-fourth)}")
                        if np.isnan(first_vp[0]):
                            nans += 1
                        else:
                            errors[filei-1, n_frame, 0, 0] = np.abs(first_vp - first)[0]
                            errors[filei-1, n_frame, 1, 0] = np.abs(second_vp - second)[0]
                            errors[filei-1, n_frame, 2, 0] = np.abs(third_vp - third)[0]
                            errors[filei-1, n_frame, 3, 0] = np.abs(fourth_vp - fourth)[0]
                            errors[filei-1, n_frame, 0, 1] = np.abs(first_vp - first)[1]
                            errors[filei-1, n_frame, 1, 1] = np.abs(second_vp - second)[1]
                            errors[filei-1, n_frame, 2, 1] = np.abs(third_vp - third)[1]
                            errors[filei-1, n_frame, 3, 1] = np.abs(fourth_vp - fourth)[1]
                n_frame += 1
        else:
            break
    cap.release()
    # errors3_x = [x for x in errors[2, :, :, 0].ravel() if x is not np.nan]
    # errors3_y = [x for x in errors[2, :, :, 1].ravel() if x is not np.nan]
    print(f"Avg error of Run 1, coordinate X: {np.mean(errors[0, :, :, 0])} with Std {sem(errors[0, :, :, 0].ravel())}")
    print(f"Avg error of Run 1, coordinate Y: {np.mean(errors[0, :, :, 1])} with Std {sem(errors[0, :, :, 1].ravel())}")
    print(f"Avg error of Run 2, coordinate X: {np.mean(errors[1, :, :, 0])} with Std {sem(errors[1, :, :, 0].ravel())}")
    print(f"Avg error of Run 2, coordinate Y: {np.mean(errors[1, :, :, 1])} with Std {sem(errors[1, :, :, 1].ravel())}")
    print(f"Avg error of Run 3, coordinate X: {np.mean(errors[2, :, :, 0])} with Std {sem(errors[2, :, :, 0].ravel())}")
    print(f"Avg error of Run 3, coordinate Y: {np.mean(errors[2, :, :, 1])} with Std {sem(errors[2, :, :, 1].ravel())}")
    print(f"Number of times corners were not found automatically: {nans}")


if action_i == "8":
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    h, w = frame.shape[:2]
    out = cv.VideoWriter('outpy.avi', cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (w, h))
    for i in range(500):
        ret, frame = vid.read()
        if ret:
            out.write(frame)
    vid.release()
    out.release()
    cv.destroyAllWindows()
