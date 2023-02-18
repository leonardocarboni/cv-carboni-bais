import numpy as np
import cv2 as cv
from scipy.stats import sem

CHESSBOARD_VERTICES = (9, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
                 for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)


def find_chessboard(img):
    """
    It converts the image to gray scale, finds the corners of the chessboard, and then augments the
    precision of the corners

    :param img: the image to be processed
    :return: ret is a boolean value that is true if the corners were found and false if they were not.
    corners is a list of the coordinates of the corners.
    """
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
    :return: The image with the cube
    """
    image = np.copy(img)
    cv.polylines(image, [v[:4]], True, (0, 255, 0), 2)
    cv.polylines(image, np.array([v[i::4]
                 for i in range(4)]), False, (0, 0, 255), 2)
    cv.polylines(image, [v[4:8]], True, (255, 0, 0), 2)
    return image


def draw_axis(img, corners, vp):
    """
    It draws the three axes of the plane on the image

    :param img: The image to draw the axis on
    :param corners: The corners of the chessboard
    :param vp: vanishing points
    :return: The image with the axis
    """
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(vp[0].ravel()), (255, 255, 0), 5)
    img = cv.line(img, corner, tuple(vp[1].ravel()), (0, 255, 255), 5)
    img = cv.line(img, corner, tuple(vp[2].ravel()), (255, 0, 255), 5)
    return img


def show_image(img, name="chessboard"):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# preliminary phase
mtx, dist = [], []
corners_list, object_points = [], []

with np.load(f'corners.npz') as file:
    corners_list, object_points = [file[i]
                                   for i in ['corners', 'punti_oggetto']]

if len(corners_list) == 0:
    print("No npz found, Run offline script first.")
    exit(1)

print("Select what to do: ")
print("1. Webcam with cube")
print("2. Webcam with cube and corners")
print("3. Images with cube")
print("4. Images with cube and corners")
print("5. 3 Runs comparison with pre-recorded video.")
print("6. See camera parameters")
print("7. Shoot new video for runs comparison")
print("8. See standard deviations of estimated instrinsic parameters")
choice_action = input("Select action: ")


# actions that require training
if choice_action in ["1", "2", "3", "4"]:
    print("Select training set: ")
    print("1. All images")
    print("2. 10 images where corners are found automatically")
    print("3. 5 images where corners are found automatically")
    choice_training = input("Select action: ")

    with np.load(f'camera_matrix_Run{choice_training}.npz') as file:
        mtx, dist = [file[i] for i in ['mtx', 'dist']]

    # webcam
    if choice_action in ["1", "2"]:
        vid = cv.VideoCapture(0)

        while(True):
            ret, frame = vid.read()
            retC, corners = find_chessboard(frame)
            if retC:
                _, rvec, tvec = cv.solvePnP(
                    objp, corners, mtx, dist)
                vp, _ = cv.projectPoints(cube_vertices(
                    0, 0, 0, 2), rvec, tvec, mtx, dist)

                # draw corners too
                if choice_action == "2":
                    cv.drawChessboardCorners(
                        frame, CHESSBOARD_VERTICES, corners, True)

                axis = np.float32(
                    [[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
                vp_axis, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

                image = draw_cube(frame, vp.round().astype(np.int32))
                image = draw_axis(image, corners.round().astype(
                    np.int32), vp_axis.round().astype(np.int32))

                cv.imshow('frame', image)
            else:
                cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv.destroyAllWindows()

    elif choice_action in ["3", "4"]:
        # read images 1 to 30
        for i in range(1, 31):
            img = cv.imread(
                f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
            h, w = img.shape[:2]

            # draw corners too
            if choice_action == "4":
                cv.drawChessboardCorners(
                    img, CHESSBOARD_VERTICES, corners_list[i-1], True)

            _, rvec, tvec = cv.solvePnP(
                objp, corners_list[i-1], mtx, dist)
            vp, _ = cv.projectPoints(cube_vertices(
                0, 0, 0, 2), rvec, tvec, mtx, dist)

            axis = np.float32(
                [[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
            vp_axis, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

            img = draw_cube(img, vp.round().astype(np.int32))
            img = draw_axis(
                img, corners_list[i-1].round().astype(np.int32), vp_axis.round().astype(np.int32))

            show_image(img)

# 3 runs comparison with pre-recorded video
elif choice_action == "5":
    cap = cv.VideoCapture("./video.avi")
    reprojection_error = [0, 0, 0]
    n_frame = 0
    objpoints = []
    cube_errors = np.zeros((3, int(cap.get(cv.CAP_PROP_FRAME_COUNT)), 4, 2))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            objpoints.append(objp)
            retC, corners = cv.findChessboardCorners(
                frame, CHESSBOARD_VERTICES)

            if retC:
                first, second, third, fourth = (
                    corners[0, 0], corners[2, 0], corners[18, 0], corners[20, 0])

                for filei in range(1, 4):
                    with np.load(f'camera_matrix_Run{filei}.npz') as file:

                        mtx, dist = [file[i] for i in ['mtx', 'dist']]
                        _, rvec, tvec = cv.solvePnP(
                            objpoints[n_frame], corners, mtx, dist)

                        # reprojection error
                        imgpoints2, _ = cv.projectPoints(
                            objpoints[n_frame], rvec, tvec, mtx, dist)
                        error = cv.norm(corners, imgpoints2,
                                        cv.NORM_L2)/len(imgpoints2)
                        reprojection_error[filei-1] += error

                        # manual test with cube
                        vp_cube, _ = cv.projectPoints(cube_vertices(
                            0, 0, 0, 2), rvec, tvec, mtx, dist)

                        first_vp, second_vp, third_vp, fourth_vp = (
                            vp_cube[0, 0], vp_cube[1, 0], vp_cube[3, 0], vp_cube[2, 0])

                        cube_errors[filei-1, n_frame, 0,
                                    0] = np.abs(first_vp - first)[0]
                        cube_errors[filei-1, n_frame, 1,
                                    0] = np.abs(second_vp - second)[0]
                        cube_errors[filei-1, n_frame, 2,
                                    0] = np.abs(third_vp - third)[0]
                        cube_errors[filei-1, n_frame, 3,
                                    0] = np.abs(fourth_vp - fourth)[0]
                        cube_errors[filei-1, n_frame, 0,
                                    1] = np.abs(first_vp - first)[1]
                        cube_errors[filei-1, n_frame, 1,
                                    1] = np.abs(second_vp - second)[1]
                        cube_errors[filei-1, n_frame, 2,
                                    1] = np.abs(third_vp - third)[1]
                        cube_errors[filei-1, n_frame, 3,
                                    1] = np.abs(fourth_vp - fourth)[1]
                n_frame += 1
        else:
            break

    cap.release()

    print("Reprojection error run 1: {}".format(
        reprojection_error[0]/len(objpoints)))
    print("Reprojection error run 2: {}".format(
        reprojection_error[1]/len(objpoints)))
    print("Reprojection error run 3: {}".format(
        reprojection_error[2]/len(objpoints)))

    print(
        f"Avg error of Run 1, coordinate X: {np.mean(cube_errors[0, :, :, 0])} with Std {sem(cube_errors[0, :, :, 0].ravel())}")
    print(
        f"Avg error of Run 1, coordinate Y: {np.mean(cube_errors[0, :, :, 1])} with Std {sem(cube_errors[0, :, :, 1].ravel())}")
    print(
        f"Avg error of Run 2, coordinate X: {np.mean(cube_errors[1, :, :, 0])} with Std {sem(cube_errors[1, :, :, 0].ravel())}")
    print(
        f"Avg error of Run 2, coordinate Y: {np.mean(cube_errors[1, :, :, 1])} with Std {sem(cube_errors[1, :, :, 1].ravel())}")
    print(
        f"Avg error of Run 3, coordinate X: {np.mean(cube_errors[2, :, :, 0])} with Std {sem(cube_errors[2, :, :, 0].ravel())}")
    print(
        f"Avg error of Run 3, coordinate Y: {np.mean(cube_errors[2, :, :, 1])} with Std {sem(cube_errors[2, :, :, 1].ravel())}")

# see camera parameters
elif choice_action == "6":
    with np.load(f'camera_matrix_Run1.npz') as file:
        mtxRun1, distRun1 = [file[i] for i in ['mtx', 'dist']]
    with np.load(f'camera_matrix_Run2.npz') as file:
        mtxRun2, distRun2 = [file[i] for i in ['mtx', 'dist']]
    with np.load(f'camera_matrix_Run3.npz') as file:
        mtxRun3, distRun3 = [file[i] for i in ['mtx', 'dist']]
    print(f"Camera Parameters for Run 1:\n {mtxRun1}")
    print(f"Camera Parameters for Run 2:\n {mtxRun2}")
    print(f"Camera Parameters for Run 3:\n {mtxRun3}")

# shoot new video
elif choice_action == "7":
    vid = cv.VideoCapture(0)
    ret, frame = vid.read()
    h, w = frame.shape[:2]
    out = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (w, h))
    for i in range(500):
        ret, frame = vid.read()
        if ret:
            out.write(frame)
    vid.release()
    out.release()
    cv.destroyAllWindows()

# calculate standard deviations of estimated intrinsic parameters
elif choice_action == '8':
    w, h = (1920, 1080) # images size
    i = 0 # start of subset
    j = 10 # end of subset
    matrixes = [] # list of matrixes for each calibration
    for _ in range(10): # 10 calibration runs
        matrixes.append(cv.calibrateCamera(object_points[i:j], corners_list[i:j], (w, h), None, None)[1]) # calibrate camera using a subset of images and extract matrix
        i+=2 # next subset
        j+=2 # next subset
    focal_points_x = []
    focal_points_y = []
    center_x = []
    center_y = []
    for i in range(10):
        focal_points_x.append(matrixes[i][0][0])
        focal_points_y.append(matrixes[i][1][1])
        center_x.append(matrixes[i][0][2])
        center_y.append(matrixes[i][1][2])
    print(f"Focal point standard deviation is: ({np.std(focal_points_x)}, {np.std(focal_points_y)})")
    print(f"Origin standard deviation is: ({np.std(center_x)}, {np.std(center_y)})")
