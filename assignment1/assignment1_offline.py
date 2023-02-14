import cv2 as cv
import numpy as np

CHESSBOARD_VERTICES = (9, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

corners_list = []

# run_number = input("Insert run number: ")


def click_event(event, x, y, flags, params):
    ''' function to display the coordinates of the points clicked on the image '''
    if event == cv.EVENT_LBUTTONDOWN and len(edges) < 4:
        edges.append((x, y))
        print(f"Edge set: ({x}, {y}), Select {4-len(edges)} more corners")
    # if event == cv.EVENT_LBUTTONDOWN and len(edges) >= 4:
    #     print("4 edges selected. Press any key to continue.")


def find_chessboard(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray_img, CHESSBOARD_VERTICES)
    if ret:
        corners = cv.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
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


def interpolate_corners(image, edges):

    dst_size = (CHESSBOARD_VERTICES[0] * 200, CHESSBOARD_VERTICES[1] * 200)

    # Define the corners of the rectified image
    dst_corners = np.array([[0, 0], [dst_size[0], 0], [0, dst_size[1]], [
                           dst_size[0], dst_size[1]]], dtype=np.float32)

    rectified_corners = np.array([[0, 0], [(CHESSBOARD_VERTICES[0]-1) * 200, 0], [(CHESSBOARD_VERTICES[0]-1)
                                 * 200, (CHESSBOARD_VERTICES[1]-1) * 200], [0, (CHESSBOARD_VERTICES[1]-1) * 200]], dtype=np.float32)
    
    # Compute the perspective transformation matrix
    M = cv.getPerspectiveTransform(
        np.array(edges, dtype=np.float32), dst_corners)

    dx = (rectified_corners[2][0] - rectified_corners[0]
          [0]) / (CHESSBOARD_VERTICES[0]-1)
    print(f"dx = {dx}")
    dy = (rectified_corners[2][1] - rectified_corners[0]
          [1]) / (CHESSBOARD_VERTICES[1]-1)
    print(f"dy = {dy}")
    transformed_vertices = np.zeros(
        (CHESSBOARD_VERTICES[1], CHESSBOARD_VERTICES[0], 2))
    for i in range(CHESSBOARD_VERTICES[1]):
        for j in range(CHESSBOARD_VERTICES[0]):
            transformed_vertices[i, j, 0] = rectified_corners[0][0] + j * dx
            transformed_vertices[i, j, 1] = rectified_corners[0][1] + i * dy

    M_inv = np.linalg.inv(M)
    # original_corners = cv.perspectiveTransform(rectified_corners.reshape(1, -1, 2), M_inv)
    # original_corners = original_corners.reshape(-1, 2)
    original_vertices = cv.perspectiveTransform(
        transformed_vertices.reshape(1, -1, 2), M_inv)
    original_vertices = original_vertices.reshape(-1, 2)

    # Draw the rectangle in the original image
    # for vertice in original_vertices:
    #     cv.circle(img, (int(vertice[0]), int(vertice[1])),  radius=0, color=(0, 0, 255), thickness=5)

    # Rectify the image using the transformation matrix
    # rectified_image = cv.warpPerspective(image, M, dst_size)

    # show rectified image
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.namedWindow('chessboard', cv.WINDOW_NORMAL)
    # cv.imshow("chessboard", rectified_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # palle = (CHESSBOARD_VERTICES[0] - 1, CHESSBOARD_VERTICES[1] - 1)
    # # Find the chessboard corners in the rectified image
    # ret, rectified_corners = cv.findChessboardCorners(rectified_image, palle)
    # print(ret)
    original_vertices = cv.cornerSubPix(
            image, original_vertices, (11, 11), (-1, -1), TERMINATION_CRITERIA)
    
    return np.array(original_vertices, dtype=np.float32)


punti_oggetto = []
po = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)


# corner detection
for i in range(1, 31):
    img = cv.imread(
        f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    ret, corners = find_chessboard(img)

    # if i == 23:
    #     cornerizzata = img.copy()
    #     cv.drawChessboardCorners(
    #         cornerizzata, CHESSBOARD_VERTICES, corners, True)
    #     show_image(cornerizzata)
    #     print(corners)
    if not ret:
        edges = []
        print(f"Corners not in image {i} found, Select them manually")
        cv.namedWindow("chessboard", cv.WINDOW_NORMAL)
        cv.imshow("chessboard", img)
        cv.setMouseCallback('chessboard', click_event)
        cv.waitKey(0)
        cv.destroyAllWindows()
        corners = np.array(interpolate_corners(img, edges))[:, np.newaxis]
        # corners = interpolate_corners(edges)

    print(f"{i}: n corners = {len(corners)}")
    punti_oggetto.append(po)
    corners_list.append(corners)

# run 1 train on first 25 images
# punti_test = [(195, 69), (155, 407), (498, 406), (460, 54)]

# Run 1 (all corners)
err_run1, matIntr_run1, distCoeff_run1, rotEstr_run1, traEstr_run1 = cv.calibrateCamera(
    punti_oggetto, corners_list, (w, h), None, None)
print(f"Run #1: {matIntr_run1}")
# Run 2 (only 10)
err_run2, matIntr_run2, distCoeff_run2, rotEstr_run2, traEstr_run2 = cv.calibrateCamera(
    punti_oggetto[:10], corners_list[:10], (w, h), None, None)
print(f"Run #2: {matIntr_run2}")
# Run 3 (only 5)
err_run3, matIntr_run3, distCoeff_run3, rotEstr_run3, traEstr_run3 = cv.calibrateCamera(
    punti_oggetto[:5], corners_list[:5], (w, h), None, None)
print(f"Run #3: {matIntr_run3}")

#cv.drawChessboardCorners(img2, CHESSBOARD_VERTICES, corners, True)
# show_image(img2)
# fcorrezione distorsione
# punti_immagine.append(corners)
# punti_oggetto.append(po)

# err, matIntr, distCoeff, rotEstr, traEstr = cv.calibrateCamera(punti_oggetto, punti_immagine, (w,h), None, None)
# print(f'Errore medio di riproiezione = {err:.4f}')
# print('Parametri intrinseci della telecamera:')
# print(matIntr)
# print('Coefficienti di distorsione: ', distCoeff.ravel())
# print('Parametri di rotazione e traslazione per ciascuna immagine:')
# print(*[f'{i}: {r.ravel()} {t.ravel()}' for i,(r,t) in enumerate(zip(rotEstr,traEstr))], sep='\n')
# alpha = 0.01
# matIntr2, _ = cv.getOptimalNewCameraMatrix(matIntr,distCoeff,(w,h), alpha)
# img_ret = cv.undistort(img2, matIntr, distCoeff, None, matIntr2)
# points = np.array([punti_test], np.float32)
# dPt = cv.undistortPoints(points, matIntr, distCoeff, None, matIntr2)
# dPt = dPt.round().astype(int)
# # cv.polylines(img_ret, [dPt], True, (0,0,255))


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