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
    cv.polylines(img, [v[:4]], True, (0, 255, 0), 5)
    cv.polylines(img, np.array([v[i::4]
                 for i in range(4)]), False, (0, 0, 255), 5)
    cv.polylines(img, [v[4:8]], True, (255, 0, 0), 5)


def show_image(img, name="chessboard"):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow("chessboard", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_images(imgs, name="chessboards"):
    print("balle")


def interpolate_corners(image, edges):

    # top left and bottom right
    tl_edge = edges[0]
    print(tl_edge)
    br_edge = edges[3]
    print(br_edge)
    # distances
    dx = (br_edge[0] - tl_edge[0]) / (CHESSBOARD_VERTICES[0])
    dy = (br_edge[1] - tl_edge[1]) / (CHESSBOARD_VERTICES[1])

    # Initialize the array to store the X and Y coordinates of the vertices
    vertices = np.zeros((CHESSBOARD_VERTICES[1] * CHESSBOARD_VERTICES[0], 2))

    # Calculate the X and Y coordinates of the vertices
    for i in range(CHESSBOARD_VERTICES[1]):
        for j in range(CHESSBOARD_VERTICES[0]):
            vertices[i * CHESSBOARD_VERTICES[0] + j, 0] = tl_edge[0] + j * dx
            vertices[i * CHESSBOARD_VERTICES[0] + j, 1] = tl_edge[1] + i * dy

    return np.array(vertices, dtype=np.float32)


for i in range(1, 30):
    img = cv.imread(
        f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    punti_test = [(195, 69), (155, 407), (498, 406), (460, 54)]
    punti_immagine, punti_oggetto, po = [], [], []
    po = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
                  for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

    ret, corners = find_chessboard(img)

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

    corners_list.append(corners)

for i in range(1, 30):
    corners = corners_list[i-1]
    img2 = cv.imread(
        f'test_chessboard_images/{i}.jpg', cv.COLOR_BGR2GRAY)
    cv.drawChessboardCorners(img2, CHESSBOARD_VERTICES, corners, True)
    show_image(img2)
    # correzione distorsione
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
    # img_ret = cv.undistort(img, matIntr, distCoeff, None, matIntr2)
    # points = np.array([punti_test], np.float32)
    # dPt = cv.undistortPoints(points, matIntr, distCoeff, None, matIntr2)
    # dPt = dPt.round().astype(int)
    # # cv.polylines(img_ret, [dPt], True, (0,0,255))
    # tmp = img_ret.copy()
    # vp, _ = cv.projectPoints(cube_vertices(0,0,0,2), rotEstr[0], traEstr[0], matIntr, distCoeff)
    # res = draw_cube(tmp, vp.round().astype(np.int32))
    # show_image(tmp, "Temp")
