import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET

max_images = 25

dim_square = 115

axis = np.float32([[dim_square * 3, 0, 0], [0, dim_square * 3, 0], [0, 0, -dim_square * 3]]).reshape(-1, 3)

def save_xml(xmlpath, matrix, distorsion):
    xmltemplate = f"""
    <?xml version="1.0"?>
<opencv_storage>
<CameraMatrix type_id="opencv-matrix">
  <rows>{matrix.shape[0]}</rows>
  <cols>{matrix.shape[1]}</cols>
  <dt>f</dt>
  <data>
  {matrix}
  </data>
</CameraMatrix>
<DistortionCoeffs type_id="opencv-matrix">
  <rows>{distorsion.shape[0]}</rows>
  <cols>{distorsion.shape[1]}</cols>
  <dt>f</dt>
  <data>
    {distorsion}
  </data>
</DistortionCoeffs>
</opencv_storage>
"""

    mytree = ET.fromstring(xmltemplate)

    mytree.write(xmlpath)


def click_event(event, x, y, flags, params):
    """
    callback function to manually annotate the four outmost corner points
    """
    if event == cv.EVENT_LBUTTONDOWN and len(edges) < 4:
        edges.append((x, y))
        print(f"Edge set: ({x}, {y}), Select {4-len(edges)} more corners")


def interpolate_corners(image, image_edges):
    """
    It estimates internal corner positions through linear interpolation

    :param image: the image 
    :param image_edges: the edges of the chessboard in the original image
    :return: the coordinates of the corners of the chessboard in the original image.
    """

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
    
    for i in range(CHESSBOARD_VERTICES[1]):  # for each row of the chessboard
        # for each column of the chessboard
        for j in range(CHESSBOARD_VERTICES[0]):
            # store the estimated x coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j,
                                 0] = rectified_corners[0][0] + j * dim_square
            # store the estimated y coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j,
                                 1] = rectified_corners[0][1] + i * dim_square
            

    # sugo = cv.warpPerspective(image, M, (image.shape[0], image.shape[1]))
    # sugo = cv.drawChessboardCorners(sugo, CHESSBOARD_VERTICES, np.array(
    #     transformed_vertices.reshape(1, -1, 2), dtype=np.float32), True)
    # cv.imshow("sugo", sugo)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # extract the inverse transformation matrix
    M_inv = np.linalg.inv(M)

    # invert the rectified corners to map in the original image
    original_vertices = np.float32(cv.perspectiveTransform(
        transformed_vertices.reshape(1, -1, 2), M_inv))
    original_vertices = original_vertices.reshape(-1, 2)

    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # original_vertices = cv.cornerSubPix(
    #     image, original_vertices, (11, 11), (-1, -1), TERMINATION_CRITERIA)

    return np.array(original_vertices, dtype=np.float32)

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

CHESSBOARD_VERTICES = (8, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

op = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32) * dim_square
op_3 = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[0])
               for x in range(CHESSBOARD_VERTICES[1])], dtype=np.float32) * dim_square
        
for camera_i in range(1, 5):
    image_points_intrinsics = []
    object_points_intrinsics = []
    path = f"./data/cam{camera_i}/intrinsics.avi"
    cap = cv.VideoCapture(path)
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0
    found = 0
    while found < max_images:
        retF, frame = cap.read()
        if retF:
            if camera_i != 3:
                retC, corners = cv.findChessboardCorners(
                    frame, CHESSBOARD_VERTICES)
            elif i > 194:
                retC, corners = cv.findChessboardCorners(
                    frame, (6, 8))
            if retC:
                found += 1
                image_points_intrinsics.append(corners)
                if camera_i != 3:
                    object_points_intrinsics.append(op)
                else: 
                    object_points_intrinsics.append(op_3)
            print(f"{i}: {retC}")
            i += 1
        else:
            break
    cap.release()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points_intrinsics, image_points_intrinsics, (w, h), None, None)

    #save_xml(f"/data/cam{camera_i}/intrisics.xml", camera_matrix, dist_coeffs)
    if camera_i != 3:
        retval_extr, rvec_extr, tvec_extr = cv.solvePnP(op, corners, camera_matrix, dist_coeffs, flags = cv.SOLVEPNP_ITERATIVE)
    else:
        retval_extr, rvec_extr, tvec_extr = cv.solvePnP(op_3, corners, camera_matrix, dist_coeffs, flags = cv.SOLVEPNP_ITERATIVE)
    vp_axis, _ = cv.projectPoints(axis, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
    frame = draw_axis(frame, corners.round().astype(np.int32), vp_axis.round().astype(np.int32))
    cv.namedWindow("PUTTANA TUA MAMMA", cv.WINDOW_NORMAL)
    cv.imshow("PUTTANA TUA MAMMA", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    # extrinsics
    image_points_extrinsics = []
    object_points_extrinsics = []
    cap = cv.VideoCapture(f"./data/cam{camera_i}/checkerboard.avi")
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while cap.isOpened():
        retF, frame = cap.read()
        # frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        if retF:
            if camera_i != 3:
                retC, corners = cv.findChessboardCorners(
                    frame, CHESSBOARD_VERTICES)
            else:
                retC, corners = cv.findChessboardCorners(
                frame, (6, 8))
            if not retC:
                edges = []  # list of outmost corners to be annotated
                print(f"Corners not in image {i} found, Select them manually")
                cv.namedWindow("insert points", cv.WINDOW_NORMAL)
                cv.imshow("insert points", frame)
                cv.setMouseCallback('insert points', click_event)
                cv.waitKey(0)
                cv.destroyAllWindows()
                corners = np.array(interpolate_corners(frame, edges))[
                    :, np.newaxis]
                frame_copy = frame.copy()
                cv.namedWindow("PUTTANA TUA MAMMA", cv.WINDOW_NORMAL)
                cv.imshow("PUTTANA TUA MAMMA", cv.drawChessboardCorners(
                    frame_copy, (CHESSBOARD_VERTICES[0], CHESSBOARD_VERTICES[1]), corners, True))
                cv.waitKey(0)
                cv.destroyAllWindows()

            image_points_extrinsics.append(corners)
            object_points_extrinsics.append(op)

        break
    cap.release()
    retval_extr, rvec_extr, tvec_extr = cv.solvePnP(
        np.array(object_points_extrinsics, dtype = np.float32), np.array(image_points_extrinsics, dtype = np.float32)[0], camera_matrix, dist_coeffs, flags = cv.SOLVEPNP_ITERATIVE)

    R = cv.Rodrigues(rvec_extr)[0]
    vp_axis, _ = cv.projectPoints(axis, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
    frame = draw_axis(frame, corners.round().astype(np.int32), vp_axis.round().astype(np.int32))
    cv.namedWindow("PUTTANA TUA MAMMA", cv.WINDOW_NORMAL)
    cv.imshow("PUTTANA TUA MAMMA", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    np.savez(f"data/cam{camera_i}/config", camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs, rvec_extr=rvec_extr, tvec_extr=tvec_extr, R = R)
