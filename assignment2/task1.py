import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET

max_images = 41

dim_square = 115


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


def interpolate_corners(image, edges):
    """
    It estimates internal corner positions through linear interpolation

    :param image: the image 
    :param edges: the edges of the chessboard in the original image
    :return: the coordinates of the corners of the chessboard in the original image.
    """

    # number of squares for each row
    horizontal_squares = CHESSBOARD_VERTICES[0] - 1
    # number of squares for each column
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
        np.array(edges, dtype=np.float32), dst_corners)

    # all corners coordinates in the rectified image
    transformed_vertices = np.zeros(
        (CHESSBOARD_VERTICES[1], CHESSBOARD_VERTICES[0], 2))

    for i in range(CHESSBOARD_VERTICES[1]):  # for each row of the chessboard
        # for each column of the chessboard
        for j in range(CHESSBOARD_VERTICES[0]):
            # store the estimated x coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j, 0] = rectified_corners[0][0] + j * dim_square
            # store the estimated y coordinate of the (i,j) edge using linear interpolation
            transformed_vertices[i, j, 1] = rectified_corners[0][1] + i * dim_square

    # extract the inverse transformation matrix
    M_inv = np.linalg.inv(M)

    # invert the rectified corners to map in the original image
    original_vertices = np.float32(cv.perspectiveTransform(
        transformed_vertices.reshape(1, -1, 2), M_inv))
    original_vertices = original_vertices.reshape(-1, 2)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    original_vertices = cv.cornerSubPix(
        image, original_vertices, (11, 11), (-1, -1), TERMINATION_CRITERIA)

    return np.array(original_vertices, dtype=np.float32)


CHESSBOARD_VERTICES = (8, 6)
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


op = np.array([(x, y, 0) for y in range(CHESSBOARD_VERTICES[1])
               for x in range(CHESSBOARD_VERTICES[0])], dtype=np.float32)

for camera_i in range(1, 5):
    image_points_intrinsics = []
    object_points_intrinsics = []
    path = f"./data/cam{camera_i}/intrinsics.avi"
    cap = cv.VideoCapture(path)
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while i < max_images:
        retF, frame = cap.read()
        if retF:
            retC, corners = cv.findChessboardCorners(
                frame, CHESSBOARD_VERTICES)
            if retC:
                image_points_intrinsics.append(corners)
                object_points_intrinsics.append(op)
            print(f"{i}: {retC}")
            i += 1
        else:
            break
    cap.release()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points_intrinsics, image_points_intrinsics, (w, h), None, None)

    #save_xml(f"/data/cam{camera_i}/intrisics.xml", camera_matrix, dist_coeffs)

    # extrinsics
    image_points_extrinsics = []
    object_points_extrinsics = []
    cap = cv.VideoCapture(f"./data/cam{camera_i}/checkerboard.avi")
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while cap.isOpened():
        retF, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        if retF:
            retC, corners = cv.findChessboardCorners(
                frame, CHESSBOARD_VERTICES)
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
                cv.namedWindow("PUTTANA TUA MAMMA", cv.WINDOW_NORMAL)
                cv.imshow("PUTTANA TUA MAMMA", cv.drawChessboardCorners(frame, CHESSBOARD_VERTICES, corners, True))
                cv.waitKey(0)
                cv.destroyAllWindows()

            image_points_extrinsics.append(corners)
            object_points_extrinsics.append(op)
            if i == 2:
                break
            i+=1
        else:
            break
    cap.release()
    
    retval_extr, rvec_extr, tvec_extr = cv.solvePnP(
        object_points_extrinsics, image_points_extrinsics, camera_matrix, dist_coeffs, rvecs, tvecs, False, cv.SOLVEPNP_ITERATIVE)

    np.savez(f"data/cam{camera_i}/config", camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs, rvec_extr=rvec_extr, tvec_extr=tvec_extr)
