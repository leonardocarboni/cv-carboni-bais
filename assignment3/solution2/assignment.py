import glm
import random
import numpy as np
import cv2 as cv
from utils import *

# global variables
block_size = 1.0
voxel_size = 45.0   # voxel every 3cm
lookup_table = []
camera_handles = []
background_models = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

# generate the floor grid locations
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


# determines which voxels should be set
def set_voxel_positions(width, height, depth, curr_time):
    if len(lookup_table) == 0:
        create_lookup_table(width, height, depth)

    # initialize voxel list
    voxel_list = []
    
    # swap y and z
    voxel_grid = np.ones((width, depth, height), np.float32)
    
    for i_camera in range(4):

        if curr_time == 0:
            # train MOG2 on background video, remove shadows, default learning rate
            background_models.append(cv.createBackgroundSubtractorMOG2())
            background_models[i_camera].setShadowValue(0)

            # open background.avi
            camera_handle = cv.VideoCapture(cameras_videos_info[i_camera][0])
            num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

            # train background model on each frame
            for i_frame in range(num_frames):
                ret, image = camera_handle.read()
                if ret:
                    background_models[i_camera].apply(image)

            # close background.avi
            camera_handle.release()

            # open video.avi
            camera_handles.append(cv.VideoCapture(cameras_videos_info[i_camera][2]))
            num_frames = int(camera_handles[i_camera].get(cv.CAP_PROP_FRAME_COUNT))
        
        # read frame
        ret, image = camera_handles[i_camera].read()

        # determine foreground
        foreground_image = background_subtraction(image, background_models[i_camera])

        # set voxel to off if it is not visible in the camera, or is not in the foreground
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_grid[x, z, y]:
                        continue
                    voxel_index = z + y * depth + x * (depth * height)

                    projection_x = int(lookup_table[i_camera][voxel_index][0][0])
                    projection_y = int(lookup_table[i_camera][voxel_index][0][1])
                    if projection_x < 0 or projection_y < 0 or projection_x >= foreground_image.shape[1] or projection_y >= foreground_image.shape[0] or not foreground_image[projection_y, projection_x]:
                        voxel_grid[x, z, y] = 0.0
    colors = []
    # put voxels that are on in list
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_grid[x, z, y] > 0:
                    voxel_list.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
    voxels_to_cluster = np.array([[x[0], x[2]]
                                    for x in voxel_list], dtype=np.float32)
    compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    for i, _ in enumerate(voxel_list):
            label = labels[i][0]
            colors.append(labels_to_color[label])
    return voxel_list, colors


# create lookup table
def create_lookup_table(width, height, depth):
    # create 3d voxel grid
    voxel_space_3d = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel_space_3d.append([voxel_size * (x * block_size - width / 2), voxel_size * (z * block_size - depth / 2), - voxel_size * (y * block_size)])

    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)

        # use config.xml to read camera calibration
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        mtx = file_handle.getNode('camera_matrix').mat()
        dist = file_handle.getNode('dist_coeffs').mat()
        rvec = file_handle.getNode('rvec_extr').mat()
        tvec = file_handle.getNode('tvec_extr').mat()
        file_handle.release()
        
        # project voxel 3d points to 2d in each camera
        voxel_space_2d, jac = cv.projectPoints(np.array(voxel_space_3d, np.float32), rvec, tvec, mtx, dist)
        lookup_table.append(voxel_space_2d)

# applies background subtraction to obtain foreground mask
def background_subtraction(image, background_model):
    foreground_image = background_model.apply(image, learningRate=0)
    
    # remove noise through dilation and erosion
    erosion_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilation_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)
    contours, _ = cv.findContours(
                foreground_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # Sort the remaining contours by size (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]

    result = np.zeros_like(foreground_image)
    cv.fillPoly(result, contours, color=255)
    return result


# Gets stored camera positions
def get_cam_positions():
    cam_positions = []
    
    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        tvec = file_handle.getNode('tvec_extr').mat()
        rvec = file_handle.getNode('rvec_extr').mat()
        file_handle.release()

        # obtain positions
        rotation_matrix = cv.Rodrigues(rvec)[0]
        positions = -np.matrix(rotation_matrix).T * np.matrix(tvec)
        cam_positions.append([positions[0][0], -positions[2][0], positions[1][0]])
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# Gets stored camera rotations
def get_cam_rotation_matrices():
    cam_rotations = []
    
    for i in range(4):
        camera_path = './data/cam' + str(i + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        rvec = file_handle.getNode('rvec_extr').mat()
        file_handle.release()
        
        # normalize rotations
        angle = np.linalg.norm(rvec)
        axis = rvec / angle
        
        # apply rotation to compensate for difference between OpenCV and OpenGL
        transform = glm.rotate(-0.5 * np.pi, [0, 0, 1]) * glm.rotate(-angle, glm.vec3(axis[0][0], axis[1][0], axis[2][0]))
        transform_to = glm.rotate(0.5 * np.pi, [1, 0, 0])
        transform_from = glm.rotate(-0.5 * np.pi, [1, 0, 0])
        cam_rotations.append(transform_to * transform * transform_from)
    return cam_rotations

def offline(width, height, depth):
    if len(lookup_table) == 0:
        create_lookup_table(width, height, depth)

    # initialize voxel list
    voxel_list = []
    
    # swap y and z
    voxel_grid = np.ones((width, depth, height), np.float32)
    
    for i_camera in range(4):

        # train MOG2 on background video, remove shadows, default learning rate
        background_models.append(cv.createBackgroundSubtractorMOG2())
        background_models[i_camera].setShadowValue(0)

        # open background.avi
        camera_handle = cv.VideoCapture(cameras_videos_info[i_camera][0])
        num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

        # train background model on each frame
        for i_frame in range(num_frames):
            ret, image = camera_handle.read()
            if ret:
                background_models[i_camera].apply(image)

        # close background.avi
        camera_handle.release()

        # open video.avi
        camera_handles.append(cv.VideoCapture(cameras_videos_info[i_camera][2]))
        num_frames = int(camera_handles[i_camera].get(cv.CAP_PROP_FRAME_COUNT))
        
        # read frame
        ret, image = camera_handles[i_camera].read()

        # determine foreground
        foreground_image = background_subtraction(image, background_models[i_camera])

        # set voxel to off if it is not visible in the camera, or is not in the foreground
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_grid[x, z, y]:
                        continue
                    voxel_index = z + y * depth + x * (depth * height)

                    projection_x = int(lookup_table[i_camera][voxel_index][0][0])
                    projection_y = int(lookup_table[i_camera][voxel_index][0][1])
                    if projection_x < 0 or projection_y < 0 or projection_x >= foreground_image.shape[1] or projection_y >= foreground_image.shape[0] or not foreground_image[projection_y, projection_x]:
                        voxel_grid[x, z, y] = 0.0
    colors = []
    # put voxels that are on in list
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxel_grid[x, z, y] > 0:
                    voxel_list.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
    voxels_to_cluster = np.array([[x[0], x[2]]
                                    for x in voxel_list], dtype=np.float32)
    compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    for i, _ in enumerate(voxel_list):
            label = labels[i][0]
            colors.append(labels_to_color[label])



    file_handle = cv.FileStorage('./data/cam2/config.xml', cv.FileStorage_READ)
    mtx = file_handle.getNode('camera_matrix').mat()
    dist = file_handle.getNode('dist_coeffs').mat()
    rvec = file_handle.getNode('rvec_extr').mat()
    tvec = file_handle.getNode('tvec_extr').mat()
    file_handle.release()
    imgpoints = []
    for i_label, vox in enumerate(voxel_list):
        x_3D, y_3D, z_3D = (
            int(vox[0]*voxel_size),  int(vox[2]*voxel_size), int(-vox[1]*voxel_size))
        img_points, _ = cv.projectPoints(np.array(
            [x_3D, y_3D, z_3D], dtype=np.float32), rvec, tvec, mtx, dist)
        x, y = (int(img_points.ravel()[0]), int(
            img_points.ravel()[1]))  # x is < 644, y is < 486
        # tuple (2d pixel, clustering label, original color)
        imgpoints.append(((x,y), labels[i_label][0], frames[camera_i][chosen_frame][y, x]))