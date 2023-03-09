import glm
import random
import numpy as np
import cv2 as cv
from time import time
import os

from sklearn.preprocessing import normalize
from utils import *
import matplotlib.pyplot as plt


show = True

block_size = 1.0
n_frame = 0
visible_voxels = []
lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {'0': (255, 0, 0), '1': (0, 255, 0),
                   '2': (0, 0, 255), '3': (255, 0, 255)}

cap = cv.VideoCapture(cameras_videos_info[1][2]) # video of camera 2
retF, frame = cap.read() # get first frame (used for color model)
frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
cap.release()

MGGs = {'0': None, '1': None, '2': None, '3': None}

with np.load('./data/cam1/masks.npz') as file:
    mask1 = file['masks']
with np.load('./data/cam2/masks.npz') as file:
    mask2 = file['masks']
with np.load('./data/cam3/masks.npz') as file:
    mask3 = file['masks']
with np.load('./data/cam4/masks.npz') as file:
    mask4 = file['masks']

# loading parameters of cam2 for color model
s = cv.FileStorage(
    f"./data/cam2/config.xml", cv.FileStorage_READ)
camera_matrix = s.getNode('camera_matrix').mat()
dist_coeffs = s.getNode('dist_coeffs').mat()
rvec_extr = s.getNode('rvec_extr').mat()
tvec_extr = s.getNode('tvec_extr').mat()
s.release()


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -
                        block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    global lookup_table
    MGGs = {0: None, 1: None, 2: None, 3: None}
    # global frames

    masks = []
    for camera_i in range(4):
        m = get_mask(frames[camera_i][cam_to_frame[camera_i]], camera_i)
        masks.append(m)
        if show:
            leo = 0
            # show_image(m)
            # show_image(frames[camera_i][0])
            # show_image(cv.bitwise_and(frames[camera_i][0], frames[camera_i][0], mask = m))

    all_visible_voxels = []
    all_labels = []
    start_reconstruction = time()
    for _ in range(4): # 4 reconstructions for the 4 frames nedded for the color models
        visible_voxels = []
        for vox in range(voxel_positions.shape[0]):  # for each voxel id
            flag = True  # the voxel is foreground for all cameras (flag)
            for camera_i in range(4):  # for each camera
                # extract voxel 3D and 2D coordinates for that camera
                x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, camera_i]
                x = int(x)
                y = int(y)
                # check if the pixel is foreground for all cameras
                if masks[camera_i][y, x] == 0:
                    flag = False

            if flag:  # if it is foreground for all cameras
                # adapt to glm format, scale and add to reconstruction
                visible_voxels.append(
                    [x_voxels/75, -z_voxels/75, y_voxels/75])
        print(f"time to reconstruct all: {time()-start_reconstruction}")
        all_visible_voxels.append(visible_voxels)

        # COLORS
        colors = []
        for i, _ in enumerate(visible_voxels):
            label = labels[i][0]
            colors.append(labels_to_color[str(label)])
            
        voxels_to_cluster = np.array([[x[0], x[2]]
                                    for x in visible_voxels], dtype=np.float32)
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        all_labels.append(labels) # list of 4 lists, that has all labels for each visible voxel
    # end of reconstructions

    pixels_colors = [] # list of length 4, for each camera its 2d visible pixels, its clustering label and its original color
    color_model = {0: [], 1: [], 2: [], 3: []}
    
    for camera_i, vis_voxs in enumerate(all_visible_voxels):
        imgpoints = []
        chosen_frame = cam_to_frame[camera_i]
        for i_label, vox in enumerate(vis_voxs):
            x_3D, y_3D, z_3D = (
                int(vox[0]*75),  int(vox[2]*75), int(-vox[1]*75))
            # pos = np.where((lookup_table[:, 0, 1] == x) & (lookup_table[:, 1, 1] == y) & (lookup_table[:, 2, 1] == z))
            # print("hello", pos)
            # pixels = lookup_table[pos, 3:5, 1]
            img_points, _ = cv.projectPoints(np.array(
                [x_3D, y_3D, z_3D], dtype=np.float32), rvecs_extr[camera_i], tvecs_extr[camera_i], camera_matrixes[camera_i], dist_coeffs[camera_i])
            x, y = (int(img_points.ravel()[0]), int(
                img_points.ravel()[1]))  # x is < 644, y is < 486
            # tuple (2d pixel, clustering label, original color)
            imgpoints.append(((x,y), all_labels[camera_i][i_label][0], frames[camera_i][chosen_frame][y, x])) # 2d coords,clustering label, original color
        pixels_colors.append(imgpoints)
        
    
    for camera_i, infos in enumerate(pixels_colors):
        chosen_frame = cam_to_frame[camera_i]
        frame_copy = frames[camera_i][chosen_frame].copy()
        for pc in infos:
            cv.circle(frame_copy, pc[0], 2, labels_to_color[pc[1]], 2)
            color_model[pc[1]].append(pc[2].tolist())
            # print(color_model[3]) # list of np array hsv
        for person in color_model:
            MGGs[person] = cv.ml.EM_create()
            MGGs[person].setClustersNumber(3)
            MGGs[person].trainEM(
                np.array(color_model[person], dtype=np.float32))

        for person in color_model:
            loglik1 = 0
            loglik2 = 0
            loglik3 = 0
            loglik4 = 0
            for pixel in color_model[person]:
                loglik1 += MGGs[0].predict2(np.array(pixel,
                                                    dtype=np.float32))[0][0]
                loglik2 += MGGs[1].predict2(np.array(pixel,
                                                    dtype=np.float32))[0][0]
                loglik3 += MGGs[2].predict2(np.array(pixel,
                                                    dtype=np.float32))[0][0]
                loglik4 += MGGs[3].predict2(np.array(pixel,
                                                    dtype=np.float32))[0][0]
            print(person, loglik1, loglik2, loglik3, loglik4)
        show_image(frame_copy, "silhouttes")
    return visible_voxels, MGGs
        return visible_voxels, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    positions = []
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        tvec_extr = s.getNode('tvec_extr').mat()
        R = s.getNode('R_MAT').mat()
        positions.append(np.dot(-R.T, tvec_extr))
        s.release()
    positions = np.stack(positions)
    # positions = normalize(positions.squeeze(), axis=0)
    positions = positions.squeeze()
    # normalization
    positions = normalize(positions.squeeze(), axis=0) * 64
    # converting from opencv space to glm space
    # swap y and z
    positions[:, [1, 2]] = positions[:, [2, 1]]
    # abs y positions
    positions[:, 1] = np.abs(positions[:, 1])

    return positions, [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_rotations = []
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        tvec_extr = s.getNode('tvec_extr').mat()
        R = s.getNode('R_MAT').mat()

        t1 = np.hstack((R, tvec_extr))
        t1 = np.vstack((t1, [0, 0, 0, 1]))
        # swap y and z
        t1[:, [1, 2]] = t1[:, [2, 1]]
        # invert x rotation of the camears
        t1[:, 1] = -t1[:, 1]
        # transform to mat4
        rot = glm.mat4(t1.T)
        # rotate cameras by 90 degrees because they point on the wrong side
        rot = glm.rotate(rot, -90 * np.pi / 180, [0, 1, 0])

        cam_rotations.append(rot)
    return cam_rotations


def create_cube(width, height, depth):
    "creates a solid with resolution 100x100x100 with the current inputs"
    cube = []
    for x in np.arange(-width//4, 3*width//4, 80):
        for y in np.arange(-depth//2, depth//2, 80):
            for z in np.arange(-height, height, 80):
                cube.append([x, y, z])
    return cube


def create_lookup_table(width, height, depth):
    "Returns the lookup table mapping each voxels to its correspongin 3D and 2D coordinates for each camera"
    "Inputs are for the placement of the solid for the reconstruction"
    # create solid
    voxel_positions = np.array(create_cube(
        width, height, depth), dtype=np.float32)
    # initialize look up table
    lookup_table = np.zeros((voxel_positions.shape[0], 5, 4))
    for camera_i in range(1, 5):  # for each camera
        # load parameters
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        camera_matrix = s.getNode('camera_matrix').mat()
        dist_coeffs = s.getNode('dist_coeffs').mat()
        tvec_extr = s.getNode('tvec_extr').mat()
        rvec_extr = s.getNode('rvec_extr').mat()
        s.release()
        # for each 3D point of the voxel solid
        for i, pos in enumerate(voxel_positions):
            # project the point in the 2D image plane for this camera
            imgpoint, _ = cv.projectPoints(
                pos, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            imgpoint = imgpoint.ravel()
            x = imgpoint[0]
            y = imgpoint[1]
            if x >= 0 and x < 644 and y >= 0 and y < 486:  # if the 2D pixel is in range of the frame
                # store 2D and 3D coordinates
                lookup_table[i, :, camera_i -
                             1] = [int(pos[0]), int(pos[1]), int(pos[2]), int(x), int(y)]

    np.savez('data/lookup_table', lookup_table=lookup_table)
    return voxel_positions, lookup_table

# print(create_lookup_table(1500, 3000, 3000)[1])
# set_voxel_positions(1500, 3000,3000)

# show_image(mask1[0], "jnas")
# show_image(mask2[0], "jnas")
# show_image(mask3[0], "jnas")
# show_image(mask4[0], "jnas")
