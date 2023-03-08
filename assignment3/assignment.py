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
n_frame = 10
lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {'0': (255, 0, 0), '1': (0, 255, 0),
                   '2': (0, 0, 255), '3': (255, 0, 255)}

MGGs = {'0': None, '1': None, '2': None, '3': None}

frames = []
backgrounds = []


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


def get_mask(frame, camera_i):
    w, h, _ = frame.shape
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    background_pixels_hsv = cv.cvtColor(backgrounds[camera_i], cv.COLOR_BGR2HSV)
    foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)

    hue, saturation, value = best_masks[str(camera_i+1)]
    best_mask = np.zeros((w, h), dtype=np.uint8)
    for x in range(foreground_hsv.shape[0]):
        for y in range(foreground_hsv.shape[1]):
            if foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[x, y, 2] > value:
                best_mask[x, y] = 255

    best_mask = cv.morphologyEx(
        best_mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13)))
    best_mask = cv.morphologyEx(
        best_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv.findContours(
        best_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Sort the remaining contours by size (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:4]

    result = np.zeros_like(best_mask)
    cv.fillPoly(result, contours, color=255)
    return result


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.

    global n_frame
    global lookup_table
    global MGGs
    # global frames

    masks = []
    for camera_i in range(4):
        m = get_mask(frames[camera_i][n_frame], camera_i)
        masks.append(m)
        #show_image(m)
        #show_image(frames[camera_i][n_frame])
        #show_image(cv.bitwise_and(frames[camera_i][n_frame], frames[camera_i][n_frame], mask = m))
    
    visible_voxels = []
    start_reconstruction = time()
    
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


    # COLORS
    voxels_to_cluster = np.array([[x[0], x[2]]
                                  for x in visible_voxels], dtype=np.float32)
    compactness, labels, centers = cv.kmeans(
        voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    colors = []
    for i, _ in enumerate(visible_voxels):
        label = labels[i][0]
        colors.append(labels_to_color[str(label)])
    n_frame += 1
    return visible_voxels, colors

    # FOR EVERY FRAME OF THE VIDEO (5 frames per sec)
    # cap = cv.VideoCapture(cameras_videos_info[1][2])  # video of camera 2
    # camera_i = 1

    # for i in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))):
    #     retF, frame = cap.read()  # get first frame (used for color model)
    #     if i % 10 == 0:
    #         frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #         start_reconstruction = time()
    #         for vox in range(voxel_positions.shape[0]):  # for each voxel id
    #             flag = True  # the voxel is foreground for all cameras (flag)
    #             for i in range(4):  # for each camera
    #                 # extract voxel 3D and 2D coordinates for that camera
    #                 x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i]
    #                 x = int(x)
    #                 y = int(y)
    #                 mask = get_mask(frame, i)
    #                 # check if the pixel is foreground for all cameras
    #                 flag = mask[y, x] == 0

    #             if flag:  # if it is foreground for all cameras
    #                 # adapt to glm format, scale and add to reconstruction
    #                 visible_voxels.append(
    #                     [x_voxels/75, -z_voxels/75, y_voxels/75])
    #         print(f"time to reconstruct all: {time()-start_reconstruction}")

    # cap.release()

    # first frame encountered
    # if n_frame == 0:
    #     exists = os.path.isfile('./data/lookup_table.npz')
    #     if exists:  # if lookup table already exists, load it
    #         with np.load(f'./data/lookup_table.npz') as file:
    #             lookup_table = file['lookup_table']
    #         voxel_positions = np.array(create_cube(
    #             3000, 6000, 6000), dtype=np.float32)
    #         print(f"time to load/create lookup table: {time()-start_lookup}")
    #     else:  # if it does not, create it and save the file
    #         voxel_positions, lookup_table = create_lookup_table(
    #             3000, 6000, 6000)
    #         print(f"time to load/create lookup table: {time()-start_lookup}")

    #     start_reconstruction = time()
    #     for vox in range(voxel_positions.shape[0]):  # for each voxel id
    #         flag = True  # the voxel is foreground for all cameras (flag)
    #         for i in range(4):  # for each camera
    #             # extract voxel 3D and 2D coordinates for that camera
    #             x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i]
    #             x = int(x)
    #             y = int(y)
    #             # check if the pixel is foreground for all cameras
    #             if i == 0 and mask1[0][y, x] == 0:
    #                 flag = False
    #             if i == 1 and mask2[0][y, x] == 0:
    #                 flag = False
    #             if i == 2 and mask3[0][y, x] == 0:
    #                 flag = False
    #             if i == 3 and mask4[0][y, x] == 0:
    #                 flag = False
    #         if flag:  # if it is foreground for all cameras
    #             # adapt to glm format, scale and add to reconstruction
    #             visible_voxels.append([x_voxels/75, -z_voxels/75, y_voxels/75])
    #     print(f"time to reconstruct all: {time()-start_reconstruction}")

    #     # n_frame += 1  # next frame
    #     flags = cv.KMEANS_RANDOM_CENTERS
    #     voxels_to_cluster = np.array([[x[0], x[2]]
    #                                  for x in visible_voxels], dtype=np.float32)
    #     compactness, labels, centers = cv.kmeans(
    #         voxels_to_cluster, 4, None, criteria, 20, flags)
    #     colors = []
    #     for i, _ in enumerate(visible_voxels):
    #         label = labels[i][0]
    #         colors.append(labels_to_color[str(label)])
    #     frame_copy = frame.copy()
    #     pixels_colors = []
    #     color_model = {'0': [], '1': [], '2': [], '3': []}
    #     for i_label, vox in enumerate(visible_voxels):
    #         x_3D, y_3D, z_3D = (
    #             int(vox[0]*75),  int(vox[2]*75), int(-vox[1]*75))
    #         # pos = np.where((lookup_table[:, 0, 1] == x) & (lookup_table[:, 1, 1] == y) & (lookup_table[:, 2, 1] == z))
    #         # print("hello", pos)
    #         # pixels = lookup_table[pos, 3:5, 1]
    #         img_points, _ = cv.projectPoints(np.array(
    #             [x_3D, y_3D, z_3D], dtype=np.float32), rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
    #         x, y = (int(img_points.ravel()[0]), int(
    #             img_points.ravel()[1]))  # x is < 644, y is < 486
    #         # tuple (2d pixel, clustering label, original color)
    #         pixels_colors.append(((x, y), labels[i_label][0], frame_hsv[y, x]))
    #     for pc in pixels_colors:
    #         cv.circle(frame_copy, pc[0], 2, labels_to_color[str(pc[1])], 2)
    #         color_model[str(pc[1])].append(pc[2].tolist())
    #     # print(color_model['3']) # list of np array hsv
    #     if show:
    #         for person in color_model:
    #             MGGs[person] = cv.ml.EM_create()
    #             MGGs[person].setClustersNumber(3)
    #             MGGs[person].trainEM(
    #                 np.array(color_model[person], dtype=np.float32))

    #         for person in color_model:
    #             loglik1 = 0
    #             loglik2 = 0
    #             loglik3 = 0
    #             loglik4 = 0
    #             for pixel in color_model[person]:
    #                 loglik1 += MGGs['0'].predict2(np.array(pixel,
    #                                               dtype=np.float32))[0][0]
    #                 loglik2 += MGGs['1'].predict2(np.array(pixel,
    #                                               dtype=np.float32))[0][0]
    #                 loglik3 += MGGs['2'].predict2(np.array(pixel,
    #                                               dtype=np.float32))[0][0]
    #                 loglik4 += MGGs['3'].predict2(np.array(pixel,
    #                                               dtype=np.float32))[0][0]
    #             print(person, loglik1, loglik2, loglik3, loglik4)
    #         # show_image(frame_copy, "silhouttes")
    #     return visible_voxels, colors
    # else:  # a frame other than the first, we perform optimization by only looking at changed pixels

    #     # load last and current frames' masks for each camera
    #     current_mask1 = np.array(mask1[n_frame], dtype=np.int8)
    #     last_mask1 = np.array(mask1[n_frame-1], dtype=np.int8)
    #     current_mask2 = np.array(mask2[n_frame], dtype=np.int8)
    #     last_mask2 = np.array(mask2[n_frame-1], dtype=np.int8)
    #     current_mask3 = np.array(mask3[n_frame], dtype=np.int8)
    #     last_mask3 = np.array(mask3[n_frame-1], dtype=np.int8)
    #     current_mask4 = np.array(mask4[n_frame], dtype=np.int8)
    #     last_mask4 = np.array(mask4[n_frame-1], dtype=np.int8)

    #     # differences on the masks between the frames for each camera
    #     differences = [current_mask1 - last_mask1, current_mask2 -
    #                    last_mask2, current_mask3 - last_mask3, current_mask4 - last_mask4]

    #     start_reconstruction_opt = time()
    #     removed = 0
    #     added = 0
    #     for n_cam in range(len(differences)):
    #         d = differences[n_cam]  # differences in mask N
    #         xs, ys = np.where((d == 1) | (d == -1))  # voxels that have changed
    #         coords = np.stack((xs, ys), axis=1)
    #         # lookup table for the current camera
    #         lookup = lookup_table[:, :, n_cam]
    #         for coord in coords:  # for each 2D changed pixel
    #             x, y = coord[1], coord[0]
    #             all_pos = np.where((lookup[:, 3] == x) & (lookup[:, 4] == y))[
    #                 0]  # get all corresponding voxels ID
    #             for pos in all_pos:  # for each voxel ID
    #                 # same as frame 1 but with the current mask
    #                 flag = True
    #                 for i in range(4):
    #                     x_voxels, y_voxels, z_voxels, x, y = lookup_table[pos, :, i]
    #                     x = int(x)
    #                     y = int(y)
    #                     vox = [x_voxels/75, -z_voxels/75, y_voxels/75]
    #                     if i == 0 and mask1[n_frame][y, x] == 0:
    #                         flag = False
    #                     if i == 1 and mask2[n_frame][y, x] == 0:
    #                         flag = False
    #                     if i == 2 and mask3[n_frame][y, x] == 0:
    #                         flag = False
    #                     if i == 3 and mask4[n_frame][y, x] == 0:
    #                         flag = False
    #                 if not flag and vox in visible_voxels:  # if at least a camera said background
    #                     visible_voxels.remove(vox)
    #                     removed += 1
    #                     # print("Removed voxel: ", vox)
    #                 if flag:  # if it's foreground
    #                     visible_voxels.append(
    #                         [x_voxels/75, -z_voxels/75, y_voxels/75])
    #                     # print("Appended voxel: ", [x_voxels/75, -z_voxels/75, y_voxels/75])
    #                     added += 1
    #     print(f"time reconstruct optimally: {time()-start_reconstruction_opt}")
    #     print(
    #         f"voxels removed this frame: {removed}, voxels added this frame: {added}")
    #     # n_frame += 1
    #     return visible_voxels


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


start_lookup = time()

# LOOKUP TABLE
exists = os.path.isfile('./data/lookup_table.npz')
if exists:  # if lookup table already exists, load it
    with np.load(f'./data/lookup_table.npz') as file:
        lookup_table = file['lookup_table']
    voxel_positions = np.array(create_cube(
        3000, 6000, 6000), dtype=np.float32)
    print(f"time to load/create lookup table: {time()-start_lookup}")
else:  # if it does not, create it and save the file
    voxel_positions, lookup_table = create_lookup_table(
        3000, 6000, 6000)
    print(f"time to load/create lookup table: {time()-start_lookup}")


for camera_i in range(4):
    # FOR EVERY FRAME OF THE VIDEO (5 frames per sec)
    frames_c = []
    cap = cv.VideoCapture(cameras_videos_info[camera_i][2])
    for i in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2):
        retF, frame = cap.read()  # get first frame (used for color model)
        if i % 10 == 0:
            frames_c.append(frame)
    frames.append(frames_c)
    cap.release()
    
    cap = cv.VideoCapture(cameras_videos_info[camera_i][0])
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames_background = 40
    imgs_list_background = [cap.read()[1] for i in range(n_frames_background)]
    cap.release()

    # get average of baground
    background = average_images(imgs_list_background)
    backgrounds.append(background)

