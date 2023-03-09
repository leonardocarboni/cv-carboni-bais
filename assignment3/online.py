import numpy as np
import cv2 as cv
from utils import *
from time import time
import os

# TODO: Voxel space più grosso
# TODO: Finire implementazione lables corrette
# TODO:

lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

cam_to_frame = {0 : 10, 1: 0, 2:41, 3: 52}
frames = []
backgrounds = []

show = True

# loading parameters of cam2 for color model
camera_matrixes = []
dist_coeffs = []
rvecs_extr = []
tvecs_extr = []

for i in range(4):
    s = cv.FileStorage(
        f"./data/cam{i+1}/config.xml", cv.FileStorage_READ)
    camera_matrixes.append(s.getNode('camera_matrix').mat())
    dist_coeffs.append(s.getNode('dist_coeffs').mat())
    rvecs_extr.append(s.getNode('rvec_extr').mat())
    tvecs_extr.append(s.getNode('tvec_extr').mat())
    s.release()


plane = np.zeros((128, 128, 3))


def get_mask(frame, camera_i):

    w, h, _ = frame.shape
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    background_pixels_hsv = cv.cvtColor(
        backgrounds[camera_i], cv.COLOR_BGR2HSV)
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


def create_cube(width, height, depth):
    "creates a solid with resolution 100x100x100 with the current inputs"
    cube = []
    for x in np.arange(-width, width, 80):
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


def get_color_model():
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


def start_online(MGGs):
    # for each frame we considered
    for n_frame in range(0, len(frames[0]), 10):
        masks = []
        for camera_i in range(4):
            m = get_mask(frames[camera_i][n_frame], camera_i)
            masks.append(m)
            if show:
                a = 3
                # show_image(m)
                # show_image(frames[camera_i][n_frame])
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
        clusters_to_tracking(centers, labels)
        pixels_colors = []
        color_model = {0: [], 1: [], 2: [], 3: []}
        for i_label, vox in enumerate(visible_voxels):
            x_3D, y_3D, z_3D = (
                int(vox[0]*75),  int(vox[2]*75), int(-vox[1]*75))
            img_points, _ = cv.projectPoints(np.array(
                [x_3D, y_3D, z_3D], dtype=np.float32), rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            x, y = (int(img_points.ravel()[0]), int(
                img_points.ravel()[1]))  # x is < 644, y is < 486
            # tuple (2d pixel, clustering label, original color)
            pixels_colors.append(
                ((x, y), labels[i_label][0], frames[1][n_frame][y, x]))
        for pc in pixels_colors:
            color_model[pc[1]].append(pc[2].tolist())
        correct_labels = np.zeros(4)
        for person in color_model:
            logliks = np.zeros(4)
            for pixel in color_model[person]:
                logliks[0] += MGGs[0].predict2(np.array(pixel,
                                                          dtype=np.float32))[0][0]
                logliks[1] += MGGs[1].predict2(np.array(pixel,
                                                          dtype=np.float32))[0][0]
                logliks[2] += MGGs[2].predict2(np.array(pixel,
                                                          dtype=np.float32))[0][0]
                logliks[3] += MGGs[3].predict2(np.array(pixel,
                                                          dtype=np.float32))[0][0]
            correct_labels[int(person)] = np.argmax(
                logliks)  # current label to true label
        frame_copy = frames[1][n_frame].copy()
        frame_copy2 = frames[1][n_frame].copy()
        for pc in pixels_colors:
            current_label = pc[1]
            offline_label = int(correct_labels[current_label])
            color = labels_to_color[offline_label]
            online_color = labels_to_color[int(current_label)]
            cv.circle(frame_copy, pc[0], 2, color, 2)
            cv.circle(frame_copy2, pc[0], 2, online_color, 2)
        show_image(frame_copy, "silhouttes")
        show_image(frame_copy2, "silhouttes")
    return


last_points = []


def clusters_to_tracking(centers, labels):
    global plane
    global last_points
    older = last_points

    last_points = []
    for i in range(len(centers)):
        x, y = centers[i]
        x = int(x)+64
        y = int(y)+64
        last_points.append((x, y))
        plane[y, x] = labels_to_color[i]
        if len(older) != 0:
            cv.line(plane, (x, y), older[i], color=labels_to_color[i])
    show_image(plane, "silhouttes")


# execution
start_lookup = time()
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

MGGs = get_color_model()[1]

start_online(MGGs)  
