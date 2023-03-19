import os
from time import time

import glm
from scipy.spatial import distance
from sklearn.preprocessing import normalize

from utils import *

# flag to perform tracking
tracking = False

# centers coordinates to be drawn for tracking
centers_list = []

# 4 GMMs for the matching task
gaussian_mixture_models = None

# misc for the reconstruction
block_size = 1.0
lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# map converting labels to colors for the visualization
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

# map to select the hand-picked frame to build the color model for each camera
cam_to_frame = {0: 10, 1: 0, 2: 41, 3: 52}

# loading parameters of the cameras and masks of all frames
camera_matrices = []
distance_coefficients = []
rotation_vector_extrinsic = []
translation_vector_extrinsic = []

masks_all_frames = []
for i in range(4):
    with np.load(f'./data/cam{i + 1}/masks.npz') as file:
        masks_all_frames.append(file['masks'])

for i in range(4):
    s = cv.FileStorage(
        f"./data/cam{i + 1}/config.xml", cv.FileStorage_READ)
    camera_matrices.append(s.getNode('camera_matrix').mat())
    distance_coefficients.append(s.getNode('dist_coeffs').mat())
    rotation_vector_extrinsic.append(s.getNode('rvec_extr').mat())
    translation_vector_extrinsic.append(s.getNode('tvec_extr').mat())
    s.release()


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -
            block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def reconstruct_voxels(n_frame):
    # function to get the 3D reconstruction during the online section
    # input: frame to reconstruct
    # output: voxels positions and labels, clustering centers

    start_reconstruction = time()
    visible_voxels = []

    for vox in range(voxel_positions.shape[0]):  # for each voxel id
        flag = True  # the voxel is foreground for all cameras (flag)
        x_voxels, y_voxels, z_voxels = 0, 0, 0
        for i_camera in range(4):  # for each camera
            # extract voxel 3D and 2D coordinates for that camera
            x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i_camera]
            x = int(x)
            y = int(y)
            # check if the pixel is foreground for all cameras
            mask = masks_all_frames[i_camera][n_frame]
            if mask[y, x] == 0:
                flag = False

        if flag:  # if it is foregrounded for all cameras
            # adapt to glm format, scale and add to reconstruction
            visible_voxels.append(
                [x_voxels / 75, -z_voxels / 75, y_voxels / 75])

    # extract x and y coordinates for clustering (ignore vertical)
    voxels_to_cluster = np.array([[x[0], x[2]]
                                  for x in visible_voxels], dtype=np.float32)
    # clustering
    _, labels, centers = cv.kmeans(
        voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    # removing ghost voxels
    voxels_to_remove = []

    # for each cluster
    for label_center, center in enumerate(centers):
        # get voxels belonging to the cluster
        filtered_list = [lst for lst, current_lab in zip(
            visible_voxels, labels) if current_lab == label_center]

        avg_dist_from_center = 0

        # for each voxel of the cluster
        for x, z, y in filtered_list:
            # get its euclidean distance from the center
            avg_dist_from_center += distance.euclidean((x, y), center)
        avg_dist_from_center = avg_dist_from_center / \
                               len(filtered_list)  # calculate average

        # for each voxel of the cluster
        for x, z, y in filtered_list:
            # check if the voxel is too distant from the center
            if distance.euclidean((x, y), center) > avg_dist_from_center * 1.5:
                voxels_to_remove.append([x, z, y])  # store the voxel

    # for each voxel marked for removal in all clusters
    for vox in voxels_to_remove:
        visible_voxels.remove(vox)  # delete it

    # re-clustering
    voxels_to_cluster = np.array([[x[0], x[2]]
                                  for x in visible_voxels], dtype=np.float32)
    _, labels, centers = cv.kmeans(
        voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    print(
        f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")

    return visible_voxels, labels, centers


def set_voxel_positions(width, height, depth, n_frame):
    # function to handle the online section:
    # for each frame, it gets the 3D reconstruction, selects the least occluded camera
    # and performs the matching task
    # input: frame to reconstruct (width, height, depth are not used as the lookup table is global, built on execution)
    # output: voxel positions and colors

    # offline trained GMMs and lookup table
    global gaussian_mixture_models
    global lookup_table

    # 3D reconstruction for the frame
    visible_voxels, labels, centers = reconstruct_voxels(n_frame)

    # 2D coordinates of the projected voxels for each camera
    images_points = []

    # number of connected components for each camera (used for occlusion detection)
    counts = []

    # 2D coordinates of the pixels for each person and for each camera
    people_pixels = []

    # for each camera
    for n_camera in range(4):
        image_points = []  # 2D coordinates of the projected voxels for this camera
        frame_camera = frames[n_camera][n_frame]  # frame for this camera
        mask = masks_all_frames[n_camera][n_frame]  # mask for this camera

        # for each visible voxel
        for i_label, vox in enumerate(visible_voxels):
            # scaling back to original values
            x_3d, y_3d, z_3d = (
                int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
            coordinates_2d, _ = cv.projectPoints(np.array(
                [x_3d, y_3d, z_3d], dtype=np.float32), rotation_vector_extrinsic[n_camera],
                translation_vector_extrinsic[n_camera],
                camera_matrices[n_camera], distance_coefficients[n_camera])  # projecting the voxel
            x_2d, y_2d = (int(coordinates_2d.ravel()[0]), int(
                coordinates_2d.ravel()[1]))  # rounding
            image_points.append(
                ((x_2d, y_2d), labels[i_label][0]))  # store the 2D coordinate

        # store all 2D coordinates for this camera
        images_points.append(image_points)

        # storing all 2D pixel coordinates for each person for this camera
        person_pixels = {0: [], 1: [], 2: [], 3: []}
        for pixel, label_of_pixel in image_points:
            person_pixels[label_of_pixel].append(pixel)
        people_pixels.append(person_pixels)

        # get number of connected component for this camera and store it
        count, _, _, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        counts.append(count)

    # choosing the least occluded camera
    n_camera_with_best_separation = np.argmax(counts)

    # map to match online labels to the offline labels, built polling the GMMs
    # also stores the associated log likelihood score (online label -> (offline label, log likelihood))
    best_person = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]

    # get the frame from the least occluded camera
    frame_camera = frames[n_camera_with_best_separation][n_frame]

    # for each person clustered in this view
    for person in people_pixels[n_camera_with_best_separation]:
        # mask to remove the trousers
        mask = np.zeros(frame_camera.shape[:2], np.uint8)
        # get the 2D coordinates of the person
        pixels = people_pixels[n_camera_with_best_separation][person]
        # calculate approximate y-waist coordinate
        waist = np.max([x[1] for x in pixels]) - np.min([x[1]
                                                         for x in pixels]) // 1.5

        # build the mask
        for x_pixel, y_pixel in pixels:
            if y_pixel < waist:
                mask[y_pixel, x_pixel] = 255

        log_likelihoods = []  # log likelihoods of the 4 GMMs for this person

        # for each GMM
        for i_gmm in range(4):
            log_likelihood = 0

            # for each pixel of the person in HSV
            for pixel in cv.cvtColor(frames[n_camera_with_best_separation][n_frame], cv.COLOR_BGR2HSV)[
                mask == 255].tolist():
                # get the likelihood score for this model and accumulate it for all pixels
                log_likelihood += gaussian_mixture_models[i_gmm].predict2(np.array(pixel, dtype=np.float32))[0][
                    0]
            # normalize the total score by the amount of pixels for this person
            log_likelihoods.append(
                log_likelihood / len(frames[n_camera_with_best_separation][n_frame][mask == 255].tolist()))

        # store the best guess and its likelihood score for this online label
        best_person[person] = (
            np.argmax(log_likelihoods), max(log_likelihoods))

    # removing duplicate guesses, performed twice for corner cases of double duplicates
    for _ in range(2):
        unique_best_person = np.unique(
            [x[0] for x in best_person])  # get all uniques guesses
        if len(unique_best_person) < 4:  # if there is at least one missing
            count = [0, 0, 0, 0]
            missing_person = [x for x in range(
                4) if x not in unique_best_person][0]  # get who is missing

            # for each person
            for person, _ in best_person:
                # count occurrences
                if person in unique_best_person:
                    count[person] += 1

            # for each person
            for person in range(4):
                if count[person] > 1:  # if it is duplicate
                    probs_of_person = []
                    # extracting likelihood scores of the duplicates
                    for x_pixel in best_person:
                        if x_pixel[0] == person:  # the duplicate
                            probs_of_person.append(x_pixel[1])
                        else:  # append value large enough to not be chosen for other people
                            probs_of_person.append(1)
                    # get the duplicate guess with the lowest score
                    index_of_missing_person = np.argmin(probs_of_person)
                    best_person[index_of_missing_person] = (missing_person, min(
                        probs_of_person))  # substitute it with the missing person

    # tracking section
    centers_right = [(0, 0), (0, 0), (0, 0), (0, 0)]

    # relabel centers to match the offline labels
    for i_label in range(4):
        new_label = best_person[i_label][0]
        x_pixel = centers[i_label][0]
        y_pixel = centers[i_label][1]
        centers_right[new_label] = (x_pixel, y_pixel)

    centers_list.append(centers_right)

    # coloring the voxels
    colors = []
    # for each voxel
    for i_vox, vox in enumerate(visible_voxels):
        # convert online label to matching offline label
        color = labels_to_color[best_person[labels[i_vox][0]][0]]
        colors.append((color[2], color[1], color[0]))  # store it in RGB
    return visible_voxels, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    positions = []
    for i_camera in range(1, 5):
        fs = cv.FileStorage(
            f"data/cam{i_camera}/config.xml", cv.FileStorage_READ)
        tvec_extr = fs.getNode('tvec_extr').mat()
        R = fs.getNode('R_MAT').mat()
        positions.append(np.dot(-R.T, tvec_extr))
        fs.release()
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
    cam_rotations = []
    for i_camera in range(1, 5):
        fs = cv.FileStorage(
            f"data/cam{i_camera}/config.xml", cv.FileStorage_READ)
        tvec_extr = fs.getNode('tvec_extr').mat()
        R = fs.getNode('R_MAT').mat()

        t1 = np.hstack((R, tvec_extr))
        t1 = np.vstack((t1, [0, 0, 0, 1]))
        # swap y and z
        t1[:, [1, 2]] = t1[:, [2, 1]]
        # invert x rotation of the cameras
        t1[:, 1] = -t1[:, 1]
        # transform to mat4
        rot = glm.mat4(t1.T)
        # rotate cameras by 90 degrees because they point on the wrong side
        rot = glm.rotate(rot, -90 * np.pi / 180, [0, 1, 0])

        cam_rotations.append(rot)
    return cam_rotations


def reconstruct_all_voxels():
    # function to reconstruct all the 4 frames in the offline section
    # they will be used to build the color models
    # output: visible voxels of the 4 reconstructions and their clustering labels

    all_visible_voxels = []
    all_labels = []
    start_reconstruction = time()
    # 4 reconstructions for the 4 frames needed for the color models
    for j_camera in range(4):
        visible_voxels = []
        for vox in range(voxel_positions.shape[0]):  # for each voxel id
            flag = True  # the voxel is foreground for all cameras (flag)
            x_voxels, y_voxels, z_voxels = 0, 0, 0
            for i_camera in range(4):  # for each camera
                # extract voxel 3D and 2D coordinates for that camera
                x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i_camera]
                x = int(x)
                y = int(y)
                # check if the pixel is foreground for all cameras
                mask = masks_all_frames[i_camera][cam_to_frame[j_camera]]
                if mask[y, x] == 0:
                    flag = False

            if flag:  # if it is foregrounded for all cameras
                # adapt to glm format, scale and add to reconstruction
                visible_voxels.append(
                    [x_voxels / 75, -z_voxels / 75, y_voxels / 75])

        # clustering  and removing ghost voxels (same as above online phase)
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        _, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        voxels_to_remove = []
        for label_center, center in enumerate(centers):
            filtered_list = [lst for lst, current_lab in zip(
                visible_voxels, labels) if current_lab == label_center]
            avg_dist_from_center = 0
            for x, z, y in filtered_list:
                avg_dist_from_center += distance.euclidean((x, y), center)
            avg_dist_from_center = avg_dist_from_center / len(filtered_list)
            for x, z, y in filtered_list:
                if distance.euclidean((x, y), center) > avg_dist_from_center * 1.5:
                    voxels_to_remove.append([x, z, y])
        for vox in voxels_to_remove:
            visible_voxels.remove(vox)
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        _, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        # list of 4 lists, that has all labels for each visible voxel
        all_labels.append(labels)
        all_visible_voxels.append(visible_voxels)
    print(
        f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")

    return all_visible_voxels, all_labels


def create_cube(width, height, depth):
    """creates a solid with resolution 100x100x100 with the current inputs"""
    cube = []
    for x in np.arange(-width, width, 80):
        for y in np.arange(-depth // 2, depth // 2, 80):
            for z in np.arange(-height, height, 80):
                cube.append([x, y, z])
    return cube


def get_gaussian_mixture_models():
    # function that handles the offline phase of creating the color models
    # output: the 4 GMMs
    global lookup_table

    # for each camera, stores the 2D coordinates of each person
    person_to_colors = [{0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []},
                        {0: [], 1: [], 2: [], 3: []}]

    # initialize the 4 GMMs
    gmm_s = {0: cv.ml.EM_create(
    ), 1: cv.ml.EM_create(), 2: cv.ml.EM_create(), 3: cv.ml.EM_create()}

    # set the cluster number of each GMM to 3
    for person_i in range(4):
        gmm_s[person_i].setClustersNumber(3)

    # get the reconstructions of the 4 frames
    all_visible_voxels, all_labels = reconstruct_all_voxels()

    # list of length 4, for each camera its 2d visible pixels, its clustering label and its original color
    pixels_colors = []

    # for each visible voxel in each camera's hand picked frame
    for i_camera, visible_voxels in enumerate(all_visible_voxels):

        # projecting the voxels (same as online phase above)
        image_points = []
        chosen_frame = cam_to_frame[i_camera]
        for i_label, vox in enumerate(visible_voxels):
            x_3d, y_3d, z_3d = (
                int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
            coordinates_2d, _ = cv.projectPoints(np.array(
                [x_3d, y_3d, z_3d], dtype=np.float32), rotation_vector_extrinsic[i_camera],
                translation_vector_extrinsic[i_camera],
                camera_matrices[i_camera], distance_coefficients[i_camera])
            x_2d, y_2d = (int(coordinates_2d.ravel()[0]), int(
                coordinates_2d.ravel()[1]))
            # store the information needed:
            # tuple (2d pixel, clustering label, original color)
            image_points.append(
                ((x_2d, y_2d), all_labels[i_camera][i_label][0], frames[i_camera][chosen_frame][y_2d, x_2d]))
        pixels_colors.append(image_points)  # store information of each camera

    # getting the coordinates of each person for each camera
    for i_camera, pixels_color in enumerate(pixels_colors):
        # for each person
        for pixel, label_pixel, color in pixels_color:
            person_to_colors[i_camera][label_pixel].append(pixel)

    # hand built map to match the offline labels of the 4 cameras
    best_person = [[0, 1, 2, 3], [2, 1, 3, 0], [2, 0, 3, 1], [2, 3, 1, 0]]

    # group the colors of each person coming from all 4 views
    person_training_data = [[], [], [], []]

    # for each camera
    for i_camera, cameras in enumerate(person_to_colors):
        # for each person
        for person in cameras:
            pixels = cameras[person]  # all the 2D coordinates of the person

            # building and applying a mask for the trousers (same as online above)
            mask = np.zeros(
                frames[i_camera][cam_to_frame[i_camera]].shape[:2], np.uint8)
            waist = np.max([x[1] for x in pixels]) - \
                    np.min([x[1] for x in pixels]) // 1.5
            for x, y in pixels:
                if y < waist:
                    mask[y, x] = 255
            person_pixels = cv.cvtColor(frames[i_camera][cam_to_frame[i_camera]], cv.COLOR_BGR2HSV)[
                mask == 255].tolist()

            # store the colors in the corresponding person for each view using the map
            person_training_data[best_person[i_camera]
            [person]].append(person_pixels)

    # training the 4 GMMs
    for i_person in range(4):
        # flatten the data
        data = [item for sublist in person_training_data[i_person]
                for item in sublist]
        # train the gaussian mixture model
        gmm_s[i_person].trainEM(
            np.array(data, dtype=np.float32))

    return gmm_s


def create_lookup_table(width, height, depth):
    """
    Returns the lookup table mapping each voxel to its corresponding 3D and 2D coordinates for each camera
    Inputs are for the placement of the solid for the reconstruction
    """
    # create solid
    voxel_positions_list = np.array(create_cube(
        width, height, depth), dtype=np.float32)
    # initialize look up table
    lut = np.zeros((voxel_positions_list.shape[0], 5, 4))
    for i_camera in range(1, 5):  # for each camera
        # load parameters
        fs = cv.FileStorage(
            f"data/cam{i_camera}/config.xml", cv.FileStorage_READ)
        camera_matrix = fs.getNode('camera_matrix').mat()
        dist_coeffs = fs.getNode('dist_coeffs').mat()
        tvec_extr = fs.getNode('tvec_extr').mat()
        rvec_extr = fs.getNode('rvec_extr').mat()
        fs.release()
        # for each 3D point of the voxel solid
        for i_voxel, pos in enumerate(voxel_positions_list):
            # project the point in the 2D image plane for this camera
            image_point, _ = cv.projectPoints(
                pos, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            image_point = image_point.ravel()
            x = image_point[0]
            y = image_point[1]
            if 0 <= x < 644 and 0 <= y < 486:  # if the 2D pixel is in range of the frame
                # store 2D and 3D coordinates
                lut[i_voxel, :, i_camera - 1] = [int(pos[0]), int(pos[1]), int(pos[2]), int(x), int(y)]

    np.savez('data/lookup_table', lookup_table=lut)
    return voxel_positions_list, lut


# EXECUTION

# creating the lookup table for reconstruction at the start of execution
start_lookup = time()
exists = os.path.isfile('./data/lookup_table.npz')
if exists:  # if lookup table already exists, load it
    with np.load(f'./data/lookup_table.npz') as file:
        lookup_table = file['lookup_table']
    voxel_positions = np.array(create_cube(
        3000, 6000, 6000), dtype=np.float32)
    print(f"time to load/create lookup table: {time() - start_lookup}")
else:  # if it does not, create it and save the file
    voxel_positions, lookup_table = create_lookup_table(
        3000, 6000, 6000)
    print(f"time to load/create lookup table: {time() - start_lookup}")

# loading all the frames used from all 4 views (5 FPS)
frames = []
for camera_i in range(4):
    frames_c = []
    cap = cv.VideoCapture(cameras_videos_info[camera_i][2])
    for i in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 2):
        retF, frame = cap.read()  # get first frame (used for color model)
        if i % 10 == 0:
            frames_c.append(frame)
    frames.append(frames_c)
    cap.release()

# offline execution, get the 4 GMMs
gaussian_mixture_models = get_gaussian_mixture_models()

# next section is for the tracking task, which is done without visualization
# run this file rather than the executable file with the tracking flag on to get the tracking
if tracking:
    # perform the online for all frames
    for f in range(0, 270):
        try:
            _, _ = set_voxel_positions(1, 2, 3, f)
        except:
            print("fail at frame", f)

    paths = np.zeros((500, 500, 3), dtype=np.uint8)
    paths2 = np.zeros((500, 500, 3), dtype=np.uint8)
    paths3 = np.zeros((500, 500, 3), dtype=np.uint8)
    paths4 = np.zeros((500, 500, 3), dtype=np.uint8)
    old_c = []
    for c_in_frame in centers_list:
        old_c = c_in_frame
        for label, c in enumerate(c_in_frame):
            # normalize the coordinates
            x = (c[0]) / 128 * 1000
            y = (c[1]) / 128 * 1000
            old_x = (old_c[label][0] + 64) / 128 * 1000
            old_y = (old_c[label][1] + 64) / 128 * 1000

            cv.circle(paths, (int(x), int(y)), 3, labels_to_color[label], -1)

            if len(old_c) > 0:
                # calculate the distance between the current and the previous position
                dist = np.linalg.norm(np.array(c) - np.array(old_c[label]))
                if dist < 10:
                    cv.line(paths2, (int(x), int(y)), (int(old_x),
                                                       int(old_y)), labels_to_color[label], 2)
                if dist < 20:
                    cv.line(paths3, (int(x), int(y)), (int(old_x),
                                                       int(old_y)), labels_to_color[label], 2)
                if dist < 8:
                    cv.line(paths4, (int(x), int(y)), (int(old_x),
                                                       int(old_y)), labels_to_color[label], 2)

    cv.imwrite("paths_v2.png", paths)
    cv.imwrite("paths_v3_10.png", paths2)
    cv.imwrite("paths_v3_20.png", paths3)
    cv.imwrite("paths_v3_8.png", paths4)
