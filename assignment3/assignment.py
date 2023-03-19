import os
from time import time

import glm
from scipy.spatial import distance
from sklearn.preprocessing import normalize

from utils import *

show = True

centers_list = []

gaussian_mixture_models = None
block_size = 1.0
n_frame = 0
lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

cap = cv.VideoCapture(cameras_videos_info[1][2])  # video of camera 2
retF, frame = cap.read()  # get first frame (used for color model)
frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
cap.release()

MGGs = {'0': None, '1': None, '2': None, '3': None}
cam_to_frame = {0: 10, 1: 0, 2: 41, 3: 52}

# loading parameters of cam2 for color model
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

plane = np.zeros((128, 128, 3))


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
    all_visible_voxels = []
    all_labels = []
    start_reconstruction = time()
    # 4 reconstructions for the 4 frames needed for the color models
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

    # COLORS
    voxels_to_cluster = np.array([[x[0], x[2]]
                                  for x in visible_voxels], dtype=np.float32)
    compactness, labels, centers = cv.kmeans(
        voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    voxels_to_remove = []
    for label, center in enumerate(centers):
        filtered_list = [lst for lst, current_lab in zip(visible_voxels, labels) if current_lab == label]
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
    compactness, labels, centers = cv.kmeans(
        voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    print(f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")

    return visible_voxels, labels, centers


def set_voxel_positions(width, height, depth, n_frame):
    global gaussian_mixture_models
    global lookup_table

    visible_voxels, labels, centers = reconstruct_voxels(n_frame)

    images_points = []
    counts = []
    people_pixels = []

    for n_camera in range(4):
        image_points = []
        frame_camera = frames[n_camera][n_frame]
        mask = masks_all_frames[n_camera][n_frame]

        for i_label, vox in enumerate(visible_voxels):
            x_3d, y_3d, z_3d = (int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
            coordinates_2d, _ = cv.projectPoints(np.array(
                [x_3d, y_3d, z_3d], dtype=np.float32), rotation_vector_extrinsic[n_camera],
                translation_vector_extrinsic[n_camera],
                camera_matrices[n_camera], distance_coefficients[n_camera])
            x_2d, y_2d = (int(coordinates_2d.ravel()[0]), int(
                coordinates_2d.ravel()[1]))
            image_points.append(
                ((x_2d, y_2d), labels[i_label][0]))
        images_points.append(image_points)

        person_pixels = {0: [], 1: [], 2: [], 3: []}
        for pixel, label in image_points:
            person_pixels[label].append(pixel)
        people_pixels.append(person_pixels)
        count, _, _, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        counts.append(count)

    n_camera_with_best_separation = np.argmax(counts)
    best_person = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
    frame_camera = frames[n_camera_with_best_separation][n_frame]
    for person in people_pixels[n_camera_with_best_separation]:
        # chosen_frame = cv.cvtColor(frame_camera, cv.COLOR_BGR2HSV)
        mask = np.zeros(frame_camera.shape[:2], np.uint8)
        pixels = people_pixels[n_camera_with_best_separation][person]
        waist = np.max([x[1] for x in pixels]) - np.min([x[1] for x in pixels]) // 1.5
        for x, y in pixels:
            if y < waist:
                mask[y, x] = 255
        probabilities = []
        for i_gmm in range(4):
            log_likelihood = 0
            for pixel in cv.cvtColor(frames[n_camera_with_best_separation][n_frame], cv.COLOR_BGR2HSV)[
                mask == 255].tolist():
                log_likelihood += gaussian_mixture_models[i_gmm].predict2(np.array(pixel, dtype=np.float32))[0][
                    0]
            probabilities.append(
                log_likelihood / len(frames[n_camera_with_best_separation][n_frame][mask == 255].tolist()))
        # people[n_camera_with_best_separation] = probabilities
        best_person[person] = (np.argmax(probabilities), max(probabilities))

    # best people is a list of dictionaries, each dictionary contains, for each person/label, the best cluster
    # number (of that camera) and the probability of that cluster

    # get only the unique values of the best person
    for i in range(2):
        unique_best_person = np.unique([x[0] for x in best_person])
        if len(unique_best_person) < 4:
            count = [0, 0, 0, 0]
            missing_person = [x for x in range(4) if x not in unique_best_person][0]
            for person, _ in best_person:
                if person in unique_best_person:
                    count[person] += 1
            for person in range(4):
                if count[person] > 1:
                    probs_of_person = []
                    for x in best_person:
                        if x[0] == person:
                            probs_of_person.append(x[1])
                        else:
                            probs_of_person.append(1)

                    index_of_missing_person = np.argmin(probs_of_person)
                    best_person[index_of_missing_person] = (missing_person, min(probs_of_person))

    frames_x = []
    for i_camera in range(4):
        frame_x = frames[i_camera][n_frame].copy()
        for pixel, label in images_points[i_camera]:
            cv.circle(frame_x, pixel, 2, labels_to_color[best_person[label][0]], -1)
        frames_x.append(frame_x)
    image_to_show = np.concatenate((np.concatenate((frames_x[0], frames_x[1]), axis=1),
                                    np.concatenate((frames_x[2], frames_x[3]), axis=1)), axis=0)

    centers_right = [(0, 0), (0, 0), (0, 0), (0, 0)]
    for i_label in range(4):
        new_label = best_person[i_label][0]
        x = centers[i_label][0]
        y = centers[i_label][1]
        centers_right[new_label] = (x, y)

    centers_list.append(centers_right)

    colors = []

    for i_vox, vox in enumerate(visible_voxels):
        color = labels_to_color[best_person[labels[i_vox][0]][0]]
        colors.append((color[2], color[1], color[0]))

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


def reconstruct_all_voxels():
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

        # COLORS
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        voxels_to_remove = []
        for label, center in enumerate(centers):
            filtered_list = [lst for lst, current_lab in zip(visible_voxels, labels) if current_lab == label]
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
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        all_labels.append(labels)  # list of 4 lists, that has all labels for each visible voxel
        all_visible_voxels.append(visible_voxels)
    print(f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")
    # end of reconstructions
    return all_visible_voxels, all_labels


def create_cube(width, height, depth):
    "creates a solid with resolution 100x100x100 with the current inputs"
    cube = []
    for x in np.arange(-width, width, 80):
        for y in np.arange(-depth // 2, depth // 2, 80):
            for z in np.arange(-height, height, 80):
                cube.append([x, y, z])
    return cube


def get_gaussian_mixture_models():
    global lookup_table

    person_to_colors = [{0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []},
                        {0: [], 1: [], 2: [], 3: []}]
    gaussian_mixture_models = {0: cv.ml.EM_create(), 1: cv.ml.EM_create(), 2: cv.ml.EM_create(), 3: cv.ml.EM_create()}

    for person_i in range(4):
        gaussian_mixture_models[person_i].setClustersNumber(3)

    all_visible_voxels, all_labels = reconstruct_all_voxels()

    # list of length 4, for each camera its 2d visible pixels, its clustering label and its original color
    pixels_colors = []

    # for each visible voxel in each camera's best frame
    for i_camera, visible_voxels in enumerate(all_visible_voxels):

        image_points = []
        chosen_frame = cam_to_frame[i_camera]
        frame_copy = frames[i_camera][chosen_frame].copy()
        for i_label, vox in enumerate(visible_voxels):
            x_3d, y_3d, z_3d = (int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
            coordinates_2d, _ = cv.projectPoints(np.array(
                [x_3d, y_3d, z_3d], dtype=np.float32), rotation_vector_extrinsic[i_camera],
                translation_vector_extrinsic[i_camera],
                camera_matrices[i_camera], distance_coefficients[i_camera])
            x_2d, y_2d = (int(coordinates_2d.ravel()[0]), int(
                coordinates_2d.ravel()[1]))  # x is < 644, y is < 486
            # tuple (2d pixel, clustering label, original color)
            image_points.append(
                ((x_2d, y_2d), all_labels[i_camera][i_label][0], frames[i_camera][chosen_frame][y_2d, x_2d]))
            cv.circle(frame_copy, (x_2d, y_2d), 5, labels_to_color[all_labels[i_camera][i_label][0]], -1)
        pixels_colors.append(image_points)
        # show_image(frame_copy, "offline labels")

    # AT THIS POINT WE HAVE THE 2D PIXELS, THEIR CLUSTERING LABELS AND THEIR ORIGINAL COLORS

    # for each camera
    for i_camera, pixels_color in enumerate(pixels_colors):
        # for each person
        for pixel, label, color in pixels_color:
            # if the pixel is of the current person
            person_to_colors[i_camera][label].append(pixel)

    # AT THIS POINT WE HAVE THE 2D PIXELS OF EACH PERSON FOR EACH CAMERA

    best_person = [[0, 1, 2, 3], [2, 1, 3, 0], [2, 0, 3, 1], [2, 3, 1, 0]]
    person_training_data = [[], [], [], []]
    for i_camera, cameras in enumerate(person_to_colors):
        # for each person

        for person in cameras:
            # calculate the histogram
            pixels = cameras[person]
            mask = np.zeros(frames[i_camera][cam_to_frame[i_camera]].shape[:2], np.uint8)

            waist = np.max([x[1] for x in pixels]) - np.min([x[1] for x in pixels]) // 1.5

            for x, y in pixels:
                if y < waist:
                    mask[y, x] = 255

            person_pixels = cv.cvtColor(frames[i_camera][cam_to_frame[i_camera]], cv.COLOR_BGR2HSV)[
                mask == 255].tolist()
            # person_pixels = frames[i_camera][cam_to_frame[i_camera]][mask == 255].tolist()

            person_training_data[best_person[i_camera][person]].append(person_pixels)

    for i_person in range(4):
        # flatten the data
        data = [item for sublist in person_training_data[i_person] for item in sublist]
        # train the gaussian mixture model
        gaussian_mixture_models[i_person].trainEM(np.array(data, dtype=np.float32))

    # AT THIS POINT WE HAVE THE HISTOGRAMS OF EACH PERSON FOR EACH CAMERA
    # training the gaussian mixture models
    for i_person in range(4):
        # flatten the data
        data = [item for sublist in person_training_data[i_person] for item in sublist]
        # train the gaussian mixture model
        gaussian_mixture_models[i_person].trainEM(np.array(data, dtype=np.float32))

    return gaussian_mixture_models


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

frames = []
backgrounds = []
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

gaussian_mixture_models = get_gaussian_mixture_models()

for f in range(0, 270):
    try:
        _, _ = set_voxel_positions(1, 2, 3, f)
    except:
        print("fail at frame", f)

np.savez('centers_list', centers_list=centers_list)

paths = np.zeros((500, 500, 3), dtype=np.uint8)
paths2 = np.zeros((500, 500, 3), dtype=np.uint8)
paths3 = np.zeros((500, 500, 3), dtype=np.uint8)
paths4 = np.zeros((500, 500, 3), dtype=np.uint8)
old_c = []
for c_in_frame in centers_list:
    old_c = c_in_frame
    for label, c in enumerate(c_in_frame):
        # normalize the coordinates to be in a 500 x 500 image
        x = (c[0]) / 128 * 1000
        y = (c[1]) / 128 * 1000
        # normalize the coordinates to be in a 500 x 500 image

        old_x = (old_c[label][0] + 64) / 128 * 1000
        old_y = (old_c[label][1] + 64) / 128 * 1000

        cv.circle(paths, (int(x), int(y)), 3, labels_to_color[label], -1)

        if len(old_c) > 0:
            # calculate the distance between the current and the previous position
            dist = np.linalg.norm(np.array(c) - np.array(old_c[label]))
            if dist < 10:
                cv.line(paths2, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)
            if dist < 20:
                cv.line(paths3, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)
            if dist < 8:
                cv.line(paths4, (int(x), int(y)), (int(old_x), int(old_y)), labels_to_color[label], 2)

cv.imwrite("paths_v2.png", paths)
cv.imwrite("paths_v3_10.png", paths2)
cv.imwrite("paths_v3_20.png", paths3)
cv.imwrite("paths_v3_8.png", paths4)
