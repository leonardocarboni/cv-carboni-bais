import os
from time import time

import matplotlib

from utils import *

matplotlib.use('TkAgg')

lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

# best frames for each camera
cam_to_frame = {0: 10, 1: 0, 2: 41, 3: 52}
frames = []
backgrounds = []

show = True

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


def get_mask(frame, camera_i):
    w, h, _ = frame.shape
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    background_pixels_hsv = cv.cvtColor(
        backgrounds[camera_i], cv.COLOR_BGR2HSV)
    foreground_hsv = cv.absdiff(frame_hsv, background_pixels_hsv)

    hue, saturation, value = best_masks[str(camera_i + 1)]
    best_mask = np.zeros((w, h), dtype=np.uint8)
    for x in range(foreground_hsv.shape[0]):
        for y in range(foreground_hsv.shape[1]):
            if foreground_hsv[x, y, 0] > hue and foreground_hsv[x, y, 1] > saturation and foreground_hsv[
                x, y, 2] > value:
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
        for y in np.arange(-depth // 2, depth // 2, 80):
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

        all_visible_voxels.append(visible_voxels)

        # COLORS
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        all_labels.append(labels)  # list of 4 lists, that has all labels for each visible voxel
    print(f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")
    # end of reconstructions
    return all_visible_voxels, all_labels


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

    print(f"Voxel Reconstruction completed in {time() - start_reconstruction} seconds.")

    return visible_voxels, labels


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
        x_3d, y_3d, z_3d = 0, 0, 0

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

        pixels_colors.append(image_points)

    # AT THIS POINT WE HAVE THE 2D PIXELS, THEIR CLUSTERING LABELS AND THEIR ORIGINAL COLORS

    # for each camera
    for i_camera, pixels_color in enumerate(pixels_colors):
        # for each person
        for pixel, label, color in pixels_color:
            # if the pixel is of the current person
            person_to_colors[i_camera][label].append(pixel)

    # AT THIS POINT WE HAVE THE 2D PIXELS OF EACH PERSON FOR EACH CAMERA

    best_person = [0, 1, 2, 3, 2, 3, 0, 1, 2, 0, 3, 1, 1, 3, 2, 0]
    # {0: (0, []), 1: (1, histograms[0][1]), 2: (2, histograms[0][2]),
    #  3: (3, histograms[0][3])},
    # {0: (2, histograms[1][0]), 1: (3, histograms[1][1]), 2: (0, histograms[1][2]),
    #  3: (1, histograms[1][3])},
    # {0: (2, histograms[2][0]), 1: (0, histograms[2][1]), 2: (3, histograms[2][2]),
    #  3: (1, histograms[2][3])},
    # {0: (1, histograms[3][0]), 1: (3, histograms[3][1]), 2: (2, histograms[3][2]),
    #  3: (0, histograms[3][3])}]

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

            person_pixels = frames[i_camera][cam_to_frame[i_camera]][mask == 255].tolist()

            person_training_data[best_person[i_camera * 4 + person]].append(person_pixels)

    for i_person in range(4):
        # flatten the data
        data = [item for sublist in person_training_data[i_person] for item in sublist]
        # train the gaussian mixture model
        gaussian_mixture_models[i_person].trainEM(np.array(data, dtype=np.float32))

    # AT THIS POINT WE HAVE THE HISTOGRAMS OF EACH PERSON FOR EACH CAMERA

    return gaussian_mixture_models


def start_online(gaussian_mixture_models):
    global lookup_table

    for n_frame in range(5, len(frames[0])):
        visible_voxels, labels = reconstruct_voxels(n_frame)
        best_people = []
        images_points = []
        for n_camera in range(4):
            image_points = []

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

            best_person = {0: (-1, -1), 1: (-1, -1), 2: (-1, -1), 3: (-1, -1)}
            for person in person_pixels:
                mask = np.zeros(frames[n_camera][n_frame].shape[:2], np.uint8)
                pixels = person_pixels[person]
                waist = np.max([x[1] for x in pixels]) - np.min([x[1] for x in pixels]) // 1.5
                for x, y in pixels:
                    if y < waist:
                        mask[y, x] = 255
                probabilities = []
                for i_gmm in range(4):
                    log_likelihood = 0
                    for pixel in frames[n_camera][n_frame][mask == 255].tolist():
                        log_likelihood += gaussian_mixture_models[i_gmm].predict2(np.array(pixel, dtype=np.float32))[0][
                            0]
                    probabilities.append(log_likelihood / len(frames[n_camera][n_frame][mask == 255].tolist()))

                best_person[person] = (np.argmax(probabilities), np.max(probabilities))
            best_people.append(best_person)

        print(best_people)

        right_labels = []
        for i_person in range(4):  # persona assoluta [label vera]
            tumadre = []
            for i_camera in range(4):  # camera
                tumadre.append(best_people[i_camera][i_person])
            # take the one with the second parameter (probability) higher

            sicuro = max(tumadre, key=lambda x: x[1])[0]
            right_labels.append(sicuro)

        for i_camera in range(4):
            frame_x = frames[i_camera][n_frame].copy()
            for pixel, label in images_points[i_camera]:
                a = np.where(right_labels == label)[0][0]
                cv.circle(frame_x, pixel, 2, labels_to_color[a], -1)

            show_image(frame_x, f"frame {n_frame} camera {i_camera}")


last_points = []

if __name__ == '__main__':
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

        # get average of background
        background = average_images(imgs_list_background)
        backgrounds.append(background)

    gmm_s = get_gaussian_mixture_models()

    start_online(gmm_s)
