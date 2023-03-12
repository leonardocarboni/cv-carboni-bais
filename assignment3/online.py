import os
from time import time

import matplotlib.pyplot as plt

from utils import *

# TODO: Voxel space più grosso
# TODO: Finire implementazione lables corrette
# TODO:

lookup_table = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
labels_to_color = {0: (255, 0, 0), 1: (0, 255, 0),
                   2: (0, 0, 255), 3: (255, 0, 255)}

cam_to_frame = {0: 10, 1: 0, 2: 41, 3: 52}
frames = []
backgrounds = []

show = True

# loading parameters of cam2 for color model
camera_matrixes = []
dist_coeffs = []
rvecs_extr = []
tvecs_extr = []

masks_all_frames = []
for i in range(4):
    with np.load(f'./data/cam{i + 1}/masks.npz') as file:
        masks_all_frames.append(file['masks'])

for i in range(4):
    s = cv.FileStorage(
        f"./data/cam{i + 1}/config.xml", cv.FileStorage_READ)
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


def get_histogram(colors):
    channel_1 = [x[0] for x in colors]
    channel_2 = [x[1] for x in colors]
    channel_3 = [x[2] for x in colors]
    data = [channel_1, channel_2, channel_3]
    plt.hist(channel_1, bins=64, color="b")
    plt.hist(channel_2, bins=64, color="g")
    plt.hist(channel_3, bins=64, color="r")
    plt.show()
    return data


def get_color_model():
    global lookup_table
    MGGs = [{0: None, 1: None, 2: None, 3: None}] * 4
    # global frames

    all_visible_voxels = []
    all_labels = []
    start_reconstruction = time()
    for i in range(4):  # 4 reconstructions for the 4 frames nedded for the color models
        visible_voxels = []
        for vox in range(voxel_positions.shape[0]):  # for each voxel id
            flag = True  # the voxel is foreground for all cameras (flag)
            for camera_i in range(4):  # for each camera
                # extract voxel 3D and 2D coordinates for that camera
                x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, camera_i]
                x = int(x)
                y = int(y)
                # check if the pixel is foreground for all cameras
                mask = masks_all_frames[camera_i][cam_to_frame[i]]
                if mask[y, x] == 0:
                    flag = False

            if flag:  # if it is foreground for all cameras
                # adapt to glm format, scale and add to reconstruction
                visible_voxels.append(
                    [x_voxels / 75, -z_voxels / 75, y_voxels / 75])
        print(f"time to reconstruct all: {time() - start_reconstruction}")
        all_visible_voxels.append(visible_voxels)

        # COLORS
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
        all_labels.append(labels)  # list of 4 lists, that has all labels for each visible voxel
    # end of reconstructions

    pixels_colors = []  # list of length 4, for each camera its 2d visible pixels, its clustering label and its original color
    person_to_colors = [{0: [], 1: [], 2: [], 3: []}] * 4
    histograms = [{0: [], 1: [], 2: [], 3: []}] * 4
    for camera_i, vis_voxs in enumerate(all_visible_voxels):
        imgpoints = []
        chosen_frame = cam_to_frame[camera_i]
        for i_label, vox in enumerate(vis_voxs):
            x_3D, y_3D, z_3D = (
                int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
            img_points, _ = cv.projectPoints(np.array(
                [x_3D, y_3D, z_3D], dtype=np.float32), rvecs_extr[camera_i], tvecs_extr[camera_i],
                camera_matrixes[camera_i], dist_coeffs[camera_i])
            x, y = (int(img_points.ravel()[0]), int(
                img_points.ravel()[1]))  # x is < 644, y is < 486
            # tuple (2d pixel, clustering label, original color)
            imgpoints.append(((x, y), all_labels[camera_i][i_label][0],
                              frames[camera_i][chosen_frame][y, x]))  # 2d coords,clustering label, original color
        pixels_colors.append(imgpoints)
    # print(pixels_colors[0][0], pixels_colors[0][1], pixels_colors[0][2])
    # print("-----------------")
    # print(pixels_colors[1][0], pixels_colors[1][1], pixels_colors[1][2])

    print(person_to_colors)
    print("-----------")
    for camera_i, infos in enumerate(pixels_colors):
        chosen_frame = cam_to_frame[camera_i]
        frame_copy = frames[camera_i][chosen_frame].copy()
        for point_i, pc in enumerate(infos):
            # if (camera_i == 0 or camera_i == 1) and pc[1] == 0:
            #     print(camera_i , pc)
            cv.circle(frame_copy, pc[0], 2, labels_to_color[pc[1]], 2)
            person_to_colors[0][pc[1]].append(pc[2].tolist())
            if point_i > 1000 and point_i < 1020:
                print('--------------')
                print(person_to_colors[0][0])
                print(person_to_colors[1][0])
        # print(person_to_colors)
        # print(person_to_colors[0][0])
        # print("-----------")
        # print(person_to_colors[1][0])
        for person in person_to_colors[camera_i]:
            print("person ", person)
            print("camera ", camera_i)
            histograms[camera_i][person] = get_histogram(person_to_colors[camera_i][person])  # store a 3XN matrix
            # print(histograms[camera_i][person])
            # print("------------------------------")
        #     MGGs[camera_i][person] = cv.ml.EM_create()
        #     MGGs[camera_i][person].setClustersNumber(3)
        #     MGGs[camera_i][person].trainEM(
        #         np.array(person_to_colors[camera_i][person], dtype=np.float32))

        # for person in person_to_colors[camera_i]:
        #     logliks = np.zeros(4)
        #     for pixel in person_to_colors[camera_i][person]:
        #         logliks[0] += MGGs[camera_i][0].predict2(np.array(pixel,
        #                                             dtype=np.float32))[0][0]
        #         logliks[1] += MGGs[camera_i][1].predict2(np.array(pixel,
        #                                             dtype=np.float32))[0][0]
        #         logliks[2] += MGGs[camera_i][2].predict2(np.array(pixel,
        #                                             dtype=np.float32))[0][0]
        #         logliks[3] += MGGs[camera_i][3].predict2(np.array(pixel,
        #                                             dtype=np.float32))[0][0]
        #     print(person, np.argmax(logliks))

        show_image(frame_copy, "silhouttes")
    for camera_hist in histograms:
        for person_hist in camera_hist:
            # print(histograms[0][person_hist][0])
            # print("--------------")
            # print(histograms[1][person_hist][0])
            channel1_distance = np.corrcoef(histograms[0][person_hist][0], histograms[1][person_hist][0])
            channel2_distance = np.corrcoef(histograms[0][person_hist][1], histograms[1][person_hist][1])
            channel3_distance = np.corrcoef(histograms[0][person_hist][2], histograms[1][person_hist][2])
            print(channel1_distance)
            print(channel2_distance)
            print(channel3_distance)
            print(channel3_distance + channel1_distance + channel2_distance)
    return visible_voxels, MGGs


def reconstruct_voxels():
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


def get_color_model_2():
    global lookup_table
    mgg_list = [{0: None, 1: None, 2: None, 3: None}] * 4
    person_to_colors = [{0: [], 1: [], 2: [], 3: []}] * 4
    histograms = [{0: [], 1: [], 2: [], 3: []}] * 4

    all_visible_voxels, all_labels = reconstruct_voxels()

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
                [x_3d, y_3d, z_3d], dtype=np.float32), rvecs_extr[i_camera], tvecs_extr[i_camera],
                camera_matrixes[i_camera], dist_coeffs[i_camera])
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
        for i_person in range(4):
            # for each pixel
            for pixel, label, color in pixels_color:
                # if the pixel is of the current person
                if label == i_person:
                    person_to_colors[i_camera][i_person].append(pixel)

    # AT THIS POINT WE HAVE THE 2D PIXELS OF EACH PERSON FOR EACH CAMERA

    # for each camera
    for i_camera, cameras in enumerate(person_to_colors):
        # for each person
        for i_person, person in enumerate(cameras):
            # calculate the histogram
            mask = np.zeros(frames[i_camera][cam_to_frame[i_camera]].shape[:2], np.uint8)
            mask[person] = 255
            # parameters: image, channels, mask, histSize, ranges
            hist = cv.calcHist([frames[i_camera][cam_to_frame[i_camera]]], [0, 1, 2], mask, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])

            # normalize the histogram
            hist = cv.normalize(hist, hist).flatten()

            plt.plot(hist)
            plt.title("Normalized Histogram")
            plt.show()

            histograms[i_camera][i_person] = hist

    for i_camera, pixels_color in enumerate(pixels_colors):
        frame_copy = frames[i_camera][cam_to_frame[i_camera]].copy()
        for pixel, label, color in pixels_color:
            cv.circle(frame_copy, pixel, 3, labels_to_color[label], -1)
        show_image(frame_copy, "frame")


def start_online(MGGs):
    # for each frame we considered
    for n_frame in range(0, len(frames[0]), 10):

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
                mask = masks_all_frames[camera_i][n_frame]
                if mask[y, x] == 0:
                    flag = False

            if flag:  # if it is foreground for all cameras
                # adapt to glm format, scale and add to reconstruction
                visible_voxels.append(
                    [x_voxels / 75, -z_voxels / 75, y_voxels / 75])
        print(f"time to reconstruct all: {time() - start_reconstruction}")

        # COLORS
        voxels_to_cluster = np.array([[x[0], x[2]]
                                      for x in visible_voxels], dtype=np.float32)
        compactness, labels, centers = cv.kmeans(
            voxels_to_cluster, 4, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

        pixels_colors = []
        color_model = [{0: [], 1: [], 2: [], 3: []}] * 4
        correct_labels = np.zeros((4, 4), dtype=tuple)
        for camera_i in range(4):
            imgpoints = []
            for i_label, vox in enumerate(visible_voxels):
                x_3D, y_3D, z_3D = (
                    int(vox[0] * 75), int(vox[2] * 75), int(-vox[1] * 75))
                img_points, _ = cv.projectPoints(np.array(
                    [x_3D, y_3D, z_3D], dtype=np.float32), rvecs_extr[camera_i], tvecs_extr[camera_i],
                    camera_matrixes[camera_i], dist_coeffs[camera_i])
                x, y = (int(img_points.ravel()[0]), int(
                    img_points.ravel()[1]))  # x is < 644, y is < 486
                # tuple (2d pixel, clustering label, original color)
                imgpoints.append(
                    ((x, y), labels[i_label][0], frames[camera_i][n_frame][y, x]))
            pixels_colors.append(imgpoints)
            for pc in pixels_colors[camera_i]:
                color_model[camera_i][pc[1]].append(pc[2].tolist())
            for person in color_model[camera_i]:
                logliks = np.zeros(4)
                for pixel in color_model[camera_i][person]:
                    logliks[0] += MGGs[camera_i][0].predict2(np.array(pixel,
                                                                      dtype=np.float32))[0][0]
                    logliks[1] += MGGs[camera_i][1].predict2(np.array(pixel,
                                                                      dtype=np.float32))[0][0]
                    logliks[2] += MGGs[camera_i][2].predict2(np.array(pixel,
                                                                      dtype=np.float32))[0][0]
                    logliks[3] += MGGs[camera_i][3].predict2(np.array(pixel,
                                                                      dtype=np.float32))[0][0]
                print(logliks)
                correct_labels[camera_i][person] = (np.argmax(logliks), np.max(
                    logliks))  # store the best score for each person and for each camera (16 guesses)

        final_labels = np.zeros(4)
        print("correct_label", correct_labels)
        for i in range(4):
            all_logliks = [x[1] for x in correct_labels[:, i]]  # 4 different scores for person i, one for each camera
            best_guess = np.argmax(all_logliks)  # which camera had the most confident guess
            print("best guess: ", best_guess)
            # print(np.where(correct_labels[best_guess] == correct_labels[best_guess, i]))
            final_labels[i] = correct_labels[best_guess, i][0]
        print(final_labels)

        frame_copy = frames[best_guess][n_frame].copy()
        frame_copy2 = frames[best_guess][n_frame].copy()
        for pc in pixels_colors[best_guess]:
            current_label = pc[1]
            offline_label = final_labels[current_label]
            color = labels_to_color[offline_label]
            online_color = labels_to_color[int(current_label)]
            cv.circle(frame_copy, pc[0], 2, color, 2)
            cv.circle(frame_copy2, pc[0], 2, online_color, 2)
        show_image(frame_copy, "silhouttes")
        show_image(frame_copy2, "silhouttes")
        clusters_to_tracking(centers, final_labels)
    return


last_points = []


def clusters_to_tracking(centers, labels):
    global plane
    global last_points
    older = last_points

    last_points = []
    for i in range(len(centers)):
        x, y = centers[i]
        x = int(x) + 64
        y = int(y) + 64
        last_points.append((x, y))
        plane[y, x] = labels_to_color[labels[i]]
        if len(older) != 0:
            cv.line(plane, (x, y), older[i], color=labels_to_color[labels[i]])
    show_image(plane, "silhouttes")


# execution
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

    # get average of baground
    background = average_images(imgs_list_background)
    backgrounds.append(background)

# MGGs = get_color_model()[1]
print(get_color_model_2())
# start_online(MGGs)

# TODO: build 16 models, and calcualte a golbal loglik to choose the best guess for each pixel corresponding to each person.
