import glm
import random
import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
from engine.config import config

block_size = 115


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -
                        block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    seta = [(4, 6, 13), (9, 12, 6)]
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                data.append([x*block_size - width/2, y *
                             block_size, z*block_size - depth/2])
    return data


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
    positions = positions.squeeze() / 115
    # converting from opencv space to glm space
    # swap y and z
    positions[:, [1, 2]] = positions[:, [2, 1]]
    # abs y positions
    positions[:, 1] = np.abs(positions[:, 1])

    return positions


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


def create_lookup_table():
    "Creates file for the lookup table mapping 3D voxels to 2D image coordinates for each camera"
    w, h, d = 7, 14, 14 # 7 chessboard squares (cols)
    # only center
    # voxel_positions = [(0,0,0)]
    # cube
    voxel_positions = []
    for x in range(w):
        for y in range(d):
            for z in range(h):
                voxel_positions.append(
                    [x * block_size, y * block_size, z * block_size])
    voxel_positions = np.array(voxel_positions, dtype=np.float32)
    lookup_table = {}
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        camera_matrix = s.getNode('camera_matrix').mat()
        dist_coeffs = s.getNode('dist_coeffs').mat()
        tvec_extr = s.getNode('tvec_extr').mat()
        rvec_extr = s.getNode('rvec_extr').mat()
        s.release()
        for pos in voxel_positions:  # for each 3D point of the voxel cube
            # project the point in the 2D image plane for this camera
            imgpoint, _ = cv.projectPoints(
                pos, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            lookup_table[(int(imgpoint.ravel()[0]), int(imgpoint.ravel()[1]), camera_i)] = (
                int(pos[0]), int(pos[1]), int(pos[2]))  # store voxel

    with open(f'data/lookup_table.txt', 'w+') as f:
        f.write(str(lookup_table))


def voxel_reconstruction():
    " returns voxel reconstruction of the horseman"

    # Reading the stored lookup table
    lookup_table = ''
    with open(f'data/lookup_table.txt', 'r') as f:
        for i in f.readlines():
            lookup_table = i  # string
    lookup_table = eval(lookup_table)

    # Visible voxels for each camera
    visible_voxels = {}
    for camera_i in range(1, 5):  # for each camera
        cube = []
        with np.load(f'./data/cam{camera_i}/mask.npz') as file:
            mask = file['mask']
        for x, y, i in lookup_table:  # for each 2D point  that corresponds to a 3D voxel
            if camera_i == i:  # only 2D points for a specific camera plane
                cube.append([x, y])
                if mask[x, y] != 0: # if it is foreground TODO: never enters here for camera 1
                    x_voxels, y_voxels, z_voxels = lookup_table[(x, y, camera_i)] # extract corresponding 3D voxel for that camera
                    visible_voxels[(x_voxels, y_voxels, z_voxels, camera_i-1)] = True # this voxel is foreground for the camera

        mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        for point in cube:
            cv.circle(mask2, tuple(point), 5, (255, 0, 0), -1)
        cv.imshow("cubo", mask2)
        cv.waitKey(0)
        cv.destroyAllWindows()
    reconstruction = []
    voxel_positions = np.array(set_voxel_positions(
        config['world_width'], config['world_height'], config['world_depth']), dtype=np.float32)
    print(visible_voxels)
    for pos in voxel_positions:  # for each 3D voxel point of the cube
        # pos = pos / 115
        # temp = pos[1]
        # pos[1] = pos[2]
        # pos[2] = temp
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        # if the 3D point is foreground for all cameras
        if (x, y, z, 0) in visible_voxels and (x, y, z, 1) in visible_voxels and (x, y, z, 2) in visible_voxels and (x, y, z, 3) in visible_voxels:
            reconstruction.append([x, y, z])
    return reconstruction


create_lookup_table()
print(voxel_reconstruction())
