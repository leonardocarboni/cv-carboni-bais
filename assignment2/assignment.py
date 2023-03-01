import glm
import random
import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
from engine.config import config

block_size = 1
block_size2 = 115


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
    # Reading the stored lookup table
    voxel_positions, lookup_table = create_lookup_table()
    print(voxel_positions.shape)
    # lookup_table = ''
    # with open(f'data/lookup_table.txt', 'r') as f:
    #     for i in f.readlines():
    #         lookup_table = i  # string
    # lookup_table = eval(lookup_table)

    # Visible voxels for each camera
    visible_voxels = []
      # for each camera
    with np.load('./data/cam1/mask.npz') as file:
        mask1 =file['mask']
    with np.load('./data/cam2/mask.npz') as file:
        mask2 =file['mask']
    with np.load('./data/cam3/mask.npz') as file:
        mask3 =file['mask']
    with np.load('./data/cam4/mask.npz') as file:
        mask4 =file['mask']
    
    lookup_table[0]
    for i, x_voxels, y_voxels, z_voxels, x, y in lookup_table:  # for each 2D point  that corresponds to a 3D voxel
        i = int(i)
        x = int(x)
        y = int(y)
        flag = True
          # to show cube in front of mask
        if i == 0 and mask1[y, x] == 0:
            flag = False
        if i == 1 and mask2[y, x] == 0:
            flag = False
        if i == 2 and mask3[y, x] == 0:
            flag = False
        if i == 3 and mask4[y, x] == 0:
            flag = False
        if flag:
            visible_voxels.append([x_voxels/115, -z_voxels/115, y_voxels/115])
    visible_voxels

    # mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) 
    # for x, y in lookup_table.values():
    #     for camera_i in range(1, 5):
    #             cv.circle(mask2, (x, y), 5, (255, 0, 0), -1)
    # cv.imshow("cubo", mask2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # reconstruction = []
    # # voxel_positions = create_cube()  # x, y, z
    # for pos in voxel_positions:  # for each 3D voxel point of the cube
    #     x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
    #     # if the 3D point is foreground for all cameras
    #     if (x, y, z) in visible_voxels[0] and (x, y, z) in visible_voxels[1] and (x, y, z) in visible_voxels[2] and (x, y, z) in visible_voxels[3]:
            
            
    #         reconstruction.append([x/115, -z/115, y/115])
    return visible_voxels


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


def create_cube():
    w, h, d = 750, 1500, 1500  # 7 chessboard squares (cols)
    cube = []
    for x in np.arange(0, w, 50):
        for y in np.arange(-d//2, d//2, 50):
            for z in np.arange(-h, h, 10):
                    # negative because TODO: find out why
                cube.append([x, y, z])
    return cube


def create_lookup_table():
    "Creates file for the lookup table mapping 3D voxels to 2D image coordinates for each camera"

    voxel_positions = np.array(create_cube(), dtype=np.float32)
    lookup_table = np.zeros((voxel_positions.shape[0], 6))
    for camera_i in range(1, 5):
        s = cv.FileStorage(
            f"data/cam{camera_i}/config.xml", cv.FileStorage_READ)
        camera_matrix = s.getNode('camera_matrix').mat()
        dist_coeffs = s.getNode('dist_coeffs').mat()
        tvec_extr = s.getNode('tvec_extr').mat()
        rvec_extr = s.getNode('rvec_extr').mat()
        s.release()
        for i, pos in enumerate(voxel_positions):  # for each 3D point of the voxel cube
            # project the point in the 2D image plane for this camera
            imgpoint, _ = cv.projectPoints(
                pos, rvec_extr, tvec_extr, camera_matrix, dist_coeffs)
            imgpoint = imgpoint.ravel()
            x = imgpoint[0]
            y = imgpoint[1]
            if x >= 0 and x <= 644 and y >= 0 and y < 486:
                lookup_table[i, :] = [camera_i-1, int(pos[0]), int(pos[1]), int(pos[2]), int(x), int(y)]  # store voxel (glm)

    np.savez('data/lookup_table', lookup_table = lookup_table)
    # with open(f'data/lookup_table.txt', 'w+') as f:
    #     f.write(str(lookup_table))
    return voxel_positions, lookup_table

# create_lookup_table()
# print(set_voxel_positions(1,2,3))