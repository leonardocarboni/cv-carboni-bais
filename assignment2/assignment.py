import glm
import random
import numpy as np
import cv2 as cv
from time import time as millis

block_size = 1
block_size2 = 115
n_frame = 0
visible_voxels = []

with np.load('./data/cam1/masks.npz') as file:
        mask1 =file['masks']
with np.load('./data/cam2/masks.npz') as file:
        mask2 =file['masks']
with np.load('./data/cam3/masks.npz') as file:
        mask3 =file['masks']
with np.load('./data/cam4/masks.npz') as file:
        mask4 =file['masks']

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
    global n_frame
    start = millis()
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    # Reading the stored lookup table
    # voxel_positions, lookup_table = create_lookup_table(750, 1500, 1500)
    with np.load(f'./data/lookup_table.npz') as file:
        lookup_table = file['lookup_table']
    voxel_positions = np.array(create_cube(750, 1500, 1500), dtype = np.float32)
    print(millis()-start)
    print(voxel_positions.shape)
    # lookup_table = ''
    # with open(f'data/lookup_table.txt', 'r') as f:
    #     for i in f.readlines():
    #         lookup_table = i  # string
    # lookup_table = eval(lookup_table)

    # Visible voxels for each camera
      # for each camera
    
    
    if n_frame == 0:

        for vox in range(voxel_positions.shape[0]):
            flag = True
            for i in range(4):
                x_voxels, y_voxels, z_voxels, x, y = lookup_table[vox, :, i]
                x = int(x)
                y = int(y)
                # to show cube in front of mask
                if i == 0 and mask1[0][y, x] == 0:
                    flag = False
                if i == 1 and mask2[0][y, x] == 0:
                    flag = False
                if i == 2 and mask3[0][y, x] == 0:
                    flag = False
                if i == 3 and mask4[0][y, x] == 0:
                    flag = False
            if flag:
                visible_voxels.append([x_voxels/75, -z_voxels/75, y_voxels/75])
        print(millis()-start)
        # visible_voxels

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
        n_frame += 1
        return visible_voxels
    else:
        current_mask1 = mask1[n_frame]
        last_mask1 = mask1[n_frame-1]
        current_mask2 = mask2[n_frame]
        last_mask2 = mask2[n_frame-1]
        current_mask3 = mask3[n_frame]
        last_mask3 = mask3[n_frame-1]
        current_mask4 = mask4[n_frame]
        last_mask4 = mask4[n_frame-1]
        differences_1 = current_mask1 - last_mask1
        x = 0
        y = 0
        remove_pos1 = lookup_table[(lookup_table[:, :, 0][:, 3] == x) & (lookup_table[:, :, 0][:, 4] == y)][]
        print(remove_pos1)
        add_pos1 = np.where(visible_voxels[differences_1 == 255])
        differences_2 = current_mask2 - last_mask2
        remove_pos2 = np.where(visible_voxels[differences_2 == -255])
        add_pos2 = np.where(visible_voxels[differences_2 == 255])
        differences_3 = current_mask3 - last_mask3
        remove_pos3 = np.where(visible_voxels[differences_3 == -255])
        add_pos3 = np.where(visible_voxels[differences_3 == 255])
        differences_4 = current_mask4 - last_mask4
        remove_pos4 = np.where(visible_voxels[differences_4 == -255])
        add_pos4 = np.where(visible_voxels[differences_4 == 255])
        print(remove_pos1)
        # visible_voxels.remove(visible_voxels[]
        # visible_voxels.append(np.where(visible_voxels[differences_1 == 255 and differences_2 == 255 and differences_3 == 255 and differences_4 == 255]))
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


def create_cube(width, height, depth):
    # w, h, d = 750, 1500, 1500  # 7 chessboard squares (cols)
    cube = []
    for x in np.arange(0, width, 7):
        for y in np.arange(-depth//2, depth//2, 15):
            for z in np.arange(-height, height, 15):
                    # negative because TODO: find out why
                cube.append([x, y, z])
    return cube


def create_lookup_table(width, height, depth):
    "Creates file for the lookup table mapping 3D voxels to 2D image coordinates for each camera"

    voxel_positions = np.array(create_cube(width, height, depth), dtype=np.float32)
    lookup_table = np.zeros((voxel_positions.shape[0], 5, 4))
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
                lookup_table[i, :, camera_i-1] = [int(pos[0]), int(pos[1]), int(pos[2]), int(x), int(y)]  # store voxel (glm)

    np.savez('data/lookup_table', lookup_table = lookup_table)
    # with open(f'data/lookup_table.txt', 'w+') as f:
    #     f.write(str(lookup_table))
    return voxel_positions, lookup_table

# create_lookup_table()
# print(set_voxel_positions(1,2,3))